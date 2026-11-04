#pragma once
// AudioSR speech super-resolution: full AX650 chain.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include "ddim_assets.h"
#include "dsp.hpp"
#include "model_runner.hpp"

namespace audiosr {
constexpr int LAT_C = 16, LAT_T = 64, LAT_F = 32;
constexpr float SCALE = 0.3342401385307312f;
constexpr float UNCOND = -11.4981f;
constexpr int MEL_T = 512;

class AudioSR {
public:
    bool load(const std::string& dir, const std::string& mel_filter) {
        auto P = [&](const char* n) { return dir + "/" + n; };
        if (!enc_.load(P("audiosr_vae_encoder_nhwc.axmodel"))) return false;
        if (!unet_.load(P("audiosr_unet.axmodel"))) return false;
        if (!dec_.load(P("audiosr_vae_decoder.axmodel"))) return false;
        const char* voc[] = {"voc_pre_fp32.axmodel", "voc_up0a_mix.axmodel", "voc_up0b_mix.axmodel",
            "voc_up1a_mix.axmodel", "voc_up1b_mix.axmodel", "voc_split_nhwc_up2_mix.axmodel",
            "voc_split_nhwc_up3_mix.axmodel", "voc_split_nhwc_up4_mix.axmodel",
            "voc_split_nhwc_post_fp32.axmodel"};
        for (int i = 0; i < 9; ++i) if (!voc_[i].load(P(voc[i]))) return false;
        fb_ = load_mel_filter(mel_filter);
        return !fb_.empty();
    }

    std::vector<float> enhance(const std::vector<float>& wav, int sr) {
        auto x = resample(wav, sr, SR);
        // normalize + 5.12 s pad / truncate
        double mean = 0; for (float v : x) mean += v; mean /= std::max<size_t>(1, x.size());
        float peak = 1e-8f; for (float v : x) peak = std::max(peak, std::abs((float)(v - mean)));
        for (float& v : x) v = (float)((v - mean) / peak * 0.5);
        const int n512 = SR * 512 / 100;  // 245760
        x.resize(n512, 0.f);
        auto spec = stft(x);
        auto mel = logmel(spec, fb_);
        float cutoff = cutoff_hz(spec);
        auto low = lowpass(x, cutoff);
        auto low_mel = logmel(stft(low), fb_);

        std::mt19937 gen(42); std::normal_distribution<float> nd(0.f, 1.f);
        auto noise = [&](size_t n) { std::vector<float> z(n); for (auto& v : z) v = nd(gen); return z; };
        auto zc = encode(mel, noise(LAT_C*LAT_T*LAT_F));
        auto zl = encode(low_mel, noise(LAT_C*LAT_T*LAT_F));

        std::vector<float> latent = noise(LAT_C*LAT_T*LAT_F);
        std::vector<float> uncond(LAT_C*LAT_T*LAT_F, UNCOND);
        for (int i = 0; i < DDIM_STEPS; ++i) {
            const int step = (int)timesteps[DDIM_STEPS - 1 - i];
            std::vector<float> vu, vc;
            if (!unet_run(latent, uncond, step, vu) || !unet_run(latent, zc, step, vc)) return {};
            std::vector<float> v(vu.size());
            for (size_t j = 0; j < v.size(); ++j) v[j] = vu[j] + 3.5f * (vc[j] - vu[j]);
            const float sc = sqrt_alphas_cumprod[step], so = sqrt_one_minus_alphas_cumprod[step];
            const int idx = DDIM_STEPS - 1 - i;
            const float at = alphas[idx], ap = alphas_prev[idx], sig = sigmas[idx];
            auto eps = noise(latent.size());
            for (size_t j = 0; j < latent.size(); ++j) {
                const float et = sc * v[j] + so * latent[j];
                const float px0 = sc * latent[j] - so * v[j];
                latent[j] = std::sqrt(ap) * px0 + std::sqrt(std::max(0.f, 1.f - ap - sig*sig)) * et + sig * eps[j];
            }
        }
        std::vector<float> dm;
        if (!dec_.set_input("z", latent.data(), latent.size()) || !dec_.run() ||
            !dec_.get_output(0, dm, MEL_T * N_MEL)) return {};
        // mel_replace_ops: copy low-pass mel for the first cutoff frames
        int cb = 0; double e = 0; for (int t = 0; t < MEL_T; ++t) e += std::exp(low_mel[(size_t)t*N_MEL]);
        const double limit = e * 0.985; double acc = 0;
        for (int t = 0; t < MEL_T; ++t) { acc += std::exp(low_mel[(size_t)t*N_MEL]); if (acc >= limit) { cb = t + 1; break; } }
        for (int t = 0; t < cb; ++t)
            std::copy(low_mel.begin() + (size_t)t*N_MEL, low_mel.begin() + (size_t)(t+1)*N_MEL, dm.begin() + (size_t)t*N_MEL);

        std::vector<float> v = dm;  // [512,256] -> pre
        if (!run_voc(v)) return {};
        // v now wav [245776]
        if (std::getenv("AUDIOSR_RAW")) { v.resize(low.size()); return v; }
        auto y = postprocess(v, low);
        return y;
    }

private:
    AxModel enc_, unet_, dec_;
    std::array<AxModel, 9> voc_;
    std::vector<float> fb_;

    std::vector<float> encode(const std::vector<float>& mel, const std::vector<float>& eps) {
        std::vector<float> o;
        if (!enc_.set_input("mel", mel.data(), mel.size()) || !enc_.run()) return {};
        std::vector<float> mu, lv;
        if (!enc_.get_output(0, mu, LAT_C*LAT_T*LAT_F) || !enc_.get_output(1, lv, LAT_C*LAT_T*LAT_F)) return {};
        for (size_t i = 0; i < mu.size(); ++i) mu[i] = (mu[i] + eps[i] * std::exp(0.5f * lv[i])) * SCALE;
        return mu;
    }
    bool unet_run(const std::vector<float>& x, const std::vector<float>& c, int t, std::vector<float>& out) {
        std::vector<float> xc(2*LAT_C*LAT_T*LAT_F);
        std::copy(x.begin(), x.end(), xc.begin()); std::copy(c.begin(), c.end(), xc.begin()+LAT_C*LAT_T*LAT_F);
        const float tt = (float)t;
        if (!unet_.set_input("x", xc.data(), xc.size()) || !unet_.set_input("t", &tt, 1) || !unet_.run()) return false;
        return unet_.get_output(0, out, LAT_C*LAT_T*LAT_F);
    }
    bool run_voc(std::vector<float>& x) {
        const char* in[] = {"mel","x_up0a","x_up0b","x_up1a","x_up1b","x_up2","x_up3","x_up4","x_post"};
        const size_t outn[] = {512*1536, 3072*768, 3072*768, 15361*384, 15361*384, 61444*192, 122888*96, 245776*48, 245776};
        for (int i = 0; i < 9; ++i) {
            std::vector<float> o;
            if (!voc_[i].set_input(in[i], x.data(), x.size()) || !voc_[i].run() || !voc_[i].get_output(0, o, outn[i])) return false;
            x.swap(o);
        }
        return true;
    }
    static std::vector<float> postprocess(const std::vector<float>& out, const std::vector<float>& low) {
        auto gt = stft(low), sp = stft(out);
        const float fc = cutoff_hz(gt); const int kb = std::min(BINS-1, (int)(fc / 24000.f * 1024.f));
        for (size_t t = 0; t < sp.size()/BINS; ++t)
            for (int k = 0; k < kb; ++k) sp[t*BINS+k] = gt[t*BINS+k];
        auto y = istft(sp); y.resize(low.size());
        float p = 1e-8f; for (float v : y) p = std::max(p, std::abs(v));
        for (float& v : y) v = v * 0.5f / p;
        double m = 0; for (float v : y) m += v; m /= std::max<size_t>(1, y.size());
        for (float& v : y) v -= (float)m;
        return y;
    }
};
}  // namespace audiosr
