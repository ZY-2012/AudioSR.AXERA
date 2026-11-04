#pragma once
// AudioSR host DSP: resample, STFT/ISTFT, 256-bin log-mel, Butterworth low-pass.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <vector>
#include "third_party/kissfft/kiss_fftr.h"

namespace audiosr {
constexpr int SR = 48000, NFFT = 2048, HOP = 480, N_MEL = 256, BINS = NFFT / 2 + 1;
constexpr int PAD = (NFFT - HOP) / 2;
constexpr double PI = 3.14159265358979323846;

inline std::vector<float> hann(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i) w[i] = 0.5f - 0.5f * (float)std::cos(2.0 * PI * i / n);
    return w;
}
class FFT {
public:
    explicit FFT(int n) : n_(n) { f_ = kiss_fftr_alloc(n, 0, nullptr, nullptr); i_ = kiss_fftr_alloc(n, 1, nullptr, nullptr); }
    ~FFT() { free(f_); free(i_); }
    void forward(const float* x, std::vector<std::complex<float>>& out) {
        out.resize(n_ / 2 + 1); kiss_fftr(f_, x, reinterpret_cast<kiss_fft_cpx*>(out.data()));
    }
    std::vector<float> inverse(const std::vector<std::complex<float>>& s) {
        std::vector<float> y(n_); kiss_fftri(i_, reinterpret_cast<const kiss_fft_cpx*>(s.data()), y.data());
        for (auto& v : y) v /= (float)n_; return y;
    }
private:
    int n_; kiss_fftr_cfg f_ = nullptr, i_ = nullptr;
};

inline std::vector<float> reflect_pad(const std::vector<float>& x, int left, int right) {
    const int n = (int)x.size(); std::vector<float> o(n + left + right);
    for (int i = 0; i < left; ++i) o[i] = x[std::min(left - i, n - 1)];
    std::copy(x.begin(), x.end(), o.begin() + left);
    for (int i = 0; i < right; ++i) o[left + n + i] = x[std::max(0, n - 2 - i)];
    return o;
}

// STFT: [frames, 1025], hann periodic, reflect pad.
inline std::vector<std::complex<float>> stft(const std::vector<float>& x) {
    auto xp = reflect_pad(x, PAD, PAD);
    const int frames = (int)(xp.size() - NFFT) / HOP + 1;
    const auto win = hann(NFFT);
    std::vector<std::complex<float>> spec((size_t)frames * BINS);
    FFT fft(NFFT); std::vector<float> f(NFFT); std::vector<std::complex<float>> s;
    for (int t = 0; t < frames; ++t) {
        for (int i = 0; i < NFFT; ++i) f[i] = xp[t * HOP + i] * win[i];
        fft.forward(f.data(), s);
        std::copy(s.begin(), s.end(), spec.begin() + (size_t)t * BINS);
    }
    return spec;
}

inline std::vector<float> istft(const std::vector<std::complex<float>>& spec) {
    const int frames = (int)spec.size() / BINS;
    const int total = (frames - 1) * HOP + NFFT;
    const auto win = hann(NFFT);
    std::vector<float> y(total, 0.f), env(total, 0.f), all((size_t)frames * NFFT);
    FFT fft(NFFT);
#pragma omp parallel for schedule(static)
    for (int t = 0; t < frames; ++t) {
        std::vector<std::complex<float>> s(spec.begin() + (size_t)t * BINS, spec.begin() + (size_t)(t + 1) * BINS);
        auto f = fft.inverse(s);
        std::copy(f.begin(), f.end(), all.begin() + (size_t)t * NFFT);
    }
    for (int t = 0; t < frames; ++t) {
        const int off = t * HOP; const float* f = all.data() + (size_t)t * NFFT;
        for (int i = 0; i < NFFT; ++i) { y[off+i] += f[i] * win[i]; env[off+i] += win[i] * win[i]; }
    }
    y = std::vector<float>(y.begin() + PAD, y.end() - PAD);
    env = std::vector<float>(env.begin() + PAD, env.end() - PAD);
    for (size_t i = 0; i < y.size(); ++i) y[i] /= std::max(env[i], 1e-12f);
    return y;
}

inline std::vector<float> load_mel_filter(const std::string& path) {
    std::vector<float> fb((size_t)N_MEL * BINS); std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open mel filter %s\n", path.c_str()); return {}; }
    f.read((char*)fb.data(), (std::streamsize)(fb.size() * sizeof(float))); return fb;
}

// [T, 256] row-major; log(max(mel,1e-5))
inline std::vector<float> logmel(const std::vector<std::complex<float>>& spec, const std::vector<float>& fb) {
    const int T = (int)spec.size() / BINS; std::vector<float> out((size_t)T * N_MEL);
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t) {
        for (int m = 0; m < N_MEL; ++m) {
            float acc = 0.f; const float* row = fb.data() + (size_t)m * BINS;
            for (int k = 0; k < BINS; ++k) acc += std::abs(spec[(size_t)t*BINS+k]) * row[k];
            out[(size_t)t*N_MEL+m] = std::log(std::max(acc, 1e-5f));
        }
    }
    return out;
}

inline float cutoff_hz(const std::vector<std::complex<float>>& spec) {
    const int T = (int)spec.size() / BINS; std::vector<double> energy(BINS, 0.0);
    for (int t = 0; t < T; ++t) for (int k = 0; k < BINS; ++k) energy[k] += std::abs(spec[(size_t)t*BINS+k]);
    const double total = std::accumulate(energy.begin(), energy.end(), 0.0);
    int bin = 0; double acc = 0.0;
    for (int k = 0; k < BINS; ++k) { acc += energy[k]; if (acc >= total * 0.985) { bin = k; break; } }
    float hz = (float)bin / 1024.0f * 24000.0f; return hz < 1000.f ? 24000.f : hz;
}

// 8th-order Butterworth, 4 RBJ biquads, forward only (stable and deterministic).
inline std::vector<float> lowpass(const std::vector<float>& x, float fc) {
    if (fc >= SR * 0.499f) return x;
    const float w0 = 2.f * (float)PI * fc / SR, cw = std::cos(w0), sw = std::sin(w0);
    const float q[4] = {0.50979558f, 0.60134489f, 0.89997622f, 2.56291545f};
    std::vector<float> y = x;
    for (int s = 0; s < 4; ++s) {
        const float alpha = sw / (2.f * q[s]);
        const float b0 = (1.f - cw) * 0.5f, b1 = 1.f - cw, b2 = b0;
        const float a0 = 1.f + alpha, a1 = -2.f * cw, a2 = 1.f - alpha;
        const float nb0=b0/a0, nb1=b1/a0, nb2=b2/a0, na1=a1/a0, na2=a2/a0;
        float z1=0.f, z2=0.f;
        for (float& v : y) { const float o = nb0*v + z1; z1 = nb1*v - na1*o + z2; z2 = nb2*v - na2*o; v = o; }
    }
    return y;
}

inline std::vector<float> resample(const std::vector<float>& x, int orig, int target) {
    if (orig == target) return x;
    int g = std::gcd(orig, target); const int o = orig / g, n = target / g;
    const double rolloff = 0.99, width_scale = 6.0, base = std::min(o, n) * rolloff;
    const int width = (int)std::ceil(width_scale * o / base), ksize = 2 * width + o;
    std::vector<float> kernel((size_t)n * ksize);
    for (int p = 0; p < n; ++p) for (int k = 0; k < ksize; ++k) {
        double t = ((double)(k - width) / o) - (double)p / n;
        t = std::max(-width_scale, std::min(width_scale, t * base));
        const double win = std::pow(std::cos(t * PI / width_scale / 2.0), 2.0);
        const double sinc = std::abs(t) < 1e-12 ? 1.0 : std::sin(PI * t) / (PI * t);
        kernel[(size_t)p*ksize+k] = (float)(sinc * win * base / o);
    }
    std::vector<float> padded(width + x.size() + width + o, 0.f); std::copy(x.begin(), x.end(), padded.begin() + width);
    const int nout = ((int)padded.size() - ksize) / o + 1; std::vector<float> y((size_t)nout * n);
    for (int p = 0; p < n; ++p) for (int j = 0; j < nout; ++j) {
        float acc = 0.f; const int off = j * o; const float* row = kernel.data() + (size_t)p * ksize;
        for (int k = 0; k < ksize; ++k) acc += padded[off+k] * row[k]; y[(size_t)j*n+p] = acc;
    }
    y.resize(std::min<int>(y.size(), (int)std::ceil((double)n * x.size() / o))); return y;
}

inline float rms(const std::vector<float>& x) {
    double s = 0; for (float v : x) s += (double)v*v; return x.empty() ? 0.f : (float)std::sqrt(s/x.size());
}
}  // namespace audiosr
