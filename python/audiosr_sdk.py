"""AudioSR speech super-resolution on AX650 (axengine).

The SDK accepts 8/16/24/48 kHz mono/stereo WAV and produces 48 kHz output.
It implements the upstream AudioSR preprocessing (5.12 s padding, log-mel,
low-pass conditioning, DDIM-10) in NumPy/librosa, then runs the AX650
axmodels with axengine.
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt
import librosa

SR = 48000
N_FFT = 2048
HOP = 480
WIN = 2048
N_MEL = 256
FMIN = 20
FMAX = 24000
SEG = 5.12
SCALE = 0.3342401385307312
UNCOND = -11.4981


def _normalize(y):
    y = y - np.mean(y)
    return (y / (np.max(np.abs(y)) + 1e-8) * 0.5).astype(np.float32)


def _pad_wav(y, target):
    out = np.zeros(target, dtype=np.float32)
    out[:min(target, len(y))] = y[:target]
    return out


def _mel(y):
    # same reflect padding as audiosr.utils.mel_spectrogram_train
    p = (N_FFT - HOP) // 2
    yy = np.pad(y, (p, p), mode="reflect")
    stft = librosa.stft(yy, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                        window="hann", center=False)
    mag = np.abs(stft)
    basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MEL, fmin=FMIN, fmax=FMAX)
    mel = np.log(np.maximum(basis @ mag, 1e-5))
    return mel.astype(np.float32), mag.astype(np.float32)


def _cutoff(stft_mag, percentile=0.985):
    energy = np.cumsum(stft_mag.sum(axis=1))  # [freq], sum over time
    idx = 0
    for i in range(1, energy.shape[0]):
        if energy[-i] < energy[-1] * percentile:
            idx = energy.shape[0] - i
            break
    cutoff = idx / 1024.0 * 24000.0
    return 24000.0 if cutoff < 1000.0 else cutoff


def _lowpass(y, cutoff):
    sos = butter(8, cutoff, btype="lowpass", fs=SR, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def make_batch(path):
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)
    if sr != SR:
        y = resample_poly(y, SR, sr).astype(np.float32)
    duration = len(y) / SR
    pad_duration = duration + ((SEG - duration % SEG) if duration % SEG else 0.0)
    target_frame = int(pad_duration * 100)
    y = _normalize(y)
    y = _pad_wav(y, int(SR * pad_duration))

    mel, stft = _mel(y)
    mel = mel.T  # [T, 256]
    stft = stft.T
    if mel.shape[0] < target_frame:
        mel = np.pad(mel, ((0, target_frame - mel.shape[0]), (0, 0)))
        stft = np.pad(stft, ((0, target_frame - stft.shape[0]), (0, 0)))
    else:
        mel, stft = mel[:target_frame], stft[:target_frame]
    if mel.shape[-1] % 2:
        mel, stft = mel[..., :-1], stft[..., :-1]

    cutoff = _cutoff(stft, 0.985)
    low = _lowpass(y, cutoff)
    low_mel, _ = _mel(low)
    low_mel = low_mel.T
    if low_mel.shape[0] < target_frame:
        low_mel = np.pad(low_mel, ((0, target_frame - low_mel.shape[0]), (0, 0)))
    else:
        low_mel = low_mel[:target_frame]
    if low_mel.shape[-1] % 2:
        low_mel = low_mel[..., :-1]
    return mel[None, None], low_mel[None, None], low, cutoff


class AudioSR:
    def __init__(self, model_dir, seed=42, ddim="model_convert/audiosr_ddim_10.npz"):
        import axengine
        self.axengine = axengine
        self.dir = Path(model_dir)
        self.enc = axengine.InferenceSession(str(self.dir / "audiosr_vae_encoder_nhwc.axmodel"))
        self.unet = axengine.InferenceSession(str(self.dir / "audiosr_unet.axmodel"))
        self.dec = axengine.InferenceSession(str(self.dir / "audiosr_vae_decoder.axmodel"))
        self.voc = [axengine.InferenceSession(str(self.dir / n)) for n in [
            "voc_pre_fp32.axmodel", "voc_up0a_mix.axmodel", "voc_up0b_mix.axmodel",
            "voc_up1a_mix.axmodel", "voc_up1b_mix.axmodel", "voc_split_nhwc_up2_mix.axmodel",
            "voc_split_nhwc_up3_mix.axmodel", "voc_split_nhwc_up4_mix.axmodel",
            "voc_split_nhwc_post_fp32.axmodel"]]
        self.d = np.load(ddim)
        self.rng = np.random.RandomState(seed)

    def _encode(self, mel):
        o = self.enc.run(None, {"mel": np.ascontiguousarray(mel[0, 0, :, :, None].astype(np.float32))})
        return (o[0] + self.rng.randn(*o[0].shape).astype(np.float32) *
                np.exp(0.5 * o[1])).astype(np.float32) * SCALE

    def enhance(self, path):
        mel, low_mel, low_wav, _ = make_batch(path)
        zc = self._encode(mel)
        zl = self._encode(low_mel)
        x = self.rng.randn(1, 16, 64, 32).astype(np.float32)
        uncond = np.full((1, 16, 64, 32), UNCOND, np.float32)
        ts = np.flip(self.d["timesteps"])
        n = len(ts)
        for i, step in enumerate(ts):
            t = np.array([float(step)], np.float32)
            def run(c):
                return self.unet.run(None, {"x": np.ascontiguousarray(np.concatenate([x, c], 1)), "t": t})[0]
            vu, vc = run(uncond), run(zc)
            v = vu + 3.5 * (vc - vu)
            sc = float(self.d["sqrt_alphas_cumprod"][step])
            so = float(self.d["sqrt_one_minus_alphas_cumprod"][step])
            et = sc * v + so * x
            px0 = sc * x - so * v
            idx = n - 1 - i
            at, ap = float(self.d["alphas"][idx]), float(self.d["alphas_prev"][idx])
            sig = float(self.d["sigmas"][idx])
            x = (np.sqrt(ap) * px0 + np.sqrt(max(0.0, 1.0 - ap - sig * sig)) * et +
                 sig * self.rng.randn(*x.shape).astype(np.float32)).astype(np.float32)
        dm = self.dec.run(None, {"z": np.ascontiguousarray(x / SCALE)})[0]
        # mel bin cutoff, using upstream mel_replace_ops
        e = np.cumsum(np.exp(low_mel[0]).sum(axis=1))
        limit = e[-1] * 0.985
        cb = 0
        for i in range(1, e.shape[0]):
            if e[-i] < limit:
                cb = e.shape[0] - i
                break
        dm[0, :, :cb, :] = low_mel[0, :cb, :]
        voc = dm.squeeze(1).transpose(0, 2, 1).astype(np.float32)  # [1, 256, 512]
        x = np.ascontiguousarray(voc.transpose(0, 2, 1))  # NHWC [1, 512, 256]
        for s in self.voc:
            x = s.run(None, {s.get_inputs()[0].name: x})[0]
        wav = x[0, 0][:low_wav.size]
        return postprocess(wav, low_wav), SR


def postprocess(out, src):
    stft_gt = librosa.stft(src)
    stft_out = librosa.stft(out)
    e = np.cumsum(np.sum(np.abs(stft_gt), axis=-1))
    limit = e[-1] * 0.985
    c = 0
    for i in range(1, e.shape[0]):
        if e[-i] < limit:
            c = e.shape[0] - i
            break
    er = np.mean(np.sum(np.abs(stft_gt[c])) / np.sum(np.abs(stft_out[c])))
    er = min(max(er, 0.8), 1.2)
    stft_out[:c] = stft_gt[:c] / er
    out = librosa.istft(stft_out, length=out.shape[0])
    out = 0.5 * out / max(float(np.max(np.abs(out))), 1e-8)
    return (out - float(np.mean(out))).astype(np.float32)
