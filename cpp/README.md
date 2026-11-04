# AudioSR.AXERA C++ runtime

Full AX650 C++17 runtime for the AudioSR chain: WAV → resample → STFT/log-mel →
VAE encoder → DDIM-10 UNet → VAE decoder → split vocoder → postprocess.

No Python, PyTorch or ONNX Runtime dependency. The AX Engine V2 API is used for
all 12 axmodels; KissFFT (BSD-3-Clause) provides the host FFT.

## Build

```bash
TOOLCHAIN_ROOT=/data/shared/huyuan/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu \
BSP_MSP_DIR=/data/shared/huyuan/toolchains/ax650n_bsp_sdk/msp/out \
bash cpp/build_ax650.sh
```

The binary is written to `cpp/bin/audiosr_ax650`.

## Run on board

```bash
export LD_LIBRARY_PATH=/soc/lib:$LD_LIBRARY_PATH
./cpp/bin/audiosr_ax650 models samples/listen/16k/speech_2_input_16k.wav out.wav
```

`models/` must contain the 12 axmodels plus `mel_filters_256x1025.bin` (all on
HuggingFace). `--bench` averages 3 runs. The C++ implementation uses `std::mt19937`
with seed 42; the Python SDK uses NumPy RandomState, so the exact waveform of a
different seed is not sample-identical even though both use the same models.

Measured on AX650N / NPU3, 5.12 s input:

| runtime | time | RTF |
|---|---:|---:|
| Python SDK | ~20–22 s | ~4.0–4.3 |
| **C++** | **~7.3 s** | **~1.43** |
