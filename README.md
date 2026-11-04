# AudioSR.AXERA

**AX650 / NPU3 语音超分：低采样率语音 → 48 kHz。完整链路 ONNX 拆分 + Pulsar2 混合量化，vocoder 板端相似度 0.9984，完整链路 vs PyTorch 0.9917。**

![AX650](https://img.shields.io/badge/AX650-NPU3-blue) ![vocoder](https://img.shields.io/badge/vocoder%20corr-0.9984-brightgreen) ![full](https://img.shields.io/badge/full%20chain%20corr-0.9917-brightgreen) [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-AudioSR.AXERA-yellow)](https://huggingface.co/HY-2012/AudioSR.AXERA)

- HuggingFace（模型从这里下载）：<https://huggingface.co/HY-2012/AudioSR.AXERA>
- GitHub：<https://github.com/ZY-2012/AudioSR_AXERA>
- 上游：<https://github.com/haoheliu/versatile_audio_super_resolution>
- 输入：8 / 16 / 24 / 48 kHz 单声道或立体声 WAV；输出：48 kHz
- 5.12 s 输入，AudioSR DDIM-10；vocoder 拆成 9 个小模型以适配 AX650 内存和精度

## 目录

- 快速开始
- 结果
- 推理速度
- 模型
- 模型转换
- 数据来源
- License

## 快速开始（AX650 板端）

从 HuggingFace 下载整个仓库，或只下载 `models/`，放在包根目录旁：

```bash
# 模型目录结构
# AudioSR_AXERA/
# └── models/
#     ├── audiosr_unet.axmodel
#     ├── audiosr_vae_encoder_nhwc.axmodel
#     ├── audiosr_vae_decoder.axmodel
#     └── voc_*.axmodel

pip install -r requirements.txt

python python/demo.py \
  samples/listen/16k/speech_2_input_16k.wav \
  -o output_48k.wav
```

输入 8 kHz / 16 kHz 均可；SDK 会重采样到 48 kHz，并按 AudioSR 的 5.12 s 窗口处理后送 NPU。

## 工程结构

```text
AudioSR_AXERA/
├── model_convert/       # ONNX 导出 → 拆分 → 校准数据 → Pulsar2 U16 量化
├── python/              # Python 板端推理 SDK（axengine）
├── cpp/                 # C++ 板端推理源码 + 交叉编译脚本
├── models/              # 仅 README，量化模型从 HuggingFace 下载
├── samples/             # 中英文示例输入
└── requirements.txt
```

## 结果

### 精度

| 模块 / 链路 | 板端 vs ONNX/PyTorch |
|---|---:|
| VAE encoder（NHWC） | 0.99975 |
| UNet | 0.99989 |
| VAE decoder | 0.99999 |
| vocoder 9-model chain | **0.99838** |
| full AudioSR chain | **0.99169** |

### 示例

| 输入 | 输出 |
|---|---|
| [English 16 kHz](samples/listen/16k/speech_2_input_16k.wav) | [48 kHz](samples/listen/output/speech_2_output_48k.wav) |
| [English 8 kHz](samples/listen/8k/OSR_us_000_0010_input_8k.wav) | 运行 SDK 生成 |
| [中文 8 kHz](samples/listen/8k/OSR_cn_000_0072_input_8k.wav) | 运行 SDK 生成 |

## 推理速度

5.12 s 输入、DDIM-10、Python SDK（AX650 NPU3）：

```text
约 20–22 s / 5.12 s 音频，RTF ≈ 4.0–4.3
```

短音频主要受 DDIM 10 步 UNet 和 vocoder 拆分模型调度开销影响；这是当前 Python 参考实现的实测值。

## 模型

模型不放进 GitHub，统一从 HuggingFace 下载：

- `audiosr_unet.axmodel`：273 MB
- `audiosr_vae_encoder_nhwc.axmodel`：102 MB
- `audiosr_vae_decoder.axmodel`：143 MB
- `voc_pre_fp32.axmodel`：11 MB
- `voc_up0a_mix.axmodel`：55 MB
- `voc_up0b_mix.axmodel`：202 MB
- `voc_up1a_mix.axmodel`：12 MB
- `voc_up1b_mix.axmodel`：52 MB
- `voc_split_nhwc_up2_mix.axmodel`：17 MB
- `voc_split_nhwc_up3_mix.axmodel`：7.7 MB
- `voc_split_nhwc_up4_mix.axmodel`：5.8 MB
- `voc_split_nhwc_post_fp32.axmodel`：99 KB

详见 [`models/README.md`](models/README.md)。

## 模型转换

```bash
# 1. 上游 AudioSR 代码 / checkpoint
# 2. 导出 VAE encoder / UNet / decoder / vocoder 子图
# 3. 校准数据
# 4. Pulsar2 8.0 量化
PULSAR2_IMAGE=docker-registry.aitsw.axera-tech.com/pulsar2:8.0 \
  bash model_convert/compile.sh voc_pre_fp32 voc_up0a_mix ...
```

具体命令、BF16/U16/S8 配置和各阶段校准见 [`model_convert/README.md`](model_convert/README.md)。

## 数据来源

- 上游 AudioSR：<https://github.com/haoheliu/versatile_audio_super_resolution>
- 示例英文 / 中文语音：Open Speech Repository、公开 16 kHz 人声样例
- 示例输出由本项目 AX650 Python SDK 生成

## License

AudioSR 上游代码与模型 checkpoint 遵循其原始许可证；本仓库的转换脚本和 SDK 以 Apache-2.0 提供。模型权重请遵循上游许可证。
