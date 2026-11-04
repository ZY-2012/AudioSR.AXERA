# Model conversion

AudioSR speech (`haoheliu/audiosr_speech`) → AX650 NPU3.

## 流程

```text
AudioSR checkpoint
  -> export VAE encoder (NHWC), UNet, VAE decoder
  -> export / split HiFi-GAN vocoder
  -> calibration features
  -> Pulsar2 8.0 U16 / mixed-precision
  -> precision gate (0.98)
```

## 目录

```text
model_convert/
├── configs/                         # Pulsar2 输入配置
├── compile.sh                       # 逐个编译 + precision gate
├── precision_gate.py                # EndToEnd 输出门禁
├── export_vocoder_split_nhwc.py     # pre/up0..up4/post 初版拆分
├── export_up01_split.py             # up0/up1 再拆成 a/b
├── make_*_calib.py                  # 板端校准特征
├── prepare_*.py                     # 生成 Pulsar2 JSON
├── make_ddim_assets.py              # DDIM-10 常量
└── audiosr_ddim_10.npz
```

## 运行

```bash
# 需要先准备上游 AudioSR 源码和 checkpoint
python model_convert/export_vocoder_split_nhwc.py
python model_convert/export_up01_split.py
python model_convert/make_voc_split_nhwc_calib.py
python model_convert/make_up01_calib.py
python model_convert/prepare_voc_split_nhwc.py
python model_convert/prepare_up01.py

PULSAR2_IMAGE=docker-registry.aitsw.axera-tech.com/pulsar2:8.0 \
  bash model_convert/compile.sh \
  voc_pre_fp32 voc_up0a_mix voc_up0b_mix voc_up1a_mix voc_up1b_mix \
  voc_split_nhwc_up2_mix voc_split_nhwc_up3_mix voc_split_nhwc_up4_mix \
  voc_split_nhwc_post_fp32
```

`compile.sh` 会执行 `precision_gate.py`，每个子模型最终输出低于 0.98 即停止。vocoder 拆分的目的是把大模型误差和内存压力分散到 pre / up / post 小模型；pre 和 post 用 FP32，up0/up1 再拆成 a/b 并混合精度。

VAE encoder 使用 NHWC 输入 `[1,512,256,1]`；Pulsar2 对 NCHW 的 1×1×512×256 输入会出现明显板端布局误差，NHWC 版板端 mean/logvar 均约 0.9998。
