# Models

GitHub 仓库不包含 axmodel。请从 HuggingFace 下载：

<https://huggingface.co/HY-2012/AudioSR.AXERA/tree/main/models>

需要的文件：

```text
audiosr_unet.axmodel
audiosr_vae_encoder_nhwc.axmodel
audiosr_vae_decoder.axmodel
voc_pre_fp32.axmodel
voc_up0a_mix.axmodel
voc_up0b_mix.axmodel
voc_up1a_mix.axmodel
voc_up1b_mix.axmodel
voc_split_nhwc_up2_mix.axmodel
voc_split_nhwc_up3_mix.axmodel
voc_split_nhwc_up4_mix.axmodel
voc_split_nhwc_post_fp32.axmodel
```

所有模型均为 Pulsar2 8.0 / AX650 NPU3 编译。vocoder 采用拆分 + 混合精度：
`pre/post` 为 FP32，`up0/up1` 拆成两段，其余 up2/up3/up4 为混合精度。
