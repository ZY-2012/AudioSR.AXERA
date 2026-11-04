import json
from pathlib import Path
OUT=Path('model_convert/configs')
shapes={'pre':('mel',(512,256)),'up0':('x_up0',(512,1536)),'up1':('x_up1',(3072,768)),'up2':('x_up2',(15361,384)),'up3':('x_up3',(61444,192)),'up4':('x_up4',(122888,96)),'post':('x_post',(245776,48))}
for n,(t,(l,c)) in shapes.items():
 cfg={"model_type":"ONNX","target_hardware":"AX650","npu_mode":"NPU3","input":f"/workspace/models/voc_split_nhwc_{n}.onnx","output_dir":f"/workspace/build/compile/voc_split_nhwc_{n}","output_name":"model.axmodel","work_dir":f"/workspace/build/compile/voc_split_nhwc_{n}_work","input_shapes":f"{t}:1x{l}x{c}","onnx_opt":{"disable_onnx_optimization":False,"enable_onnxsim":False,"model_check":True},"quant":{"calibration_method":"MinMax","precision_analysis":True,"precision_analysis_method":"EndToEnd","highest_mix_precision":False,"layer_configs":[{"start_tensor_names":["DEFAULT"],"end_tensor_names":["DEFAULT"],"data_type":"U16","weight_data_type":"S8","output_data_type":"U16"}],"input_configs":[{"tensor_name":t,"calibration_dataset":f"/workspace/build/calib/voc_split_nhwc/{n}.tar.gz","calibration_format":"Numpy","calibration_size":8,"calibration_mean":[],"calibration_std":[]}]},"input_processors":[],"compiler":{"check":0}}
 p=OUT/f'voc_split_nhwc_{n}.json'; p.write_text(json.dumps(cfg,indent=2)+'\n'); print(p)
