import json
from pathlib import Path
OUT=Path('model_convert/configs')
shapes={'up0a':('x_up0a',(512,1536)),'up0b':('x_up0b',(3072,768)),'up1a':('x_up1a',(3072,768)),'up1b':('x_up1b',(15361,384))}
for n,(t,(l,c)) in shapes.items():
 cfg={"model_type":"ONNX","target_hardware":"AX650","npu_mode":"NPU3","input":f"/workspace/models/voc_{n}.onnx","output_dir":f"/workspace/build/compile/voc_{n}","output_name":"model.axmodel","work_dir":f"/workspace/build/compile/voc_{n}_work","input_shapes":f"{t}:1x{l}x{c}","onnx_opt":{"disable_onnx_optimization":False,"enable_onnxsim":False,"model_check":True},"quant":{"calibration_method":"MinMax","precision_analysis":True,"precision_analysis_method":"EndToEnd","highest_mix_precision":False,"layer_configs":[{"start_tensor_names":["DEFAULT"],"end_tensor_names":["DEFAULT"],"data_type":"U16","weight_data_type":"S8","output_data_type":"U16"}],"input_configs":[{"tensor_name":t,"calibration_dataset":f"/workspace/build/calib/up01/{n}.tar.gz","calibration_format":"Numpy","calibration_size":8,"calibration_mean":[],"calibration_std":[]}]},"input_processors":[],"compiler":{"check":0}}
 p=OUT/f'voc_{n}.json'; p.write_text(json.dumps(cfg,indent=2)+'\n'); print(p)
