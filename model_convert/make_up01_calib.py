import tarfile
from pathlib import Path
import numpy as np
import onnxruntime as ort
base=Path('build/calib/up01'); base.mkdir(parents=True,exist_ok=True)
inputs={n:[] for n in ['up0a','up0b','up1a','up1b']}
for i in range(8):
 x=np.load(f'build/calib/voc_mel/{i:04d}.npy').astype(np.float32).transpose(0,2,1)
 for n in ['pre','up0a','up0b','up1a','up1b']:
  if n!='pre': inputs[n].append(x.copy())
  s=ort.InferenceSession('models/voc_split_nhwc_pre.onnx' if n=='pre' else f'models/voc_{n}.onnx',providers=['CPUExecutionProvider']); x=s.run(None,{s.get_inputs()[0].name:x})[0]
for n,arrs in inputs.items():
 d=base/n; d.mkdir(parents=True,exist_ok=True)
 for i,a in enumerate(arrs): np.save(d/f'{i:04d}.npy',a)
 with tarfile.open(base/f'{n}.tar.gz','w:gz') as t:
  for f in sorted(d.glob('*.npy')): t.add(f,arcname=f.name)
 print(n,len(arrs),arrs[0].shape)
