import tarfile
from pathlib import Path
import numpy as np
import onnxruntime as ort
order=['pre','up0','up1','up2','up3','up4','post']
sess={n:ort.InferenceSession(f'models/voc_split_nhwc_{n}.onnx',providers=['CPUExecutionProvider']) for n in order}
inputs={n:[] for n in order}
for i in range(8):
 x=np.load(f'build/calib/voc_mel/{i:04d}.npy').astype(np.float32).transpose(0,2,1)
 for n in order:
  inputs[n].append(x.copy()); s=sess[n]; x=s.run(None,{s.get_inputs()[0].name:x})[0]
base=Path('build/calib/voc_split_nhwc'); base.mkdir(parents=True,exist_ok=True)
for n,arrs in inputs.items():
 d=base/n; d.mkdir(parents=True,exist_ok=True)
 for i,a in enumerate(arrs): np.save(d/f'{i:04d}.npy',a)
 with tarfile.open(base/f'{n}.tar.gz','w:gz') as t:
  for f in sorted(d.glob('*.npy')): t.add(f,arcname=f.name)
 print(n,len(arrs),arrs[0].shape)
