import os,sys
from pathlib import Path
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
ROOT=Path(__file__).resolve().parents[1]; UP=ROOT/'origin/versatile_audio_super_resolution'; sys.path.insert(0,str(UP))
import numpy as np; np.float=float; np.complex=complex
import torch, torch.nn as nn, torch.nn.functional as F, audiosr.pipeline as ap
ap.download_checkpoint=lambda model_name='speech': str(ROOT/'origin/checkpoints/speech/pytorch_model.bin')
m=ap.build_model(model_name='speech',device='cpu'); v=m.first_stage_model.vocoder.eval()
try: v.remove_weight_norm()
except Exception: pass
class Front(nn.Module):
 def __init__(self,v,i): super().__init__(); self.v=v; self.i=i
 def forward(self,x):
  x=F.leaky_relu(x.transpose(1,2),0.1); return self.v.ups[self.i](x).transpose(1,2)
class Blocks(nn.Module):
 def __init__(self,v,i): super().__init__(); self.v=v; self.i=i
 def forward(self,x):
  x=x.transpose(1,2); xs=None
  for j in range(self.v.num_kernels):
   r=self.v.resblocks[self.i*self.v.num_kernels+j](x)
   xs=r if xs is None else xs+r
  return (xs/self.v.num_kernels).transpose(1,2)
mods={'up0a':Front(v,0),'up0b':Blocks(v,0),'up1a':Front(v,1),'up1b':Blocks(v,1)}
shapes={'up0a':(1,512,1536),'up0b':(1,3072,768),'up1a':(1,3072,768),'up1b':(1,15361,384)}
names={'up0a':('x_up0a','x_up0a_out'),'up0b':('x_up0b','x_up0b_out'),'up1a':('x_up1a','x_up1a_out'),'up1b':('x_up1b','x_up1b_out')}
for name,mod in mods.items():
 x=torch.randn(*shapes[name]); mod.eval()
 with torch.no_grad(): y=mod(x)
 p=ROOT/'models'/f'voc_{name}.onnx'; inp,out=names[name]
 torch.onnx.export(mod,(x,),str(p),input_names=[inp],output_names=[out],opset_version=17,do_constant_folding=True,dynamo=False)
 import onnxruntime as ort
 s=ort.InferenceSession(str(p),providers=['CPUExecutionProvider']); got=s.run(None,{inp:x.numpy()})[0]
 print(name,tuple(x.shape),'->',tuple(y.shape),'err',float(np.abs(got-y.numpy()).max()),'size',p.stat().st_size/1e6)
