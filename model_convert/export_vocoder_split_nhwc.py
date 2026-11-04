import os,sys
from pathlib import Path
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
ROOT=Path(__file__).resolve().parents[1]; UP=ROOT/'origin/versatile_audio_super_resolution'; sys.path.insert(0,str(UP))
import numpy as np; np.float=float; np.complex=complex
import torch, torch.nn as nn, torch.nn.functional as F, audiosr.pipeline as ap
ap.download_checkpoint=lambda model_name='speech': str(ROOT/'origin/checkpoints/speech/pytorch_model.bin')
m=ap.build_model(model_name='speech',device='cpu'); voc=m.first_stage_model.vocoder.eval()
try: voc.remove_weight_norm()
except Exception: pass
class Pre(nn.Module):
 def __init__(self,v): super().__init__(); self.v=v
 def forward(self,x): return self.v.conv_pre(x.transpose(1,2)).transpose(1,2)
class Up(nn.Module):
 def __init__(self,v,i): super().__init__(); self.v=v; self.i=i
 def forward(self,x):
  x=x.transpose(1,2); x=F.leaky_relu(x,0.1); x=self.v.ups[self.i](x)
  xs=None
  for j in range(self.v.num_kernels):
   r=self.v.resblocks[self.i*self.v.num_kernels+j](x)
   xs=r if xs is None else xs+r
  return (xs/self.v.num_kernels).transpose(1,2)
class Post(nn.Module):
 def __init__(self,v): super().__init__(); self.v=v
 def forward(self,x): return torch.tanh(self.v.conv_post(F.leaky_relu(x.transpose(1,2))))
mods={'pre':Pre(voc),'up0':Up(voc,0),'up1':Up(voc,1),'up2':Up(voc,2),'up3':Up(voc,3),'up4':Up(voc,4),'post':Post(voc)}
shapes={'pre':(1,512,256),'up0':(1,512,1536),'up1':(1,3072,768),'up2':(1,15361,384),'up3':(1,61444,192),'up4':(1,122888,96),'post':(1,245776,48)}
names={'pre':('mel','x_pre'),'up0':('x_up0','x0'),'up1':('x_up1','x1'),'up2':('x_up2','x2'),'up3':('x_up3','x3'),'up4':('x_up4','x4'),'post':('x_post','wav')}
for name,mod in mods.items():
 x=torch.randn(*shapes[name]); mod.eval()
 with torch.no_grad(): y=mod(x)
 p=ROOT/'models'/f'voc_split_nhwc_{name}.onnx'; inp,out=names[name]
 torch.onnx.export(mod,(x,),str(p),input_names=[inp],output_names=[out],opset_version=17,do_constant_folding=True,dynamo=False)
 import onnxruntime as ort
 s=ort.InferenceSession(str(p),providers=['CPUExecutionProvider']); got=s.run(None,{inp:x.numpy()})[0]
 print(name,tuple(x.shape),'->',tuple(y.shape),'err',float(np.abs(got-y.numpy()).max()),'size',p.stat().st_size/1e6)
