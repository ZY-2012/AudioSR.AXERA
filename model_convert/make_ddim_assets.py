import os,sys,numpy as np
from pathlib import Path
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'; ROOT=Path('.'); sys.path.insert(0,'origin/versatile_audio_super_resolution')
np.float=float; np.complex=complex
import audiosr.pipeline as ap
from audiosr.latent_diffusion.models.ddim import make_ddim_timesteps, make_ddim_sampling_parameters
ap.download_checkpoint=lambda model_name='speech': str(ROOT/'origin/checkpoints/speech/pytorch_model.bin')
m=ap.build_model(model_name='speech',device='cpu')
for steps in [10,20]:
 ts=make_ddim_timesteps('uniform',steps,m.num_timesteps,verbose=True)
 sig,alpha,alpha_prev=make_ddim_sampling_parameters(alphacums=m.alphas_cumprod.cpu(),ddim_timesteps=ts,eta=1.0,verbose=False)
 np.savez(f'assets/audiosr_ddim_{steps}.npz',timesteps=np.asarray(ts,dtype=np.int64),alphas=np.asarray(alpha,dtype=np.float32),alphas_prev=np.asarray(alpha_prev,dtype=np.float32),sigmas=np.asarray(sig,dtype=np.float32),sqrt_one_minus=np.sqrt(1-np.asarray(alpha,dtype=np.float32)).astype(np.float32),sqrt_alphas_cumprod=m.sqrt_alphas_cumprod.cpu().numpy().astype(np.float32),sqrt_one_minus_alphas_cumprod=m.sqrt_one_minus_alphas_cumprod.cpu().numpy().astype(np.float32))
 print(steps,ts)
