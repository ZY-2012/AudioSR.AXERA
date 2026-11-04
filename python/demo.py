#!/usr/bin/env python3
import argparse, os, sys, time
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audiosr_sdk import AudioSR

def main():
    p=argparse.ArgumentParser(description="AudioSR speech super-resolution on AX650")
    p.add_argument("input")
    p.add_argument("-o","--output",default=None)
    p.add_argument("--models",default=os.path.join(os.path.dirname(__file__),"..","models"))
    p.add_argument("--ddim",default=os.path.join(os.path.dirname(__file__),"..","model_convert","audiosr_ddim_10.npz"))
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    m=AudioSR(os.path.abspath(a.models),seed=a.seed,ddim=os.path.abspath(a.ddim))
    t=time.time(); y,sr=m.enhance(a.input); dt=time.time()-t
    out=a.output or os.path.splitext(a.input)[0]+"_48k.wav"
    sf.write(out,y,sr,subtype="PCM_16")
    d=len(y)/sr
    print(f"AudioSR: {sr} Hz, {d:.2f}s, {dt:.2f}s inference, RTF {dt/max(d,1e-9):.3f}")
    print("output:",out)

if __name__=="__main__": main()
