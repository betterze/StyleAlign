#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 18:16:02 2021

@author: wuzongze
"""

def PadBE(imgs,num_extreme):
    
    b=np.repeat(imgs[:1],num_extreme,axis=0)
    e=np.repeat(imgs[-1:],num_extreme,axis=0)
    
    tmp=[b,imgs,e,imgs[::-1],b]
    
    tmp1=np.concatenate(tmp)
    return tmp1

from dnnlib import tflib
import numpy as np
import pickle
import imageio
import argparse

def LoadModel(model_path):
    # Initialize TensorFlow.
    tflib.init_tf()
    with open(model_path, 'rb') as f:
        _, _, Gs = pickle.load(f)
    Gs.print_layers()
    return Gs

def lerp(a,b,t):
     return a + (b - a) * t

def Truncation(src_dlatents,dlatent_avg,truncation_psi,truncation_cutoff):
    layer_idx = np.arange(src_dlatents.shape[1])[np.newaxis, :, np.newaxis]
    ones = np.ones(layer_idx.shape, dtype=np.float32)
    
    if truncation_cutoff is None:
        coefs = ones*truncation_psi
    else:
        coefs = np.where(layer_idx < truncation_cutoff, truncation_psi * ones, ones)
    src_dlatents_np=lerp(dlatent_avg, src_dlatents, coefs)
    return src_dlatents_np

class MergeFace():
    def __init__(self,source_pkl,target_pkl,source_latent,target_latent,target_is_z):
        
        self.Gs=LoadModel(source_pkl)
        self.Gs1=LoadModel(source_pkl)
        self.Gs2=LoadModel(target_pkl)
        
        
        self.w_plus1=np.load(source_latent)
        
        if target_is_z:
            z=np.load(target_latent)['dlatents']
            w_plus2= self.Gs2.components.mapping.run(z, None)
            
            dlatent_avg=self.Gs2.get_var('dlatent_avg')
            truncation_psi=0.5  #default value of StyleGAN2
            truncation_cutoff=None #default value of StyleGAN2
            self.w_plus2 =Truncation(w_plus2,dlatent_avg,truncation_psi,truncation_cutoff)
            
            
        else:
            self.w_plus2=np.load(target_latent)
        
        self.GetWeightName()
        self.fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        
    def GetWeightName(self):
        #merge two network
        vars1=self.Gs1.vars
        vars2=self.Gs2.vars
        names=list(vars2.keys())
        self.wnames=[]
        for name in names:
            if 'G_synthesis' in name:
                self.wnames.append(name)
        
        d1={}
        for name in self.wnames:
            tmp=vars1[name]
            d1[name]=tmp
        self.d1=tflib.run(d1)
        
        d2={}
        for name in self.wnames:
            tmp=vars2[name]
            d2[name]=tmp
        self.d2=tflib.run(d2)
        
    def merge(self,ts1,ts2,M):
        
        full_out=[]
        for i in range(len(ts1)):
            
            t=ts1[i]
            w_plus=lerp(self.w_plus1,self.w_plus2,t)
            
            t=ts2[i]
            d={}
            for name in self.wnames:
                tmp=self.Gs.vars[name]
                a=self.d1[name]
                b=self.d2[name]
                tmp1=lerp(a,b,t)
                d[tmp]=tmp1
            tflib.set_vars(d)
            
            out = self.Gs.components.synthesis.run(w_plus,output_transform=self.fmt)
            full_out.append(out)
        full_out=np.concatenate(full_out[1:])
        
        return full_out
    
    
    

def main():
    parser = argparse.ArgumentParser(
        description='combine proj latent codes',
    )
    parser.add_argument('--source_pkl',     help='', required=True)
    parser.add_argument('--target_pkl',      help='', required=True)
    parser.add_argument('--source_latent',     help='', required=True)
    parser.add_argument('--target_latent',     help='', required=True)
    parser.add_argument('--target_is_z', action='store_true', )
    
    M=MergeFace(**vars(parser.parse_args()))
    
    num_step=110
    ts1=[0]+list(np.linspace(0,1,num_step))
    ts2=ts1
    
    full_out=M.merge(ts1,ts2,M)
    print('full_out:', full_out.shape)
    
    duration=8
    fps=int(len(full_out)/duration)
    imageio.mimwrite('./morphing.mp4', full_out , fps = fps)
    
    



#%%

if __name__ == "__main__":
    
    main()
    
    


















