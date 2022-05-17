#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 05:34:51 2021

@author: wuzongze
"""


import os
import argparse
import pickle
import numpy as np

import dnnlib
import dnnlib.tflib as tflib
from PIL import Image

#%%

def lerp(a,b,t):
     return a + (b - a) * t

def Truncation(src_dlatents,dlatent_avg,truncation_psi,truncation_cutoff):
    layer_idx = np.arange(src_dlatents.shape[1])[np.newaxis, :, np.newaxis]
    ones = np.ones(layer_idx.shape, dtype=np.float32)
    
    if truncation_cutoff is None:
        coefs = ones*truncation_psi
    else:
        coefs = np.where(layer_idx > truncation_cutoff, truncation_psi * ones, ones)
    src_dlatents_np=lerp(dlatent_avg, src_dlatents, coefs)
    return src_dlatents_np



def main():
    parser = argparse.ArgumentParser(
        description='combine proj latent codes',
    )
    parser.add_argument('--network',     help='Network pickle filename', required=True)
    parser.add_argument('--source_path',      help='path to source inverted latent codes', required=True)
    parser.add_argument('--target_path',     help='path to target inverted latent codes', required=True)
    
    
    args = parser.parse_args()
    network=args.network
    source_path=args.source_path
    target_path=args.target_path
    
    tflib.init_tf()
    print('Loading networks from "%s"...' % network)
    with dnnlib.util.open_url(network) as fp:
        _, _, Gs = pickle.load(fp)
        
    
    names=os.listdir(source_path)
    names=sorted(names)

    for name in names:
        if name[-4:]=='.npz' :
            tmp=np.load(source_path+name)
            z=tmp['dlatents']
#            print(z.shape)
            img=Gs.run(z,None,output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
#            print(img.shape)
            name1=name[:-4]+'.jpg'
            
            img=Image.fromarray(img[0]).save(target_path+name1)
            
#%%
if __name__ == "__main__":
    main()


