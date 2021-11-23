#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 23:58:36 2021

@author: wuzongze
"""
import os
import numpy as np 
from PIL import Image
import argparse

def CombineLatent(file_path,save_path):

    names=os.listdir(file_path)
    names=sorted(names)
#    print(names)
    
    codes=[]
    imgs=[]
    for name in names:
        if '.npz' in name:
            tmp=np.load(file_path+name)
            tmp1=tmp['dlatents']
            codes.append(tmp1)
        elif '.png' in name:
            img=Image.open(file_path+name)
            img=np.array(img)
            imgs.append(img)
    
    codes=np.concatenate(codes)
    np.save(save_path+'z',codes)
    
    imgs=np.array(imgs)
    np.save(save_path+'invert_img',imgs)


def main():
    parser = argparse.ArgumentParser(
        description='combine proj latent codes',
    )
    parser.add_argument('--file_path',     help='path to a set of invert codes', required=True)
    parser.add_argument('--save_path',     help='path for saving the combine latent codes', required=True)
    
    
    CombineLatent(**vars(parser.parse_args()))

#%%
if __name__ == "__main__":
    main()




    
