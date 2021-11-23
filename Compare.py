#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 04:37:10 2021

@author: wuzongze
"""

import argparse
import os 

import numpy as np
from PIL import Image

from visualizer import HtmlPageVisualizer



def Vis(out,save_path,rownames=None,colnames=None):
    num_images=out.shape[0]
    step=out.shape[1]
    
    if colnames is None:
        colnames=[f'Step {i:02d}' for i in range(1, step + 1)]
    if rownames is None:
        rownames=[str(i) for i in range(num_images)]
    
    
    visualizer = HtmlPageVisualizer(
      num_rows=num_images, num_cols=step + 1, viz_size=256)
    visualizer.set_headers(
      ['Name'] +colnames)
    
    for i in range(num_images):
        visualizer.set_cell(i, 0, text=rownames[i])
    
    for i in range(num_images):
        for k in range(step):
            image=out[i,k,:,:,:]
            visualizer.set_cell(i, 1+k, image=image)
    
    # Save results.
    visualizer.save(save_path)
    

def LoadImgs(file_path,names):
    imgs=[]
    for name in names:
        img=Image.open(file_path+name)
        img=np.array(img)
        imgs.append(img)
    imgs=np.array(imgs)
    return imgs


def main():
    parser = argparse.ArgumentParser(
        description='combine proj latent codes',
    )
    parser.add_argument('--source_img_path',     help='', required=True)
    parser.add_argument('--source_path',      help='path to source inverted images', required=True)
    parser.add_argument('--target_path',     help='path for saving translated images', required=True)
    parser.add_argument('--save_path',     help='path for saving translated images', required=True)
    
    args = parser.parse_args()
    source_img_path=args.source_img_path
    source_path=args.source_path
    target_path=args.target_path
    save_path=args.save_path
    
    names=os.listdir(target_path)
    names=sorted(names)
    
    
    imgs=LoadImgs(source_img_path,names)[:,None]
    invert=LoadImgs(source_path,names)[:,None]
    target=LoadImgs(target_path,names)[:,None]
    
    out=np.concatenate([imgs,invert,target],axis=1)
    
    
    colnames=['input','invert','translate']
    Vis(out,save_path,colnames=colnames)
#    print('save_path',save_path)


#%%
if __name__ == "__main__":
    main()












