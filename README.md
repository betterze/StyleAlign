# StyleAlign: Analysis and Applications of Aligned StyleGAN Models



<p align="center">
  <a href="https://youtu.be/gjjo11IncP4?t=6"><img src='https://github.com/betterze/StyleAlign/blob/main/img/model_progress/ffhq2dog0.gif'   width=600  ></a
    <a> 
</p> 

Check our video here: <a href="https://youtu.be/gjjo11IncP4"><img src="https://img.shields.io/badge/-YouTube-red?&style=for-the-badge&logo=youtube&logoColor=white" height=20></a>
    
> **StyleAlign: Analysis and Applications of Aligned StyleGAN Models**<br>
> [Zongze Wu](https://www.cs.huji.ac.il/~wuzongze/), [Yotam Nitzan](https://yotamnitzan.github.io/), [Eli Shechtman](https://research.adobe.com/person/eli-shechtman/), [Dani Lischinski](https://pages.cs.huji.ac.il/danix/) <br>
> https://openreview.net/pdf?id=Qg2vi4ZbHM9 <br>
>
>**Abstract:** In this paper, we perform an in-depth study of the properties and applications of **aligned generative models**.
We refer to two models as aligned if they share the same architecture, and one of them (the **child**) is obtained from the other (the **parent**) via fine-tuning to another domain, a common practice in transfer learning.
Several works already utilize some basic properties of aligned StyleGAN models to perform image-to-image translation.
Here, we perform the first detailed exploration of model alignment, also focusing on StyleGAN. First, we empirically analyze aligned models and provide answers to important questions regarding their nature. In particular, we find that the child model's latent spaces are semantically aligned with those of the parent, inheriting incredibly rich semantics, even for distant data domains such as human faces and churches.
Second, equipped with this better understanding, we leverage aligned models to solve a diverse set of tasks.
In addition to image translation, we demonstrate fully automatic cross-domain image morphing.
We further show that zero-shot vision tasks may be performed in the child domain, while relying exclusively on supervision in the parent domain.
We demonstrate qualitatively and quantitatively that our approach yields state-of-the-art results, while requiring only simple fine-tuning and inversion. 

## Requirements
The codes are based on [StyleGAN2-ada](https://github.com/NVlabs/stylegan2-ada#requirements). The environmental requirements (from StyleGAN2-ada) are listed below:

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 64-bit Python 3.6 or 3.7. We recommend Anaconda3 with numpy 1.14.3 or newer.
* We recommend TensorFlow 1.14, which we used for all experiments in the paper, but TensorFlow 1.15 is also supported on Linux. TensorFlow 2.x is not supported.
* On Windows you need to use TensorFlow 1.14, as the standard 1.15 installation does not include necessary C++ headers.
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of GPU memory, NVIDIA drivers, CUDA 10.0 toolkit and cuDNN 7.5.
* Docker users: use the [provided Dockerfile](./Dockerfile) to build an image with the required library dependencies.

The generator and discriminator networks rely heavily on custom TensorFlow ops that are compiled on the fly using NVCC. On Windows, the compilation requires Microsoft Visual Studio to be in `PATH`. We recommend installing [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/) and adding it into `PATH` using `"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"`.

## usage
Train a parent [StyleGAN2](https://github.com/NVlabs/stylegan2-ada) or [StyleGAN2-ada] model in domain A, then use the parent model weights as initiation for child model (by adding the --resume flag) and fine tune it in domain B. In this way, we obtain the aligned parent and child models, and we could perform image translation or morphing using the following codes. 
	
## pretrained checkpoint
The pretrained checkpoints could be downloaded from [here](https://drive.google.com/drive/folders/1MqCHQ6Yx-eon-3fu1g_AGjpyAUmzH6Jy?usp=sharing). The FFHQ model is from [StyleGAN2 repo](https://github.com/NVlabs/stylegan2). The FFHQ512, FFHQ512_dog, FFHQ512_cat, FFHQ512_wild models are from [StyleGAN2-ada repo](https://github.com/NVlabs/stylegan2-ada). Other models are trained or fine tuning by ourselves.

To download all checkpoints:
	
 ```
gdown --fuzzy 'https://drive.google.com/drive/folders/1MqCHQ6Yx-eon-3fu1g_AGjpyAUmzH6Jy?usp=sharing' -O /checkpoint --folder
  ```
	
	
## Image-to-Image Translation 

  ```
source_img_path='./example/dog/'   
source_path='./img_invert/ffhq512_dog/z/'  # path for saving inverted latent codes and images
target_path='./img_invert/ffhq512_dog/translate/cat/' #path for saving translation images 

source_pkl='./checkpoint/ffhq512_dog.pkl'
target_pkl='./checkpoint/ffhq512_dog_cat.pkl'

compare_html='./img_invert/ffhq512_dog/translate/cat.html'

python projector_z.py --outdir=$source_path  \
   		      --target=$source_img_path \
  		      --network=$source_pkl


python I2I.py --network $target_pkl \
		      --source_path $source_path \
		      --target_path $target_path	


python Compare.py --source_img_path $source_img_path \
		      --source_path $source_path \
		      --target_path $target_path \
		      --save_path $compare_html 	
  
  ```

## Cross-domain Image Morphing
To morph image from different domains, please train an [e4e](https://github.com/omertov/encoder4editing) encoder in each doamin, and invert the images into w+ space. We provide pretrained e4e models for FFHQ512, FFHQ512_dog, FFHQ512_dog_cat in [here](https://drive.google.com/drive/folders/1MqCHQ6Yx-eon-3fu1g_AGjpyAUmzH6Jy). We use w+ space for better image reconstruction (compared to z space). 
	
  ```
source_pkl='./checkpoint/ffhq512_dog.pkl'
target_pkl='./checkpoint/ffhq512_dog_cat.pkl'
source_latent='./img_invert/ffhq512_dog/e4e_w_plus/flickr_dog_000043.npy' #w_plus 
target_latent='./img_invert/ffhq512_dog_cat/e4e_w_plus/flickr_cat_000008.npy' #w_plus 

python MergeFace.py --source_pkl $source_pkl --target_pkl $target_pkl --source_latent $source_latent --target_latent $target_latent
  ```
	
We can also translate an image from source to target domian and create a smooth video. We use w+ space in source domain for better reconstruction, and z space in target domain for better translation. Please add --target_is_z flag in the end. 
	
 ```
source_pkl='./checkpoint/ffhq512_dog.pkl'
target_pkl='./checkpoint/ffhq512_dog_cat.pkl'
source_latent='./img_invert/ffhq512_dog/e4e_w_plus/flickr_dog_000045.npy'  # w
target_latent='./img_invert/ffhq512_dog/z/flickr_dog_000045.npz'  # z

python MergeFace.py --source_pkl $source_pkl --target_pkl $target_pkl --source_latent $source_latent --target_latent $target_latent --target_is_z
  ```
	
	
  

## Shared Semantic Controls Between Parent and Child Models

<p align="center">
  <a href=https://youtu.be/gjjo11IncP4?t=21"><img src='https://github.com/betterze/StyleAlign/blob/main/img/aligned_direction/ffhq2cartoon2.gif'   width=800  ></a
</p> 
<p align="center">
  <a href="https://youtu.be/gjjo11IncP4?t=21"><img src='https://github.com/betterze/StyleAlign/blob/main/img/aligned_direction/ffhq2afhq2.gif'   width=800  ></a
</p> 
    
## Image Translation

<p align="center">
  <a ><img src='https://github.com/betterze/StyleAlign/blob/main/img/translate/ffhq2dog.jpg'   width=800  ></a
</p> 
<p align="center">
    <a ><img src='https://github.com/betterze/StyleAlign/blob/main/img/translate/ffhq2mega2.jpg'   width=800  ></a
</p> 
 
## Cross-domain Image Morphing

<p align="center">
  <a href="https://youtu.be/gjjo11IncP4?t=40"><img src='https://github.com/betterze/StyleAlign/blob/main/img/morphing/dog2cat2_c.gif'   width=400  ></a
    <a href="https://youtu.be/gjjo11IncP4?t=68"><img src='https://github.com/betterze/StyleAlign/blob/main/img/morphing/ffhq2dog2_c.gif'   width=400  ></a
</p> 

## Knowledge Transfer from Parent to Child Domain
  <p align="center">
    <a ><img src='https://github.com/betterze/StyleAlign/blob/main/img/knowledge.jpg'   width=800  ></a
</p>  
		   
		   
## Citation

If you use this code for your research, please cite our paper:

```
@article{wu2021stylealign,
  title={StyleAlign: Analysis and Applications of Aligned StyleGAN Models},
  author={Wu, Zongze and Nitzan, Yotam and Shechtman, Eli and Lischinski, Dani},
  journal={arXiv preprint arXiv:2110.11323},
  year={2021}
}
```
		   
