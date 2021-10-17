# StyleAlign: Analysis and Applications of Aligned StyleGAN Models



<p align="center">
  <a><img src='https://github.com/betterze/StyleAlign/blob/main/img/model_progress/dog2cat0.gif'   width=400  ></a
    <a> </a
    <a><img src='https://github.com/betterze/StyleAlign/blob/main/img/model_progress/ffhq2dog0.gif'   width=400 ></a
</p> 
    

> **StyleAlign: Analysis and Applications of Aligned StyleGAN Models**<br>
> Zongze Wu, Yotam Nitzan, Eli Shechtman, Dani Lischinski <br>
>
>**Abstract:** In this paper, we perform an in-depth study of the properties and applications of \emph{aligned generative models}.
We refer to two models as aligned if they share the same architecture, and one of them (the \emph{child}) is obtained from the other (the \emph{parent}) via fine-tuning to another domain, a common practice in transfer learning.
Several works already utilize some basic properties of aligned StyleGAN models to perform image-to-image translation.
Here, we perform the first detailed exploration of model alignment, also focusing on StyleGAN. First, we empirically analyze aligned models and provide answers to important questions regarding their nature. In particular, we find that the child model's latent spaces are semantically aligned with those of the parent, inheriting incredibly rich semantics, even for distant data domains such as human faces and churches.
Second, equipped with this better understanding, we leverage aligned models to solve a diverse set of tasks.
In addition to image translation, we demonstrate fully automatic cross-domain image morphing.
We further show that zero-shot vision tasks may be performed in the child domain, while relying exclusively on supervision in the parent domain.
We demonstrate qualitatively and quantitatively that our approach yields state-of-the-art results, while requiring only simple fine-tuning and inversion. 
