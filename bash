#!/bin/sh



source_img_path='./example/dog/'   
source_path='./img_invert/ffhq512_dog/z/'  # path for save inverted latent codes and images
target_path='./img_invert/ffhq512_dog/translate/cat/' #path for translation images 

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


#python CombineLatent.py --file_path ./img_invert/ffhq512_dog/z/ \
#			--save_path  ./img_invert/ffhq512_dog/



source_img_path='./example/dog/'   
source_path='./img_invert/ffhq512_dog/z/' 
source_pkl='./checkpoint/ffhq512_dog.pkl'

target_img_path='./example/cat/'   
target_path='./img_invert/ffhq512_dog_cat/z/' 
target_pkl='./checkpoint/ffhq512_dog_cat.pkl'

compare_html='./img_invert/dog_cat.html'



python projector_z.py --outdir=$source_path  \
   		      --target=$source_img_path \
  		      --network=$source_pkl

python projector_z.py --outdir=$target_path  \
   		      --target=$target_img_path \
  		      --network=$target_pkl












