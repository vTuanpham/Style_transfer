# Arbitrary style transfer using adaptive Instance Normalization

## This project was built based on the following papers:

 * [ A neural algorithm of artistic style](https://arxiv.org/abs/1508.06576)
 * [ An Improved Style Transfer Algorithm Using Feedforward Neural Network for Real-Time Image Conversion](https://www.mdpi.com/2071-1050/11/20/5673)
 * [ Arbitrary style transfer in real time with adaptive instance norm](https://arxiv.org/abs/1703.06868)
 * [ Controlling colors of GAN-Genrated and real images via color histograms](https://arxiv.org/abs/2011.11731)
 * [ Demystifying neural style transfer](https://arxiv.org/abs/1701.01036)
 * [ Exploring the structure of a real-time, arbitrary neural artistic stylization network](https://arxiv.org/abs/1705.06830)
 * [ Instance normalization the missing ingredient for fast stylization](https://arxiv.org/abs/1607.08022)
 * [ Learning deep embedding with histogram loss](https://arxiv.org/abs/1611.00822)
 * [ Learning linear transformation for fast arbitrary style transfer](https://arxiv.org/abs/1808.04537)
 * [ Universal style transform via feature transform](https://arxiv.org/abs/1705.08086)

# With a couple modifications
  * The AdaIN module have trainable params(EPS) (Better training stability)
  * Use pretrained images recovery in the decoder for faster training
  * Added histogram loss and variance loss to better guide the model
  * Add new augmentations for content and style

## Results:
  ![image](https://github.com/vTuanpham/Style_transfer/assets/82665400/be3aef53-6ff2-42cd-8ff7-1e00c3dedabf)
  ![image](https://github.com/vTuanpham/Style_transfer/assets/82665400/94db9f2c-d569-4890-9b4b-cf574e399f6e)
  ![image](https://github.com/vTuanpham/Style_transfer/assets/82665400/5cccd992-abe5-480b-9981-fc4f8aaee884)
  ![image](https://github.com/vTuanpham/Style_transfer/assets/82665400/16e6929a-860c-4038-8df0-924a98331cc7)
  ![image](https://github.com/vTuanpham/Style_transfer/assets/82665400/ebc3b5e4-9b27-4e78-b0cf-261562fb764b)

## Training
  Install dependencies first, this might take awhile..
  ```
  pip install -r requirements.txt
  ```
  To train, modified the script in the src/scripts folder
  ```
  bash src/scripts/train.sh 
  ```

  * This project is heavily support for wandb logging:
    ### Every epochs, eval the model performance by inferencing all the images in the src/data/eval_dir   
    * ![image](https://github.com/vTuanpham/Style_transfer/assets/82665400/051c1ed1-2402-4b70-84dd-aab9711afb39)
    ### Auto log saved checkpoint to wandb with wandb artifacts
    * ![image](https://github.com/vTuanpham/Style_transfer/assets/82665400/a3838187-fb5a-4983-bce7-6ee8a23f5603)

## Test
  #### Modified the alpha value (higher mean higher emphasis on the style)
  ```
  python src/inference.py --path_to_save_cpkt "" --alpha 
  ```

#### Leave a star ⭐ if you find this useful!

### TO DO:
  * Include model checkpoint
  * Easier inference
  * Add docs on all args for training
  * Longer training might reduce noise ?
  * Output image is a bit less saturate than the style
  * Sleep
   




