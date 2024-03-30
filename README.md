# GNR 638 Mini Project 2

## Image Deblurring

1. Download sharp images from here: [Google Drive Link](https://drive.usercontent.google.com/download?id=1YLksKtMhd2mWyVSkvhDaDLWSc1qYNCz-&export=download&authuser=0)

2. Downscale the images to (256,448). (Set A)

3. Create a set of images by applying different Gaussian filters to each image: (Set B)
   a. Kernel size = 3x3, sigma = 0.3
   b. Kernel size = 7x7, sigma = 1
   c. Kernel size = 11x11, sigma = 1.6

4. Design a network to deblur images (Set B -> Set A) with an upper limit of 15M parameters.

5. Test set will be provided along with ground truth later. We will also provide an evaluation script. Please report PSNR score according to this score.

6. Please use:

   - numpy==1.24.4
   - PIL==10.2
   - scikit-image==0.21 \
     for preprocessing and running evaluation script.

7. You need to submit:
   - Codes
   - Final checkpoint
   - Report (contains model architecture, training details, training curves, qualitative and quantitative results)
