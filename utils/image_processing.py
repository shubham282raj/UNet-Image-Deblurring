import numpy as np
from PIL import Image
from skimage.filters import gaussian
import os
import random
import numpy as np
import matplotlib.pyplot as plt

def process_image(re_process=False):
    """
    Maked blurred and clear images folders in "dataset/train/train_sharp_processed"
    By default images are not re-processed
    Pass 'True' to force re-process the images
    """

    dataset_path = "dataset/train/train_sharp"
    dataset_blurred = "dataset/train/train_sharp_processed/blurred"
    dataset_clear = "dataset/train/train_sharp_processed/clear"

    if not os.path.exists(dataset_path):
        print("Dataset not found")
        import sys
        sys.exit(1)

    if not os.path.exists(dataset_blurred):
        os.makedirs(dataset_blurred)
    elif not re_process:
        print("Blurred Images Dataset Folder Already Exist! pass 'True' as argument to re-process images")
        import sys
        sys.exit()

    if not os.path.exists(dataset_clear):
        os.makedirs(dataset_clear)
    elif not re_process:
        print("Clear Images Dataset Folder Already Exist! pass 'True' as argument to re-process images")
        import sys
        sys.exit()

    img_num = 1

    for folder in os.listdir(dataset_path):

        folder_path = os.path.join(dataset_path, folder)
        
        for file in os.listdir(folder_path):

            image_path = os.path.join(folder_path, file)
            image = Image.open(image_path)

            # downscales the image to 256x448
            # image = image.resize((256, 448))
            image = image.resize((448, 256))
            
            # image to np array
            image = np.array(image)
            
            # applying filter on each image
            image1 = gaussian(image, sigma=0.3)
            image2 = gaussian(image, sigma=1)
            image3 = gaussian(image, sigma=1.6)
            
            image = Image.fromarray(image)
            image1 = Image.fromarray((image1 * 255).astype(np.uint8))
            image2 = Image.fromarray((image2 * 255).astype(np.uint8))
            image3 = Image.fromarray((image3 * 255).astype(np.uint8))

            # save the image
            image.save(dataset_clear + "/" + str(img_num) + ".png")

            image1.save(dataset_blurred + "/" + str(img_num) + "_1.png")
            image2.save(dataset_blurred + "/" + str(img_num) + "_2.png")
            image3.save(dataset_blurred + "/" + str(img_num) + "_3.png")

            img_num += 1

            if(img_num%1000==0):
                print(f"Image processed: ${img_num} / 24000")