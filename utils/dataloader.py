import os
from torch.utils.data import Dataset, DataLoader
import numpy as np

def dataloader():
    dataset_path = "dataset/train/train_sharp"

    if not os.path.exists(dataset_path):
        print("Dataset not found")
        import sys
        sys.exit(1)

    dataset = {}

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        images = []
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            images.append(image_path)
        dataset[folder] = images

    return dataset

def processed_dataset():

    dataset_blurred = "dataset/train/train_sharp_processed/blurred"
    dataset_clear = "dataset/train/train_sharp_processed/clear"

    if (not os.path.exists(dataset_blurred)) or (not os.path.exists(dataset_clear)):
        print("processed images not available")
        import sys
        sys.exit(1)

    dataset = []

    for blur_img in os.listdir(dataset_blurred):
        
        clear_img = blur_img.split('_')[0] + ".png"

        blur_img = os.path.join(dataset_blurred, blur_img)
        clear_img = os.path.join(dataset_clear, clear_img)

        dataset.append((blur_img, clear_img))

    dataset = np.array(dataset)
    return dataset

class BlurDataset(Dataset):
    def __init__(self, processed_dataset):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        return self.n_samples