import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

def load_base_dataset():
    '''
    returns dictionary of format <folder_name: [list of image paths]>
    '''
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
    
    print("Dataset loaded successfully")
    print(f"Number of folders: {len(dataset.keys())}")
    print(f"Number of images: {sum([len(images) for images in dataset.values()])}")

    return dataset

def load_processed_dataset():
    '''
    Load the processed dataset
    returns a numpy array of shape (n_samples, 2)
    format <blur_img, clear_img>
    '''

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

    print("Processed dataset loaded successfully")
    print(f"Number of images: {dataset.shape[0]}")
    print("<blur_img, clear_img>")
    return dataset

class Blur_Clear_Dataset(Dataset):
    def __init__(self, processed_dataset):
        self.x = processed_dataset[:, 0]
        self.y = processed_dataset[:, 1]
        self.n_samples = processed_dataset.shape[0]

    def __getitem__(self, index):
        blur_img = Image.open(self.x[index])
        clear_img = Image.open(self.y[index])
        return blur_img, clear_img

    def __len__(self):
        return self.n_samples
    
def create_torch_dataloader(processed_dataset, batch_size=32, shuffle=True):
    dataset = Blur_Clear_Dataset(
        processed_dataset=processed_dataset
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return dataloader