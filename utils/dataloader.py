import os

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