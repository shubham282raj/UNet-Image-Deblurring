import torch

def load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path)

def save_checkpoint(checkpoint, path):
    return torch.save(checkpoint, path)