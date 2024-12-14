# 
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class IRDataset(Dataset):
    """Dataset class for infrared images."""
    def __init__(self, sim_dir, real_dir, transform=None, phase='train'):
        """
        Args:
            sim_dir (str): Directory with simulated IR images
            real_dir (str): Directory with real IR images
            transform (callable, optional): Transform to be applied on images
            phase (str): 'train', 'val', or 'test'
        """
        self.sim_dir = os.path.join(sim_dir, phase)
        self.real_dir = os.path.join(real_dir, phase)
        
        self.sim_images = sorted(os.listdir(self.sim_dir))
        self.real_images = sorted(os.listdir(self.real_dir))
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return max(len(self.sim_images), len(self.real_images))

    def __getitem__(self, idx):
        # Handle different dataset sizes by cycling through smaller dataset
        sim_idx = idx % len(self.sim_images)
        real_idx = idx % len(self.real_images)
        
        # Load simulated image
        sim_path = os.path.join(self.sim_dir, self.sim_images[sim_idx])
        sim_img = Image.open(sim_path).convert('L')  # Convert to grayscale
        sim_img = self.transform(sim_img)
        
        # Load real image
        real_path = os.path.join(self.real_dir, self.real_images[real_idx])
        real_img = Image.open(real_path).convert('L')
        real_img = self.transform(real_img)
        
        return {'sim': sim_img, 'real': real_img}
