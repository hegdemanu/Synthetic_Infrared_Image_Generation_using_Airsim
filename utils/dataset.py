import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class IRDataset(Dataset):
    """Dataset class for handling infrared images."""
    
    def __init__(self, data_path, transform=None):
        self.data_path = Path(data_path)
        self.image_files = list(self.data_path.glob('*.png')) + \
                          list(self.data_path.glob('*.jpg'))
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image
