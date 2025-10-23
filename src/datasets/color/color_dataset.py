import os
from typing import List, Optional, Callable

import torch
from PIL import Image


class ColorDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, indices: List[int], transform: Optional[Callable] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.dataset = []
        
        # Load all images directly from the dataset directory
        if os.path.exists(data_dir):
            color_dirs = os.listdir(data_dir)
            for color_dir in color_dirs:
                color_path = os.path.join(data_dir, color_dir)
                if os.path.isdir(color_path):  # Only process directories
                    color_name = color_dir
                    images = os.listdir(color_path)
                    self.dataset.extend([(os.path.join(color_path, image), color_name) for image in images])
        
        # Create label mapping
        self.unique_labels = sorted(list(set([item[1] for item in self.dataset])))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        
        # Filter the dataset using the indices
        self.dataset = [self.dataset[i] for i in indices]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image_path, color_name = self.dataset[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.label_to_idx[color_name]
        return image, label