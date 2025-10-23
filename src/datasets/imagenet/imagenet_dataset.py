import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable


class ImageNetDataset(Dataset):
    def __init__(self, mapping_csv_path: str, data_dir: str, transform: Optional[Callable] = None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load mapping CSV
        self.mapping_df = pd.read_csv(mapping_csv_path)
        
        # Create dataset
        self.dataset = []
        for _, row in self.mapping_df.iterrows():
            image_path = os.path.join(self.data_dir, row['image_path'])
            target_wnid = row['target_wnid']
            self.dataset.append((image_path, target_wnid))
        
        # Create label mapping
        self.unique_labels = sorted(list(set([item[1] for item in self.dataset])))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.num_classes = len(self.unique_labels)
        
        print(f"Loaded {len(self.dataset)} images with {self.num_classes} classes")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image_path, target_wnid = self.dataset[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label like color dataset
        label = self.label_to_idx[target_wnid]
        
        return image, label
