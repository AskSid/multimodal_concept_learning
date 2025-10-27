import os
from typing import Optional, Callable

import pandas as pd
import torch
from PIL import Image


class ColorDataset(torch.utils.data.Dataset):
    def __init__(self, mapping_csv_path: str, data_dir: str, transform: Optional[Callable] = None, return_synset: bool = False):
        self.data_dir = data_dir
        self.transform = transform
        self.return_synset = return_synset

        mapping_df = pd.read_csv(mapping_csv_path)

        self.dataset = []
        for _, row in mapping_df.iterrows():
            image_path = os.path.join(self.data_dir, row["image_path"])
            class_name = row["class_name"]
            self.dataset.append((image_path, class_name))

        self.unique_labels = sorted({item[1] for item in self.dataset})
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.num_classes = len(self.unique_labels)

        print(f"Loaded {len(self.dataset)} images with {self.num_classes} classes")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image_path, class_name = self.dataset[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        if self.return_synset:
            return image, class_name
        else:
            label = self.label_to_idx[class_name]
            return image, label
