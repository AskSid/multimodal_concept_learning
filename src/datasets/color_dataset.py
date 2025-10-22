import argparse
import os
import random
import yaml
from typing import List, Tuple, Optional, Callable

import torch
from PIL import Image, ImageDraw
from src.datasets.color_dataset_config import ColorDatasetConfig

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
        
        
        # Filter the dataset using the indices
        self.dataset = [self.dataset[i] for i in indices]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image_path, color_name = self.dataset[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, color_name


def generate_color_dataset(config: ColorDatasetConfig) -> str:
    """Generate a color dataset with circles of different colors, positions, and radii."""
    # Create output directory structure
    dataset_dir = os.path.join(config.data_dir, config.dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    min_radius, max_radius = config.radius_range
    
    for rgb in config.colors:
        # Create color directory with format r{R}g{G}b{B}
        color_name = f"r{rgb[0]}g{rgb[1]}b{rgb[2]}"
        
        # Create color directory directly in dataset directory
        color_dir = os.path.join(dataset_dir, color_name)
        os.makedirs(color_dir, exist_ok=True)
        
        # Generate images for this color
        for idx in range(config.n_images_per_color):
            # Random intensity factor
            factor = random.uniform(config.min_intensity, config.max_intensity)
            
            # Scale color intensity
            r, g, b = rgb
            sr = int(round(r * factor))
            sg = int(round(g * factor))
            sb = int(round(b * factor))
            # Ensure bounds and avoid becoming fully black or fully white
            sr = max(1 if r > 0 else 0, min(254 if r == 255 else 255, sr))
            sg = max(1 if g > 0 else 0, min(254 if g == 255 else 255, sg))
            sb = max(1 if b > 0 else 0, min(254 if b == 255 else 255, sb))
            scaled_rgb = (sr, sg, sb)
            
            # Create image with random circle
            img = Image.new("RGB", (config.image_size, config.image_size), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # Random circle parameters
            radius = random.randint(min_radius, max_radius)
            margin = radius
            cx = random.randint(margin, config.image_size - margin)
            cy = random.randint(margin, config.image_size - margin)
            bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
            draw.ellipse(bbox, fill=scaled_rgb, outline=None)
            
            # Save image directly
            filename = f"circle_{color_name}_{idx:05d}.png"
            img_path = os.path.join(color_dir, filename)
            img.save(img_path, format="PNG")
    
    print(f"Generated {config.n_images_per_color} images for each of {len(config.colors)} colors.")
    print(f"Dataset saved to: {os.path.abspath(dataset_dir)}")
    
    return dataset_dir


def main():
    # Load config
    parser = argparse.ArgumentParser(description="Generate color dataset from config file")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file")
    
    args = parser.parse_args()
    
    with open(args.config_path, "r") as f:
        config = ColorDatasetConfig.from_params(yaml.safe_load(f))
    
    # Generate the dataset
    output_path = generate_color_dataset(config)
    print(f"Dataset generation completed. Output: {output_path}")


# Example usage
if __name__ == "__main__":
    main()
