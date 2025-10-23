import argparse
import csv
import os
import random
import yaml
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw
from src.datasets.color.color_dataset_config import ColorDatasetConfig


def compute_split_counts(num_items: int, ratios: List[float]) -> List[int]:
    """Distribute items across splits while respecting the provided ratios."""
    if num_items == 0:
        return [0] * len(ratios)

    raw_counts = [num_items * ratio for ratio in ratios]
    counts = [int(count) for count in raw_counts]
    remainder = num_items - sum(counts)

    if remainder > 0:
        sorted_indices = sorted(
            range(len(ratios)),
            key=lambda idx: (raw_counts[idx] - counts[idx], -idx),
            reverse=True,
        )
        idx = 0
        while remainder > 0:
            counts[sorted_indices[idx % len(sorted_indices)]] += 1
            remainder -= 1
            idx += 1

    return counts


def generate_color_dataset(config: ColorDatasetConfig) -> str:
    """Generate a color dataset with circles of different colors, positions, and radii."""
    # Create output directory structure
    dataset_dir = os.path.join(config.data_dir, config.dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    min_radius, max_radius = config.radius_range
    
    color_image_map: Dict[str, List[str]] = {}

    for rgb in config.colors:
        # Create color directory with format r{R}g{G}b{B}
        color_name = f"r{rgb[0]}g{rgb[1]}b{rgb[2]}"

        # Create color directory directly in dataset directory
        color_dir = os.path.join(dataset_dir, color_name)
        os.makedirs(color_dir, exist_ok=True)

        color_image_map[color_name] = []

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
            relative_path = os.path.relpath(img_path, dataset_dir)
            color_image_map[color_name].append(relative_path)

    print(f"Generated {config.n_images_per_color} images for each of {len(config.colors)} colors.")

    # Prepare split ratios
    split_names = ["train", "val", "test"]
    if len(config.train_val_test_split) != len(split_names):
        raise ValueError("train_val_test_split must contain three values for train/val/test ratios")

    total_ratio = sum(config.train_val_test_split)
    if total_ratio <= 0:
        raise ValueError("train_val_test_split must sum to a positive value")

    normalized_ratios = [ratio / total_ratio for ratio in config.train_val_test_split]
    split_records: Dict[str, List[Tuple[str, str]]] = {name: [] for name in split_names}

    # Split images per color into train/val/test
    for color_name, image_paths in color_image_map.items():
        shuffled_paths = image_paths.copy()
        random.shuffle(shuffled_paths)

        train_count, val_count, test_count = compute_split_counts(len(shuffled_paths), normalized_ratios)

        train_split = shuffled_paths[:train_count]
        val_split = shuffled_paths[train_count:train_count + val_count]
        test_split = shuffled_paths[train_count + val_count:]

        split_records["train"].extend((path, color_name) for path in train_split)
        split_records["val"].extend((path, color_name) for path in val_split)
        split_records["test"].extend((path, color_name) for path in test_split)

    # Write mapping CSVs
    for split_name in split_names:
        mapping_path = os.path.join(dataset_dir, f"{split_name}_mapping.csv")
        with open(mapping_path, "w", newline="") as mapping_file:
            writer = csv.writer(mapping_file)
            writer.writerow(["image_path", "color_name"])
            writer.writerows(split_records[split_name])
        print(f"Created {split_name} mapping with {len(split_records[split_name])} images: {mapping_path}")

    print(f"Dataset saved to: {os.path.abspath(dataset_dir)}")

    return dataset_dir


def main():
    # Load config
    parser = argparse.ArgumentParser(description="Generate color dataset from config file")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file")
    
    args = parser.parse_args()
    
    with open(args.config_path, "r") as f:
        config = ColorDatasetConfig.from_params(yaml.safe_load(f))
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Generate the dataset
    output_path = generate_color_dataset(config)
    print(f"Dataset generation completed. Output: {output_path}")


# Example usage
if __name__ == "__main__":
    main()
