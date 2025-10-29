import argparse
import os
import random
import json
import yaml
from pathlib import Path
from typing import List, Dict, Set
import glob

from src.datasets.imagenet.imagenet_dataset_config import ImageNetDatasetConfig


def load_wnid_to_name_mapping(data_dir: str) -> Dict[str, str]:
    """Load mapping from WNID to human-readable names."""
    words_path = os.path.join(data_dir, "ILSVRC2012_devkit_t12", "data", "words.txt")
    wnid_to_name = {}
    
    with open(words_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                wnid = parts[0]
                name = ' '.join(parts[1:])
                wnid_to_name[wnid] = name
    
    return wnid_to_name


def get_imagenet1k_wnids(train_dir: str) -> List[str]:
    """Get all ImageNet-1K WNIDs from train directory structure."""
    train_path = Path(train_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    
    wnids = []
    for synset_dir in train_path.iterdir():
        if synset_dir.is_dir() and synset_dir.name.startswith('n'):
            wnids.append(synset_dir.name)
    
    return sorted(wnids)


def create_ood_labels(num_ood: int, wnids: List[str], rng: random.Random) -> Set[str]:
    """Randomly select WNIDs to be OOD labels."""
    if num_ood > len(wnids):
        raise ValueError(f"num_ood ({num_ood}) cannot be greater than total WNIDs ({len(wnids)})")
    
    return set(rng.sample(wnids, num_ood))


def create_labels_mapping(wnids: List[str], wnid_to_name: Dict[str, str], 
                         ood_wnids: Set[str]) -> Dict[str, str]:
    """Create labels mapping with OOD labels for selected WNIDs."""
    labels_mapping = {}
    
    for wnid in wnids:
        if wnid in ood_wnids:
            # Create OOD label using semantic name instead of WNID
            full_name = wnid_to_name.get(wnid, wnid)
            first_name = full_name.split(',')[0].strip()
            labels_mapping[wnid] = f"<ood_{first_name}>"
        else:
            # Use semantic label - strip commas to match CSV format
            full_name = wnid_to_name.get(wnid, wnid)
            # Extract only the first name (before first comma)
            first_name = full_name.split(',')[0].strip()
            labels_mapping[wnid] = first_name
    
    return labels_mapping


def create_imagenet1k_config(data_dir: str, output_dir: str, dataset_name: str,
                            wnids: List[str], per_class_train: int, per_class_val: int,
                            per_class_test: int, seed: int) -> ImageNetDatasetConfig:
    """Create ImageNet-1K dataset configuration."""
    return ImageNetDatasetConfig(
        data_dir=data_dir,
        train_dir="train",
        val_dir="val", 
        val_ground_truth_file="ILSVRC2012_validation_ground_truth.txt",
        output_dir=output_dir,
        dataset_name=dataset_name,
        target_synsets=wnids,
        per_class_train=per_class_train,
        per_class_val=per_class_val,
        per_class_test=per_class_test,
        allow_shortfall=True,
        seed=seed
    )


def save_config(config: ImageNetDatasetConfig, output_path: str):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    config_dict = {
        "data_dir": config.data_dir,
        "train_dir": config.train_dir,
        "val_dir": config.val_dir,
        "val_ground_truth_file": config.val_ground_truth_file,
        "output_dir": config.output_dir,
        "dataset_name": config.dataset_name,
        "target_synsets": config.target_synsets,
        "per_class_train": config.per_class_train,
        "per_class_val": config.per_class_val,
        "per_class_test": config.per_class_test,
        "allow_shortfall": config.allow_shortfall,
        "seed": config.seed
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    print(f"Saved ImageNet-1K config to: {output_path}")


def save_labels_mapping(labels_mapping: Dict[str, str], output_path: str):
    """Save labels mapping to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(labels_mapping, f, indent=2, sort_keys=True)
    
    print(f"Saved labels mapping to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create ImageNet-1K dataset config and labels mapping")
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="Path to ImageNet data directory")
    parser.add_argument("--dataset_name", type=str, required=True,
                       help="Name for the dataset")
    parser.add_argument("--num_ood", type=int, default=100,
                       help="Number of WNIDs to assign as OOD labels")
    parser.add_argument("--per_class_train", type=int, default=500,
                       help="Number of training images per class")
    parser.add_argument("--per_class_val", type=int, default=50,
                       help="Number of validation images per class")
    parser.add_argument("--per_class_test", type=int, default=50,
                       help="Number of test images per class")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for OOD selection")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    rng = random.Random(args.seed)
    
    print(f"Creating ImageNet-1K configs with {args.num_ood} OOD labels...")
    print(f"Data directory: {args.data_dir}")
    print(f"Random seed: {args.seed}")
    
    # Load WNID to name mapping
    print("Loading WNID to name mapping...")
    wnid_to_name = load_wnid_to_name_mapping(args.data_dir)
    
    # Get all ImageNet-1K WNIDs
    print("Discovering ImageNet-1K WNIDs...")
    train_dir = os.path.join(args.data_dir, "train")
    wnids = get_imagenet1k_wnids(train_dir)
    print(f"Found {len(wnids)} ImageNet-1K WNIDs")
    
    # Create OOD labels
    print(f"Randomly selecting {args.num_ood} WNIDs for OOD labels...")
    ood_wnids = create_ood_labels(args.num_ood, wnids, rng)
    
    # Create labels mapping
    print("Creating labels mapping...")
    labels_mapping = create_labels_mapping(wnids, wnid_to_name, ood_wnids)
    
    # Create dataset config
    print("Creating dataset configuration...")
    config = create_imagenet1k_config(
        data_dir=args.data_dir,
        output_dir=os.path.join("data", "multimodal_concept_learning", "imagenet1k"),
        dataset_name=args.dataset_name,
        wnids=wnids,
        per_class_train=args.per_class_train,
        per_class_val=args.per_class_val,
        per_class_test=args.per_class_test,
        seed=args.seed
    )
    
    # Save files in appropriate experiment directories
    config_path = os.path.join("experiments", "datasets", "imagenet", f"{args.dataset_name}.yaml")
    labels_path = os.path.join("experiments", "multimodal", "imagenet", f"{args.dataset_name}_labels_mapping.json")
    
    save_config(config, config_path)
    save_labels_mapping(labels_mapping, labels_path)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total WNIDs: {len(wnids)}")
    print(f"OOD WNIDs: {len(ood_wnids)}")
    print(f"Semantic WNIDs: {len(wnids) - len(ood_wnids)}")
    print(f"Config file: {config_path}")
    print(f"Labels mapping: {labels_path}")
    print("\nOOD WNIDs selected:")
    for wnid in sorted(ood_wnids):
        print(f"  {wnid}: {wnid_to_name.get(wnid, 'Unknown')}")


if __name__ == "__main__":
    main()
