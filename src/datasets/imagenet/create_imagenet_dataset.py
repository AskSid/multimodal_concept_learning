import argparse
import os
import random
import csv
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
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


def load_ilsvrc_to_wnid_mapping(data_dir: str) -> Dict[int, str]:
    """Load mapping from ILSVRC2012_ID to WNID."""
    import scipy.io
    
    meta_path = os.path.join(data_dir, "ILSVRC2012_devkit_t12", "data", "meta.mat")
    meta = scipy.io.loadmat(meta_path)
    synsets = meta['synsets']
    
    ilsvrc_to_wnid = {}
    for synset in synsets:
        ilsvrc_id = int(synset[0][0][0][0])
        wnid = str(synset[0][1][0])
        ilsvrc_to_wnid[ilsvrc_id] = wnid
    
    return ilsvrc_to_wnid


def load_parent_child_relationships(data_dir: str) -> Dict[str, List[str]]:
    """Load parent-child relationships from wordnet.is_a.txt."""
    isa_path = os.path.join(data_dir, "ILSVRC2012_devkit_t12", "data", "wordnet.is_a.txt")
    parent_to_children = {}
    
    with open(isa_path, 'r') as f:
        for line in f:
            parent, child = line.strip().split()
            if parent not in parent_to_children:
                parent_to_children[parent] = []
            parent_to_children[parent].append(child)
    
    return parent_to_children


def load_imagenet1k_wnids(data_dir: str) -> set:
    """Load all ImageNet-1K WNIDs from meta.mat."""
    import scipy.io
    
    meta_path = os.path.join(data_dir, "ILSVRC2012_devkit_t12", "data", "meta.mat")
    meta = scipy.io.loadmat(meta_path)
    synsets = meta['synsets']
    
    ilsvrc_wnids = set()
    for synset in synsets:
        wnid = str(synset[0][1][0])
        ilsvrc_wnids.add(wnid)
    
    return ilsvrc_wnids


def get_leaf_synsets(synset: str, parent_to_children: Dict[str, List[str]]) -> List[str]:
    """Get all leaf synsets under a given synset."""
    if synset not in parent_to_children:
        return [synset]  # Already a leaf
    
    leaves = []
    for child in parent_to_children[synset]:
        leaves.extend(get_leaf_synsets(child, parent_to_children))
    return leaves


def get_imagenet1k_leaf_synsets(synset: str, parent_to_children: Dict[str, List[str]], 
                                ilsvrc_wnids: set) -> List[str]:
    """Get all ImageNet-1K leaf synsets under a given synset."""
    if synset not in parent_to_children:
        return [synset] if synset in ilsvrc_wnids else []
    
    leaves = []
    for child in parent_to_children[synset]:
        if child in ilsvrc_wnids:
            leaves.append(child)
        else:
            leaves.extend(get_imagenet1k_leaf_synsets(child, parent_to_children, ilsvrc_wnids))
    return leaves


def get_all_imagenet1k_descendants(synset: str, parent_to_children: Dict[str, List[str]], 
                                   ilsvrc_wnids: set) -> List[str]:
    """Get all ImageNet-1K synsets that are descendants of the given synset."""
    descendants = []
    
    def find_descendants(current_synset):
        if current_synset in ilsvrc_wnids:
            descendants.append(current_synset)
        
        if current_synset in parent_to_children:
            for child in parent_to_children[current_synset]:
                find_descendants(child)
    
    find_descendants(synset)
    return descendants


def harvest_train_images(train_dir: str, target_synsets: List[str], parent_to_children: Dict[str, List[str]], 
                        ilsvrc_wnids: set) -> Dict[str, List[str]]:
    """Harvest training images for target synsets by finding their leaf children."""
    synset_images = {}
    
    for target_synset in target_synsets:
        # Get all ImageNet-1K synsets that are descendants of this target synset
        imagenet_descendants = get_all_imagenet1k_descendants(target_synset, parent_to_children, ilsvrc_wnids)
        all_images = []
        
        for leaf_synset in imagenet_descendants:
            leaf_dir = os.path.join(train_dir, leaf_synset)
            if os.path.exists(leaf_dir):
                images = glob.glob(os.path.join(leaf_dir, "*.JPEG"))
                all_images.extend([os.path.relpath(img, train_dir) for img in images])
        
        if all_images:
            synset_images[target_synset] = all_images
            print(f"Found {len(all_images)} training images for synset {target_synset} from {len(imagenet_descendants)} ImageNet-1K descendants")
        else:
            print(f"Warning: No training images found for synset {target_synset}")
    
    return synset_images


def harvest_test_images(val_dir: str, val_ground_truth_file: str, target_synsets: List[str], 
                       ilsvrc_to_wnid: Dict[int, str], parent_to_children: Dict[str, List[str]], 
                       ilsvrc_wnids: set) -> Dict[str, List[str]]:
    """Harvest test images for target synsets by finding their leaf children."""
    # Load ground truth
    with open(val_ground_truth_file, 'r') as f:
        ground_truth = [int(line.strip()) for line in f]
    
    # Get validation images
    val_images = sorted(glob.glob(os.path.join(val_dir, "ILSVRC2012_val_*.JPEG")))
    
    if len(val_images) != len(ground_truth):
        raise ValueError(f"Number of validation images ({len(val_images)}) doesn't match ground truth length ({len(ground_truth)})")
    
    # Build mapping from ImageNet-1K synsets to target synsets
    leaf_to_target = {}
    for target_synset in target_synsets:
        imagenet_descendants = get_all_imagenet1k_descendants(target_synset, parent_to_children, ilsvrc_wnids)
        for descendant in imagenet_descendants:
            leaf_to_target[descendant] = target_synset
    
    # Map to target synsets
    synset_images = {synset: [] for synset in target_synsets}
    
    for img_path, gt_id in zip(val_images, ground_truth):
        if gt_id in ilsvrc_to_wnid:
            wnid = ilsvrc_to_wnid[gt_id]
            if wnid in leaf_to_target:
                target_synset = leaf_to_target[wnid]
                synset_images[target_synset].append(os.path.basename(img_path))
    
    return synset_images


def sample_images(synset_images: Dict[str, List[str]], target_per_class: int, 
                 allow_shortfall: bool, rng: random.Random) -> List[Tuple[str, str]]:
    """Sample images from synsets with target per class."""
    records = []
    
    for synset, images in synset_images.items():
        if not images:
            if not allow_shortfall:
                raise ValueError(f"No images found for synset {synset}")
            continue
            
        if len(images) < target_per_class and not allow_shortfall:
            raise ValueError(f"Not enough images for synset {synset}: {len(images)} < {target_per_class}")
        
        # Sample images
        n_samples = min(target_per_class, len(images))
        sampled = rng.sample(images, n_samples)
        records.extend([(img, synset) for img in sampled])
    
    return records


def split_train_val(train_synset_images: Dict[str, List[str]], per_class_train: int, per_class_val: int, 
                   rng: random.Random) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Split training images into train and validation sets."""
    train_split = {}
    val_split = {}
    
    for synset, images in train_synset_images.items():
        if not images:
            train_split[synset] = []
            val_split[synset] = []
            continue
        
        # Shuffle images
        shuffled_images = images.copy()
        rng.shuffle(shuffled_images)
        
        # Split into train/val
        total_needed = per_class_train + per_class_val
        if len(shuffled_images) >= total_needed:
            train_split[synset] = shuffled_images[:per_class_train]
            val_split[synset] = shuffled_images[per_class_train:per_class_train + per_class_val]
        else:
            # Not enough images, use what we have
            train_split[synset] = shuffled_images[:min(per_class_train, len(shuffled_images))]
            val_split[synset] = shuffled_images[per_class_train:min(per_class_train + per_class_val, len(shuffled_images))]
    
    return train_split, val_split


def create_mapping_csv(records: List[Tuple[str, str]], target_synsets: List[str], 
                      wnid_to_name: Dict[str, str], output_path: str, split_name: str):
    """Create CSV mapping file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'target_synset', 'class_name'])
        
        for img_path, original_wnid in records:
            # Find target synset index
            target_idx = target_synsets.index(original_wnid)
            target_wnid = target_synsets[target_idx]
            class_name = wnid_to_name.get(target_wnid, target_wnid)
            
            writer.writerow([img_path, target_wnid, class_name])
    
    print(f"Created {split_name} mapping with {len(records)} images: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create ImageNet dataset mapping CSV")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config_path, 'r') as f:
        config = ImageNetDatasetConfig.from_params(yaml.safe_load(f))
    
    # Set random seed
    random.seed(config.seed)
    rng = random.Random(config.seed)
    
    # Load mappings
    wnid_to_name = load_wnid_to_name_mapping(config.data_dir)
    ilsvrc_to_wnid = load_ilsvrc_to_wnid_mapping(config.data_dir)
    parent_to_children = load_parent_child_relationships(config.data_dir)
    ilsvrc_wnids = load_imagenet1k_wnids(config.data_dir)
    
    # Create paths
    train_dir = os.path.join(config.data_dir, config.train_dir)
    val_dir = os.path.join(config.data_dir, config.val_dir)
    val_ground_truth_file = os.path.join(config.data_dir, "ILSVRC2012_devkit_t12", "data", config.val_ground_truth_file)
    
    # Harvest images
    print("Harvesting training images...")
    train_synset_images = harvest_train_images(train_dir, config.target_synsets, parent_to_children, ilsvrc_wnids)
    
    print("Harvesting test images...")
    test_synset_images = harvest_test_images(val_dir, val_ground_truth_file, config.target_synsets, ilsvrc_to_wnid, parent_to_children, ilsvrc_wnids)
    
    # Split training data into train/val
    print("Splitting training data into train/val...")
    train_split, val_split = split_train_val(train_synset_images, config.per_class_train, config.per_class_val, rng)
    
    # Sample images
    print("Sampling train images...")
    train_records = sample_images(train_split, config.per_class_train, config.allow_shortfall, rng)
    
    print("Sampling validation images...")
    val_records = sample_images(val_split, config.per_class_val, config.allow_shortfall, rng)
    
    print("Sampling test images...")
    test_records = sample_images(test_synset_images, config.per_class_test, config.allow_shortfall, rng)
    
    # Create output directory
    output_dir = os.path.join(config.output_dir, config.dataset_name)
    
    # Create mapping CSVs
    train_csv = os.path.join(output_dir, "train_mapping.csv")
    val_csv = os.path.join(output_dir, "val_mapping.csv")
    test_csv = os.path.join(output_dir, "test_mapping.csv")
    
    create_mapping_csv(train_records, config.target_synsets, wnid_to_name, train_csv, "train")
    create_mapping_csv(val_records, config.target_synsets, wnid_to_name, val_csv, "validation")
    create_mapping_csv(test_records, config.target_synsets, wnid_to_name, test_csv, "test")
    
    print(f"Dataset creation completed. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
