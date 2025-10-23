from dataclasses import dataclass
from typing import Optional, List
import os

@dataclass
class ImageNetDatasetConfig:
    """Configuration for creating ImageNet dataset mappings."""
    
    # Dataset source paths
    data_dir: str
    train_dir: str
    val_dir: str
    val_ground_truth_file: str
    
    # Output
    output_dir: str
    dataset_name: str
    
    # Target synsets to use
    target_synsets: List[str]
    
    # Sampling parameters
    per_class_train: int
    per_class_val: int
    per_class_test: int
    allow_shortfall: bool
    seed: int
    
    @classmethod
    def from_params(cls, params: Optional[dict]) -> "ImageNetDatasetConfig":
        params = params or {}
        return cls(
            data_dir=params.get("data_dir", "/tmp/data"),
            train_dir=params.get("train_dir", "train"),
            val_dir=params.get("val_dir", "val"),
            val_ground_truth_file=params.get("val_ground_truth_file", "ILSVRC2012_validation_ground_truth.txt"),
            output_dir=params.get("output_dir", "/tmp/output"),
            dataset_name=params.get("dataset_name", "imagenet_dataset"),
            target_synsets=params.get("target_synsets", []),
            per_class_train=int(params.get("per_class_train", 3000)),
            per_class_val=int(params.get("per_class_val", 50)),
            per_class_test=int(params.get("per_class_test", 50)),
            allow_shortfall=bool(params.get("allow_shortfall", False)),
            seed=int(params.get("seed", 42)),
        )
