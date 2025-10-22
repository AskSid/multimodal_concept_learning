from dataclasses import dataclass
from typing import Optional, List
import torch

@dataclass
class VisionTrainingConfig:
    """Configuration for the vision training experiment."""
    
    # Model architecture parameters
    model_name: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_labels: int
    patch_size: int
    hidden_dropout_prob: float
    attention_dropout_prob: float
    num_attention_heads: int

    # Dataset parameters
    data_dir: str
    dataset_name: str
    
    # Training parameters
    epochs: int
    learning_rate: float
    batch_size: int
    effective_batch_size: int
    weight_decay: float
    image_size: int
    label_smoothing: float
    val_split: float
    num_workers: int
    prefetch_factor: int
    augreg_l: int
    augreg_m: int
    augreg_alpha: float
    results_dir: str
    
    # Additional parameters
    seed: int
    device: str
    disable_tqdm: bool
    disable_wandb: bool
    wandb_project: str
    wandb_run_name: str

    @classmethod
    def from_params(cls, params: Optional[dict]) -> "VisionTrainingConfig":
        params = params or {}
        return cls(
            model_name=params.get("model_name", "vit"),
            hidden_size=params.get("hidden_size", 768),
            intermediate_size=params.get("intermediate_size", 3072),
            num_hidden_layers=params.get("num_hidden_layers", 12),
            num_labels=params.get("num_labels", 100),
            patch_size=params.get("patch_size", 16),
            hidden_dropout_prob=params.get("hidden_dropout_prob", 0.1),
            attention_dropout_prob=params.get("attention_dropout_prob", 0.1),
            num_attention_heads=params.get("num_attention_heads", 8),
            data_dir=params.get("data_dir", "/tmp/data"),
            dataset_name=params.get("dataset_name", "color"),
            epochs=params.get("epochs", 300),
            learning_rate=params.get("learning_rate", 1e-4),
            batch_size=params.get("batch_size", 128),
            effective_batch_size=params.get("effective_batch_size", 4096),
            weight_decay=params.get("weight_decay", 0.1),
            image_size=params.get("image_size", 224),
            label_smoothing=params.get("label_smoothing", 0.0),
            val_split=params.get("val_split", 0.1),
            num_workers=params.get("num_workers", 8),
            prefetch_factor=params.get("prefetch_factor", 2),
            augreg_l=params.get("augreg_l", 2),
            augreg_m=params.get("augreg_m", 10),
            augreg_alpha=params.get("augreg_alpha", 0.2),
            results_dir=params.get("results_dir", "/tmp/results"),
            seed=params.get("seed", 42),
            device=params.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            disable_tqdm=params.get("disable_tqdm", True),
            disable_wandb=params.get("disable_wandb", False),
            wandb_project=params.get("wandb_project", None),
            wandb_run_name=params.get("wandb_run_name", None),
        )