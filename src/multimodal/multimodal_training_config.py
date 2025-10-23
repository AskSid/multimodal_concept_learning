from dataclasses import dataclass
from typing import Optional, List, Dict, Union
import torch

@dataclass
class MultimodalTrainingConfig:
    """Configuration for multimodal training experiment."""
    
    # Dataset parameters
    mapping_path: str
    extra_mapping_path: Optional[str]
    image_root: str
    ood_labels_path: str
    prompt_template: str
    val_split: float
    dataset_name: str
    
    # Model parameters
    vision_model_name: str
    language_model_name: str
    vision_path: Optional[str]
    num_vision_tokens: int
    num_labels: int
    trainable_vision_layers: List[int]
    trainable_language_layers: List[int]
    trainable_language_embeddings: bool
    trainable_projector: bool
    use_fast_tokenizer: bool
    attn_implementation: str
    torch_dtype: Optional[str]
    
    # Training parameters
    epochs: int
    batch_size: int
    effective_batch_size: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    lr_scheduler_type: Optional[str]
    warmup_steps: int
    optimizer_type: str
    gradient_accumulation_steps: int
    
    # Training settings
    seed: int
    device: str
    mixed_precision: Optional[str]
    disable_tqdm: bool
    supervision_type: str
    
    # Data loading
    num_workers: int
    prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool
    image_size: int
    train_transforms: List[Union[str, Dict[str, Union[int, float, List[float], List[int]]]]]
    val_transforms: List[Union[str, Dict[str, Union[int, float, List[float], List[int]]]]]
    transform_params: Dict[str, Dict[str, Union[int, float, List[float], List[int]]]]
    normalize_mean: Optional[List[float]]
    normalize_std: Optional[List[float]]

    # Saving and logging
    save_dir: str
    run_name: str
    save_every_epoch: bool
    save_best_only: bool
    
    # Evaluation
    eval_steps: Optional[int]
    eval_strategy: str
    
    # Weights & Biases
    use_wandb: bool
    wandb_project: str
    wandb_run_name: Optional[str]
    
    # Distributed training
    use_accelerate: bool
    num_processes: Optional[int]
    split_batches: bool

    @classmethod
    def from_params(cls, params: Optional[dict]) -> "MultimodalTrainingConfig":
        params = params or {}
        return cls(
            mapping_path=params.get("mapping_path", "/users/sboppana/data/sboppana/multimodal_concept_mapping/data/imagenet100_v2/train_mapping.csv"),
            extra_mapping_path=params.get("extra_mapping_path", "/users/sboppana/data/sboppana/multimodal_concept_mapping/data/imagenet100_v2/train_mapping_separate.csv"),
            image_root=params.get("image_root", "/users/sboppana/data/sboppana/multimodal_concept_mapping/data/imagenet"),
            ood_labels_path=params.get("ood_labels_path", "/users/sboppana/data/sboppana/multimodal_concept_mapping/data/imagenet100_v2/separate_and_ood_separate_synsets.txt"),
            prompt_template=params.get("prompt_template", "Is a {class_name} in the image?"),
            val_split=float(params.get("val_split", 0.1)),
            dataset_name=params.get("dataset_name", "imagenet_multimodal"),
            vision_model_name=params.get("vision_model_name", "google/vit-base-patch16-224-in21k"),
            language_model_name=params.get("language_model_name", "google/gemma-3-1b-it"),
            vision_path=params.get("vision_path", None),
            num_vision_tokens=int(params.get("num_vision_tokens", 197)),
            num_labels=int(params.get("num_labels", 100)),
            trainable_vision_layers=params.get("trainable_vision_layers", []),
            trainable_language_layers=params.get("trainable_language_layers", []),
            trainable_language_embeddings=bool(params.get("trainable_language_embeddings", True)),
            trainable_projector=bool(params.get("trainable_projector", True)),
            use_fast_tokenizer=bool(params.get("use_fast_tokenizer", True)),
            attn_implementation=params.get("attn_implementation", "eager"),
            torch_dtype=params.get("torch_dtype", "bfloat16" if torch.cuda.is_available() else None),
            epochs=int(params.get("epochs", 25)),
            batch_size=int(params.get("batch_size", 4)),
            effective_batch_size=int(params.get("effective_batch_size", 256)),
            learning_rate=float(params.get("learning_rate", 5e-4)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
            max_grad_norm=float(params.get("max_grad_norm", 1.0)),
            lr_scheduler_type=params.get("lr_scheduler_type", None),
            warmup_steps=int(params.get("warmup_steps", 0)),
            optimizer_type=params.get("optimizer_type", "adamw"),
            gradient_accumulation_steps=int(params.get("gradient_accumulation_steps", 1)),
            seed=int(params.get("seed", 42)),
            device=params.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            mixed_precision=params.get("mixed_precision", "bf16" if torch.cuda.is_available() else "no"),
            disable_tqdm=bool(params.get("disable_tqdm", True)),
            supervision_type=params.get("supervision_type", "answer_only"),
            num_workers=int(params.get("num_workers", 4)),
            prefetch_factor=int(params.get("prefetch_factor", 2)),
            pin_memory=bool(params.get("pin_memory", True)),
            persistent_workers=bool(params.get("persistent_workers", True)),
            image_size=int(params.get("image_size", 224)),
            train_transforms=params.get(
                "train_transforms",
                [
                    {"name": "Resize", "size": [256, 256]},
                    {
                        "name": "RandomResizedCrop",
                        "size": 224,
                        "scale": [0.8, 1.0],
                    },
                    "RandomHorizontalFlip",
                    {
                        "name": "ColorJitter",
                        "brightness": 0.2,
                        "contrast": 0.2,
                        "saturation": 0.2,
                        "hue": 0.1,
                    },
                    "ToTensor",
                    "Normalize",
                ],
            ),
            val_transforms=params.get(
                "val_transforms",
                [
                    {"name": "Resize", "size": [224, 224]},
                    "ToTensor",
                    "Normalize",
                ],
            ),
            transform_params=params.get("transform_params", {}),
            normalize_mean=params.get(
                "normalize_mean",
                [0.485, 0.456, 0.406],
            ),
            normalize_std=params.get(
                "normalize_std",
                [0.229, 0.224, 0.225],
            ),
            save_dir=params.get("save_dir", "/users/sboppana/data/sboppana/multimodal_concept_learning/results/multimodal"),
            run_name=params.get("run_name", "mllm_imagenet100_ood"),
            save_every_epoch=bool(params.get("save_every_epoch", False)),
            save_best_only=bool(params.get("save_best_only", True)),
            eval_steps=params.get("eval_steps", None),
            eval_strategy=params.get("eval_strategy", "epoch"),
            use_wandb=bool(params.get("use_wandb", False)),
            wandb_project=params.get("wandb_project", "multimodal-concept-learning"),
            wandb_run_name=params.get("wandb_run_name", None),
            use_accelerate=bool(params.get("use_accelerate", True)),
            num_processes=params.get("num_processes", None),
            split_batches=bool(params.get("split_batches", True)),
        )
