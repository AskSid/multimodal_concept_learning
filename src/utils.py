import random
from typing import TYPE_CHECKING

import numpy as np
import torch
from torchvision import transforms


if TYPE_CHECKING:
    from src.multimodal.multimodal_training_config import MultimodalTrainingConfig
    from src.vision.vision_training_config import VisionTrainingConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_multimodal_transforms(
    config: "MultimodalTrainingConfig",
    is_train: bool = True,
):
    """Image transforms for multimodal training."""

    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def create_vision_transforms(
    config: "VisionTrainingConfig",
    is_train: bool = True,
):
    """Image transforms for vision-only training."""

    if config.dataset_name in {"imagenet", "imagenet100"}:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    transform_map = {
        "RandomResizedCrop": lambda: transforms.RandomResizedCrop(config.image_size),
        "RandomHorizontalFlip": lambda: transforms.RandomHorizontalFlip(),
        "ColorJitter": lambda: transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        ),
        "RandomRotation": lambda: transforms.RandomRotation(degrees=15),
        "RandomAffine": lambda: transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=0,
        ),
        "RandomPerspective": lambda: transforms.RandomPerspective(
            distortion_scale=0.2,
            p=0.5,
        ),
        "RandomErasing": lambda: transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
        ),
        "Resize": lambda: transforms.Resize((config.image_size, config.image_size)),
        "ToTensor": lambda: transforms.ToTensor(),
        "Normalize": lambda: transforms.Normalize(mean, std),
    }

    transform_list = config.train_transforms if is_train else config.val_transforms

    transform_objects = []
    for transform_name in transform_list:
        if transform_name not in transform_map:
            raise ValueError(f"Unknown transform: {transform_name}")
        transform_objects.append(transform_map[transform_name]())

    return transforms.Compose(transform_objects)
