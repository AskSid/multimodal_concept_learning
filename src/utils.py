import random
from typing import TYPE_CHECKING, Any, Dict, Union

import numpy as np
import torch
from torchvision import transforms


if TYPE_CHECKING:
    from src.multimodal.multimodal_training_config import MultimodalTrainingConfig
    from src.vision.vision_training_config import VisionTrainingConfig


_TransformSpec = Union[str, Dict[str, Any]]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ensure_tuple(value: Any):
    if isinstance(value, list):
        return tuple(value)
    return value


def _resolve_mean_std(config, params: Dict[str, Any]):
    mean = params.get("mean", getattr(config, "normalize_mean", None))
    std = params.get("std", getattr(config, "normalize_std", None))

    if mean is None or std is None:
        dataset_name = getattr(config, "dataset_name", None)
        if dataset_name in {"imagenet", "imagenet100", "imagenet_multimodal"}:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

    return mean, std


def _build_resize(config, params: Dict[str, Any]):
    size = _ensure_tuple(params.get("size"))
    if size is None:
        size = getattr(config, "image_size", 224)
    return transforms.Resize(size)


def _build_random_resized_crop(config, params: Dict[str, Any]):
    size = params.get("size")
    if size is None:
        size = getattr(config, "image_size", 224)
    scale = params.get("scale")
    ratio = params.get("ratio")
    kwargs = {"size": size}
    if scale is not None:
        kwargs["scale"] = tuple(scale) if isinstance(scale, (list, tuple)) else scale
    if ratio is not None:
        kwargs["ratio"] = tuple(ratio) if isinstance(ratio, (list, tuple)) else ratio
    return transforms.RandomResizedCrop(**kwargs)


def _build_random_horizontal_flip(_config, params: Dict[str, Any]):
    return transforms.RandomHorizontalFlip(p=params.get("p", 0.5))


def _build_color_jitter(_config, params: Dict[str, Any]):
    defaults = {
        "brightness": 0.4,
        "contrast": 0.4,
        "saturation": 0.4,
        "hue": 0.1,
    }
    defaults.update(params)
    return transforms.ColorJitter(
        brightness=defaults.get("brightness"),
        contrast=defaults.get("contrast"),
        saturation=defaults.get("saturation"),
        hue=defaults.get("hue"),
    )


def _build_random_rotation(_config, params: Dict[str, Any]):
    return transforms.RandomRotation(degrees=params.get("degrees", 15))


def _build_random_affine(_config, params: Dict[str, Any]):
    return transforms.RandomAffine(
        degrees=params.get("degrees", 0),
        translate=params.get("translate", (0.1, 0.1)),
        scale=params.get("scale", (0.9, 1.1)),
        shear=params.get("shear", 0),
    )


def _build_random_perspective(_config, params: Dict[str, Any]):
    return transforms.RandomPerspective(
        distortion_scale=params.get("distortion_scale", 0.2),
        p=params.get("p", 0.5),
    )


def _build_random_erasing(_config, params: Dict[str, Any]):
    return transforms.RandomErasing(
        p=params.get("p", 0.25),
        scale=params.get("scale", (0.02, 0.33)),
        ratio=params.get("ratio", (0.3, 3.3)),
    )


def _build_to_tensor(_config, _params: Dict[str, Any]):
    return transforms.ToTensor()


def _build_normalize(config, params: Dict[str, Any]):
    mean, std = _resolve_mean_std(config, params)
    return transforms.Normalize(mean=mean, std=std)


_TRANSFORM_FACTORIES = {
    "Resize": _build_resize,
    "RandomResizedCrop": _build_random_resized_crop,
    "RandomHorizontalFlip": _build_random_horizontal_flip,
    "ColorJitter": _build_color_jitter,
    "RandomRotation": _build_random_rotation,
    "RandomAffine": _build_random_affine,
    "RandomPerspective": _build_random_perspective,
    "RandomErasing": _build_random_erasing,
    "ToTensor": _build_to_tensor,
    "Normalize": _build_normalize,
}


def create_transforms(
    config: Union["MultimodalTrainingConfig", "VisionTrainingConfig"],
    is_train: bool = True,
):
    """Create image transforms based on configuration-provided specs."""

    transform_entries = getattr(
        config,
        "train_transforms" if is_train else "val_transforms",
        None,
    )
    if not transform_entries:
        raise ValueError("Transform list is empty or undefined in config.")

    transform_params = getattr(config, "transform_params", {}) or {}

    transforms_to_apply = []
    for entry in transform_entries:
        if isinstance(entry, dict):
            name = entry.get("name")
            if not name:
                raise ValueError("Transform dict entries must include a 'name' key.")
            entry_params = {k: v for k, v in entry.items() if k != "name"}
        elif isinstance(entry, str):
            name = entry
            entry_params = {}
        else:
            raise TypeError(
                "Transform entries must be either strings or dictionaries with a 'name' key."
            )

        factory = _TRANSFORM_FACTORIES.get(name)
        if factory is None:
            raise ValueError(f"Unknown transform: {name}")

        params = dict(transform_params.get(name, {}))
        params.update(entry_params)
        transforms_to_apply.append(factory(config, params))

    return transforms.Compose(transforms_to_apply)
