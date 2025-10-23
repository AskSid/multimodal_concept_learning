from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class ColorDatasetConfig:
    """Configuration for generating color datasets."""
    
    # Dataset parameters
    dataset_name: str
    image_size: int
    colors: List[Tuple[int, int, int]]  # RGB values
    radius_range: Tuple[int, int]  # (min_radius, max_radius)
    n_images_per_color: int
    
    # Color variation parameters
    min_intensity: float
    max_intensity: float
    
    # Split parameters
    train_val_test_split: List[float]

    # Output parameters
    data_dir: str
    seed: int
    
    @classmethod
    def from_params(cls, params: Optional[dict]) -> "ColorDatasetConfig":
        params = params or {}
        return cls(
            dataset_name=params.get("dataset_name", "color_dataset"),
            image_size=params.get("image_size", 224),
            colors=params.get("colors", [(255, 0, 0), (0, 255, 0), (0, 0, 255)]),  # Red, Green, Blue
            radius_range=params.get("radius_range", (10, 50)),
            n_images_per_color=params.get("n_images_per_color", 100),
            min_intensity=params.get("min_intensity", 0.3),
            max_intensity=params.get("max_intensity", 0.95),
            train_val_test_split=params.get("train_val_test_split", [0.7, 0.15, 0.15]),
            data_dir=params.get("data_dir", "/users/sboppana/data/sboppana/data/multimodal_concept_learning/"),
            seed=int(params.get("seed", 42)),
        )
