import argparse
import sys
import os
import torch
import yaml
import time
import warnings
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb
from accelerate import Accelerator

# Suppress pydantic warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import set_seed
from src.vision.vision_training_config import VisionTrainingConfig
from src.datasets.color.color_dataset import ColorDataset
from src.datasets.imagenet.imagenet_dataset import ImageNetDataset
from transformers import ViTForImageClassification, ViTConfig


def create_transforms(config: VisionTrainingConfig, is_train: bool = True):
    """Create image transforms for training or validation."""
    if config.dataset_name == "imagenet" or config.dataset_name == "imagenet100":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet normalization
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]  # Default normalization
    
    # Transform mapping dictionary
    transform_map = {
        "RandomResizedCrop": lambda: transforms.RandomResizedCrop(config.image_size),
        "RandomHorizontalFlip": lambda: transforms.RandomHorizontalFlip(),
        "ColorJitter": lambda: transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        "RandomRotation": lambda: transforms.RandomRotation(degrees=15),
        "RandomAffine": lambda: transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0),
        "RandomPerspective": lambda: transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        "RandomErasing": lambda: transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        "Resize": lambda: transforms.Resize((config.image_size, config.image_size)),
        "ToTensor": lambda: transforms.ToTensor(),
        "Normalize": lambda: transforms.Normalize(mean, std)
    }
    
    # Get transform list based on train/val
    transform_list = config.train_transforms if is_train else config.val_transforms
    
    # Create transforms from the list
    transform_objects = []
    for transform_name in transform_list:
        if transform_name in transform_map:
            transform_objects.append(transform_map[transform_name]())
        else:
            raise ValueError(f"Unknown transform: {transform_name}")
    
    return transforms.Compose(transform_objects)


def load_split_datasets(
    dataset_cls,
    mapping_dir: str,
    data_dir: str,
    train_transform,
    val_transform,
):
    mapping_paths = {
        "train": os.path.join(mapping_dir, "train_mapping.csv"),
        "val": os.path.join(mapping_dir, "val_mapping.csv"),
        "test": os.path.join(mapping_dir, "test_mapping.csv"),
    }

    for split_name, mapping_path in mapping_paths.items():
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Missing {split_name} mapping CSV at {mapping_path}")

    train_dataset = dataset_cls(mapping_paths["train"], data_dir, transform=train_transform)
    val_dataset = dataset_cls(mapping_paths["val"], data_dir, transform=val_transform)
    test_dataset = dataset_cls(mapping_paths["test"], data_dir, transform=val_transform)

    return train_dataset, val_dataset, test_dataset


def init_model(config: VisionTrainingConfig):
    """Initialize the model."""
    if config.model_name == "vit":
        # Create HuggingFace ViT config
        vit_config = ViTConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_labels=config.num_labels,
            patch_size=config.patch_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_dropout_prob=config.attention_dropout_prob,
            num_attention_heads=config.num_attention_heads,
        )
        model = ViTForImageClassification(vit_config)
    else:
        raise ValueError(f"Model {config.model_name} not supported.")
    
    return model

def run_training(model: ViTForImageClassification, train_loader: DataLoader, val_loader: DataLoader, config: VisionTrainingConfig, accelerator: Accelerator):
    """Run the train/val loop."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01)
    
    # Use label smoothing if specified
    if config.label_smoothing > 0:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    best_loss = float("inf")
    
    # Prepare model, optimizer, and dataloaders with accelerate
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    
    # Initialize wandb if not disabled
    if not config.disable_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config)
        )
    
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        
        # Train loop
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Calculate gradient accumulation steps
        accumulation_steps = config.effective_batch_size // config.batch_size
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]", disable=config.disable_tqdm)
        for batch_idx, (images, labels) in enumerate(train_pbar):
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            accelerator.backward(loss)
            
            # Update weights only after accumulating gradients
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Val loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]", disable=config.disable_tqdm)
        with torch.no_grad():
            for images, labels in val_pbar:
                outputs = model(images)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.join(config.results_dir, "models"), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config.results_dir, "models", "best_model.pt"))
        
        # Log metrics
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'learning_rate': current_lr,
            'epoch_time': epoch_time
        }
        
        # Print metrics
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{config.epochs} completed in {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            print("-" * 50)
        
        # Log to wandb
        if not config.disable_wandb:
            wandb.log(metrics)
        
        scheduler.step()
    
    if accelerator.is_main_process:
        print(f"Best val loss: {best_loss:.4f}")
    torch.save(model.state_dict(), os.path.join(config.results_dir, "models", "final_model.pt"))
    
    if not config.disable_wandb:
        wandb.finish()

def evaluate_model(model: ViTForImageClassification, test_loader: DataLoader, config: VisionTrainingConfig, accelerator: Accelerator):
    """Evaluate model on test set and return metrics."""
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    # Use label smoothing if specified
    if config.label_smoothing > 0:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Prepare test loader with accelerate
    test_loader = accelerator.prepare(test_loader)
    
    test_pbar = tqdm(test_loader, desc="Testing", disable=config.disable_tqdm)
    with torch.no_grad():
        for images, labels in test_pbar:
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            test_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.logits.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * test_correct / test_total
    
    if accelerator.is_main_process:
        print(f"Test Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
    
    return {
        'test_loss': test_loss,
        'test_acc': test_acc
    }

def main():
    # Initialize accelerate
    accelerator = Accelerator()
    
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)

    args = parser.parse_args()
    
    # Load config from YAML file
    with open(args.config_path, "r") as f:
        config = VisionTrainingConfig.from_params(yaml.safe_load(f))
    
    # Set seed
    set_seed(config.seed)

    # Create transforms
    train_transform = create_transforms(config, is_train=True)
    val_transform = create_transforms(config, is_train=False)
    
    # Load dataset
    if config.dataset_name == "color":
        train_dataset, val_dataset, test_dataset = load_split_datasets(
            ColorDataset,
            mapping_dir=config.data_dir,
            data_dir=config.data_dir,
            train_transform=train_transform,
            val_transform=val_transform,
        )

    elif config.dataset_name == "imagenet100":
        train_dataset, val_dataset, test_dataset = load_split_datasets(
            ImageNetDataset,
            mapping_dir="/users/sboppana/data/sboppana/data/multimodal_concept_learning/imagenet100",
            data_dir=config.data_dir,
            train_transform=train_transform,
            val_transform=val_transform,
        )

    else:
        raise ValueError(f"Dataset {config.dataset_name} not supported.")

    if hasattr(train_dataset, "num_classes"):
        config.num_labels = train_dataset.num_classes


    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    if accelerator.is_main_process:
        print(f"Loaded {config.dataset_name} dataset with {len(train_dataset)} train samples, {len(val_dataset)} validation samples, and {len(test_dataset)} test samples.")
    
    # Create results directory
    os.makedirs(config.results_dir, exist_ok=True)
    
    # init model
    model = init_model(config)
    
    # Run training
    run_training(model, train_loader, val_loader, config, accelerator)
    
    # Evaluate on test set
    if accelerator.is_main_process:
        print("\n" + "="*50)
        print("FINAL EVALUATION ON TEST SET")
        print("="*50)
    test_metrics = evaluate_model(model, test_loader, config, accelerator)
    
    # Log test metrics to wandb if enabled
    if not config.disable_wandb:
        wandb.log(test_metrics)



if __name__ == "__main__":
    main()
