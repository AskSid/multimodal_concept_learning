import argparse
import os
import torch
import yaml
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

from src.utils import set_seed
from src.vision.vision_training_config import VisionTrainingConfig
from src.datasets.color_dataset import ColorDataset
from transformers import ViTForImageClassification, ViTConfig


def create_transforms(config: VisionTrainingConfig, is_train: bool = True):
    """Create image transforms for training or validation."""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    return transform

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
    
    model.to(config.device)
    return model

def run_training(model: ViTForImageClassification, train_loader: DataLoader, val_loader: DataLoader, config: VisionTrainingConfig):
    """Run the train/val loop."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01)
    
    # Use label smoothing if specified
    if config.label_smoothing > 0:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    best_loss = float("inf")
    
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
            images, labels = images.to(config.device), labels.to(config.device)
            
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
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
                images, labels = images.to(config.device), labels.to(config.device)
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
        print(f"Epoch {epoch+1}/{config.epochs} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        print("-" * 50)
        
        # Log to wandb
        if not config.disable_wandb:
            wandb.log(metrics)
        
        scheduler.step()
    
    print(f"Best val loss: {best_loss:.4f}")
    torch.save(model.state_dict(), os.path.join(config.results_dir, "models", "final_model.pt"))
    
    if not config.disable_wandb:
        wandb.finish()

def evaluate_model(model: ViTForImageClassification, test_loader: DataLoader, config: VisionTrainingConfig):
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
    
    test_pbar = tqdm(test_loader, desc="Testing", disable=config.disable_tqdm)
    with torch.no_grad():
        for images, labels in test_pbar:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            test_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.logits.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * test_correct / test_total
    
    print(f"Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    return {
        'test_loss': test_loss,
        'test_acc': test_acc
    }

def main():
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
        # First, get the actual size by loading all data
        temp_dataset = []
        if os.path.exists(config.data_dir):
            color_dirs = os.listdir(config.data_dir)
            for color_dir in color_dirs:
                color_path = os.path.join(config.data_dir, color_dir)
                if os.path.isdir(color_path):
                    color_name = color_dir
                    images = os.listdir(color_path)
                    temp_dataset.extend([(os.path.join(color_path, image), color_name) for image in images])
        
        actual_size = len(temp_dataset)
        
        # Split into train/val/test
        indices = list(range(actual_size))
        labels = [temp_dataset[i][1] for i in indices]
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=config.seed, stratify=labels)
        
        # Second split: train vs val from train+val
        train_val_labels = [labels[i] for i in train_val_idx]
        train_idx, val_idx = train_test_split(train_val_idx, test_size=config.val_split, random_state=config.seed, stratify=train_val_labels)
        
        # Create datasets
        train_dataset = ColorDataset(config.data_dir, indices=train_idx, transform=train_transform)
        val_dataset = ColorDataset(config.data_dir, indices=val_idx, transform=val_transform)
        test_dataset = ColorDataset(config.data_dir, indices=test_idx, transform=val_transform)
        
    else:
        raise ValueError(f"Dataset {config.dataset_name} not supported.")
    
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

    print(f"Loaded {config.dataset_name} dataset with {len(train_dataset)} train samples, {len(val_dataset)} validation samples, and {len(test_dataset)} test samples.")
    
    # Create results directory
    os.makedirs(config.results_dir, exist_ok=True)
    
    # init model
    model = init_model(config)
    
    # Run training
    run_training(model, train_loader, val_loader, config)
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50)
    test_metrics = evaluate_model(model, test_loader, config)
    
    # Log test metrics to wandb if enabled
    if not config.disable_wandb:
        wandb.log(test_metrics)



if __name__ == "__main__":
    main()