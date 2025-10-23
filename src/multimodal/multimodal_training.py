import argparse
import sys
import os
import torch
import yaml
import time
import warnings
import json
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup

# Suppress pydantic warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import set_seed
from src.multimodal.multimodal_training_config import MultimodalTrainingConfig
from src.multimodal.mllm import MLLM
from src.datasets.imagenet.imagenet_dataset import ImageNetDataset, MultimodalCollator


def load_multimodal_dataset(config: MultimodalTrainingConfig):
    """Load multimodal dataset with train/val split."""
    # Load main mapping
    df = pd.read_csv(config.mapping_path)
    print(f"Loaded main mapping from {config.mapping_path} with {len(df)} rows.")
    
    # Load extra mapping if provided
    if config.extra_mapping_path and os.path.exists(config.extra_mapping_path):
        print(f"Loading extra mapping from {config.extra_mapping_path}")
        extra_df = pd.read_csv(config.extra_mapping_path)
        print(f"Loaded extra mapping with {len(extra_df)} rows.")
        df = pd.concat([df, extra_df], ignore_index=True)
        print(f"After concatenation, total rows: {len(df)}")
    
    # Create train/val split
    train_idx, val_idx = train_test_split(
        df.index,
        test_size=config.val_split,
        random_state=config.seed,
        shuffle=True,
        stratify=df['target_wnid'] if 'target_wnid' in df.columns else None
    )
    
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    
    print(f"Train set: {len(train_df)} rows, Val set: {len(val_df)} rows.")
    
    # Create datasets
    train_dataset = ImageNetDataset(train_df, config.image_root, transform=None, return_synset=True)
    val_dataset = ImageNetDataset(val_df, config.image_root, transform=None, return_synset=True)
    
    return train_dataset, val_dataset


def create_transforms(config: MultimodalTrainingConfig, is_train: bool = True):
    """Create image transforms for training or validation."""
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    return transform


def init_model(config: MultimodalTrainingConfig):
    """Initialize the MLLM model."""
    model = MLLM(
        vision_model_name=config.vision_model_name,
        language_model_name=config.language_model_name,
        vision_path=config.vision_path,
        num_vision_tokens=config.num_vision_tokens,
    )
    
    # Set trainable parameters
    model.set_trainable_params(
        trainable_vision_layers=config.trainable_vision_layers,
        trainable_language_layers=config.trainable_language_layers,
        trainable_language_embeddings=config.trainable_language_embeddings,
        trainable_projector=config.trainable_projector,
    )
    
    return model


def run_training(model: MLLM, train_loader: DataLoader, val_loader: DataLoader, config: MultimodalTrainingConfig, accelerator: Accelerator):
    """Run the train/val loop."""
    # Set up optimizer
    if config.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Optimizer {config.optimizer_type} not supported.")
    
    # Set up scheduler
    if config.lr_scheduler_type == "linear":
        total_steps = len(train_loader) * config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
    else:
        scheduler = None
    
    # Prepare model, optimizer, and dataloaders with accelerate
    if scheduler is not None:
        model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
            model, optimizer, scheduler, train_loader, val_loader
        )
    else:
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
    
    # Initialize wandb if not disabled
    if config.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or config.run_name,
            config=vars(config)
        )
    
    best_loss = float("inf")
    
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        
        # Train loop
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]", disable=config.disable_tqdm)
        for batch_idx, batch in enumerate(train_pbar):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(
                        images=batch["images"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    loss = outputs.loss
                
                train_loss += loss.item()
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                
                if scheduler is not None and accelerator.sync_gradients:
                    scheduler.step()
                
                optimizer.zero_grad()
        
        train_loss /= len(train_loader)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]", disable=config.disable_tqdm)
        with torch.no_grad():
            for batch in val_pbar:
                with accelerator.autocast():
                    outputs = model(
                        images=batch["images"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    val_loss += outputs.loss.item()
        
        val_loss /= len(val_loader)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save best model
        if val_loss < best_loss and accelerator.is_main_process:
            best_loss = val_loss
            os.makedirs(os.path.join(config.save_dir, "models"), exist_ok=True)
            
            # Save unwrapped model
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(config.save_dir, "models", "best_model.pt"))
            
            # Save training config
            with open(os.path.join(config.save_dir, "models", "training_config.json"), "w") as f:
                json.dump(vars(config), f, indent=2)
        
        # Save every epoch if requested
        if config.save_every_epoch and accelerator.is_main_process:
            epoch_save_path = os.path.join(config.save_dir, "models", f"epoch_{epoch+1}")
            os.makedirs(epoch_save_path, exist_ok=True)
            
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(epoch_save_path, "model.pt"))
        
        # Log metrics
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_loss,
            'learning_rate': current_lr,
            'epoch_time': epoch_time
        }
        
        # Print metrics
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{config.epochs} completed in {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Best Val Loss: {best_loss:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print("-" * 50)
        
        # Log to wandb
        if config.use_wandb:
            wandb.log(metrics)
    
    if accelerator.is_main_process:
        print(f"Best val loss: {best_loss:.4f}")
    
    # Save final model
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), os.path.join(config.save_dir, "models", "final_model.pt"))
    
    if config.use_wandb:
        wandb.finish()


def evaluate_model(model: MLLM, test_loader: DataLoader, config: MultimodalTrainingConfig, accelerator: Accelerator):
    """Evaluate model on test set and return metrics."""
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    test_pbar = tqdm(test_loader, desc="Testing", disable=config.disable_tqdm)
    with torch.no_grad():
        for batch in test_pbar:
            with accelerator.autocast():
                outputs = model(
                    images=batch["images"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                test_loss += outputs.loss.item()
                
                # Calculate accuracy (simplified - in practice you'd need more sophisticated evaluation)
                # This is a placeholder for proper multimodal evaluation
                predictions = torch.argmax(outputs.logits, dim=-1)
                labels = batch["labels"]
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    if accelerator.is_main_process:
        print(f"Test Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
    
    return {
        'test_loss': test_loss,
        'test_acc': test_acc
    }


def main():
    # Initialize accelerate with DDP kwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=1,  # Will be set from config
        split_batches=True
    )
    
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    
    # Load config from YAML file
    with open(args.config_path, "r") as f:
        config = MultimodalTrainingConfig.from_params(yaml.safe_load(f))
    
    # Set seed
    set_seed(config.seed)
    
    # Create transforms
    train_transform = create_transforms(config, is_train=True)
    val_transform = create_transforms(config, is_train=False)
    
    # Load dataset
    train_dataset, val_dataset = load_multimodal_dataset(config)
    
    if accelerator.is_main_process:
        print(f"Loaded multimodal dataset with {len(train_dataset)} train samples and {len(val_dataset)} validation samples.")
    
    # Create results directory
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(os.path.join(config.save_dir, "models"), exist_ok=True)
    
    # Initialize model
    model = init_model(config)
    
    # Print model info
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Create collator
    collator = MultimodalCollator(
        image_processor=model.image_processor,
        tokenizer=model.tokenizer,
        num_vision_tokens=config.num_vision_tokens,
        prompt_template=config.prompt_template,
        all_class_names=train_dataset.unique_labels,
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )
    
    # Run training
    run_training(model, train_loader, val_loader, config, accelerator)
    
    # Evaluate on validation set (as test set)
    if accelerator.is_main_process:
        print("\n" + "="*50)
        print("FINAL EVALUATION ON VALIDATION SET")
        print("="*50)
    test_metrics = evaluate_model(model, val_loader, config, accelerator)
    
    # Log test metrics to wandb if enabled
    if config.use_wandb:
        wandb.log(test_metrics)


if __name__ == "__main__":
    main()
