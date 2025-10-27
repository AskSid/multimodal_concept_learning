import argparse
import sys
import os
import torch
import yaml
import time
import warnings
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from transformers import get_linear_schedule_with_warmup

# Suppress pydantic warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import set_seed, create_transforms
from src.multimodal.multimodal_training_config import MultimodalTrainingConfig
from src.multimodal.mllm import MLLM
from src.datasets.imagenet.imagenet_dataset import ImageNetDataset, MultimodalCollator
from src.datasets.color.color_dataset import ColorDataset


def load_split_datasets(
    dataset_cls,
    mapping_dir: str,
    data_dir: str,
    train_transform,
    val_transform,
):
    """Load train/val/test datasets using pre-constructed splits."""
    mapping_paths = {
        "train": os.path.join(mapping_dir, "train_mapping.csv"),
        "val": os.path.join(mapping_dir, "val_mapping.csv"),
        "test": os.path.join(mapping_dir, "test_mapping.csv"),
    }

    for split_name, mapping_path in mapping_paths.items():
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Missing {split_name} mapping CSV at {mapping_path}")

    train_dataset = dataset_cls(mapping_paths["train"], data_dir, transform=train_transform, return_synset=True)
    val_dataset = dataset_cls(mapping_paths["val"], data_dir, transform=val_transform, return_synset=True)
    test_dataset = dataset_cls(mapping_paths["test"], data_dir, transform=val_transform, return_synset=True)

    return train_dataset, val_dataset, test_dataset



def init_model(config: MultimodalTrainingConfig):
    """Initialize the MLLM model."""
    model = MLLM(
        vision_model_name=config.vision_model_name,
        language_model_name=config.language_model_name,
        vision_path=config.vision_path,
        num_vision_tokens=config.num_vision_tokens,
        labels_mapping_path=config.labels_mapping_path,
    )
    
    # Set trainable parameters
    model.set_trainable_params(config.trainable_params_setting)
    
    return model


def run_training(model: MLLM, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, config: MultimodalTrainingConfig, accelerator: Accelerator):
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
    
    # Save initial model
    if accelerator.is_main_process:
        os.makedirs(os.path.join(config.results_dir, "models"), exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), os.path.join(config.results_dir, "models", "initial_model.pt"))
        unwrapped_model.tokenizer.save_pretrained(os.path.join(config.results_dir, "models", "tokenizer"))
    
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
            os.makedirs(os.path.join(config.results_dir, "models"), exist_ok=True)
            
            # Save unwrapped model
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(config.results_dir, "models", "best_model.pt"))
            
            # Save training config
            with open(os.path.join(config.results_dir, "models", "training_config.json"), "w") as f:
                json.dump(vars(config), f, indent=2)
        
        # Save every epoch if requested
        if config.save_every_epoch and accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(config.results_dir, "models", f"epoch_{epoch}_model.pt"))

        
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
        if config.use_wandb and accelerator.is_main_process:
            wandb.log(metrics)
    
    if accelerator.is_main_process:
        print(f"Best val loss: {best_loss:.4f}")
    
    # Load best model for final evaluation
    if accelerator.is_main_process:
        print("\n" + "="*50)
        print("FINAL EVALUATION ON VALIDATION SET")
        print("="*50)
        
        # Load best model
        best_model_path = os.path.join(config.results_dir, "models", "best_model.pt")
        if os.path.exists(best_model_path):
            unwrapped_model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
            print(f"Loaded best model from {best_model_path}")
        else:
            print("Best model not found, using final model")
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, config, accelerator)
    
    if accelerator.is_main_process:
        print(f"\nFinal Test Results:")
        print(f"Test Loss: {test_metrics['test_loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['test_acc']:.4f}")
    
    if config.use_wandb and accelerator.is_main_process:
        wandb.finish()


def evaluate_model(model: MLLM, test_loader: DataLoader, config: MultimodalTrainingConfig, accelerator: Accelerator):
    """Evaluate model on test set and return metrics."""
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    # Get the unwrapped model to access tokenizer
    unwrapped_model = accelerator.unwrap_model(model)
    
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
                
                # Evaluate using string matching for yes/no tasks
                logits = outputs.logits
                labels = batch["labels"]
                predicted_ids = torch.argmax(logits, dim=-1)

                # Process each sample in the batch
                batch_size = predicted_ids.size(0)
                for i in range(batch_size):
                    # Get valid positions (where labels != -100)
                    valid_mask = labels[i] != -100
                    if not valid_mask.any():
                        continue
                    
                    # Get the predicted and ground truth tokens for valid positions
                    pred_tokens = predicted_ids[i][valid_mask].cpu().tolist()
                    true_tokens = labels[i][valid_mask].cpu().tolist()
                    
                    # Convert to strings for comparison
                    pred_text = unwrapped_model.tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
                    true_text = unwrapped_model.tokenizer.decode(true_tokens, skip_special_tokens=True).strip()
                    
                    # Determine if prediction is correct based on yes/no
                    pred_is_yes = "yes" in pred_text.lower()
                    true_is_yes = "yes" in true_text.lower()
                    
                    # Check if the yes/no prediction matches
                    is_correct = (pred_is_yes == true_is_yes)
                    
                    if is_correct:
                        correct_predictions += 1
                    total_predictions += 1
    
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
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    
    # Load config from YAML file
    with open(args.config_path, "r") as f:
        config = MultimodalTrainingConfig.from_params(yaml.safe_load(f))

    # Calculate gradient accumulation steps from effective batch size
    assert config.effective_batch_size % config.batch_size == 0, f"effective_batch_size ({config.effective_batch_size}) must be divisible by batch_size ({config.batch_size})"
    gradient_accumulation_steps = config.effective_batch_size // config.batch_size
    
    # Initialize accelerate with configuration-driven settings
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=gradient_accumulation_steps,
        split_batches=config.split_batches,
        mixed_precision=config.mixed_precision,
    )
    
    # Set seed
    set_seed(config.seed)
    
    # Create transforms
    train_transform = create_transforms(config, is_train=True)
    val_transform = create_transforms(config, is_train=False)
    
    # Load dataset using pre-constructed splits
    # Select dataset class based on config
    if config.dataset_name == "color_multimodal":
        dataset_cls = ColorDataset
    elif config.dataset_name == "imagenet_multimodal":
        dataset_cls = ImageNetDataset
    else:
        raise ValueError(f"Dataset {config.dataset_name} not supported.")
    
    train_dataset, val_dataset, test_dataset = load_split_datasets(
        dataset_cls,
        mapping_dir=os.path.dirname(config.mapping_path),  # Extract mapping directory from mapping_path
        data_dir=config.image_root,
        train_transform=train_transform,
        val_transform=val_transform,
    )
    
    if accelerator.is_main_process:
        print(f"Loaded multimodal dataset with {len(train_dataset)} train samples, {len(val_dataset)} validation samples, and {len(test_dataset)} test samples.")
    
    # Create results directory
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(os.path.join(config.results_dir, "models"), exist_ok=True)
    
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
        tokenizer=model.tokenizer,
        num_vision_tokens=config.num_vision_tokens,
        prompt_template=config.prompt_template,
        all_class_names=train_dataset.unique_labels,
        labels_mapping=model.labels_mapping,
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )
    
    # Prepare all loaders with accelerate
    train_loader, val_loader, test_loader = accelerator.prepare(train_loader, val_loader, test_loader)
    
    # Run training (includes evaluation)
    run_training(model, train_loader, val_loader, test_loader, config, accelerator)


if __name__ == "__main__":
    main()
