import os
import random
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable, List
from transformers import AutoTokenizer


class ImageNetDataset(Dataset):
    def __init__(self, mapping_csv_path: str, data_dir: str, transform: Optional[Callable] = None, return_synset: bool = False):
        self.data_dir = data_dir
        self.transform = transform
        self.return_synset = return_synset
        
        # Load mapping CSV
        self.mapping_df = pd.read_csv(mapping_csv_path)
        
        # Create dataset
        self.dataset = []
        for _, row in self.mapping_df.iterrows():
            # Standardized CSV format: filename, target_synset, class_name
            image_path = os.path.join(self.data_dir, row['filename'])
            
            if self.return_synset:
                # For multimodal, return class_name
                class_name = row['class_name']
                self.dataset.append((image_path, class_name))
            else:
                # For vision, return target_synset
                target_synset = row['target_synset']
                self.dataset.append((image_path, target_synset))
        
        # Create label mapping
        self.unique_labels = sorted(list(set([item[1] for item in self.dataset])))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.num_classes = len(self.unique_labels)
        
        print(f"Loaded {len(self.dataset)} images with {self.num_classes} classes")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image_path, label_data = self.dataset[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.return_synset:
            return image, label_data
        else:
            label = self.label_to_idx[label_data]
            return image, label


class MultimodalCollator:
    """Collator for multimodal training with yes/no questions."""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_vision_tokens: int,
        prompt_template: str = "Is a {class_name} in the image?",
        all_class_names: Optional[List[str]] = None,
        image_processor=None,
    ):
        self.tokenizer = tokenizer
        self.num_vision_tokens = num_vision_tokens
        self.prompt_template = prompt_template
        self.all_class_names = sorted(set(all_class_names)) if all_class_names is not None else None
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = self.tokenizer.pad_token_id
        
        # Pre-tokenize answer tokens for efficiency
        self.yes_token_ids = self.tokenizer(" Yes", add_special_tokens=False).input_ids
        self.no_token_ids = self.tokenizer(" No", add_special_tokens=False).input_ids
    
    def __call__(self, batch):
        """Collate batch of (image, class_name) pairs into multimodal training format."""
        images, texts, label_token_ids = [], [], []
        
        for image, class_name in batch:
            images.append(image)
            
            # Generate prompt with random yes/no
            is_yes = random.random() < 0.5
            
            if is_yes:
                # Use actual class name
                class_name_for_prompt = class_name
            else:
                # Sample different class for negative example
                if self.all_class_names:
                    other_classes = [c for c in self.all_class_names if c != class_name]
                    class_name_for_prompt = random.choice(other_classes) if other_classes else class_name
                else:
                    class_name_for_prompt = class_name
            
            # Create prompt and answer
            prompt = self.prompt_template.format(class_name=class_name_for_prompt)
            text = prompt + (" Yes" if is_yes else " No")
            texts.append(text)
            label_token_ids.append(self.yes_token_ids if is_yes else self.no_token_ids)
        
        # Tokenize texts
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens=True,
        )
        text_input_ids = tokenized["input_ids"]
        text_attention_mask = tokenized["attention_mask"]
        
        # Prepend vision tokens
        B = text_input_ids.size(0)
        image_pad = torch.full((B, self.num_vision_tokens), self.pad_id, dtype=torch.long)
        input_ids = torch.cat([image_pad, text_input_ids], dim=1)
        image_attn = torch.ones((B, self.num_vision_tokens), dtype=torch.long)
        attention_mask = torch.cat([image_attn, text_attention_mask], dim=1)
        
        # Create labels (supervise only on answer tokens)
        labels = input_ids.clone()
        labels[:, :self.num_vision_tokens] = -100  # Mask vision tokens
        labels[:, self.num_vision_tokens:][text_input_ids == self.pad_id] = -100  # Mask padding
        
        # Mask everything except answer tokens
        for i in range(B):
            text_ids = text_input_ids[i].tolist()
            answer_tokens = label_token_ids[i]
            
            # Find answer token positions
            answer_start = None
            for j in range(len(text_ids) - len(answer_tokens) + 1):
                if text_ids[j:j+len(answer_tokens)] == answer_tokens:
                    answer_start = j
                    break
            
            if answer_start is not None:
                # Mask everything except answer tokens
                labels[i, :] = -100
                for k, token_id in enumerate(answer_tokens):
                    labels[i, self.num_vision_tokens + answer_start + k] = token_id
        
        return {
            "images": torch.stack(images),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
