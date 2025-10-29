import torch
import torch.nn as nn
import json
from transformers import (
    ViTModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
)
from typing import List, Optional, Union
from PIL import Image


class MLLM(torch.nn.Module):
    """Multimodal LLM combining pre-trained vision and language model with a projector."""
    
    def __init__(
        self,
        vision_model_name: str = "google/vit-base-patch16-224-in21k",
        language_model_name: str = "google/gemma-3-1b-it",
        vision_path: Optional[str] = None,
        num_vision_tokens: int = 197,
        labels_mapping_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name
        self.num_vision_tokens = num_vision_tokens
        
        # Load vision model
        if vision_path is not None:
            self.vision_model = ViTModel.from_pretrained(vision_path)
        else:
            # Use AutoModelForImageClassification for timm models
            if 'timm' in vision_model_name:
                self.vision_model = AutoModelForImageClassification.from_pretrained(vision_model_name)
            else:
                self.vision_model = ViTModel.from_pretrained(vision_model_name)
        
        # Load language model
        model_kwargs = {"attn_implementation": "eager", "torch_dtype": torch.bfloat16}
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_name, **model_kwargs
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            language_model_name, use_fast=True
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load labels mapping and add OOD tokens if provided
        self.labels_mapping = None
        if labels_mapping_path is not None:
            with open(labels_mapping_path, 'r') as f:
                self.labels_mapping = json.load(f)
            
            # Extract OOD tokens (those starting with "<ood")
            ood_tokens = [label for label in self.labels_mapping.values() if label.startswith("<ood")]
            
            if ood_tokens:
                # Add OOD tokens to tokenizer
                self.tokenizer.add_tokens(ood_tokens)
                # Resize language model embeddings to accommodate new tokens
                self.language_model.resize_token_embeddings(len(self.tokenizer))
                
                # Initialize new token embeddings by copying from existing tokens
                new_embeddings = self.language_model.get_input_embeddings()
                new_embeddings.weight.data[-len(ood_tokens):] = new_embeddings.weight.data[:len(ood_tokens)].clone()
        
        # Projector to align vision and language embeddings
        if 'timm' in vision_model_name:
            # For timm models, get hidden size from the timm_model component
            vision_hidden_size = self.vision_model.timm_model.embed_dim
        else:
            vision_hidden_size = self.vision_model.config.hidden_size
            
        self.projector = nn.Linear(
            vision_hidden_size,
            self.language_model.config.hidden_size
        )
        
        # EOS token for generation
        self.eos_token = self.tokenizer.decode(self.tokenizer.eos_token_id)
    
    def forward(self, images, input_ids, attention_mask, labels=None):
        """Forward pass for training."""
        # Get vision embeddings
        if 'timm' in self.vision_model_name:
            # For timm models loaded with AutoModelForImageClassification
            image_embeds = self.vision_model.timm_model.forward_features(images)
        else:
            # For standard ViT models
            vision_outputs = self.vision_model(
                pixel_values=images, 
                output_hidden_states=True, 
                return_dict=True
            )
            image_embeds = vision_outputs.last_hidden_state
        
        projected_image_embeds = self.projector(image_embeds)
        
        # Get language embeddings
        input_embedding_layer = self.language_model.get_input_embeddings()
        language_embeds = input_embedding_layer(input_ids)
        
        # Replace vision token positions with projected image embeddings
        language_embeds[:, :self.num_vision_tokens, :] = projected_image_embeds
        
        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=language_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs
    
    def get_vision_projected_embeds(self, images):
        """Get projected vision embeddings for a batch of images."""
        self.eval()
        with torch.no_grad():
            if 'timm' in self.vision_model_name:
                # For timm models loaded with AutoModelForImageClassification
                image_embeds = self.vision_model.timm_model.forward_features(images)
            else:
                # For standard ViT models
                vision_outputs = self.vision_model(
                    pixel_values=images, 
                    output_hidden_states=True, 
                    return_dict=True
                )
                image_embeds = vision_outputs.last_hidden_state
            projected_embeds = self.projector(image_embeds)
        return projected_embeds
    
    def get_vision_embeds(self, images):
        """Get raw vision embeddings for a batch of images."""
        self.eval()
        with torch.no_grad():
            if 'timm' in self.vision_model_name:
                # For timm models loaded with AutoModelForImageClassification
                return self.vision_model.timm_model.forward_features(images)
            else:
                # For standard ViT models
                vision_outputs = self.vision_model(
                    pixel_values=images, 
                    output_hidden_states=True, 
                    return_dict=True
                )
                return vision_outputs.last_hidden_state
    
    def set_trainable_params(self, trainable_params_setting: str):
        """Set trainable parameters based on the specified setting.
        
        Args:
            trainable_params_setting: One of "vision_only", "language_only", or "language_embed_only"
        """
        # First, freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Always make projector trainable
        for param in self.projector.parameters():
            param.requires_grad = True
        
        if trainable_params_setting == "vision_only":
            # Only train vision model
            for param in self.vision_model.parameters():
                param.requires_grad = True
                
        elif trainable_params_setting == "language_only":
            # Train entire language model
            for param in self.language_model.parameters():
                param.requires_grad = True
                
        elif trainable_params_setting == "language_embed_only":
            # Only train language model input embeddings
            for param in self.language_model.get_input_embeddings().parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown trainable_params_setting: {trainable_params_setting}")
        
        # Print trainable parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable params setting: {trainable_params_setting}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")