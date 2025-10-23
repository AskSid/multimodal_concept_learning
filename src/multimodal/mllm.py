import torch
import torch.nn as nn
from transformers import (
    ViTModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoImageProcessor,
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
    ):
        super().__init__()
        
        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name
        self.num_vision_tokens = num_vision_tokens
        
        # Load vision model
        if vision_path is not None:
            self.vision_model = ViTModel.from_pretrained(vision_path)
        else:
            self.vision_model = ViTModel.from_pretrained(vision_model_name)
        
        # Load language model
        model_kwargs = {"attn_implementation": "eager", "torch_dtype": torch.bfloat16}
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_name, **model_kwargs
        )
        
        # Load tokenizer and image processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            language_model_name, use_fast=True
        )
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Projector to align vision and language embeddings
        self.projector = nn.Linear(
            self.vision_model.config.hidden_size,
            self.language_model.config.hidden_size
        )
        
        # EOS token for generation
        self.eos_token = self.tokenizer.decode(self.tokenizer.eos_token_id)
    
    def forward(self, images, input_ids, attention_mask, labels=None):
        """Forward pass for training."""
        # Get vision embeddings
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
            vision_outputs = self.vision_model(
                pixel_values=images, 
                output_hidden_states=True, 
                return_dict=True
            )
            return vision_outputs.last_hidden_state
    
    
    def set_trainable_params(
        self,
        trainable_vision_layers: List[int] = None,
        trainable_language_layers: List[int] = None,
        trainable_language_embeddings: bool = True,
        trainable_projector: bool = True,
    ):
        """Set which parameters are trainable."""
        # Freeze all parameters first
        for p in self.parameters():
            p.requires_grad = False
        
        # Unfreeze vision layers
        if trainable_vision_layers is not None:
            total_vision_layers = len(self.vision_model.encoder.layer)
            for layer_idx in range(total_vision_layers):
                if layer_idx in trainable_vision_layers:
                    for p in self.vision_model.encoder.layer[layer_idx].parameters():
                        p.requires_grad = True
        
        # Unfreeze language layers
        if trainable_language_layers is not None:
            # Find language model layers (architecture-agnostic)
            lm = self.language_model
            layer_container = None
            
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                layer_container = lm.model.layers
            elif hasattr(lm, "model") and hasattr(lm.model, "decoder") and hasattr(lm.model.decoder, "layers"):
                layer_container = lm.model.decoder.layers
            elif hasattr(lm, "transformer") and hasattr(lm.transformer, "h"):
                layer_container = lm.transformer.h
            elif hasattr(lm, "layers"):
                layer_container = lm.layers
            
            if layer_container is not None:
                total_l_layers = len(layer_container)
                for layer_idx in range(total_l_layers):
                    if layer_idx in trainable_language_layers:
                        for p in layer_container[layer_idx].parameters():
                            p.requires_grad = True
        
        # Unfreeze language embeddings
        if trainable_language_embeddings:
            for p in self.language_model.get_input_embeddings().parameters():
                p.requires_grad = True
        
        # Unfreeze projector
        if trainable_projector:
            for p in self.projector.parameters():
                p.requires_grad = True
