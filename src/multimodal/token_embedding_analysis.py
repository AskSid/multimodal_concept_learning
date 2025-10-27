#!/usr/bin/env python3
"""
Token Embedding Analysis Script

This script analyzes token embeddings from trained multimodal models,
creating UMAP visualizations and distance analyses for regular and OOD tokens.
"""

# Standard library imports
import argparse
import sys
import os
import json
import warnings
import time
import re

# Third-party imports
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import plotly.graph_objects as go
import plotly.express as px

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")

# Add project root to path
current_dir = os.getcwd()
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Project imports
from src.utils import set_seed, create_transforms
from src.multimodal.multimodal_training_config import MultimodalTrainingConfig
from src.multimodal.mllm import MLLM
from src.datasets.imagenet.imagenet_dataset import ImageNetDataset, MultimodalCollator
from src.datasets.color.color_dataset import ColorDataset


def load_token_embeddings(results_dir: str):
    """
    Load token embeddings from all epochs of a trained model.
    
    Args:
        results_dir: Path to the results directory containing the trained model
        
    Returns:
        tuple: (embeddings_by_epoch, tokenizer, config)
    """
    models_dir = os.path.join(results_dir, "models")
    
    # Load training config
    config_path = os.path.join(models_dir, "training_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Training config not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create config object
    config = MultimodalTrainingConfig.from_params(config_dict)
    
    # Load saved tokenizer (includes OOD tokens that were added during training)
    tokenizer_path = os.path.join(models_dir, "tokenizer")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Saved tokenizer not found at {tokenizer_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    print(f"Loaded saved tokenizer with vocabulary size: {len(tokenizer)}")
    
    # Initialize model
    model = MLLM(
        vision_model_name=config.vision_model_name,
        language_model_name=config.language_model_name,
        vision_path=config.vision_path,
        num_vision_tokens=config.num_vision_tokens,
        labels_mapping_path=config.labels_mapping_path
    )
    
    # Dictionary to store embeddings for each epoch
    embeddings_by_epoch = {}
    
    # Load initial model
    initial_model_path = os.path.join(models_dir, "initial_model.pt")
    if os.path.exists(initial_model_path):
        model.load_state_dict(torch.load(initial_model_path, map_location='cpu'))
        model.eval()
        
        # Extract embedding matrix
        input_embeddings = model.language_model.get_input_embeddings()
        embeddings_by_epoch["initial"] = input_embeddings.weight.clone()
        print(f"Loaded initial model embeddings: {input_embeddings.weight.shape}")
    
    # Load epoch models (but not best_model.pt)
    epoch_files = [f for f in os.listdir(models_dir) if f.startswith("epoch_") and f.endswith("_model.pt")]
    epoch_files.sort(key=lambda x: int(x.split("_")[1]))  # Sort by epoch number
    
    for epoch_file in epoch_files:
        epoch_num = epoch_file.split("_")[1]
        epoch_path = os.path.join(models_dir, epoch_file)
        
        model.load_state_dict(torch.load(epoch_path, map_location='cpu'))
        model.eval()
        
        # Extract embedding matrix
        input_embeddings = model.language_model.get_input_embeddings()
        embeddings_by_epoch[f"epoch_{epoch_num}"] = input_embeddings.weight.clone()
        print(f"Loaded epoch {epoch_num} embeddings: {input_embeddings.weight.shape}")
    
    print(f"Total loaded {len(embeddings_by_epoch)} embedding matrices")
    return embeddings_by_epoch, tokenizer, config


def extract_tokens_from_saved_tokenizer(tokenizer, config):
    """
    Extract OOD and regular tokens from the saved tokenizer and labels mapping.
    
    Args:
        tokenizer: Saved tokenizer with OOD tokens already added
        config: MultimodalTrainingConfig object
        
    Returns:
        tuple: (labels_mapping, ood_tokens, regular_tokens)
    """
    # Load labels mapping
    labels_mapping = None
    ood_tokens = []
    regular_tokens = []
    
    if config.labels_mapping_path and os.path.exists(config.labels_mapping_path):
        with open(config.labels_mapping_path, 'r') as f:
            labels_mapping = json.load(f)
        
        # Extract OOD tokens (those starting with "<ood")
        ood_tokens = [label for label in labels_mapping.values() if label.startswith("<ood")]
        
        # Extract regular tokens (not OOD)
        regular_tokens = [label for label in labels_mapping.values() if not label.startswith("<ood")]
        
        print(f"Found {len(ood_tokens)} OOD tokens and {len(regular_tokens)} regular tokens in labels mapping")
        if ood_tokens:
            print(f"OOD tokens: {ood_tokens}")
    else:
        print("No labels mapping found, will extract tokens from tokenizer vocabulary")
        # If no labels mapping, try to identify OOD tokens from tokenizer vocabulary
        # This is a fallback - ideally we should have the labels mapping
        vocab = tokenizer.get_vocab()
        ood_tokens = [token for token in vocab.keys() if token.startswith("<ood")]
        regular_tokens = [token for token in vocab.keys() if not token.startswith("<ood") and not token.startswith("<") and len(token) > 1]
    
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"Regular tokens: {len(regular_tokens)}")
    print(f"OOD tokens: {len(ood_tokens)}")
    
    return labels_mapping, ood_tokens, regular_tokens


def extract_rgb_from_mapping(labels_mapping, token):
    """Extract RGB values for a given token from the labels mapping."""
    for rgb_key, token_value in labels_mapping.items():
        if token_value == token:
            # Parse RGB values from format "r{red}g{green}b{blue}"
            match = re.match(r'r(\d+)g(\d+)b(\d+)', rgb_key)
            if match:
                r, g, b = map(int, match.groups())
                return (r/255.0, g/255.0, b/255.0)  # Normalize to [0,1]
    return (0.5, 0.5, 0.5)  # Default gray if not found


def calculate_color_embedding_correlation(embeddings_by_epoch, ood_tokens, regular_tokens, 
                                         ood_token_ids, regular_token_ids, labels_mapping):
    """
    Calculate correlation between color distance and token embedding distance for ALL tokens.
    
    Args:
        embeddings_by_epoch: Dictionary of embeddings by epoch
        ood_tokens: List of OOD token names
        regular_tokens: List of regular token names
        ood_token_ids: List of OOD token IDs
        regular_token_ids: List of regular token IDs
        labels_mapping: Labels mapping dictionary
        
    Returns:
        float: Pearson correlation coefficient
    """
    # Get the last epoch (highest epoch number)
    epoch_names = [name for name in embeddings_by_epoch.keys() if name.startswith('epoch_')]
    if not epoch_names:
        print("No epoch data found, using initial embeddings")
        last_epoch_name = 'initial'
    else:
        # Sort by epoch number and get the last one
        epoch_numbers = [int(name.split('_')[1]) for name in epoch_names]
        last_epoch_num = max(epoch_numbers)
        last_epoch_name = f'epoch_{last_epoch_num}'
    
    print(f"\n=== Color-Embedding Distance Correlation Analysis ({last_epoch_name}) ===")
    
    # Get embeddings for the last epoch
    embedding_matrix = embeddings_by_epoch[last_epoch_name]
    
    # Get all token IDs and names
    all_token_ids = ood_token_ids + regular_token_ids
    all_token_names = ood_tokens + regular_tokens
    
    # Extract embeddings for all tokens
    token_embeddings = embedding_matrix[all_token_ids].detach().cpu().float().numpy()
    
    # Extract RGB colors for all tokens
    rgb_colors = []
    for token in all_token_names:
        rgb_color = extract_rgb_from_mapping(labels_mapping, token)
        rgb_colors.append(rgb_color)
    
    rgb_colors = np.array(rgb_colors)  # Shape: (n_tokens, 3)
    
    # Calculate pairwise color distances (L1 distance)
    n_tokens = len(all_token_names)
    color_distances = []
    embedding_distances = []
    
    print(f"Calculating pairwise distances for {n_tokens} tokens (regular + OOD)...")
    
    for i in range(n_tokens):
        for j in range(i + 1, n_tokens):
            # Color distance (L1 - sum of absolute differences)
            color_dist = np.sum(np.abs(rgb_colors[i] - rgb_colors[j]))
            color_distances.append(color_dist)
            
            # Embedding distance (1 - cosine similarity)
            embedding_sim = cosine_similarity([token_embeddings[i]], [token_embeddings[j]])[0][0]
            embedding_dist = 1 - embedding_sim
            embedding_distances.append(embedding_dist)
    
    # Convert to numpy arrays
    color_distances = np.array(color_distances)
    embedding_distances = np.array(embedding_distances)
    
    # Calculate Pearson correlation
    correlation = np.corrcoef(color_distances, embedding_distances)[0, 1]
    
    print(f"Number of token pairs: {len(color_distances)}")
    print(f"Color distance range: [{color_distances.min():.4f}, {color_distances.max():.4f}]")
    print(f"Embedding distance range: [{embedding_distances.min():.4f}, {embedding_distances.max():.4f}]")
    print(f"Pearson correlation coefficient: {correlation:.4f}")
    
    return correlation


def create_umap_visualization(embeddings_by_epoch, ood_tokens, regular_tokens, labels_mapping, 
                             ood_token_ids, regular_token_ids, output_dir=None):
    """
    Create UMAP visualizations for token embeddings.
    
    Args:
        embeddings_by_epoch: Dictionary of embeddings by epoch
        ood_tokens: List of OOD token names
        regular_tokens: List of regular token names
        labels_mapping: Labels mapping dictionary
        ood_token_ids: List of OOD token IDs
        regular_token_ids: List of regular token IDs
        output_dir: Directory to save plots (optional)
    """
    # Get all token IDs for regular and OOD tokens
    all_token_ids = ood_token_ids + regular_token_ids
    all_token_names = ood_tokens + regular_tokens
    token_types = ['OOD'] * len(ood_tokens) + ['Regular'] * len(regular_tokens)

    print(f"Analyzing {len(all_token_ids)} tokens across {len(embeddings_by_epoch)} epochs")

    # Create a dictionary to store embeddings for each epoch
    epoch_embeddings = {}

    for epoch_name, embedding_matrix in embeddings_by_epoch.items():
        # Extract embeddings for our tokens of interest
        token_embeddings = embedding_matrix[all_token_ids].detach().cpu().float().numpy()
        epoch_embeddings[epoch_name] = token_embeddings
        print(f"{epoch_name}: {token_embeddings.shape}")

    # Create UMAP reducer (fit on initial embeddings for consistency)
    print("\nFitting UMAP on initial embeddings...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(all_token_ids)-1))
    initial_embeddings = epoch_embeddings['initial']
    reducer.fit(initial_embeddings)

    # Transform all epochs using the same UMAP fit
    epoch_projections = {}
    for epoch_name, embeddings in epoch_embeddings.items():
        projections = reducer.transform(embeddings)
        epoch_projections[epoch_name] = projections
        print(f"Transformed {epoch_name}: {projections.shape}")

    # Extract RGB colors for all tokens
    token_colors = []
    for token in all_token_names:
        rgb_color = extract_rgb_from_mapping(labels_mapping, token)
        token_colors.append(rgb_color)

    print(f"Extracted RGB colors for {len(token_colors)} tokens")

    # Create enhanced UMAP plots with RGB colors and shape differentiation
    epochs = list(epoch_projections.keys())
    n_epochs = len(epochs)
    fig, axes = plt.subplots(1, n_epochs, figsize=(6*n_epochs, 6))

    if n_epochs == 1:
        axes = [axes]

    for i, epoch_name in enumerate(epochs):
        ax = axes[i]
        projections = epoch_projections[epoch_name]
        
        # Plot each token with its RGB color and appropriate shape
        for j, (x, y) in enumerate(projections):
            color = token_colors[j]
            token_type = token_types[j]
            token_name = all_token_names[j]
            
            # Use different shapes for regular vs OOD
            if token_type == 'OOD':
                marker = 'o'  # Circle for OOD tokens
                size = 60
            else:
                marker = 'x'  # X for regular tokens
                size = 80
            
            ax.scatter(x, y, c=[color], marker=marker, s=size, alpha=0.8, edgecolors='black', linewidth=0.5)

        ax.set_title(f'{epoch_name}', fontsize=14)
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.scatter([], [], c='black', marker='x', s=80, label='Regular', edgecolors='black')
        ax.scatter([], [], c='black', marker='o', s=60, label='OOD', edgecolors='black')
        ax.legend()

    plt.tight_layout()
    plt.suptitle('Token Embeddings Colored by RGB Values (✗=Regular, ○=OOD)', y=1.02, fontsize=16)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'token_embeddings_umap.png'), dpi=300, bbox_inches='tight')
        print(f"Saved UMAP plot to {output_dir}/token_embeddings_umap.png")
    
    plt.close()  # Close the figure instead of showing it

    # Print RGB color information
    print(f"\n=== RGB Color Information ===")
    print("Regular tokens:")
    for i, token in enumerate(regular_tokens):
        rgb = token_colors[all_token_names.index(token)]
        print(f"  {token}: RGB({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})")

    print(f"\nOOD tokens (first 5):")
    for i, token in enumerate(ood_tokens[:5]):
        rgb = token_colors[all_token_names.index(token)]
        print(f"  {token}: RGB({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})")
    print(f"  ... and {len(ood_tokens) - 5} more OOD tokens")


def create_3d_umap_visualization(embeddings_by_epoch, ood_tokens, regular_tokens, labels_mapping, 
                                 ood_token_ids, regular_token_ids, output_dir=None):
    """
    Create 3D UMAP visualization for the last epoch using Plotly.
    
    Args:
        embeddings_by_epoch: Dictionary of embeddings by epoch
        ood_tokens: List of OOD token names
        regular_tokens: List of regular token names
        labels_mapping: Labels mapping dictionary
        ood_token_ids: List of OOD token IDs
        regular_token_ids: List of regular token IDs
        output_dir: Directory to save plots (optional)
    """
    print("=== Creating 3D UMAP Visualization ===")
    
    # Get the last epoch (highest epoch number)
    epoch_names = [name for name in embeddings_by_epoch.keys() if name.startswith('epoch_')]
    if not epoch_names:
        print("No epoch data found, using initial embeddings")
        last_epoch_name = 'initial'
    else:
        # Sort by epoch number and get the last one
        epoch_numbers = [int(name.split('_')[1]) for name in epoch_names]
        last_epoch_num = max(epoch_numbers)
        last_epoch_name = f'epoch_{last_epoch_num}'
    
    print(f"Using {last_epoch_name} for 3D visualization")
    
    # Get embeddings for the last epoch
    embedding_matrix = embeddings_by_epoch[last_epoch_name]
    
    # Get all token IDs and names
    all_token_ids = ood_token_ids + regular_token_ids
    all_token_names = ood_tokens + regular_tokens
    token_types = ['OOD'] * len(ood_tokens) + ['Regular'] * len(regular_tokens)
    
    # Extract embeddings for our tokens
    token_embeddings = embedding_matrix[all_token_ids].detach().cpu().float().numpy()
    print(f"Token embeddings shape: {token_embeddings.shape}")
    
    # Create 3D UMAP reducer
    print("Fitting 3D UMAP...")
    reducer_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=min(15, len(all_token_ids)-1))
    projections_3d = reducer_3d.fit_transform(token_embeddings)
    print(f"3D projections shape: {projections_3d.shape}")
    
    # Extract RGB colors for all tokens
    token_colors = []
    for token in all_token_names:
        rgb_color = extract_rgb_from_mapping(labels_mapping, token)
        token_colors.append(rgb_color)
    
    # Convert RGB to hex colors for Plotly
    hex_colors = []
    for rgb in token_colors:
        hex_color = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
        hex_colors.append(hex_color)
    
    # Create hover text with token information
    hover_text = []
    for i, token_name in enumerate(all_token_names):
        token_type = token_types[i]
        rgb = token_colors[i]
        hover_text.append(f"Token: {token_name}<br>Type: {token_type}<br>RGB: ({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})")
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Add OOD tokens
    ood_mask = [token_type == 'OOD' for token_type in token_types]
    if any(ood_mask):
        ood_x = projections_3d[ood_mask, 0]
        ood_y = projections_3d[ood_mask, 1]
        ood_z = projections_3d[ood_mask, 2]
        ood_colors = [hex_colors[i] for i, is_ood in enumerate(ood_mask) if is_ood]
        ood_hover = [hover_text[i] for i, is_ood in enumerate(ood_mask) if is_ood]
        ood_names = [all_token_names[i] for i, is_ood in enumerate(ood_mask) if is_ood]
        
        fig.add_trace(go.Scatter3d(
            x=ood_x, y=ood_y, z=ood_z,
            mode='markers',
            marker=dict(
                size=6,
                color=ood_colors,
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            text=ood_names,
            hovertemplate='%{text}<br>%{customdata}<extra></extra>',
            customdata=ood_hover,
            name='OOD Tokens'
        ))
    
    # Add regular tokens
    regular_mask = [token_type == 'Regular' for token_type in token_types]
    if any(regular_mask):
        regular_x = projections_3d[regular_mask, 0]
        regular_y = projections_3d[regular_mask, 1]
        regular_z = projections_3d[regular_mask, 2]
        regular_colors = [hex_colors[i] for i, is_regular in enumerate(regular_mask) if is_regular]
        regular_hover = [hover_text[i] for i, is_regular in enumerate(regular_mask) if is_regular]
        regular_names = [all_token_names[i] for i, is_regular in enumerate(regular_mask) if is_regular]
        
        fig.add_trace(go.Scatter3d(
            x=regular_x, y=regular_y, z=regular_z,
            mode='markers',
            marker=dict(
                size=4,
                color=regular_colors,
                symbol='x',
                line=dict(width=2, color='black')
            ),
            text=regular_names,
            hovertemplate='%{text}<br>%{customdata}<extra></extra>',
            customdata=regular_hover,
            name='Regular Tokens'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'3D UMAP Visualization - {last_epoch_name}<br><sub>Tokens colored by RGB values (✗=Regular, ○=OOD)</sub>',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3',
            bgcolor='white'
        ),
        width=1000,
        height=800,
        showlegend=True
    )
    
    # Save as HTML file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, 'token_embeddings_3d_umap.html')
        fig.write_html(html_path)
        print(f"Saved 3D UMAP plot to {html_path}")
    
    print("3D UMAP visualization complete!")


def create_regular_tokens_analysis(embeddings_by_epoch, regular_tokens, regular_token_ids, 
                                  labels_mapping, output_dir=None):
    """
    Create UMAP analysis focused only on regular tokens.
    
    Args:
        embeddings_by_epoch: Dictionary of embeddings by epoch
        regular_tokens: List of regular token names
        regular_token_ids: List of regular token IDs
        labels_mapping: Labels mapping dictionary
        output_dir: Directory to save plots (optional)
    """
    print("=== Regular Tokens UMAP Analysis ===")

    # Extract embeddings for only regular tokens across epochs
    regular_epoch_embeddings = {}
    for epoch_name, embedding_matrix in embeddings_by_epoch.items():
        regular_token_embeddings = embedding_matrix[regular_token_ids].detach().cpu().float().numpy()
        regular_epoch_embeddings[epoch_name] = regular_token_embeddings
        print(f"{epoch_name}: {regular_token_embeddings.shape}")

    # Create UMAP reducer fit only on regular tokens from initial epoch
    print("\nFitting UMAP on regular tokens from initial epoch...")
    regular_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(5, len(regular_tokens)-1))
    initial_regular_embeddings = regular_epoch_embeddings['initial']
    regular_reducer.fit(initial_regular_embeddings)

    # Transform all epochs using the same UMAP fit
    regular_epoch_projections = {}
    for epoch_name, embeddings in regular_epoch_embeddings.items():
        projections = regular_reducer.transform(embeddings)
        regular_epoch_projections[epoch_name] = projections
        print(f"Transformed {epoch_name}: {projections.shape}")

    # Create subplots for each epoch showing only regular tokens
    epochs = list(regular_epoch_projections.keys())
    n_epochs = len(epochs)
    fig, axes = plt.subplots(1, n_epochs, figsize=(5*n_epochs, 5))

    if n_epochs == 1:
        axes = [axes]

    for i, epoch_name in enumerate(epochs):
        ax = axes[i]
        projections = regular_epoch_projections[epoch_name]
        
        # Plot each regular token with its RGB color
        for j, (x, y) in enumerate(projections):
            token_name = regular_tokens[j]
            rgb_color = extract_rgb_from_mapping(labels_mapping, token_name)
            
            ax.scatter(x, y, c=[rgb_color], marker='o', s=100, alpha=0.8, 
                      edgecolors='black', linewidth=1.5)
            
            # Add token labels
            ax.annotate(token_name, (x, y), xytext=(5, 5), textcoords='offset points', 
                       fontsize=10, alpha=0.8, fontweight='bold')

        ax.set_title(f'{epoch_name} - Regular Tokens Only', fontsize=14)
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Regular Token Embeddings Evolution (RGB Colored)', y=1.02, fontsize=16)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'regular_tokens_umap.png'), dpi=300, bbox_inches='tight')
        print(f"Saved regular tokens UMAP plot to {output_dir}/regular_tokens_umap.png")
    
    plt.close()  # Close the figure instead of showing it

    # Calculate distances between regular tokens
    print(f"\n=== Regular Token Distance Analysis ===")
    for epoch_name in epochs:
        projections = regular_epoch_projections[epoch_name]
        
        print(f"\n{epoch_name}:")
        for i, token1 in enumerate(regular_tokens):
            for j, token2 in enumerate(regular_tokens):
                if i < j:  # Avoid duplicates and self-comparison
                    dist = np.linalg.norm(projections[i] - projections[j])
                    print(f"  {token1} ↔ {token2}: {dist:.3f}")


def main():
    """Main function to run token embedding analysis."""
    parser = argparse.ArgumentParser(description='Token Embedding Analysis')
    parser.add_argument('--results_dir', type=str, 
                       default='/users/sboppana/data/sboppana/multimodal_concept_learning/results/multimodal/color/96_colors_375',
                       help='Path to results directory containing trained model')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save output plots (defaults to results_dir if not specified)')
    
    args = parser.parse_args()
    
    # Set output directory to results_dir if not specified
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    print(f"Loading embeddings from: {args.results_dir}")
    print(f"Saving plots to: {args.output_dir}")
    
    # Load embeddings, tokenizer, and config from saved results
    embeddings_by_epoch, tokenizer, config = load_token_embeddings(args.results_dir)

    # Extract OOD and regular tokens from the saved tokenizer
    labels_mapping, ood_tokens, regular_tokens = extract_tokens_from_saved_tokenizer(tokenizer, config)

    print(f"\n=== Token Analysis ===")
    print(f"Total vocabulary size: {len(tokenizer)}")
    print(f"Regular tokens: {len(regular_tokens)}")
    print(f"OOD tokens: {len(ood_tokens)}")

    # Show some examples
    if regular_tokens:
        print(f"\nSample regular tokens: {regular_tokens[:5]}")
    if ood_tokens:
        print(f"\nSample OOD tokens: {ood_tokens[:5]}")

    # Get token IDs for analysis
    ood_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in ood_tokens]
    regular_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in regular_tokens]

    print(f"\nOOD token IDs: {ood_token_ids}")
    print(f"Regular token IDs: {regular_token_ids[:10]}...")  # Show first 10

    # Create UMAP visualizations
    create_umap_visualization(embeddings_by_epoch, ood_tokens, regular_tokens, labels_mapping,
                            ood_token_ids, regular_token_ids, args.output_dir)
    
    # Create 3D UMAP visualization for the last epoch
    create_3d_umap_visualization(embeddings_by_epoch, ood_tokens, regular_tokens, labels_mapping,
                                ood_token_ids, regular_token_ids, args.output_dir)
    
    # Create regular tokens analysis
    create_regular_tokens_analysis(embeddings_by_epoch, regular_tokens, regular_token_ids,
                                 labels_mapping, args.output_dir)
    
    # Calculate color-embedding distance correlation
    correlation = calculate_color_embedding_correlation(embeddings_by_epoch, ood_tokens, regular_tokens,
                                                       ood_token_ids, regular_token_ids, labels_mapping)
    
    print(f"\n=== FINAL CORRELATION RESULT ===")
    print(f"Color-Embedding Distance Correlation: {correlation:.4f}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
