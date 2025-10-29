#!/usr/bin/env python3
"""
Token Embedding Analysis Script for ImageNet Models

This script analyzes token embeddings from trained multimodal models on ImageNet,
creating UMAP visualizations and distance analyses for regular and OOD tokens.
Unlike the color version, this script doesn't use RGB coloring since ImageNet
tokens represent object classes rather than colors.
"""

# Standard library imports
import argparse
import sys
import os
import json
import warnings

# Third-party imports
import torch
import numpy as np
from transformers import AutoTokenizer
import umap
import plotly.graph_objects as go
import plotly.express as px

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Project imports
from src.utils import set_seed, create_transforms
from src.multimodal.multimodal_training_config import MultimodalTrainingConfig
from src.multimodal.mllm import MLLM
from src.datasets.imagenet.imagenet_dataset import ImageNetDataset, MultimodalCollator
from src.datasets.color.color_dataset import ColorDataset


def load_wordnet_hierarchy(data_dir: str):
    """
    Load WordNet hierarchy relationships from wordnet.is_a.txt.
    
    Args:
        data_dir: Path to the ImageNet data directory
        
    Returns:
        tuple: (parent_to_children, child_to_parents, wnid_to_name)
    """
    # Load parent-child relationships
    isa_path = os.path.join(data_dir, "ILSVRC2012_devkit_t12", "data", "wordnet.is_a.txt")
    parent_to_children = {}
    child_to_parents = {}
    
    with open(isa_path, 'r') as f:
        for line in f:
            parent, child = line.strip().split()
            if parent not in parent_to_children:
                parent_to_children[parent] = []
            parent_to_children[parent].append(child)
            
            if child not in child_to_parents:
                child_to_parents[child] = []
            child_to_parents[child].append(parent)
    
    # Load WNID to name mapping
    words_path = os.path.join(data_dir, "ILSVRC2012_devkit_t12", "data", "words.txt")
    wnid_to_name = {}
    
    with open(words_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                wnid = parts[0]
                name = ' '.join(parts[1:])
                wnid_to_name[wnid] = name
    
    return parent_to_children, child_to_parents, wnid_to_name


def find_root_nodes(parent_to_children: dict) -> list:
    """
    Find root nodes in the WordNet hierarchy (nodes with no parents).
    
    Args:
        parent_to_children: Dictionary mapping parent to list of children
        
    Returns:
        List of root WNIDs
    """
    all_children = set()
    for children in parent_to_children.values():
        all_children.update(children)
    
    # Root nodes are parents that are not children
    root_nodes = []
    for parent in parent_to_children.keys():
        if parent not in all_children:
            root_nodes.append(parent)
    
    return root_nodes


def get_nodes_at_depth(parent_to_children: dict, root_nodes: list, depth: int) -> list:
    """
    Get all nodes at a specific depth from root nodes.
    
    Args:
        parent_to_children: Dictionary mapping parent to list of children
        root_nodes: List of root WNIDs
        depth: Depth level to extract (0 = root, 1 = first level down, etc.)
        
    Returns:
        List of WNIDs at the specified depth
    """
    if depth == 0:
        return root_nodes
    
    current_level = root_nodes
    for _ in range(depth):
        next_level = []
        for node in current_level:
            if node in parent_to_children:
                next_level.extend(parent_to_children[node])
        current_level = next_level
        if not current_level:  # No more children
            break
    
    return current_level


def get_path_to_root(wnid: str, child_to_parents: dict) -> list:
    """
    Get the path from a WNID to the root of the hierarchy.
    
    Args:
        wnid: Starting WNID
        child_to_parents: Dictionary mapping child to list of parents
        
    Returns:
        List of WNIDs in the path from wnid to root
    """
    path = [wnid]
    current = wnid
    
    while current in child_to_parents and child_to_parents[current]:
        # Take the first parent if multiple exist
        current = child_to_parents[current][0]
        path.append(current)
    
    return path


def get_path_based_colors(token_names: list, token_to_wnid: dict, parent_to_children: dict, 
                         child_to_parents: dict, wnid_to_name: dict, depth: int = 2) -> tuple:
    """
    Get colors for tokens based on which parent they pass through on their path to root.
    
    Args:
        token_names: List of token names
        token_to_wnid: Dictionary mapping token names to WNIDs
        parent_to_children: Dictionary mapping parent to list of children
        child_to_parents: Dictionary mapping child to list of parents
        wnid_to_name: Dictionary mapping WNID to name
        depth: Depth level to use for coloring (0 = root, 1 = first level down, etc.)
        
    Returns:
        tuple: (token_to_color, token_to_parent)
    """
    # Find root nodes
    root_nodes = find_root_nodes(parent_to_children)
    print(f"Found {len(root_nodes)} root nodes: {root_nodes[:5]}...")
    
    # Get potential parents at specified depth
    potential_parents = get_nodes_at_depth(parent_to_children, root_nodes, depth)
    print(f"Found {len(potential_parents)} potential parents at depth {depth}")
    
    # Create color palette
    colors = px.colors.qualitative.Set3
    if len(potential_parents) > len(colors):
        colors = colors * (len(potential_parents) // len(colors) + 1)
    
    parent_to_color = {parent: colors[i % len(colors)] for i, parent in enumerate(potential_parents)}
    
    # Map each token to its color based on path intersection
    token_to_color = {}
    token_to_parent = {}
    
    for token_name in token_names:
        if token_name in token_to_wnid:
            wnid = token_to_wnid[token_name]
            path = get_path_to_root(wnid, child_to_parents)
            
            # Find which potential parent this token's path intersects with
            intersecting_parent = None
            for parent in potential_parents:
                if parent in path:
                    intersecting_parent = parent
                    break
            
            if intersecting_parent:
                token_to_color[token_name] = parent_to_color[intersecting_parent]
                token_to_parent[token_name] = intersecting_parent
            else:
                # Fallback color if no intersection found
                token_to_color[token_name] = '#888888'
                token_to_parent[token_name] = token_name
        else:
            # Fallback for tokens not in mapping
            token_to_color[token_name] = '#888888'
            token_to_parent[token_name] = token_name
    
    return token_to_color, token_to_parent


def _extract_embeddings_for_tokens(embeddings_by_epoch, token_ids):
    """Extract embeddings for specific tokens across all epochs."""
    epoch_embeddings = {}
    for epoch_name, embedding_matrix in embeddings_by_epoch.items():
        token_embeddings = embedding_matrix[token_ids].detach().cpu().float().numpy()
        epoch_embeddings[epoch_name] = token_embeddings
        print(f"{epoch_name}: {token_embeddings.shape}")
    return epoch_embeddings


def _fit_umap_and_transform(epoch_embeddings, n_components=2, n_neighbors=None):
    """Fit UMAP on initial embeddings and transform all epochs."""
    if n_neighbors is None:
        n_neighbors = min(15, len(list(epoch_embeddings.values())[0]) - 1)
    
    print(f"\nFitting {n_components}D UMAP...")
    reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=n_neighbors)
    initial_embeddings = epoch_embeddings['initial'].detach().cpu().float().numpy()
    reducer.fit(initial_embeddings)

    # Transform all epochs using the same UMAP fit
    epoch_projections = {}
    for epoch_name, embeddings in epoch_embeddings.items():
        projections = reducer.transform(embeddings.detach().cpu().float().numpy())
        epoch_projections[epoch_name] = projections
        print(f"Transformed {epoch_name}: {projections.shape}")
    
    return epoch_projections


def _sort_epochs(epochs):
    """Sort epochs with 'initial' first, then numeric epochs."""
    if 'initial' in epochs:
        epochs.remove('initial')
        epochs.sort(key=lambda x: int(x.split('_')[1]) if x.startswith('epoch_') else 0)
        epochs.insert(0, 'initial')
    else:
        epochs.sort(key=lambda x: int(x.split('_')[1]) if x.startswith('epoch_') else 0)
    return epochs


def _create_token_traces(projections, token_names, token_type, token_to_color, token_to_parent, 
                        wnid_to_name, n, epoch_name, is_3d=False):
    """Create Plotly traces for tokens."""
    if is_3d:
        x, y, z = projections[:, 0], projections[:, 1], projections[:, 2]
        coords = dict(x=x, y=y, z=z)
        scatter_class = go.Scatter3d
    else:
        x, y = projections[:, 0], projections[:, 1]
        coords = dict(x=x, y=y)
        scatter_class = go.Scatter
    
    colors = [token_to_color.get(name, '#888888') for name in token_names]
    parents_list = [token_to_parent.get(name, name) for name in token_names]
    parent_names = [wnid_to_name.get(parent, parent) for parent in parents_list]
    hover = [f"Token: {name}<br>Type: {token_type}<br>Parent: {parent_name}<br>Epoch: {epoch_name}" 
             for name, parent_name in zip(token_names, parent_names)]
    
    marker_symbol = 'circle' if token_type == 'OOD' else 'square'
    
    # Reduce marker size for 3D plots by 3 times
    if is_3d:
        marker_size = 3 if token_type == 'OOD' else 2
    else:
        marker_size = 8 if token_type == 'OOD' else 6
    
    # Thin black border on both circles and squares
    marker_dict = dict(
        size=marker_size,
        color=colors,
        symbol=marker_symbol,
        line=dict(width=0.5, color='black')
    )
    
    return scatter_class(
        mode='markers',
        marker=marker_dict,
        text=token_names,
        hovertemplate='%{text}<br>%{customdata}<extra></extra>',
        customdata=hover,
        name=f'{token_type} Tokens',
        legendgroup=token_type,
        **coords
    )


def _create_epoch_slider(epochs, epoch_trace_indices, title_template, total_traces):
    """Create slider for epoch navigation.
    
    Args:
        epochs: List of epoch names
        epoch_trace_indices: Dictionary mapping epoch names to lists of trace indices for that epoch
        title_template: Template for title updates
        total_traces: Total number of traces
    """
    steps = []
    for epoch_name in epochs:
        step = dict(
            method="update",
            args=[{"visible": [False] * total_traces},
                  {"title": title_template.format(epoch_name)}],
            label=epoch_name
        )
        
        # Show traces for this epoch
        if epoch_name in epoch_trace_indices:
            for idx in epoch_trace_indices[epoch_name]:
                if idx < total_traces:
                    step["args"][0]["visible"][idx] = True
        
        steps.append(step)
    
    return [dict(
        active=0,
        currentvalue={"prefix": "Epoch: "},
        pad={"t": 50},
        steps=steps
    )]


def _create_filter_buttons(ood_trace_indices, regular_trace_indices, total_traces):
    """Create filter buttons for token types.
    
    Args:
        ood_trace_indices: List of trace indices for OOD tokens
        regular_trace_indices: List of trace indices for Regular tokens
        total_traces: Total number of traces
    """
    return [dict(
        type="buttons",
        direction="left",
        buttons=list([
            dict(
                args=[{"visible": [True] * total_traces}],
                label="Show All",
                method="restyle"
            ),
            dict(
                args=[{"visible": [True if i in ood_trace_indices else False for i in range(total_traces)]}],
                label="OOD Only",
                method="restyle"
            ),
            dict(
                args=[{"visible": [True if i in regular_trace_indices else False for i in range(total_traces)]}],
                label="Regular Only",
                method="restyle"
            )
        ]),
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.01,
        xanchor="left",
        y=1.02,
        yanchor="top"
    )]


def _create_legend_traces(token_to_color, token_to_parent, wnid_to_name, unique_parents):
    """Create invisible traces for legend entries."""
    legend_traces = []
    
    for parent in unique_parents:
        parent_name = wnid_to_name.get(parent, parent)
        color = token_to_color.get(parent, '#888888')
        
        # Count how many tokens belong to this parent
        count = sum(1 for token, p in token_to_parent.items() if p == parent)
        
        # Create invisible trace for legend
        trace = go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color, symbol='circle'),
            name=f"{parent_name} ({count})",
            showlegend=True,
            visible='legendonly'  # Only show in legend, not on plot
        )
        legend_traces.append(trace)
    
    return legend_traces


def load_token_embeddings(results_dir: str, max_epochs: int = None):
    """
    Load token embeddings from all epochs of a trained model.
    
    Args:
        results_dir: Path to the results directory containing the trained model
        max_epochs: Maximum number of epochs to load (None for all epochs)
        
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
    
    # Limit to max_epochs if specified
    if max_epochs is not None:
        epoch_files = epoch_files[:max_epochs]
        print(f"Limiting analysis to first {max_epochs} epochs")
    
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
            print(f"Sample OOD tokens: {ood_tokens[:5]}")
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


def create_umap_visualization(ood_embeddings, regular_embeddings, ood_tokens, regular_tokens, labels_mapping, 
                             token_to_color, token_to_parent, wnid_to_name, n, output_dir=None):
    """
    Create interactive UMAP visualization with epoch slider and token type filtering.
    
    Args:
        ood_embeddings: Dictionary of averaged OOD embeddings by epoch
        regular_embeddings: Dictionary of averaged regular embeddings by epoch
        ood_tokens: List of OOD token names
        regular_tokens: List of regular token names
        labels_mapping: Labels mapping dictionary
        token_to_color: Dictionary mapping token names to colors
        token_to_parent: Dictionary mapping token names to parent categories
        wnid_to_name: Dictionary mapping WNIDs to names
        n: Parent level for coloring
        output_dir: Directory to save plots (optional)
    """
    # Combine embeddings and token information
    all_token_names = ood_tokens + regular_tokens
    token_types = ['OOD'] * len(ood_tokens) + ['Regular'] * len(regular_tokens)

    print(f"Analyzing {len(all_token_names)} tokens across {len(ood_embeddings)} epochs")

    # Combine embeddings for UMAP fitting
    combined_embeddings = {}
    for epoch_name in ood_embeddings.keys():
        ood_emb = ood_embeddings[epoch_name]
        regular_emb = regular_embeddings[epoch_name]
        if len(ood_emb) > 0 and len(regular_emb) > 0:
            combined_embeddings[epoch_name] = torch.cat([ood_emb, regular_emb], dim=0)
        elif len(ood_emb) > 0:
            combined_embeddings[epoch_name] = ood_emb
        elif len(regular_emb) > 0:
            combined_embeddings[epoch_name] = regular_emb
        else:
            combined_embeddings[epoch_name] = torch.empty(0, ood_emb.shape[1] if len(ood_emb) > 0 else regular_emb.shape[1])

    # Fit UMAP and transform
    epoch_projections = _fit_umap_and_transform(combined_embeddings, n_components=2)
    epochs = _sort_epochs(list(epoch_projections.keys()))
    
    # Create figure
    fig = go.Figure()
    
    # Track trace indices for each epoch and token type
    epoch_trace_indices = {epoch: [] for epoch in epochs}
    ood_trace_indices = []
    regular_trace_indices = []
    
    # Add traces for each epoch
    for epoch_name in epochs:
        projections = epoch_projections[epoch_name]
        
        # Separate OOD and Regular tokens based on their positions in the combined array
        ood_start = 0
        ood_end = len(ood_tokens)
        regular_start = ood_end
        regular_end = ood_end + len(regular_tokens)
        
        # Add OOD tokens
        if len(ood_tokens) > 0:
            ood_projections = projections[ood_start:ood_end]
            trace = _create_token_traces(ood_projections, ood_tokens, 'OOD', token_to_color, 
                                       token_to_parent, wnid_to_name, n, epoch_name, is_3d=False)
            trace.update(visible=(epoch_name == epochs[0]))
            fig.add_trace(trace)
            trace_idx = len(fig.data) - 1
            epoch_trace_indices[epoch_name].append(trace_idx)
            ood_trace_indices.append(trace_idx)
        
        # Add Regular tokens
        if len(regular_tokens) > 0:
            regular_projections = projections[regular_start:regular_end]
            trace = _create_token_traces(regular_projections, regular_tokens, 'Regular', token_to_color, 
                                       token_to_parent, wnid_to_name, n, epoch_name, is_3d=False)
            trace.update(visible=(epoch_name == epochs[0]))
            fig.add_trace(trace)
            trace_idx = len(fig.data) - 1
            epoch_trace_indices[epoch_name].append(trace_idx)
            regular_trace_indices.append(trace_idx)
    
    # Create slider and filter buttons
    sliders = _create_epoch_slider(epochs, epoch_trace_indices, "Token Embeddings UMAP - {}", len(fig.data))
    updatemenus = _create_filter_buttons(ood_trace_indices, regular_trace_indices, len(fig.data))
    
    # Create legend traces
    unique_parents = list(set(token_to_parent.values()))
    unique_parents.sort()
    legend_traces = _create_legend_traces(token_to_color, token_to_parent, wnid_to_name, unique_parents)
    
    # Add legend traces to figure
    for trace in legend_traces:
        fig.add_trace(trace)
    
    # Update layout
    fig.update_layout(
        title=f"Token Embeddings UMAP - {epochs[0]}",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        ),
        sliders=sliders,
        updatemenus=updatemenus
    )
    
    # Save as HTML file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, 'token_embeddings_umap_interactive.html')
        fig.write_html(html_path)
        print(f"Saved interactive UMAP plot to {html_path}")
    
    # Print token information
    print(f"\n=== Token Information ===")
    print("Regular tokens (first 10):")
    for i, token in enumerate(regular_tokens[:10]):
        print(f"  {token}")
    if len(regular_tokens) > 10:
        print(f"  ... and {len(regular_tokens) - 10} more regular tokens")

    print(f"\nOOD tokens (first 10):")
    for i, token in enumerate(ood_tokens[:10]):
        print(f"  {token}")
    if len(ood_tokens) > 10:
        print(f"  ... and {len(ood_tokens) - 10} more OOD tokens")


def create_3d_umap_visualization(ood_embeddings, regular_embeddings, ood_tokens, regular_tokens, labels_mapping, 
                                 token_to_color, token_to_parent, wnid_to_name, n, output_dir=None):
    """
    Create interactive 3D UMAP visualization with epoch slider and token type filtering.
    
    Args:
        ood_embeddings: Dictionary of averaged OOD embeddings by epoch
        regular_embeddings: Dictionary of averaged regular embeddings by epoch
        ood_tokens: List of OOD token names
        regular_tokens: List of regular token names
        labels_mapping: Labels mapping dictionary
        token_to_color: Dictionary mapping token names to colors
        token_to_parent: Dictionary mapping token names to parent categories
        wnid_to_name: Dictionary mapping WNIDs to names
        n: Parent level for coloring
        output_dir: Directory to save plots (optional)
    """
    print("=== Creating Interactive 3D UMAP Visualization ===")
    
    # Combine embeddings and token information
    all_token_names = ood_tokens + regular_tokens
    token_types = ['OOD'] * len(ood_tokens) + ['Regular'] * len(regular_tokens)

    print(f"Analyzing {len(all_token_names)} tokens across {len(ood_embeddings)} epochs")

    # Combine embeddings for UMAP fitting
    combined_embeddings = {}
    for epoch_name in ood_embeddings.keys():
        ood_emb = ood_embeddings[epoch_name]
        regular_emb = regular_embeddings[epoch_name]
        if len(ood_emb) > 0 and len(regular_emb) > 0:
            combined_embeddings[epoch_name] = torch.cat([ood_emb, regular_emb], dim=0)
        elif len(ood_emb) > 0:
            combined_embeddings[epoch_name] = ood_emb
        elif len(regular_emb) > 0:
            combined_embeddings[epoch_name] = regular_emb
        else:
            combined_embeddings[epoch_name] = torch.empty(0, ood_emb.shape[1] if len(ood_emb) > 0 else regular_emb.shape[1])

    # Fit UMAP and transform
    epoch_projections = _fit_umap_and_transform(combined_embeddings, n_components=3)
    epochs = _sort_epochs(list(epoch_projections.keys()))
    
    # Create figure
    fig = go.Figure()
    
    # Track trace indices for each epoch and token type
    epoch_trace_indices = {epoch: [] for epoch in epochs}
    ood_trace_indices = []
    regular_trace_indices = []
    
    # Add traces for each epoch
    for epoch_name in epochs:
        projections = epoch_projections[epoch_name]
        
        # Separate OOD and Regular tokens based on their positions in the combined array
        ood_start = 0
        ood_end = len(ood_tokens)
        regular_start = ood_end
        regular_end = ood_end + len(regular_tokens)
        
        # Add OOD tokens
        if len(ood_tokens) > 0:
            ood_projections = projections[ood_start:ood_end]
            trace = _create_token_traces(ood_projections, ood_tokens, 'OOD', token_to_color, 
                                       token_to_parent, wnid_to_name, n, epoch_name, is_3d=True)
            trace.update(visible=(epoch_name == epochs[0]))
            fig.add_trace(trace)
            trace_idx = len(fig.data) - 1
            epoch_trace_indices[epoch_name].append(trace_idx)
            ood_trace_indices.append(trace_idx)
        
        # Add Regular tokens
        if len(regular_tokens) > 0:
            regular_projections = projections[regular_start:regular_end]
            trace = _create_token_traces(regular_projections, regular_tokens, 'Regular', token_to_color, 
                                       token_to_parent, wnid_to_name, n, epoch_name, is_3d=True)
            trace.update(visible=(epoch_name == epochs[0]))
            fig.add_trace(trace)
            trace_idx = len(fig.data) - 1
            epoch_trace_indices[epoch_name].append(trace_idx)
            regular_trace_indices.append(trace_idx)
    
    # Create slider and filter buttons
    sliders = _create_epoch_slider(epochs, epoch_trace_indices, "Token Embeddings 3D UMAP - {}", len(fig.data))
    updatemenus = _create_filter_buttons(ood_trace_indices, regular_trace_indices, len(fig.data))
    
    # Create legend traces
    unique_parents = list(set(token_to_parent.values()))
    unique_parents.sort()
    legend_traces = _create_legend_traces(token_to_color, token_to_parent, wnid_to_name, unique_parents)
    
    # Add legend traces to figure
    for trace in legend_traces:
        fig.add_trace(trace)
    
    # Update layout
    fig.update_layout(
        title=f"Token Embeddings 3D UMAP - {epochs[0]}",
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3',
            bgcolor='white'
        ),
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        ),
        sliders=sliders,
        updatemenus=updatemenus
    )
    
    # Save as HTML file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, 'token_embeddings_3d_umap_interactive.html')
        fig.write_html(html_path)
        print(f"Saved interactive 3D UMAP plot to {html_path}")
    
    print("Interactive 3D UMAP visualization complete!")


def create_regular_only_umap_visualization(regular_embeddings, regular_tokens, 
                                         token_to_color, token_to_parent, wnid_to_name, n, output_dir=None):
    """Create 2D UMAP visualization fit only on regular tokens and showing only regular tokens."""
    print("=== Creating Regular Tokens Only 2D UMAP Visualization ===")
    
    # Fit UMAP on regular embeddings
    epoch_projections = _fit_umap_and_transform(regular_embeddings, n_components=2, 
                                              n_neighbors=min(5, len(regular_tokens)-1))
    epochs = _sort_epochs(list(epoch_projections.keys()))
    
    # Create figure
    fig = go.Figure()
    
    # Track trace indices for each epoch
    epoch_trace_indices = {epoch: [] for epoch in epochs}
    
    # Add traces for each epoch
    for epoch_name in epochs:
        projections = epoch_projections[epoch_name]
        trace = _create_token_traces(projections, regular_tokens, 'Regular', token_to_color, 
                                   token_to_parent, wnid_to_name, n, epoch_name, is_3d=False)
        trace.update(visible=(epoch_name == epochs[0]))
        fig.add_trace(trace)
        trace_idx = len(fig.data) - 1
        epoch_trace_indices[epoch_name].append(trace_idx)
    
    # Create slider
    sliders = _create_epoch_slider(epochs, epoch_trace_indices, "Regular Tokens UMAP - {}", len(fig.data))
    
    # Create legend traces
    unique_parents = list(set(token_to_parent.values()))
    unique_parents.sort()
    legend_traces = _create_legend_traces(token_to_color, token_to_parent, wnid_to_name, unique_parents)
    
    # Add legend traces to figure
    for trace in legend_traces:
        fig.add_trace(trace)
    
    # Update layout
    fig.update_layout(
        title=f"Regular Tokens UMAP - {epochs[0]}",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        ),
        sliders=sliders
    )
    
    # Save as HTML file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, 'regular_tokens_umap_2d.html')
        fig.write_html(html_path)
        print(f"Saved regular tokens 2D UMAP plot to {html_path}")
    
    print("Regular tokens 2D UMAP visualization complete!")


def create_regular_only_3d_umap_visualization(regular_embeddings, regular_tokens, 
                                            token_to_color, token_to_parent, wnid_to_name, n, output_dir=None):
    """Create 3D UMAP visualization fit only on regular tokens and showing only regular tokens."""
    print("=== Creating Regular Tokens Only 3D UMAP Visualization ===")
    
    # Fit UMAP on regular embeddings
    epoch_projections = _fit_umap_and_transform(regular_embeddings, n_components=3, 
                                              n_neighbors=min(5, len(regular_tokens)-1))
    epochs = _sort_epochs(list(epoch_projections.keys()))
    
    # Create figure
    fig = go.Figure()
    
    # Track trace indices for each epoch
    epoch_trace_indices = {epoch: [] for epoch in epochs}
    
    # Add traces for each epoch
    for epoch_name in epochs:
        projections = epoch_projections[epoch_name]
        trace = _create_token_traces(projections, regular_tokens, 'Regular', token_to_color, 
                                   token_to_parent, wnid_to_name, n, epoch_name, is_3d=True)
        trace.update(visible=(epoch_name == epochs[0]))
        fig.add_trace(trace)
        trace_idx = len(fig.data) - 1
        epoch_trace_indices[epoch_name].append(trace_idx)
    
    # Create slider
    sliders = _create_epoch_slider(epochs, epoch_trace_indices, "Regular Tokens 3D UMAP - {}", len(fig.data))
    
    # Create legend traces
    unique_parents = list(set(token_to_parent.values()))
    unique_parents.sort()
    legend_traces = _create_legend_traces(token_to_color, token_to_parent, wnid_to_name, unique_parents)
    
    # Add legend traces to figure
    for trace in legend_traces:
        fig.add_trace(trace)
    
    # Update layout
    fig.update_layout(
        title=f"Regular Tokens 3D UMAP - {epochs[0]}",
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3',
            bgcolor='white'
        ),
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        ),
        sliders=sliders
    )
    
    # Save as HTML file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, 'regular_tokens_umap_3d.html')
        fig.write_html(html_path)
        print(f"Saved regular tokens 3D UMAP plot to {html_path}")
    
    print("Regular tokens 3D UMAP visualization complete!")


def main():
    """Main function to run token embedding analysis."""
    parser = argparse.ArgumentParser(description='Token Embedding Analysis for ImageNet Models')
    parser.add_argument('--results_dir', type=str, 
                       default='/users/sboppana/data/sboppana/multimodal_concept_learning/results/multimodal/imagenet/imagenet1k_timm_vit',
                       help='Path to results directory containing trained model')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save output plots (defaults to results_dir if not specified)')
    parser.add_argument('--max_epochs', type=int, default=None,
                       help='Maximum number of epochs to analyze (None for all epochs)')
    parser.add_argument('--parent_level', type=int, default=4,
                       help='WordNet hierarchy level to use for coloring (1=immediate parent, 2=grandparent, etc.)')
    parser.add_argument('--data_dir', type=str, 
                       default='/users/sboppana/data/sboppana/multimodal_concept_mapping/data/imagenet',
                       help='Path to ImageNet data directory containing WordNet hierarchy files')
    
    args = parser.parse_args()
    
    # Set output directory to results_dir/plots if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "plots")
    
    print(f"Loading embeddings from: {args.results_dir}")
    print(f"Saving plots to: {args.output_dir}")
    if args.max_epochs:
        print(f"Limiting analysis to first {args.max_epochs} epochs")
    print(f"Using WordNet hierarchy level {args.parent_level} for coloring")
    
    # Load WordNet hierarchy
    print("Loading WordNet hierarchy...")
    parent_to_children, child_to_parents, wnid_to_name = load_wordnet_hierarchy(args.data_dir)
    
    # Load embeddings, tokenizer, and config from saved results
    embeddings_by_epoch, tokenizer, config = load_token_embeddings(args.results_dir, args.max_epochs)

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

    # Get token IDs for analysis - handle multi-token names by averaging
    def get_averaged_embeddings(tokenizer, embeddings_by_epoch, token_names):
        """Get averaged embeddings for multi-token names."""
        averaged_embeddings = {}
        for epoch_name, embedding_matrix in embeddings_by_epoch.items():
            epoch_embeddings = []
            for token_name in token_names:
                # Tokenize the name to get all sub-tokens
                token_ids = tokenizer.encode(token_name, add_special_tokens=False)
                if token_ids:
                    # Average embeddings across all sub-tokens
                    token_embeddings = embedding_matrix[token_ids].mean(dim=0)
                    epoch_embeddings.append(token_embeddings)
                else:
                    # Fallback: use a zero embedding if tokenization fails
                    epoch_embeddings.append(torch.zeros(embedding_matrix.shape[1]))
            
            if epoch_embeddings:
                averaged_embeddings[epoch_name] = torch.stack(epoch_embeddings)
            else:
                averaged_embeddings[epoch_name] = torch.empty(0, embedding_matrix.shape[1])
        return averaged_embeddings

    # Get averaged embeddings for OOD and regular tokens
    ood_embeddings = get_averaged_embeddings(tokenizer, embeddings_by_epoch, ood_tokens)
    regular_embeddings = get_averaged_embeddings(tokenizer, embeddings_by_epoch, regular_tokens)

    print(f"\nProcessed {len(ood_tokens)} OOD tokens and {len(regular_tokens)} regular tokens with multi-token averaging")

    # Create reverse mapping from token names to WNIDs
    if labels_mapping:
        token_to_wnid = {token_name: wnid for wnid, token_name in labels_mapping.items()}
        print(f"Created token to WNID mapping for {len(token_to_wnid)} tokens")
    else:
        print("Warning: No labels mapping found, cannot create token to WNID mapping")
        token_to_wnid = {}
    
    # Get colors based on path intersection with parents at specified depth
    print(f"\nCalculating colors based on path intersection with parents at depth {args.parent_level}...")
    all_tokens = ood_tokens + regular_tokens
    
    token_to_color, token_to_parent = get_path_based_colors(
        all_tokens, token_to_wnid, parent_to_children, child_to_parents, wnid_to_name, args.parent_level
    )
    
    # Print some examples of the hierarchy mapping
    print(f"\n=== WordNet Hierarchy Examples ===")
    for i, token in enumerate(all_tokens[:5]):
        parent = token_to_parent.get(token, token)
        parent_name = wnid_to_name.get(parent, parent)
        print(f"{token} -> {parent} ({parent_name})")

    # Create interactive UMAP visualization with epoch slider and token type filtering
    create_umap_visualization(ood_embeddings, regular_embeddings, ood_tokens, regular_tokens, labels_mapping,
                            token_to_color, token_to_parent, wnid_to_name, args.parent_level, args.output_dir)
    
    # Create interactive 3D UMAP visualization with epoch slider and token type filtering
    create_3d_umap_visualization(ood_embeddings, regular_embeddings, ood_tokens, regular_tokens, labels_mapping,
                                token_to_color, token_to_parent, wnid_to_name, args.parent_level, args.output_dir)
    
    # Create regular tokens only visualizations (fit and show only regular tokens)
    create_regular_only_umap_visualization(regular_embeddings, regular_tokens,
                                         token_to_color, token_to_parent, wnid_to_name, args.parent_level, 
                                         args.output_dir)
    
    create_regular_only_3d_umap_visualization(regular_embeddings, regular_tokens,
                                            token_to_color, token_to_parent, wnid_to_name, args.parent_level, 
                                            args.output_dir)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
