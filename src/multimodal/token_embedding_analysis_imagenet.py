#!/usr/bin/env python3
"""Token embedding analysis utilities for ImageNet models.

This version streamlines the plotting experience by generating a single
interactive 2D UMAP visualization that animates across epochs. The goal is to
keep the exploratory workflow intact while removing the dense UI chrome and
redundant plots that previously made the script difficult to use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import umap
from transformers import AutoTokenizer

# Silence noisy third-party warnings that tend to clutter CLI output
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")

# Add project root to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.multimodal.mllm import MLLM
from src.multimodal.multimodal_training_config import MultimodalTrainingConfig


FALLBACK_COLOR = "#636363"
SYMBOL_MAP = {"Regular": "square", "OOD": "circle"}


def load_wordnet_hierarchy(data_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, str]]:
    """Load parent/child WordNet relationships required for coloring."""
    devkit_dir = os.path.join(data_dir, "ILSVRC2012_devkit_t12", "data")
    isa_path = os.path.join(devkit_dir, "wordnet.is_a.txt")
    words_path = os.path.join(devkit_dir, "words.txt")

    parent_to_children: Dict[str, List[str]] = {}
    child_to_parents: Dict[str, List[str]] = {}

    with open(isa_path, "r") as handle:
        for line in handle:
            parent, child = line.strip().split()
            parent_to_children.setdefault(parent, []).append(child)
            child_to_parents.setdefault(child, []).append(parent)

    wnid_to_name: Dict[str, str] = {}
    with open(words_path, "r") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if not parts:
                continue
            wnid = parts[0]
            name = " ".join(parts[1:]) if len(parts) > 1 else wnid
            wnid_to_name[wnid] = name

    return parent_to_children, child_to_parents, wnid_to_name


def find_root_nodes(parent_to_children: Dict[str, Iterable[str]]) -> List[str]:
    """Roots are parents that never appear as children."""
    all_children = {child for children in parent_to_children.values() for child in children}
    return [parent for parent in parent_to_children if parent not in all_children]


def get_nodes_at_depth(parent_to_children: Dict[str, List[str]], root_nodes: List[str], depth: int) -> List[str]:
    """Collect every node that sits exactly ``depth`` steps away from the roots."""
    if depth <= 0:
        return root_nodes

    current = list(root_nodes)
    for _ in range(depth):
        next_level: List[str] = []
        for node in current:
            next_level.extend(parent_to_children.get(node, []))
        if not next_level:
            break
        current = next_level
    return current


def get_path_to_root(wnid: str, child_to_parents: Dict[str, List[str]]) -> List[str]:
    """Follow the first parent pointer up to the root."""
    path = [wnid]
    current = wnid
    while current in child_to_parents and child_to_parents[current]:
        current = child_to_parents[current][0]
        path.append(current)
    return path


def get_path_based_colors(
    token_names: List[str],
    token_to_wnid: Dict[str, str],
    parent_to_children: Dict[str, List[str]],
    child_to_parents: Dict[str, List[str]],
    wnid_to_name: Dict[str, str],
    depth: int = 2,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Assign colors by tracing tokens to a shared ancestor at the requested depth."""
    root_nodes = find_root_nodes(parent_to_children)
    target_nodes = get_nodes_at_depth(parent_to_children, root_nodes, depth)
    if not target_nodes:
        target_nodes = list(root_nodes)

    palette = px.colors.qualitative.Set3
    if len(target_nodes) > len(palette):
        repeats = len(target_nodes) // len(palette) + 1
        palette = (palette * repeats)[: len(target_nodes)]

    parent_to_color = {node: palette[idx] for idx, node in enumerate(target_nodes)}

    token_to_parent: Dict[str, str] = {}
    token_to_color: Dict[str, str] = {}

    for token_name in token_names:
        wnid = token_to_wnid.get(token_name)
        parent_choice = wnid
        if wnid:
            path = get_path_to_root(wnid, child_to_parents)
            parent_choice = next((node for node in path if node in parent_to_color), wnid)
        token_to_parent[token_name] = parent_choice
        token_to_color[token_name] = parent_to_color.get(parent_choice, FALLBACK_COLOR)

    # Ensure every parent that appears has a color entry
    for parent in token_to_parent.values():
        parent_to_color.setdefault(parent, FALLBACK_COLOR)

    return token_to_color, token_to_parent, parent_to_color


def _to_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    return array.detach().cpu().float().numpy()


def _fit_umap_and_transform(
    epoch_embeddings: Dict[str, torch.Tensor],
    n_components: int = 2,
    n_neighbors: int | None = None,
) -> Dict[str, np.ndarray]:
    """Fit UMAP on the first epoch and project the rest with the same reducer."""
    if not epoch_embeddings:
        return {}

    fit_key = "initial" if "initial" in epoch_embeddings else sorted(epoch_embeddings.keys())[0]
    base_matrix = _to_numpy(epoch_embeddings[fit_key])
    if base_matrix.shape[0] < 3:
        raise ValueError("Need at least three tokens to run UMAP.")

    if n_neighbors is None:
        n_neighbors = min(15, base_matrix.shape[0] - 1)
    n_neighbors = max(2, min(n_neighbors, base_matrix.shape[0] - 1))

    reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=n_neighbors)
    reducer.fit(base_matrix)

    projections: Dict[str, np.ndarray] = {}
    for epoch_name, embeddings in epoch_embeddings.items():
        matrix = _to_numpy(embeddings)
        projections[epoch_name] = reducer.transform(matrix)
    return projections


def _sort_epochs(epoch_names: Iterable[str]) -> List[str]:
    """Sort epochs placing 'initial' first followed by numeric epochs."""
    names = list(epoch_names)
    has_initial = "initial" in names
    if has_initial:
        names.remove("initial")
    names.sort(key=lambda name: int(name.split("_")[1]) if name.startswith("epoch_") else name)
    if has_initial:
        names.insert(0, "initial")
    return names


def load_token_embeddings(results_dir: str, max_epochs: int | None = None):
    """Load embedding matrices from saved checkpoints."""
    models_dir = os.path.join(results_dir, "models")
    config_path = os.path.join(models_dir, "training_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Training config not found at {config_path}")

    with open(config_path, "r") as handle:
        config_dict = json.load(handle)

    config = MultimodalTrainingConfig.from_params(config_dict)

    tokenizer_path = os.path.join(models_dir, "tokenizer")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Saved tokenizer not found at {tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    print(f"Loaded tokenizer with vocabulary size {len(tokenizer)}")

    model = MLLM(
        vision_model_name=config.vision_model_name,
        language_model_name=config.language_model_name,
        vision_path=config.vision_path,
        num_vision_tokens=config.num_vision_tokens,
        labels_mapping_path=config.labels_mapping_path,
    )

    embeddings_by_epoch: Dict[str, torch.Tensor] = {}

    initial_model_path = os.path.join(models_dir, "initial_model.pt")
    if os.path.exists(initial_model_path):
        model.load_state_dict(torch.load(initial_model_path, map_location="cpu"))
        model.eval()
        embeddings = model.language_model.get_input_embeddings().weight.detach().cpu()
        embeddings_by_epoch["initial"] = embeddings.clone()
        print(f"Loaded initial embeddings {tuple(embeddings.shape)}")

    epoch_files = [f for f in os.listdir(models_dir) if f.startswith("epoch_") and f.endswith("_model.pt")]
    epoch_files.sort(key=lambda name: int(name.split("_")[1]))

    if max_epochs is not None:
        epoch_files = epoch_files[:max_epochs]
        print(f"Limiting to {len(epoch_files)} epoch checkpoints")

    for epoch_file in epoch_files:
        epoch_num = epoch_file.split("_")[1]
        model_path = os.path.join(models_dir, epoch_file)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        embeddings = model.language_model.get_input_embeddings().weight.detach().cpu()
        embeddings_by_epoch[f"epoch_{epoch_num}"] = embeddings.clone()
        print(f"Loaded epoch {epoch_num} embeddings {tuple(embeddings.shape)}")

    return embeddings_by_epoch, tokenizer, config


def extract_tokens_from_saved_tokenizer(tokenizer, config):
    """Extract OOD and regular token lists from the saved metadata."""
    labels_mapping = None
    ood_tokens: List[str] = []
    regular_tokens: List[str] = []

    if config.labels_mapping_path and os.path.exists(config.labels_mapping_path):
        with open(config.labels_mapping_path, "r") as handle:
            labels_mapping = json.load(handle)
        values = list(labels_mapping.values())
        ood_tokens = [token for token in values if token.startswith("<ood")]
        regular_tokens = [token for token in values if not token.startswith("<ood")]
        print(f"Labels mapping provided {len(regular_tokens)} regular and {len(ood_tokens)} OOD tokens")
    else:
        vocabulary = tokenizer.get_vocab()
        for token in vocabulary:
            if token.startswith("<ood"):
                ood_tokens.append(token)
            elif not token.startswith("<") and len(token) > 1:
                regular_tokens.append(token)
        print("No labels mapping found; derived token lists from tokenizer vocabulary")

    print(f"Total regular tokens considered: {len(regular_tokens)}")
    print(f"Total OOD tokens considered: {len(ood_tokens)}")
    return labels_mapping, ood_tokens, regular_tokens


def average_embeddings_for_tokens(
    tokenizer,
    embeddings_by_epoch: Dict[str, torch.Tensor],
    token_names: List[str],
) -> Dict[str, torch.Tensor]:
    """Average sub-token embeddings for each token string per epoch."""
    averaged: Dict[str, torch.Tensor] = {}
    if not embeddings_by_epoch:
        return averaged

    embedding_dim = next(iter(embeddings_by_epoch.values())).shape[1]

    for epoch_name, embedding_matrix in embeddings_by_epoch.items():
        if not token_names:
            averaged[epoch_name] = torch.empty((0, embedding_dim), dtype=embedding_matrix.dtype)
            continue

        epoch_vectors: List[torch.Tensor] = []
        for token_name in token_names:
            token_ids = tokenizer.encode(token_name, add_special_tokens=False)
            if token_ids:
                token_embedding = embedding_matrix[token_ids].mean(dim=0)
            else:
                token_embedding = torch.zeros(embedding_dim, dtype=embedding_matrix.dtype)
            epoch_vectors.append(token_embedding)
        averaged[epoch_name] = torch.stack(epoch_vectors)
    return averaged


def _build_projection_dataframe(
    epochs: List[str],
    projections: Dict[str, np.ndarray],
    token_names: List[str],
    token_types: List[str],
    token_to_parent: Dict[str, str],
    wnid_to_name: Dict[str, str],
) -> pd.DataFrame:
    """Convert projected coordinates into a tidy DataFrame for Plotly Express."""
    rows: List[Dict[str, object]] = []
    for epoch_name in epochs:
        coords = projections[epoch_name]
        if coords.shape[0] != len(token_names):
            raise ValueError(
                f"Projection for {epoch_name} has {coords.shape[0]} rows, expected {len(token_names)}"
            )
        for idx, token_name in enumerate(token_names):
            parent_id = token_to_parent.get(token_name, token_name)
            parent_label = wnid_to_name.get(parent_id, parent_id)
            rows.append(
                {
                    "epoch": epoch_name,
                    "token": token_name,
                    "token_type": token_types[idx],
                    "parent": parent_id,
                    "parent_name": parent_label,
                    "umap_x": coords[idx, 0],
                    "umap_y": coords[idx, 1],
                }
            )
    return pd.DataFrame(rows)


def create_interactive_umap(
    ood_embeddings: Dict[str, torch.Tensor],
    regular_embeddings: Dict[str, torch.Tensor],
    ood_tokens: List[str],
    regular_tokens: List[str],
    token_to_parent: Dict[str, str],
    parent_to_color: Dict[str, str],
    wnid_to_name: Dict[str, str],
    output_dir: str | None,
) -> None:
    """Generate a single lightweight HTML report with an animated UMAP scatter plot."""
    token_names = ood_tokens + regular_tokens
    if len(token_names) < 3:
        print("Not enough tokens for UMAP visualization (need at least three). Skipping plot generation.")
        return

    epoch_names = sorted(set(ood_embeddings.keys()) | set(regular_embeddings.keys()))
    epochs = _sort_epochs(epoch_names)

    combined_embeddings: Dict[str, torch.Tensor] = {}
    for epoch_name in epochs:
        pieces: List[torch.Tensor] = []
        if ood_tokens:
            if epoch_name not in ood_embeddings:
                raise KeyError(f"Missing OOD embeddings for {epoch_name}")
            pieces.append(ood_embeddings[epoch_name])
        if regular_tokens:
            if epoch_name not in regular_embeddings:
                raise KeyError(f"Missing regular embeddings for {epoch_name}")
            pieces.append(regular_embeddings[epoch_name])
        combined_embeddings[epoch_name] = torch.cat(pieces, dim=0)

    projections = _fit_umap_and_transform(combined_embeddings, n_components=2)
    token_types = ["OOD"] * len(ood_tokens) + ["Regular"] * len(regular_tokens)
    df = _build_projection_dataframe(epochs, projections, token_names, token_types, token_to_parent, wnid_to_name)

    parent_color_map = {
        wnid_to_name.get(parent, parent): color for parent, color in parent_to_color.items()
    }

    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color="parent_name",
        color_discrete_map=parent_color_map,
        symbol="token_type",
        symbol_map=SYMBOL_MAP,
        hover_data={"token": True, "token_type": True, "parent_name": True},
        animation_frame="epoch",
        category_orders={
            "epoch": epochs,
            "token_type": [label for label in SYMBOL_MAP if label in token_types],
        },
    )

    fig.update_traces(marker=dict(size=8, line=dict(width=0.6, color="#1f2933")))
    fig.update_layout(
        title="Token embeddings UMAP",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title_text="WordNet parent",
        width=900,
        height=700,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, "token_embeddings_umap.html")
        fig.write_html(html_path)
        print(f"Saved interactive UMAP visualization to {html_path}")
    else:
        print("No output directory provided; skipping HTML export")


def print_token_examples(regular_tokens: List[str], ood_tokens: List[str]) -> None:
    """Log a brief snapshot of the selected tokens."""
    print("\nSample regular tokens:")
    for token in regular_tokens[:5]:
        print(f"  {token}")
    if len(regular_tokens) > 5:
        print(f"  ... and {len(regular_tokens) - 5} more")

    print("\nSample OOD tokens:")
    for token in ood_tokens[:5]:
        print(f"  {token}")
    if len(ood_tokens) > 5:
        print(f"  ... and {len(ood_tokens) - 5} more")


def main() -> None:
    parser = argparse.ArgumentParser(description="Token Embedding Analysis for ImageNet Models")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/users/sboppana/data/sboppana/multimodal_concept_learning/results/multimodal/imagenet/imagenet1k_timm_vit",
        help="Path to results directory containing the trained model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output plots (defaults to results_dir/plots)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Maximum number of epochs to analyze (None for all epochs)",
    )
    parser.add_argument(
        "--parent_level",
        type=int,
        default=4,
        help="WordNet hierarchy level to use for coloring (1=parent, 2=grandparent, ...)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/users/sboppana/data/sboppana/multimodal_concept_mapping/data/imagenet",
        help="Path to ImageNet data directory containing WordNet hierarchy files",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "plots")

    print(f"Loading embeddings from {args.results_dir}")
    print(f"Saving plots to {args.output_dir}")
    if args.max_epochs is not None:
        print(f"Restricting to the first {args.max_epochs} epochs")

    parent_to_children, child_to_parents, wnid_to_name = load_wordnet_hierarchy(args.data_dir)
    embeddings_by_epoch, tokenizer, config = load_token_embeddings(args.results_dir, args.max_epochs)
    labels_mapping, ood_tokens, regular_tokens = extract_tokens_from_saved_tokenizer(tokenizer, config)

    print("\n=== Token Overview ===")
    print(f"Total vocabulary size: {len(tokenizer)}")
    print(f"Regular tokens selected: {len(regular_tokens)}")
    print(f"OOD tokens selected: {len(ood_tokens)}")
    print_token_examples(regular_tokens, ood_tokens)

    def averaged_embeddings(token_list: List[str]) -> Dict[str, torch.Tensor]:
        return average_embeddings_for_tokens(tokenizer, embeddings_by_epoch, token_list)

    ood_embeddings = averaged_embeddings(ood_tokens)
    regular_embeddings = averaged_embeddings(regular_tokens)

    if labels_mapping:
        token_to_wnid = {token_name: wnid for wnid, token_name in labels_mapping.items()}
    else:
        token_to_wnid = {}

    print(f"\nAssigning colors using WordNet hierarchy level {args.parent_level}")
    all_tokens = ood_tokens + regular_tokens
    token_to_color, token_to_parent, parent_to_color = get_path_based_colors(
        all_tokens,
        token_to_wnid,
        parent_to_children,
        child_to_parents,
        wnid_to_name,
        args.parent_level,
    )

    for token_name in all_tokens[:5]:
        parent_id = token_to_parent.get(token_name, token_name)
        parent_label = wnid_to_name.get(parent_id, parent_id)
        print(f"  {token_name} → {parent_label}")

    if not all_tokens:
        print("No tokens available to visualise; exiting.")
        return

    create_interactive_umap(
        ood_embeddings,
        regular_embeddings,
        ood_tokens,
        regular_tokens,
        token_to_parent,
        parent_to_color,
        wnid_to_name,
        args.output_dir,
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
