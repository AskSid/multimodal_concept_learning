#!/usr/bin/env python3
"""Token embedding analysis for ImageNet models using Matplotlib.

The script loads saved token embeddings, projects them with UMAP, and writes six
static figures (2D and 3D for three token subsets). The goal is to keep the
interface simple: no interactive UI, just small PNGs coloured by the chosen
WordNet ancestor level.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - activates 3D projection
import numpy as np
import torch
import umap
from transformers import AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.multimodal.mllm import MLLM
from src.multimodal.multimodal_training_config import MultimodalTrainingConfig


FALLBACK_COLOR = "#636363"
DEFAULT_LEGEND_MAX = 12


def load_wordnet_hierarchy(data_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, str]]:
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
    all_children = {child for children in parent_to_children.values() for child in children}
    return [parent for parent in parent_to_children if parent not in all_children]


def get_nodes_at_depth(parent_to_children: Dict[str, List[str]], root_nodes: List[str], depth: int) -> List[str]:
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
    path = [wnid]
    current = wnid
    while current in child_to_parents and child_to_parents[current]:
        current = child_to_parents[current][0]
        path.append(current)
    return path


def build_palette() -> List[str]:
    qualitative = []
    for name in ["tab20", "tab20b", "tab20c", "Set3"]:
        cmap = cm.get_cmap(name)
        qualitative.extend([cmap(i) for i in range(cmap.N)])
    hex_colors = [
        "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
        for r, g, b, *_ in qualitative
    ]
    return hex_colors or [FALLBACK_COLOR]


def get_path_based_colors(
    token_names: List[str],
    token_to_wnid: Dict[str, str],
    parent_to_children: Dict[str, List[str]],
    child_to_parents: Dict[str, List[str]],
    wnid_to_name: Dict[str, str],
    depth: int,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    root_nodes = find_root_nodes(parent_to_children)
    target_nodes = get_nodes_at_depth(parent_to_children, root_nodes, depth)
    if not target_nodes:
        target_nodes = list(root_nodes)

    palette = build_palette()
    parent_to_color: Dict[str, str] = {}
    token_to_parent: Dict[str, str] = {}
    token_to_color: Dict[str, str] = {}
    color_index = 0

    def claim_color(parent_id: str) -> str:
        nonlocal color_index
        if parent_id not in parent_to_color:
            parent_to_color[parent_id] = palette[color_index % len(palette)]
            color_index += 1
        return parent_to_color[parent_id]

    for node in target_nodes:
        claim_color(node)

    for token in token_names:
        wnid = token_to_wnid.get(token)
        parent_choice = wnid
        if wnid:
            path = get_path_to_root(wnid, child_to_parents)
            parent_choice = next((node for node in path if node in parent_to_color), wnid)
        if parent_choice is None:
            parent_choice = token
        token_to_parent[token] = parent_choice
        token_to_color[token] = claim_color(parent_choice)

    return token_to_color, token_to_parent, parent_to_color


def _to_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    return array.detach().cpu().float().numpy()


def _fit_umap(embeddings: torch.Tensor, n_components: int) -> np.ndarray:
    matrix = _to_numpy(embeddings)
    if matrix.shape[0] < max(3, n_components + 1):
        raise ValueError("Need more tokens to run UMAP for the requested dimensionality.")
    n_neighbors = max(2, min(15, matrix.shape[0] - 1))
    reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=n_neighbors)
    return reducer.fit_transform(matrix)


def _sort_epochs(epoch_names: Iterable[str]) -> List[str]:
    names = list(epoch_names)
    has_initial = "initial" in names
    if has_initial:
        names.remove("initial")
    names.sort(key=lambda name: int(name.split("_")[1]) if name.startswith("epoch_") else name)
    if has_initial:
        names.insert(0, "initial")
    return names


def load_token_embeddings(results_dir: str, max_epochs: int | None = None):
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
        vocab = tokenizer.get_vocab()
        for token in vocab:
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
    averaged: Dict[str, torch.Tensor] = {}
    if not embeddings_by_epoch:
        return averaged

    sample_epoch = next(iter(embeddings_by_epoch.values()))
    embedding_dim = sample_epoch.shape[1]

    for epoch_name, embedding_matrix in embeddings_by_epoch.items():
        if not token_names:
            averaged[epoch_name] = torch.empty((0, embedding_dim), dtype=embedding_matrix.dtype)
            continue
        vectors: List[torch.Tensor] = []
        for token_name in token_names:
            token_ids = tokenizer.encode(token_name, add_special_tokens=False)
            if token_ids:
                token_embedding = embedding_matrix[token_ids].mean(dim=0)
            else:
                token_embedding = torch.zeros(embedding_dim, dtype=embedding_matrix.dtype)
            vectors.append(token_embedding)
        averaged[epoch_name] = torch.stack(vectors)
    return averaged


def select_epoch(embeddings_by_epoch: Dict[str, torch.Tensor], requested_epoch: str | None) -> str:
    epochs = _sort_epochs(embeddings_by_epoch.keys())
    if not epochs:
        raise ValueError("No embedding checkpoints were loaded.")
    if requested_epoch and requested_epoch in embeddings_by_epoch:
        return requested_epoch
    if requested_epoch:
        print(f"Requested epoch '{requested_epoch}' not found; falling back to final epoch")
    return epochs[-1]


def add_parent_legend(ax, parents: List[str], parent_to_color: Dict[str, str], wnid_to_name: Dict[str, str]) -> None:
    counts = Counter(parents)
    top_parents = [parent for parent, _ in counts.most_common(DEFAULT_LEGEND_MAX)]
    handles = []
    labels = []
    for parent in top_parents:
        color = parent_to_color.get(parent, FALLBACK_COLOR)
        label = wnid_to_name.get(parent, parent)
        handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=color, markersize=6))
        labels.append(label)
    if handles:
        ax.legend(handles, labels, title="WordNet parent", loc="best", fontsize=8)


def save_scatter_2d(points: np.ndarray, colors: List[str], parents: List[str], parent_to_color: Dict[str, str],
                     wnid_to_name: Dict[str, str], title: str, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=16, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    add_parent_legend(ax, parents, parent_to_color, wnid_to_name)
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_scatter_3d(points: np.ndarray, colors: List[str], parents: List[str], parent_to_color: Dict[str, str],
                     wnid_to_name: Dict[str, str], title: str, output_path: str) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=16, depthshade=False)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    add_parent_legend(ax, parents, parent_to_color, wnid_to_name)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_umap_figures(
    label: str,
    epoch_name: str,
    embeddings: torch.Tensor,
    token_names: List[str],
    token_to_color: Dict[str, str],
    token_to_parent: Dict[str, str],
    parent_to_color: Dict[str, str],
    wnid_to_name: Dict[str, str],
    output_dir: str,
) -> None:
    if embeddings.shape[0] < 3:
        print(f"Not enough tokens to build {label} projections (need at least 3)")
        return

    colors = [token_to_color.get(token, FALLBACK_COLOR) for token in token_names]
    parents = [token_to_parent.get(token, token) for token in token_names]

    try:
        points_2d = _fit_umap(embeddings, n_components=2)
        title_2d = f"UMAP 2D ({label}, {epoch_name})"
        path_2d = os.path.join(output_dir, f"{epoch_name}_{label}_umap_2d.png")
        save_scatter_2d(points_2d, colors, parents, parent_to_color, wnid_to_name, title_2d, path_2d)
        print(f"  Saved {path_2d}")
    except ValueError as err:
        print(f"  Skipping 2D projection for {label}: {err}")

    try:
        points_3d = _fit_umap(embeddings, n_components=3)
        title_3d = f"UMAP 3D ({label}, {epoch_name})"
        path_3d = os.path.join(output_dir, f"{epoch_name}_{label}_umap_3d.png")
        save_scatter_3d(points_3d, colors, parents, parent_to_color, wnid_to_name, title_3d, path_3d)
        print(f"  Saved {path_3d}")
    except ValueError as err:
        print(f"  Skipping 3D projection for {label}: {err}")


def print_token_examples(regular_tokens: List[str], ood_tokens: List[str]) -> None:
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
    parser.add_argument(
        "--epoch",
        type=str,
        default=None,
        help="Specific epoch to visualise (e.g. 'initial' or 'epoch_10'). Defaults to final epoch",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(args.output_dir, exist_ok=True)

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

    def averaged_embeddings(tokens: List[str]) -> Dict[str, torch.Tensor]:
        return average_embeddings_for_tokens(tokenizer, embeddings_by_epoch, tokens)

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

    target_epoch = select_epoch(embeddings_by_epoch, args.epoch)
    print(f"\nTarget epoch for plots: {target_epoch}")

    combined_embeddings = None
    if ood_tokens and regular_tokens:
        combined_embeddings = torch.cat([
            ood_embeddings[target_epoch],
            regular_embeddings[target_epoch],
        ], dim=0)
    elif ood_tokens:
        combined_embeddings = ood_embeddings[target_epoch]
    elif regular_tokens:
        combined_embeddings = regular_embeddings[target_epoch]

    print("\nGenerating Matplotlib UMAP figures...")

    if combined_embeddings is not None and combined_embeddings.shape[0] >= 3:
        save_umap_figures(
            label="all_tokens",
            epoch_name=target_epoch,
            embeddings=combined_embeddings,
            token_names=all_tokens,
            token_to_color=token_to_color,
            token_to_parent=token_to_parent,
            parent_to_color=parent_to_color,
            wnid_to_name=wnid_to_name,
            output_dir=args.output_dir,
        )

    if regular_tokens and regular_embeddings.get(target_epoch) is not None:
        save_umap_figures(
            label="regular_tokens",
            epoch_name=target_epoch,
            embeddings=regular_embeddings[target_epoch],
            token_names=regular_tokens,
            token_to_color=token_to_color,
            token_to_parent=token_to_parent,
            parent_to_color=parent_to_color,
            wnid_to_name=wnid_to_name,
            output_dir=args.output_dir,
        )

    if ood_tokens and ood_embeddings.get(target_epoch) is not None:
        save_umap_figures(
            label="ood_tokens",
            epoch_name=target_epoch,
            embeddings=ood_embeddings[target_epoch],
            token_names=ood_tokens,
            token_to_color=token_to_color,
            token_to_parent=token_to_parent,
            parent_to_color=parent_to_color,
            wnid_to_name=wnid_to_name,
            output_dir=args.output_dir,
        )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
