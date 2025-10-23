#!/bin/bash

# Script to train vision models on all color datasets
# This script runs vision training experiments for each color dataset

set -e  # Exit on any error

# Base directory
BASE_DIR="/users/sboppana/data/sboppana/multimodal_concept_learning"
EXPERIMENTS_DIR="$BASE_DIR/experiments/vision/color"
RESULTS_DIR="$BASE_DIR/results/vision"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Array of experiment configs
EXPERIMENTS=(
    "primary_colors_10k"
    "primary_secondary_5k" 
    "12_colors_3k"
    "24_colors_1_5k"
    "48_colors_750"
)

echo "Starting vision color experiments..."
echo "=================================="

# Run each experiment
for experiment in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "Training vision model for: $experiment"
    echo "----------------------------------------"
    
    config_path="$EXPERIMENTS_DIR/${experiment}.yaml"
    
    if [ ! -f "$config_path" ]; then
        echo "ERROR: Config file not found: $config_path"
        continue
    fi
    
    echo "Config: $config_path"
    echo "Results will be saved to: $RESULTS_DIR/$experiment"
    
    # Run the training
    python "$BASE_DIR/src/vision/vision_training.py" \
        --config_path "$config_path"
    
    echo "Completed training for: $experiment"
    echo ""
done

echo "All vision color experiments completed!"
echo "Results saved in: $RESULTS_DIR"
