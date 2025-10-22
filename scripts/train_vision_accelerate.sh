#!/bin/bash

# Vision Training with Accelerate
# Usage: ./scripts/train_vision_accelerate.sh <config_path>
# Example: ./scripts/train_vision_accelerate.sh experiments/vision/vision_training_example.yaml

set -e  # Exit on any error

# Check if config path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_path>"
    echo "Example: $0 experiments/vision/vision_training_example.yaml"
    exit 1
fi

CONFIG_PATH=$1

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file '$CONFIG_PATH' not found!"
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root directory
cd "$PROJECT_ROOT"

echo "Starting vision training with accelerate..."
echo "Config: $CONFIG_PATH"
echo "Project root: $PROJECT_ROOT"
echo "=========================================="

# Run training with accelerate (no separate config file needed)
accelerate launch \
    --mixed_precision no \
    --num_processes 1 \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_port 29500 \
    src/vision/vision_training.py \
    --config_path "$CONFIG_PATH"

echo "Training completed!"
