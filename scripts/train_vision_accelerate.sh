#!/bin/bash
#SBATCH --job-name=vision_training
#SBATCH -p 3090-gcondo
#SBATCH --gres=gpu:8
#SBATCH --output=/dev/null
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -N 1

# Vision Training with Accelerate
# Usage: ./scripts/train_vision_accelerate.sh <config_path>
# Example: sbatch ./scripts/train_vision_accelerate.sh experiments/vision/imagenet100.yaml

set -e

CONFIG_PATH=$1

# Extract results_dir from config file
RESULTS_DIR=$(grep 'results_dir:' "$CONFIG_PATH" | sed 's/.*: *"\([^"]*\)".*/\1/')

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Redirect output to log.out in results directory
exec > "$RESULTS_DIR/log.out" 2>&1

# Set project root directory
PROJECT_ROOT="/users/sboppana/data/sboppana/multimodal_concept_learning"

# Change to project root directory
cd "$PROJECT_ROOT"

nvidia-smi

echo "Starting vision training with accelerate..."
echo "Config: $CONFIG_PATH"
echo "Project root: $PROJECT_ROOT"
echo "=========================================="

source "$PROJECT_ROOT/.venv/bin/activate"
accelerate launch \
    --mixed_precision fp16 \
    --num_processes 8 \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_port 29500 \
    src/vision/vision_training.py \
    --config_path "$CONFIG_PATH"

echo "Training completed!"