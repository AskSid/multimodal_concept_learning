#!/bin/bash
#SBATCH --job-name=token_analysis
#SBATCH --output=logs/token_analysis_%j.out
#SBATCH --error=logs/token_analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch


source /users/sboppana/data/sboppana/multimodal_concept_learning/.venv/bin/activate

EXPERIMENT_DIR=$1
OUTPUT_DIR=$EXPERIMENT_DIR/token_analysis
mkdir -p $OUTPUT_DIR


# Navigate to the project directory
cd /users/sboppana/data/sboppana/multimodal_concept_learning

# Set Python path to include the project root
export PYTHONPATH=/users/sboppana/data/sboppana/multimodal_concept_learning:$PYTHONPATH

# Run the token embedding analysis
uv run python src/multimodal/token_embedding_analysis.py --results_dir $EXPERIMENT_DIR --output_dir $OUTPUT_DIR

echo "Token embedding analysis completed at $(date)"
