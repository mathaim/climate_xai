#!/bin/bash
#SBATCH --job-name=dense_to_sparse
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/dense_to_sparse_%j.out
#SBATCH --error=logs/dense_to_sparse_%j.err

# Convert dense SAE latents to sparse format to free ~3.2T on scratch.
# Uses GPU for fast top-k selection.
#
# Usage:
#   sbatch scripts/slurm/convert_dense_to_sparse.slurm

eval "$(conda shell.bash hook)"
conda activate py312

cd ~/climate_xai

echo "=== Converting PlainSAE dense test latents ==="
python -m src.analysis.convert_dense_to_sparse \
    --input_dir /scratch/euh7ys/latents_plain_dense_test \
    --output_dir /scratch/euh7ys/latents_plain_sparse_test \
    --k 32

echo ""
echo "=== Converting Matryoshka dense test latents ==="
python -m src.analysis.convert_dense_to_sparse \
    --input_dir /scratch/euh7ys/latents_matryoshka_dense_test \
    --output_dir /scratch/euh7ys/latents_matryoshka_sparse_test \
    --k 32

echo ""
echo "=== Done! Verify, then delete dense dirs manually: ==="
echo "  rm -rf /scratch/euh7ys/latents_plain_dense_test"
echo "  rm -rf /scratch/euh7ys/latents_matryoshka_dense_test"
echo "  (this frees ~3.2T)"
