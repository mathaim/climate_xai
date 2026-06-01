#!/bin/bash
#SBATCH --job-name=msae_L15
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/msae_L15_%j.out
#SBATCH --error=logs/msae_L15_%j.err

# Train MatryoshkaSAE (BatchTopK) on GraphCast layer 15 activations.
# Data lives on standard storage (slower reads, but no scratch space).

mkdir -p ~/climate_xai/logs

eval "$(conda shell.bash hook)"
conda activate py312

cd ~/climate_xai

python -u -m src.training.train_matryoshka_sae \
    --layer 15 \
    --d_model 512 \
    --n_latents 4096 \
    --group_sizes 256 512 1024 2048 4096 \
    --target_l0 32 \
    --n_steps 300000 \
    --batch_size 4096 \
    --lr 3e-2 \
    --data_dir /standard/AikyamLab/madelyn/GraphCast/GraphCastData/layer15 \
    --output_dir /scratch/euh7ys/climate_xai/checkpoints/matryoshka_layer15 \
    --log_every 500 \
    --save_every 10000 \
    --metrics_every 5000

# Copy final model to standard storage (scratch gets purged after 90 days)
STANDARD_DIR=/standard/AikyamLab/madelyn/GraphCast/MatryoshkaSAE/checkpoints_layer15
mkdir -p "$STANDARD_DIR"
cp /scratch/euh7ys/climate_xai/checkpoints/matryoshka_layer15/final_model.pt "$STANDARD_DIR/"
cp /scratch/euh7ys/climate_xai/checkpoints/matryoshka_layer15/config.json "$STANDARD_DIR/"
cp /scratch/euh7ys/climate_xai/checkpoints/matryoshka_layer15/training_log.jsonl "$STANDARD_DIR/"
echo "Final model copied to $STANDARD_DIR"
