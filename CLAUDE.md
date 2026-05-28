# Climate XAI — Claude Code Context

## Project
SAE (Sparse Autoencoder) explainability applied to GraphCast weather model activations.
- Extract activations from GraphCast layers
- Train SAE variants (PlainSAE-TopK, MatryoshkaSAE-TopK) on those activations
- Validate learned features against atmospheric river (AR) ground truth from ERA5
- Ablation studies to understand feature importance

## Infrastructure
- **Compute**: UVA HPC (Rivanna/Afton) via SLURM
- **Login**: Open OnDemand web portal (https://ood.hpc.virginia.edu) or `ssh hpc`
- **Code**: `/home/euh7ys/climate_xai/` (git repo, cloned from GitHub)
- **Data/Activations/Checkpoints**: `/scratch/euh7ys/climate_xai/` (symlinked into repo, not in git)
- **Shared storage**: `/standard/AikyamLab/madelyn/` (lab shared, legacy code lives here)

## Directory Layout
```
climate_xai/
├── src/
│   ├── extraction/    # GraphCast activation extraction (layer as arg)
│   ├── models/        # SAE architectures (PlainSAE-TopK, MatryoshkaSAE-TopK)
│   ├── training/      # SAE training scripts (arch + layer as args)
│   ├── data/          # ERA5 data loading + AR ground-truth label loading
│   ├── analysis/      # Feature analysis & ablation studies
│   └── utils/         # Shared utilities
├── scripts/slurm/     # SLURM job submission scripts
├── configs/           # Experiment config files (YAML)
├── notebooks/         # Exploratory Jupyter notebooks
├── tests/             # Unit tests
├── data/              # SYMLINK → /scratch/.../data (ERA5 + AR labels)
├── activations/       # SYMLINK → /scratch/.../activations
├── checkpoints/       # SYMLINK → /scratch/.../checkpoints
├── results/           # SYMLINK → /scratch/.../results
└── logs/              # SYMLINK → /scratch/.../logs
```

## Key Design Decisions
- **Layer is always an argument**: extraction and training scripts take `--layer N` so one script covers all layers
- **Architecture is an argument**: training script takes `--arch plain|matryoshka` to select SAE variant
- **Both SAE types use TopK activation**: not vanilla ReLU
- **ERA5**: input data that feeds GraphCast; lives in `data/era5/`
- **AR labels**: atmospheric river ground truth from a separate detection algorithm; lives in `data/ar_labels/`

## Development Workflow
1. Edit code locally (Claude Code) or via Open OnDemand
2. Push changes to GitHub (`mathaim/climate_xai`, private)
3. On HPC: `cd ~/climate_xai && git pull` before submitting jobs
4. Submit SLURM jobs from `scripts/slurm/`
5. Results land in `/scratch/euh7ys/climate_xai/results/`

## SLURM Notes
- Partition: `gpu` for training/extraction, `standard` for data processing
- Request GPU with: `#SBATCH --gres=gpu:1`
- Load modules before running: `module load anaconda`
- Activate conda env: `conda activate climate_xai`

## Git Hygiene
- Never commit data files (*.nc, *.h5, *.pt, *.zarr, etc.)
- Never commit activations, checkpoints, or results
- Commit configs and scripts with every experiment
- Tag commits that correspond to paper/report results
