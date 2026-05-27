# Climate XAI — Claude Code Context

## Project
SAE (Sparse Autoencoder) deep learning explainability applied to climate models.

## Infrastructure
- **Compute**: UVA HPC (Rivanna/Afton) via SLURM
- **Login**: `ssh hpc` (alias in ~/.ssh/config → login.hpc.virginia.edu, user euh7ys)
- **Code**: `/home/euh7ys/climate_xai/` (git repo)
- **Data/Checkpoints**: `/scratch/euh7ys/climate_xai/` (symlinked into repo, not in git)

## Directory Layout
```
climate_xai/
├── src/
│   ├── models/     # SAE and other model architectures
│   ├── data/       # Dataset loading and preprocessing
│   ├── xai/        # Explainability methods
│   └── utils/      # Shared utilities
├── notebooks/      # Exploratory Jupyter notebooks
├── scripts/
│   └── slurm/      # SLURM job submission scripts
├── configs/        # Experiment config files (YAML/JSON)
├── tests/          # Unit tests
├── data/           # SYMLINK → /scratch/euh7ys/climate_xai/data
├── checkpoints/    # SYMLINK → /scratch/euh7ys/climate_xai/checkpoints
├── results/        # SYMLINK → /scratch/euh7ys/climate_xai/results
└── logs/           # SYMLINK → /scratch/euh7ys/climate_xai/logs
```

## Development Workflow
1. Edit code locally (Claude Code) or via `ssh hpc`
2. Push changes to GitHub
3. On HPC: `git pull` before submitting jobs
4. Submit SLURM jobs from `scripts/slurm/`
5. Results land in `/scratch/euh7ys/climate_xai/results/`

## SLURM Notes
- Partition: `gpu` for training, `standard` for data processing
- Request GPU with: `#SBATCH --gres=gpu:1`
- Load modules before running: `module load anaconda`
- Activate conda env: `conda activate climate_xai`

## Git Hygiene
- Never commit data files (*.nc, *.h5, *.pt, *.zarr, etc.)
- Never commit checkpoints or results
- Commit configs and scripts with every experiment
- Tag commits that correspond to paper/report results
