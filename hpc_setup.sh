#!/usr/bin/env bash
# Run this ONCE on the HPC after SSHing in: bash hpc_setup.sh
# Sets up the climate_xai project structure and git repo

set -e

PROJECT="climate_xai"
HOME_DIR="/home/euh7ys"
SCRATCH_DIR="/scratch/euh7ys"
REPO_DIR="$HOME_DIR/$PROJECT"

echo "=== Setting up $PROJECT on UVA HPC ==="

# 1. Create scratch directories for large files (not in git)
echo "[1/5] Creating scratch directories..."
mkdir -p "$SCRATCH_DIR/$PROJECT/data/raw"
mkdir -p "$SCRATCH_DIR/$PROJECT/data/processed"
mkdir -p "$SCRATCH_DIR/$PROJECT/checkpoints"
mkdir -p "$SCRATCH_DIR/$PROJECT/results"
mkdir -p "$SCRATCH_DIR/$PROJECT/logs"

# 2. Create the code repo directory structure
echo "[2/5] Creating project directory structure..."
mkdir -p "$REPO_DIR"
cd "$REPO_DIR"

mkdir -p src/models
mkdir -p src/data
mkdir -p src/xai
mkdir -p src/utils
mkdir -p notebooks
mkdir -p scripts/slurm
mkdir -p configs
mkdir -p tests

# 3. Symlink scratch folders into the repo (data/results stay on scratch)
echo "[3/5] Linking scratch storage into project..."
ln -sfn "$SCRATCH_DIR/$PROJECT/data"        "$REPO_DIR/data"
ln -sfn "$SCRATCH_DIR/$PROJECT/checkpoints" "$REPO_DIR/checkpoints"
ln -sfn "$SCRATCH_DIR/$PROJECT/results"     "$REPO_DIR/results"
ln -sfn "$SCRATCH_DIR/$PROJECT/logs"        "$REPO_DIR/logs"

# 4. Create placeholder __init__.py files
touch src/__init__.py
touch src/models/__init__.py
touch src/data/__init__.py
touch src/xai/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

# 5. Initialize git
echo "[4/5] Initializing git repo..."
git init
git config user.email "mathaimadelyn@gmail.com"
git config user.name "Madelyn Mathai"

echo "[5/5] Done! Directory structure:"
find "$REPO_DIR" -maxdepth 3 -not -path '*/.git/*' | sort

echo ""
echo "Next steps:"
echo "  1. cd $REPO_DIR"
echo "  2. git remote add origin git@github.com:YOUR_USERNAME/climate_xai.git"
echo "  3. git add . && git commit -m 'Initial project structure'"
echo "  4. git push -u origin main"
