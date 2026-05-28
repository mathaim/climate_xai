#!/usr/bin/env bash
# Run this ONCE on the HPC after cloning the repo: bash hpc_setup.sh
# Creates scratch directories and symlinks them into the repo

set -e

PROJECT="climate_xai"
SCRATCH_DIR="/scratch/euh7ys"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Setting up $PROJECT scratch storage ==="
echo "Repo:    $REPO_DIR"
echo "Scratch: $SCRATCH_DIR/$PROJECT"

# 1. Create scratch directories for large files (not in git)
echo "[1/3] Creating scratch directories..."
mkdir -p "$SCRATCH_DIR/$PROJECT/data/era5"
mkdir -p "$SCRATCH_DIR/$PROJECT/data/ar_labels"
mkdir -p "$SCRATCH_DIR/$PROJECT/activations"
mkdir -p "$SCRATCH_DIR/$PROJECT/checkpoints"
mkdir -p "$SCRATCH_DIR/$PROJECT/results"
mkdir -p "$SCRATCH_DIR/$PROJECT/logs"

# 2. Symlink scratch folders into the repo
echo "[2/3] Linking scratch storage into project..."
ln -sfn "$SCRATCH_DIR/$PROJECT/data"         "$REPO_DIR/data"
ln -sfn "$SCRATCH_DIR/$PROJECT/activations"  "$REPO_DIR/activations"
ln -sfn "$SCRATCH_DIR/$PROJECT/checkpoints"  "$REPO_DIR/checkpoints"
ln -sfn "$SCRATCH_DIR/$PROJECT/results"      "$REPO_DIR/results"
ln -sfn "$SCRATCH_DIR/$PROJECT/logs"         "$REPO_DIR/logs"

# 3. Verify
echo "[3/3] Done! Symlinks:"
ls -la "$REPO_DIR/data" "$REPO_DIR/activations" "$REPO_DIR/checkpoints" "$REPO_DIR/results" "$REPO_DIR/logs"

echo ""
echo "Scratch layout:"
find "$SCRATCH_DIR/$PROJECT" -maxdepth 2 -type d | sort

echo ""
echo "Next steps:"
echo "  1. conda env create -f environment.yml"
echo "  2. Copy ERA5 data into $SCRATCH_DIR/$PROJECT/data/era5/"
echo "  3. Copy AR labels into  $SCRATCH_DIR/$PROJECT/data/ar_labels/"
