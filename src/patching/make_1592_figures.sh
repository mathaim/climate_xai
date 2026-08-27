#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
PNG=clear_1592_final.png NPZ=clear_maps_7906b.npz KEYS="inj1,inj2,inj3" \
  DLABELS='inject $b=0.2$||inject $b=0.4$||inject $b=0.6$' \
  python src/patching/figure_1592_suff.py
PNG=bc_1592_tworow.png NPZ=bc_maps_1592.npz AMPBETA=1.5 \
  python src/patching/figure_1592_tworow.py
