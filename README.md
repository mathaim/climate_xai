# Climate XAI

Sparse autoencoder (SAE) interpretability for the GraphCast weather model. We
extract GraphCast's internal activations, train SAEs on them, validate the
learned concepts against atmospheric river (AR) ground truth from ERA5, and
causally test concepts by patching activations during forecasts.

## Repository layout

| Folder | Contents |
|---|---|
| `configs/` | Experiment configs and the conda environment (`environment.yml`, `matryoshka.json`) |
| `graphcast/` | GraphCast model tooling (e.g. the ActivationManager patch for component-filtered activation saving) |
| `extraction/` | Extract layer activations from GraphCast (`--layer N`) |
| `train/` | SAE training (`train_plain_sae.py`, `train_matryoshka_sae.py`); architectures in `train/models/` |
| `data_loaders/` | Load ERA5 fields and AR labels (`era5.py`, `ar_labels.py`, `regions.py`, `latents.py`) |
| `steering/` | Causal intervention: inject/clamp SAE concepts during forecasts and roll out (see `steering/README.md`) |
| `analysis/` | Concept analysis (see below) |
| `utils/` | Small shared helpers |

### analysis/
- `lib/` shared library: IVT computation, SAE feature loading, region definitions, binning
- `census/` AR-association census (one-sided Welch test over concept activations)
- `nesting/` concept containment and cross-layer hierarchy
- `enso/` ENSO concept analysis
- `adaptedsaebench/` SAEBench metrics ported to GraphCast SAEs: loss recovered, SCR, absorption, probing
- `tests/` unit tests

## Setup

    conda env create -f configs/environment.yml
    conda activate climate_xai

Run scripts from the repo root with the root on the path:

    PYTHONPATH=. python -m extraction.extract_activations --layer 8
    PYTHONPATH=. python -m train.train_matryoshka_sae --config configs/matryoshka.json
    PYTHONPATH=. python -m train.train_plain_sae --layer 8

## Pipeline

1. Extract GraphCast activations (`extraction/`)
2. Train SAEs per layer and architecture (`train/`)
3. Analyze concepts against AR intensity and labels (`analysis/`)
4. Steer by patching concepts during forecasts (`steering/`)

## Data alignment and regions

- Time: AR index N corresponds to activation timestep N and to
  1979-01-01 00:00 + (N-1)*6h. The AR record is a contiguous prefix of the
  activation record (56,700 AR steps vs 56,980 activation slots).
- AR definition: binary AR mask is class_masks == 2; background (0) and
  tropical cyclones (class 1) are excluded.
- Study regions: four landfall regions (Western North America, Western Europe,
  Western South America, and SE Australia as a distinct Tasman/subtropical
  regime). Exact bounding boxes are in analysis/lib/regions.py.

## Data and model

GraphCast checkpoints, extracted activations, SAE checkpoints, ERA5 inputs, and
AR labels are large and are not stored in this repository; they live on
project/scratch storage and are referenced through the data/ symlink and
graphcast_checkpoints/. SLURM job scripts are kept locally and are not part of
the repository.
