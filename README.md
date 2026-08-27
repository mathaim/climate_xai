# Climate XAI: Mechanistic Interpretability of Atmospheric Rivers in GraphCast

Code for *Beyond Finding Features: The Organization, Depth Evolution, and Causal Role of Concepts in AI Models of Climate Systems* (M. Mathai, T. B. Higgins, K. M. Grise, C. Agarwal, A. Mamalakis).

We train sparse autoencoders (SAEs) on [GraphCast](https://github.com/google-deepmind/graphcast) to identify, organize, trace, and causally test the concepts it learns for atmospheric rivers (ARs).

## Repository layout

    src/
      extraction/   # extract GraphCast mesh activations (layer as argument)
      models/       # SAE architectures: PlainSAE-TopK, MatryoshkaSAE-TopK
      training/     # SAE training (architecture + layer as arguments)
      data/         # ERA5 loading, AR labels, IVT and region definitions
      analysis/     # concept analysis, adapted SAEBench metrics, census
        ar_intensity/
      patching/     # activation steering / causal intervention on the forward pass
      utils/
    scripts/slurm/  # SLURM job submission scripts
    configs/        # experiment configs
    tests/          # unit tests

Data, activations, checkpoints, results, and generated figures are not tracked (they live on scratch storage); see `.gitignore`.

## Setup

    conda env create -f environment.yml
    conda activate climate_xai

## Reproducing the analysis

- Extract activations: `src/extraction/extract_activations.py` (per layer 0, 8, 15)
- Train SAEs: `src/training/train_plain_sae.py`, `src/training/train_matryoshka_sae.py`
- Adapted SAEBench (reconstruction / concept detection / disentanglement): `src/analysis/ar_intensity/sae_bench.py`, `sae_scr.py`
- AR census (regional vs. global AR-associated latents): `src/analysis/ar_intensity/census_*.py`
- Nesting / spatial containment (parent-child concepts): `src/analysis/ar_intensity/nesting_*.py`, `containment_*.py`
- IVT ground truth: `src/analysis/ar_intensity/ivt.py`, `ivt_pipeline.py`
- Causal steering: `src/patching/patch_predict.py` (forward-pass injector), `run_loss_recovered*.py` (loss recovered), and the `figure_*` / `plot_causal_rollouts.py` scripts (run and visualize the concept 1592 and 99 interventions)

## Data availability

GraphCast weights: Google DeepMind `dm_graphcast` bucket. ERA5: via WeatherBench 2 (originally ECMWF / Copernicus CDS). AR labels and IVT are derived from ERA5 as described in the paper.
