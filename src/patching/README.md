# Activation patching / causal intervention

Core:
- `patch_predict.py`   - GraphCast forward pass with an SAE concept edited at a layer
- `sae_to_jax.py`, `matry_delta.py` - build the steering edit (delta field) from an SAE concept
- `loss_recovered_prep.py`, `run_loss_recovered.py`, `run_loss_recovered_full.py` - loss-recovered metric

Paper causal experiments:
- `run_1592_inject.py` - inject concept 1592 into a clear day (sufficiency)
- `run_1592_dial.py`, `run_99_dial.py` - held-out bidirectional dials
- `run_rollout.py`      - 5-day rollout dials for concepts 1592 and 99

`archive/` holds exploratory and superseded scripts (kept for provenance, not needed to reproduce the paper).
