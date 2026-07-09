import torch
from src.analysis.ar_intensity.sae_features import SAES
for name, c in SAES.items():
    st = torch.load(c["ckpt"], map_location="cpu")
    st = st.get("model_state_dict", st) if isinstance(st, dict) else st
    dm = st.get("dead_mask", None); mc = st.get("miss_counts", None)
    print(f"{name}: ckpt dead_mask={int(dm.sum()) if dm is not None else 'absent'}"
          f"  max_miss={int(mc.max()) if mc is not None else 'absent'}", flush=True)
