"""Run climate_concept_map.make_map for a chosen region/concept/timestamp."""
import os, torch, datetime as DT
from src.analysis.ar_intensity.climate_concept_map import make_map
from src.analysis.ar_intensity.sae_features import load_sae
from src.analysis.ar_intensity.ivt_pipeline import load_channel_index
from src.analysis.ar_intensity.ivt import layer_thickness_pa
REGION = os.environ.get("REGION", "W_N_America"); CONCEPT = int(os.environ.get("CONCEPT", "1592"))
TS = os.environ.get("TS", "1995-12-10T12:00"); dt = DT.datetime.strptime(TS, "%Y-%m-%dT%H:%M")
dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, "->", REGION, CONCEPT, dt, flush=True)
idx, levels, qi, ui, vi, lat_i, lon_i = load_channel_index(); dp = layer_thickness_pa(levels)
m, c, fmin, frng = load_sae("plain_L8", dev)
make_map(REGION, CONCEPT, dt, m, c, dev, levels, qi, ui, vi, lat_i, lon_i, dp)
print("done")
