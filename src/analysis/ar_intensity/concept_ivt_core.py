"""Pure analysis functions for concept-IVT tracking decomposition. No I/O, no torch.
Pointwise gap method: no rolling window, no derivatives."""
import numpy as np

HEMISPHERE = {"W_N_America": "NH", "W_Europe": "NH",
              "W_S_America": "SH", "E_Australia": "SH"}
WET_MONTHS = {"NH": {11, 12, 1, 2, 3}, "SH": {5, 6, 7, 8, 9}}

def pearson_cols(A, y):
    """Pearson r between each column of A (T,F) and vector y (T,). Returns (F,)."""
    a = A - A.mean(0, keepdims=True); yi = y - y.mean()
    num = (a * yi[:, None]).sum(0)
    den = np.sqrt((a ** 2).sum(0) * (yi ** 2).sum())
    return np.divide(num, den, out=np.full(A.shape[1], np.nan), where=den > 0)

def select_top_concepts(A, y, k=3):
    """Indices of the top-k columns of A by signed Pearson r with y, plus the full r vector."""
    r = pearson_cols(A, y)
    order = np.argsort(np.nan_to_num(r, nan=-np.inf))[::-1]
    return order[:k], r

def zscore(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)

def pointwise_gap(z_act, z_ivt):
    """Per-timestep disagreement; small = concept and IVT co-deviate (tracking)."""
    return np.abs(z_act - z_ivt)

def classify_corr(gap, q_lo=0.25, q_hi=0.75):
    """Label timesteps: high_corr (gap in bottom q_lo), low_corr (gap in top 1-q_hi), else mid."""
    lo = np.quantile(gap, q_lo); hi = np.quantile(gap, q_hi)
    out = np.full(len(gap), "mid", dtype=object)
    out[gap <= lo] = "high_corr"
    out[gap >= hi] = "low_corr"
    return out

def season_label(months, hemisphere):
    wet = WET_MONTHS[hemisphere]
    return np.where(np.isin(months, list(wet)), "wet", "dry")

def ivt_regime(ivt, thr=250.0, q_mod=0.5, q_int=0.9):
    """no_ar (<thr); among >=thr split weak/moderate/intense by quantiles of AR-only IVT."""
    out = np.full(len(ivt), "no_ar", dtype=object); ar = ivt >= thr
    if ar.any():
        a = ivt[ar]; qm = np.quantile(a, q_mod); qi = np.quantile(a, q_int)
        out[ar] = np.where(a >= qi, "intense", np.where(a >= qm, "moderate", "weak"))
    return out
