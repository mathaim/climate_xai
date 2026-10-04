import numpy as np
BINS = ["bottom10","low_mid40","up_mid40","top10"]
def assign_bins(maxivt, p10, p50, p90):
    """< p10 -> bottom10 ; [p10,p50) -> low_mid40 ; [p50,p90) -> up_mid40 ; >= p90 -> top10."""
    v = np.asarray(maxivt, float)
    return np.where(v<p10,"bottom10", np.where(v<p50,"low_mid40", np.where(v<p90,"up_mid40","top10")))
