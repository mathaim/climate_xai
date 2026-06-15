import numpy as np
def cos_lat_coverage(ar_binary, lats):
    """Area-weighted AR fraction of a region.
    ar_binary: (T, nlat, nlon) in {0,1}; lats: (nlat,). Returns (T,) in [0,1]."""
    w = np.cos(np.deg2rad(np.asarray(lats, float)))
    num = (ar_binary * w[None, :, None]).sum(axis=(1, 2))
    den = w.sum() * ar_binary.shape[2]
    return num / den
