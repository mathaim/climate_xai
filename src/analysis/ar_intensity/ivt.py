import numpy as np
def layer_thickness_pa(levels_hpa):
    """Trapezoidal layer thickness (Pa) per level for vertical integration.
    levels_hpa in any order; returns Δp in the SAME order, in Pa."""
    p = np.asarray(levels_hpa, float) * 100.0          # hPa -> Pa
    order = np.argsort(p); ps = p[order]
    edges = np.empty(len(ps)+1)
    edges[1:-1] = 0.5*(ps[:-1]+ps[1:]); edges[0]=ps[0]; edges[-1]=ps[-1]
    dp = np.diff(edges)
    out = np.empty_like(dp); out[order] = dp
    return out
def ivt(q, u, v, levels_hpa, g=9.81):
    """IVT magnitude. q,u,v: (..., nlev); levels_hpa: (nlev,). Returns (...,) in kg/m/s."""
    dp = layer_thickness_pa(levels_hpa)
    iu = (q*u*dp).sum(axis=-1)/g
    iv = (q*v*dp).sum(axis=-1)/g
    return np.sqrt(iu*iu + iv*iv)
