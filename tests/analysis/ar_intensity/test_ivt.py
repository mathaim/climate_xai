import numpy as np
from src.analysis.ar_intensity.ivt import ivt, layer_thickness_pa
def test_thickness_total_equals_range():
    assert np.isclose(layer_thickness_pa([1000,500]).sum(), 50000.0)   # 500 hPa = 50000 Pa
def test_ivt_zonal_only():
    q=np.array([0.01,0.01]); u=np.array([10.,10.]); v=np.array([0.,0.])
    dp=layer_thickness_pa([1000,900])
    assert np.isclose(ivt(q,u,v,[1000,900]), (q*u*dp).sum()/9.81)
def test_ivt_vectorized_positive():
    q=np.full((3,5),0.005); u=np.full((3,5),5.); v=np.full((3,5),5.)
    out=ivt(q,u,v,[1000,850,700,500,300]); assert out.shape==(3,) and (out>0).all()
