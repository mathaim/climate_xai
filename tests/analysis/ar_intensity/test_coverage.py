import numpy as np
from src.analysis.ar_intensity.coverage import cos_lat_coverage
def test_full():
    assert np.isclose(cos_lat_coverage(np.ones((1,4,5),np.uint8),[0,1,2,3])[0], 1.0)
def test_zero():
    assert np.isclose(cos_lat_coverage(np.zeros((1,4,5),np.uint8),[0,1,2,3])[0], 0.0)
def test_cos_weighting():
    ar = np.array([[[0],[1]]], np.uint8)         # (T=1, lat=2, lon=1); AR only at 60deg
    assert np.isclose(cos_lat_coverage(ar,[0,60])[0], 0.5/(1.0+0.5))   # area-weighted = 1/3
