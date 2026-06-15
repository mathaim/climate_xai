from datetime import datetime, timedelta
import numpy as np
from src.analysis.ar_intensity.regions import REGIONS, index_to_datetime, cos_lat_weights
def test_four_regions():
    assert set(REGIONS) == {"W_N_America","W_Europe","W_S_America","E_Australia"}
def test_index_alignment():
    assert index_to_datetime(1) == datetime(1979,1,1,0,0)
    assert index_to_datetime(5) == datetime(1979,1,2,0,0)
    assert index_to_datetime(56700) == datetime(1979,1,1,0,0)+timedelta(hours=6*56699)
def test_cos_lat_weights():
    w = cos_lat_weights([0,60]); assert np.isclose(w[0],1.0) and np.isclose(w[1],0.5,atol=1e-6)
