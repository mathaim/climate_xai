from src.analysis.ar_intensity.binning import assign_bins
def test_assign():
    b = assign_bins([5,15,30,60,95], 10, 40, 90)
    assert list(b) == ["bottom10","low_mid40","low_mid40","up_mid40","top10"]
