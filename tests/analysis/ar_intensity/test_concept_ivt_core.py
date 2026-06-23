import numpy as np
from src.analysis.ar_intensity import concept_ivt_core as C

def test_select_top_concepts_ranks_tracking_column_first():
    rng = np.random.default_rng(0); T = 2000
    y = rng.normal(size=T)
    A = np.column_stack([y, -y, rng.normal(size=T)])
    idx, r = C.select_top_concepts(A, y, k=2)
    assert idx[0] == 0
    assert r[0] > 0.99 and r[1] < -0.99

def test_zscore_basic():
    z = C.zscore(np.array([1., 2., 3., 4.]))
    assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-9
    assert (C.zscore(np.array([5., 5., 5.])) == 0).all()

def test_gap_and_classify_split_tracking_vs_divergent():
    za = np.array([2., 1, 0.5, -0.5, -1, -2, 2, 1, 0.5, -0.5, -1, -2])
    zi = np.concatenate([za[:6], -za[6:]])
    gap = C.pointwise_gap(za, zi)
    lab = C.classify_corr(gap)
    assert (lab[:6] == "high_corr").sum() >= 4
    assert (lab[6:] == "low_corr").sum() >= 4

def test_ivt_regime_labels_with_no_ar():
    ivt = np.array([100., 300., 500., 900.])
    lab = C.ivt_regime(ivt, thr=250., q_mod=0.5, q_int=0.9)
    assert lab[0] == "no_ar"
    assert set(lab[1:]) == {"weak", "moderate", "intense"}

def test_season_label_hemisphere():
    months = np.array([1, 7, 12, 6])
    nh = C.season_label(months, "NH"); sh = C.season_label(months, "SH")
    assert list(nh) == ["wet", "dry", "wet", "dry"]
    assert list(sh) == ["dry", "wet", "dry", "wet"]
