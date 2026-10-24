"""Phase 5 tests: statistical helpers validated against exact or brute-force answers."""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import stats as ss

from src.data import frequency as fq
from src.utils import stats as st


def _dist(points: np.ndarray) -> np.ndarray:
    return np.sqrt(((points[:, None] - points[None]) ** 2).sum(-1))


# ------------------------------------------------------------------ Mantel --
def test_mantel_detects_monotone_relation_and_independence(correlated_distance_pair):
    d1, d2, d3 = correlated_distance_pair
    rg = np.random.default_rng(0)
    hi = st.mantel(d1, d2, 999, rg)
    assert hi.r > 0.95 and hi.p_greater == pytest.approx(1 / 1000)
    assert hi.r == pytest.approx(ss.spearmanr(st.upper(d1), st.upper(d2)).statistic)
    lo = st.mantel(d1, d3, 999, rg)
    assert abs(lo.r) < 0.2 and lo.p_two_sided > 0.05


def test_mantel_matches_exact_permutation_distribution():
    """n = 6: all 720 concept permutations enumerated -> exact one-sided p."""
    g = np.random.default_rng(5)
    a = _dist(g.normal(size=(6, 2)))
    b = a + _dist(g.normal(size=(6, 2))) * 0.8
    iu = np.triu_indices(6, 1)
    ra, rb = st.rank_square(a), st.rank_square(b)
    obs = np.corrcoef(ra[iu], rb[iu])[0, 1]
    exact = np.array([np.corrcoef(ra[iu], rb[np.ix_(p, p)][iu])[0, 1]
                      for p in map(list, itertools.permutations(range(6)))])
    p_exact = np.mean(exact >= obs - 1e-12)
    mc = st.mantel(a, b, 20000, np.random.default_rng(1))
    assert mc.r == pytest.approx(obs)
    assert mc.p_greater == pytest.approx(p_exact, abs=0.01)


def test_mantel_is_calibrated_under_the_null():
    g = np.random.default_rng(11)
    ps = [st.mantel(_dist(g.normal(size=(12, 2))), _dist(g.normal(size=(12, 2))), 199, g).p_greater
          for _ in range(300)]
    assert 0.02 <= np.mean(np.array(ps) <= 0.05) <= 0.09


def test_partial_mantel_removes_shared_cause():
    g = np.random.default_rng(2)
    z = _dist(g.normal(size=(25, 2)))
    x = z + 0.05 * _dist(g.normal(size=(25, 2)))
    y = z + 0.05 * _dist(g.normal(size=(25, 2)))
    raw = st.mantel(x, y, 199, g)
    part = st.partial_mantel(x, y, [z], 199, g)
    assert raw.r > 0.8 and abs(part.r) < raw.r / 2


# ---------------------------------------------------------- BH, bootstrap --
def test_bh_adjust_known_values():
    np.testing.assert_allclose(st.bh_adjust([0.01, 0.04, 0.03, 0.005]), [0.02, 0.04, 0.04, 0.02])
    out = st.bh_adjust([0.01, np.nan, 0.5])
    assert np.isnan(out[1]) and out[0] == pytest.approx(0.02) and out[2] == pytest.approx(0.5)


def test_bootstrap_pairs_drop_self_pairs():
    idx, r, c = st.bootstrap_pairs(30, np.random.default_rng(0))
    assert (r != c).all() and set(r) | set(c) <= set(idx)


def test_rsa_bootstrap_ci_covers_estimate(correlated_distance_pair):
    d1, d2, _ = correlated_distance_pair
    lo, hi = st.rsa_bootstrap_ci(d1, d2, 300, np.random.default_rng(3))
    r = ss.spearmanr(st.upper(d1), st.upper(d2)).statistic
    assert lo <= r <= hi and hi - lo < 0.1


# ----------------------------------------------------------- regression ----
def _pair_data(n=40, seed=0):
    g = np.random.default_rng(seed)
    x1 = _dist(g.normal(size=(n, 2)))
    x2 = _dist(g.normal(size=(n, 2)))
    y = 2 * x1 + _dist(g.normal(size=(n, 2))) * 0.5
    return y, x1, x2


def test_mrqap_power_and_calibration():
    y, x1, x2 = _pair_data()
    assert st.mrqap_dsp(y, [x1, x2], 499, np.random.default_rng(4))[0] < 0.01      # true predictor
    g = np.random.default_rng(123)                                                 # null predictor:
    ps = [st.mrqap_dsp(*(lambda d: (d[0], [d[1], d[2]]))(_pair_data(20, int(g.integers(1e9)))), 99, g)[1]
          for _ in range(120)]
    assert 0.01 <= np.mean(np.array(ps) <= 0.05) <= 0.11                           # ~5% false positives


def test_mrqap_fwl_t_equals_full_refit():
    """The Frisch-Waugh-Lovell shortcut must give the same t as refitting the full model."""
    y, x1, x2 = _pair_data(seed=1)
    iu = np.triu_indices(y.shape[0], 1)
    yv, a, b = y[iu], x1[iu], x2[iu]
    full_t = st._t_values(yv, np.column_stack([np.ones_like(yv), b, a]))[-1]
    others = np.column_stack([np.ones_like(yv), b])
    proj = np.linalg.pinv(others)
    re, ry = a - others @ (proj @ a), yv - others @ (proj @ yv)
    beta = re @ ry / (re @ re)
    s2 = (ry @ ry - beta ** 2 * (re @ re)) / (len(yv) - 3)
    assert beta / np.sqrt(s2 / (re @ re)) == pytest.approx(full_t, rel=1e-8)


def test_dyadic_se_matches_brute_force():
    g = np.random.default_rng(6)
    n = 9
    a, b = np.triu_indices(n, 1)
    D = np.column_stack([np.ones(len(a)), g.normal(size=len(a))])
    e = g.normal(size=len(a))
    meat = np.zeros((2, 2))
    for p in range(len(a)):
        for q in range(len(a)):
            if {a[p], b[p]} & {a[q], b[q]}:
                meat += np.outer(D[p] * e[p], D[q] * e[q])
    bread = np.linalg.pinv(D.T @ D)
    brute = np.sqrt(np.diag(bread @ meat @ bread))
    np.testing.assert_allclose(st.dyadic_cluster_se(D, e, a, b, n), brute)


# ------------------------------------------------------ Procrustes, Cliff --
def test_protest_identical_and_random():
    g = np.random.default_rng(7)
    a = g.normal(size=(30, 2))
    rot = np.array([[0, -1], [1, 0]])
    m2, p = st.protest(a, 3 * a @ rot + 1, 199, g)
    assert m2 == pytest.approx(0, abs=1e-10) and p == pytest.approx(1 / 200)
    m2r, pr = st.protest(a, g.normal(size=(30, 2)), 199, g)
    assert m2r > 0.7 and pr > 0.05


def test_cliffs_delta_matches_brute_force():
    g = np.random.default_rng(8)
    x, y = g.integers(0, 5, 30).astype(float), g.integers(1, 6, 25).astype(float)
    brute = np.mean([np.sign(a - b) for a in x for b in y])
    assert st.cliffs_delta(x, y) == pytest.approx(brute)
    assert st.cliffs_delta(y, x) == pytest.approx(-brute)
    lo, hi = st.cliffs_delta_ci(x, y, 500, np.random.default_rng(0))
    assert lo <= brute <= hi


# ------------------------------------------------------------- frequency ---
def test_msgpack_reader_and_zipf(tmp_path):
    import gzip
    # header map {"format": "cB", "version": 1}, then buckets: [] (0 cB), ["the"] (-1 cB), [] ... ["hand"] at -300 cB
    def s(x):
        b = x.encode()
        return bytes([0xA0 | len(b)]) + b
    buckets = [[] for _ in range(301)]
    buckets[1] = ["the"]
    buckets[300] = ["hand", "upper", "arm"]
    body = bytes([0xDC]) + (1 + len(buckets)).to_bytes(2, "big")
    body += bytes([0x82]) + s("format") + s("cB") + s("version") + bytes([1])
    for bk in buckets:
        body += bytes([0x90 | len(bk)]) + b"".join(s(w) for w in bk)
    p = tmp_path / "x.msgpack.gz"
    p.write_bytes(gzip.compress(body))
    d = fq.frequency_dict(str(p))
    assert d["hand"] == pytest.approx(1e-3) and d["the"] == pytest.approx(10 ** -0.01)
    assert fq.zipf("hand", d) == 6.0                     # 1e-3 -> log10 + 9
    assert fq.zipf("upper arm", d) == pytest.approx(round(np.log10(5e-4) + 9, 2))   # 1/f = 1000 + 1000
    assert fq.zipf("zzz", d) == 0.0
    with pytest.raises(ValueError):
        fq.zipf("hand2", d)


def test_zipf_matches_wordfreq_package():
    wordfreq = pytest.importorskip("wordfreq")
    import config
    if not config.WORDFREQ_FILE.exists():
        pytest.skip("wordfreq data not fetched")
    for w in ["the", "hand", "upper arm", "I", "housefly", "perspire"]:
        assert fq.zipf(w) == wordfreq.zipf_frequency(w, "en")


# --------------------------------------------------------- H1 integration --
def test_h1_regression_recovers_planted_colexification_effect():
    import pandas as pd
    from src.analysis import h1_alignment as h1
    g = np.random.default_rng(9)
    n = 30
    a, b = np.triu_indices(n, 1)
    fam = g.poisson(0.4, len(a)) * g.integers(1, 30, len(a))         # sparse, skewed support
    pf = pd.DataFrame({"i": a, "j": b, "n_families": fam})
    sim = np.zeros((n, n))
    sim[a, b] = 0.3 * np.log1p(fam) + g.normal(0, 0.3, len(a))
    sim = sim + sim.T
    np.fill_diagonal(sim, 1.0)
    zipf = g.normal(4, 1, n)
    field = (g.integers(0, 3, n)[:, None] == g.integers(0, 3, n)[None, :]).astype(float)
    field = np.triu(field, 1) + np.triu(field, 1).T
    tab = h1.regression("toy", sim, pf, zipf, field, n_perm=199, n_boot=200).set_index("term")
    assert tab.loc["log_families", "beta_std"] > 0.3
    assert tab.loc["log_families", "p_mrqap"] < 0.01 and tab.loc["log_families", "p_dyadic"] < 0.01
    assert tab.loc["log_families", "ci_low"] > 0
    assert abs(tab.loc["zipf_min", "beta_std"]) < 0.2


def test_cluster_agreement_perfect_and_structure():
    from src.analysis import h1_alignment as h1
    ref = np.repeat([0, 1, 2], 8)
    pts = np.repeat(np.array([[0, 0], [10, 0], [0, 10]]), 8, axis=0) + np.random.default_rng(0).normal(0, .1, (24, 2))
    dist = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    tab = h1.cluster_agreement("toy", dist, pts, ref, ref == 0, n_perm=99)
    prim = tab[tab["primary"]].iloc[0]
    assert prim["ari"] == pytest.approx(1.0) and prim["p_ari"] == pytest.approx(0.01)
    assert set(tab["subset"]) == {"all", "without_largest_community"}
    assert set(tab["method"]) == {"agglomerative", "kmeans"}
