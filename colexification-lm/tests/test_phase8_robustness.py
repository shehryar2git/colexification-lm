"""Phase 8 tests: the robustness module (axes, verdict rule, RSA helper, CLICS3 reader).

NOTE. An earlier copy of this file tested a different API (`mantel_conclusion`,
`group_test_conclusion`, `edge_share`, `summarise(tables)`) that `src/analysis/robustness.py`
has never had; its one good idea, grouping the checks by the proposal's robustness axes
(i)-(v), is now implemented as `robustness.axis_of` and tested below.
"""
from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest
from scipy import stats as ss

from src.analysis import robustness as rb

GML = """graph [
  node [
    id 0
    Gloss "HAND"
    Words "a;b"
  ]
  node [
    id 1
    Gloss "ARM"
  ]
  node [
    id 2
    Gloss "TREE"
  ]
  edge [
    source 0
    target 1
    words "x/y"
    families "A;B;C"
    FamilyWeight 3
    LanguageWeight 11
  ]
  edge [
    source 1
    target 2
    FamilyWeight 7
    LanguageWeight 9
  ]
]
"""


def _correlated(n=16, seed=0):
    g = np.random.default_rng(seed)
    x = g.normal(size=(n, 3))
    d = np.abs(x[:, [0]] - x[:, [0]].T)
    m = d + g.normal(scale=0.05, size=(n, n))
    m = (m + m.T) / 2
    np.fill_diagonal(d, 0)
    np.fill_diagonal(m, 0)
    return d, m


# ------------------------------------------------------------------- axes --
def test_every_axis_is_covered_and_mapped():
    assert set(rb.AXIS_OF.values()) | {"v", "extra"} == set(rb.AXES)
    assert rb.axis_of("typ_minfam3") == "i"                      # binary vs family-weighted
    assert rb.axis_of("typ_profile_cosine") == "ii"              # alternative metric
    assert rb.axis_of("typ_rate") == "iii"                       # non-shortest-path distance
    assert rb.axis_of("concepts_threshold_100") == "iv"          # concept-set threshold
    assert rb.axis_of("lofo_indo-european") == "v"               # leave-one-family-out
    assert rb.axis_of("h2_caliper_010") == "extra"


# ---------------------------------------------------------------- verdicts --
def test_conclusion_and_verdict_rule():
    assert rb.conclusion_of(0.2, 0.001) == "supported"
    assert rb.conclusion_of(0.2, 0.40) == "not supported"
    assert rb.conclusion_of(-0.2, 0.001) == "not supported"      # wrong sign is not support
    assert rb.conclusion_of(np.nan, 0.001) == "not computable"
    assert rb.verdict(0.2, 0.001, True) == "survives"
    assert rb.verdict(0.2, 0.40, True) == "does not survive"
    assert rb.verdict(0.01, 0.40, False) == "survives"           # both conclude "not supported"
    assert rb.verdict(0.2, 0.001, False) == "does not survive"   # variant finds what primary did not
    assert rb.verdict(np.nan, np.nan, True) == "not computable"


# --------------------------------------------------------------------- RSA --
def test_rsa_matches_spearman_and_is_reproducible():
    a, b = _correlated(seed=1)
    r = rb.rsa(a, b, ("t", "1"))
    iu = np.triu_indices(a.shape[0], 1)
    assert r["value"] == pytest.approx(ss.spearmanr(a[iu], b[iu]).statistic)
    assert r["p"] < 0.05 and r["ci_low"] <= r["value"] <= r["ci_high"]
    assert rb.rsa(a, b, ("t", "1"))["p"] == r["p"]
    assert rb.verdict(r["value"], r["p"], True) == "survives"


def test_rsa_under_noise_does_not_claim_support():
    g = np.random.default_rng(5)
    n = 16
    a = np.abs(g.normal(size=(n, n)))
    a = (a + a.T) / 2
    b = np.abs(g.normal(size=(n, n)))
    b = (b + b.T) / 2
    np.fill_diagonal(a, 0)
    np.fill_diagonal(b, 0)
    r = rb.rsa(a, b, ("t", "noise"))
    assert rb.conclusion_of(r["value"], r["p"]) in {"supported", "not supported"}
    assert r["p"] >= 1 / (rb.NPERM + 1)


def test_finite_subset_drops_one_bad_concept_not_all():
    a, b = _correlated(n=12, seed=2)
    bad = a.copy()
    bad[3, :] = np.nan
    bad[:, 3] = np.nan
    keep = rb.finite_subset(bad, b)
    assert keep.sum() == 11 and not keep[3]
    assert rb.rsa(bad, b, ("t", "nan"))["n_concepts"] == 11
    both = b.copy()
    both[7, :] = both[:, 7] = np.nan
    assert rb.finite_subset(bad, both).sum() == 10


# ------------------------------------------------------------------ CLICS3 --
def test_parse_clics3_gml(tmp_path):
    p = tmp_path / "net.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("graphs/network-3-families.gml", GML)
    e = rb.parse_clics3_gml(p, "graphs/network-3-families.gml")
    assert e.attrs["n_nodes"] == 3 and len(e) == 2
    r = e.set_index(["concept_a", "concept_b"])
    assert r.loc[("ARM", "HAND"), "n_families"] == 3          # stored with concept_a < concept_b
    assert r.loc[("ARM", "HAND"), "n_languages"] == 11
    assert r.loc[("ARM", "TREE"), "n_families"] == 7
    assert list(e.columns) == ["concept_a", "concept_b", "n_families", "n_languages"]


# ------------------------------------------------------- the written tables --
def test_robustness_tables_are_consistent():
    import config
    p = config.RESULTS_DIR / "robustness.csv"
    if not p.exists():
        pytest.skip("robustness not run yet (python -m src.analysis.robustness)")
    tab = pd.read_csv(p)
    assert tab["axis"].isin(rb.AXES).all()
    for axis in ("i", "ii", "iii", "iv", "v"):                 # the proposal's five axes are all present
        assert (tab["axis"] == axis).any(), f"no check for robustness axis ({axis})"
    # single checks: the verdict must be exactly the rule applied to that row's own numbers.
    # Rows whose verdict aggregates several runs (*_summary) or that are controls are excluded.
    single = tab[tab["verdict"].isin(["survives", "does not survive"]) & tab["value"].notna()
                 & tab["variant_conclusion"].isin(["supported", "not supported"])
                 & (tab["verdict_basis"] == "reproduces the primary analysis")
                 & ~tab["check"].str.endswith("_summary", na=False)]
    assert len(single) > 20
    for r in single.itertuples():
        assert rb.conclusion_of(r.value, r.p) == r.variant_conclusion
        assert rb.verdict(r.value, r.p, r.primary_conclusion == "supported") == r.verdict
    # H3 rows have no primary analysis to reproduce: "survives" means "supports H3"
    h3 = tab[(tab["verdict_basis"] == "tests H3 directly") & tab["value"].notna()
             & tab["verdict"].isin(["survives", "does not survive"])
             & tab["variant_conclusion"].isin(["supported", "not supported"])]
    for r in h3.itertuples():
        assert (r.verdict == "survives") == (rb.conclusion_of(r.value, r.p) == "supported")
    # summary rows survive only when every run they summarise survives
    for r in tab[tab["check"].str.endswith("_summary", na=False)].itertuples():
        runs = tab[tab["check"].str.startswith(r.check.replace("_summary", "_"), na=False)
                   & ~tab["check"].str.endswith("_summary", na=False)]
        if len(runs):
            assert (r.verdict == "survives") == bool((runs["verdict"] == "survives").all())
    summ = pd.read_csv(config.RESULTS_DIR / "robustness_summary.csv")
    assert len(summ) >= 20 and summ["conclusion"].notna().all()
    byaxis = pd.read_csv(config.RESULTS_DIR / "robustness_by_axis.csv")
    assert (byaxis["n_survive"] <= byaxis["n_checks"]).all()
