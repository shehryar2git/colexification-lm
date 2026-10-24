"""Phase 7 tests: H3 AUC, permutation identities, rarefaction, gradient, resource proxies."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as ss

from src.analysis import h3_typology as h3
from src.analysis import resource_proxies as rp


def _toy(n=12, seed=0):
    g = np.random.default_rng(seed)
    x = g.normal(size=(n, 4))
    sim = x @ x.T
    rate = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    pick = g.random(iu[0].size) < 0.25
    rate[iu[0][pick], iu[1][pick]] = g.uniform(0.05, 1, pick.sum())
    rate = rate + rate.T
    return sim, rate


def test_auc_brute_force():
    g = np.random.default_rng(1)
    s, lab = g.normal(size=60), g.random(60) < 0.3
    s[:5] = s[5]                                              # ties count one half
    pos, neg = s[lab], s[~lab]
    brute = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
    assert h3.auc(s, lab) == pytest.approx(brute)
    assert np.isnan(h3.auc(s, np.zeros(60, bool)))


def test_permutation_identities_match_naive_recomputation():
    sim, rate = _toy()
    n = sim.shape[0]
    iu = np.triu_indices(n, 1)
    res = h3.permutation_auc_spearman(sim, rate, 300, np.random.default_rng(7))
    assert res["auc"] == pytest.approx(h3.auc(sim[iu], rate[iu] > 0))
    assert res["spearman"] == pytest.approx(ss.spearmanr(sim[iu], rate[iu]).statistic)
    # naive: re-rank every permuted matrix (same random stream, one chunk)
    perms = np.argsort(np.random.default_rng(7).random((300, n)), axis=1)
    a_null = np.array([h3.auc(sim[np.ix_(p, p)][iu], rate[iu] > 0) for p in perms])
    s_null = np.array([ss.spearmanr(sim[np.ix_(p, p)][iu], rate[iu]).statistic for p in perms])
    assert res["p_auc_one_sided"] == pytest.approx(((a_null >= res["auc"] - 1e-12).sum() + 1) / 301)
    assert res["p_spearman_one_sided"] == pytest.approx(((s_null >= res["spearman"] - 1e-12).sum() + 1) / 301)


def test_permutation_null_is_calibrated():
    """Under H0 (similarity unrelated to colexification) p < .05 in about 5% of datasets."""
    g = np.random.default_rng(3)
    hits = 0
    for k in range(200):
        sim, rate = _toy(n=14, seed=1000 + k)
        sim = sim[np.ix_(g.permutation(14), g.permutation(14))]
        sim = (sim + sim.T) / 2
        hits += h3.permutation_auc_spearman(sim, rate, 199, g)["p_auc_one_sided"] < 0.05
    assert 0.01 <= hits / 200 <= 0.10


def _family():
    # 4 languages x 5 concepts; colexification instances (language, i, j)
    att = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 1, 1, 1, 0], [0, 0, 1, 1, 1]], dtype=bool)
    inst = [(0, 0, 1), (1, 0, 1), (2, 2, 3), (3, 2, 3), (2, 0, 1), (3, 3, 4)]
    lang, i, j = (np.array(v) for v in zip(*inst))
    return h3.FamilyData("fam", ["l0", "l1", "l2", "l3"], att, lang, i, j)


def test_family_subsample_edges_and_attestation():
    fd = _family()
    att, edge = fd.subsample(np.arange(4), 5, 2)
    assert att.all()
    assert edge[0, 1] and edge[1, 0] and edge[2, 3]          # 0-1 in 3 languages, 2-3 in 2
    assert not edge[3, 4]                                     # only one language
    att2, edge2 = fd.subsample(np.array([0, 3]), 5, 2)
    assert not edge2.any() and att2.tolist() == [True, True, True, True, True]
    att3, _ = fd.subsample(np.array([0]), 5, 2)
    assert att3.tolist() == [True, True, True, False, False]


def test_rarefied_auc_with_all_languages_equals_full():
    fd = _family()
    sim = np.random.default_rng(0).normal(size=(5, 5))
    sim = sim + sim.T
    r = h3.rarefied_auc(fd, {"m": sim}, 4, 3, np.random.default_rng(0))
    iu = np.triu_indices(5, 1)
    _, edge = fd.subsample(np.arange(4), 5, 2)
    assert (r["auc"] == h3.auc(sim[iu], edge[iu])).all() and len(r) == 3


def test_permutation_spearman_exact_and_mc():
    x = np.arange(6.0)
    r = h3.permutation_spearman(x, x * 2, np.random.default_rng(0), exact_max_n=9)
    assert r["rho"] == pytest.approx(1.0) and r["p_one_sided"] == pytest.approx(1 / 720)
    g = np.random.default_rng(2)
    a, b = g.normal(size=12), g.normal(size=12)
    mc = h3.permutation_spearman(a, b, g, exact_max_n=9, n_mc=20000)
    assert mc["rho"] == pytest.approx(ss.spearmanr(a, b).statistic)
    assert abs(mc["p_two_sided"] - ss.spearmanr(a, b).pvalue) < 0.05
    assert np.isnan(h3.permutation_spearman(x[:3], x[:3], g)["rho"])


def test_gradient_table_pending_and_done():
    fam = [f"f{k}" for k in range(6)]
    fit = pd.DataFrame({"family_gc": fam, "condition": "m", "auc": np.linspace(.6, .8, 6),
                        "auc_rarefied_mean": np.linspace(.6, .8, 6), "n_languages": [20, 30, 40, 50, 60, 70]})
    prox = pd.DataFrame({"family_gc": fam, "joshi_class": [0, 1, 2, 3, 4, 5], "n_languages": 1})
    prox.attrs["joshi_class_status"] = "done"
    out = h3.gradient_table(fit, prox, ["m"], np.random.default_rng(0))
    j = out[(out["proxy"] == "joshi_class") & (out["auc_kind"] == "rarefied")].iloc[0]
    assert j["rho"] == pytest.approx(1.0) and j["p_one_sided"] == pytest.approx(1 / 720)
    w = out[out["proxy"] == "wikipedia_log10_articles"]
    assert (w["status"] == "PENDING USER RUN").all() and not out["h3_supported"].any()


def test_language_agreement_toy():
    units = pd.DataFrame({"variety_id": ["v1", "v2"], "language_gc": ["a", "b"]})
    forms_u = pd.DataFrame({"variety_id": ["v1"] * 4 + ["v2"] * 3, "concept_id": list("WXYZ") + list("WXY"),
                            "language_gc": ["a"] * 4 + ["b"] * 3})
    inst = pd.DataFrame({"variety_id": ["v1"], "concept_a": ["W"], "concept_b": ["X"]})
    sim = np.array([[1, .9, .1, .2], [.9, 1, .3, .1], [.1, .3, 1, .5], [.2, .1, .5, 1]])
    out = h3.language_agreement({"units": units, "forms_u": forms_u, "inst": inst}, list("WXYZ"), {"m": sim})
    a = out.set_index("language_gc")
    assert a.loc["a", "n_pairs"] == 6 and a.loc["a", "n_colex_pairs"] == 1 and a.loc["a", "auc_m"] == 1.0
    assert a.loc["b", "n_pairs"] == 3 and np.isnan(a.loc["b", "auc_m"])


# ------------------------------------------------------------ resource proxies --
def _langs():
    return pd.DataFrame({"glottocode": ["fam1", "aaa1", "bbb1", "sub1", "fam2", "ccc1"],
                         "name": ["F1", "Alpha", "Beta", "Sub", "F2", "Gamma"],
                         "level": ["family", "language", "language", "family", "family", "language"],
                         "iso639_3": ["", "aaa", "bbb", "", "", "ccc"],
                         "family_gc": ["fam1", "fam1", "fam1", "fam1", "fam2", "fam2"]})


def test_parse_lang2tax_and_family_joshi():
    t = rp.parse_lang2tax("# header\nAlpha,5\nBeta\t2\nGamma,9\n\n")
    assert t["language_paper"].tolist() == ["Alpha", "Beta"] and t["class_low"].tolist() == [5, 2]
    t["family_gc"] = ["fam1", "fam1"]
    lb = rp.family_joshi(t, ["fam1", "fam2"], full_table=False).set_index("family_gc")
    assert lb.loc["fam1", "joshi_class"] == 5 and np.isnan(lb.loc["fam2", "joshi_class"])
    full = rp.family_joshi(t, ["fam1", "fam2"], full_table=True).set_index("family_gc")
    assert full.loc["fam2", "joshi_class"] == 0


def test_verified_joshi_table_maps_to_glottolog():
    import config
    if not (config.GLOTTOLOG_DIR / "cldf" / "languages.csv").exists():
        pytest.skip("Glottolog not fetched")
    t, status = rp.joshi_languages(rp.glottolog_languoids())
    assert t["family_gc"].notna().all() and len(t) == 37
    fam = t.set_index("language_paper")["family_gc"]
    assert fam["English"] == fam["Hindi"] == "indo1319" and fam["Swahili"] == "atla1278"


def test_edition_families_and_family_wikipedia():
    editions = pd.DataFrame({"code": ["a", "b", "zz", "simple", "x", "old"],
                             "articles": [100, 5000, 70, 900, 10, 999999],
                             "closed": [False, False, False, False, False, True]})
    wd = pd.DataFrame({"item": list("123456"), "wm_code": ["a", "b", "zz", "x", "x", "old"],
                       "glottocode": ["aaa1", "", "sub1", "aaa1", "ccc1", "ccc1"],
                       "iso639_3": ["", "bbb", "", "", "", ""]})
    e = rp.edition_families(editions, wd, _langs()).set_index("code")
    assert "simple" not in e.index and "old" not in e.index
    assert e.loc["a", "family_gc"] == "fam1"                 # via Glottocode
    assert e.loc["b", "family_gc"] == "fam1"                 # via ISO fallback
    assert e.loc["zz", "family_gc"] == "fam1"                # subgroup-level Glottocode -> family
    assert e.loc["x", "mapping"] == "ambiguous"
    w = rp.family_wikipedia(e.reset_index(), ["fam1", "fam2"]).set_index("family_gc")
    assert w.loc["fam1", "wikipedia_max_articles"] == 5000 and w.loc["fam1", "wikipedia_top_edition"] == "b"
    assert w.loc["fam2", "wikipedia_max_articles"] == 0
    assert w.loc["fam1", "wikipedia_log10_articles"] == pytest.approx(np.log10(5001))


# ------------------------------------------------------------------ dashboard --
def test_dashboard_pair_report_and_neighbours():
    from src.viz import dashboard as db
    order = ["A", "B", "C", "D"]
    sim = np.array([[1, .9, .1, .2], [.9, 1, .3, .1], [.1, .3, 1, .5], [.2, .1, .5, 1]])
    pairs = pd.DataFrame({"concept_a": ["A"], "concept_b": ["B"], "n_families": [7], "n_languages": [30],
                          "cov_families": [50]})
    audit = pd.DataFrame({"concept_a": ["A"], "concept_b": ["B"], "suggested_decision": ["present"], "decision": [""]})
    rep = db.pair_report("B", "A", order, {"w2v_gnews": sim}, pairs, audit)
    assert rep["n_families"] == 7 and rep["english"].startswith("present (suggested")
    assert rep["models"]["mutual_rank"].iloc[0] == 0.0
    assert db.neighbours(sim, order, {"B": "b"}, "A", k=2) == ["b", "D"]


def test_equal_earth_projection_reference_values():
    """Equal Earth: equator/centre at origin; x at (180°, 0°) = 2√3·π/(3·A1)."""
    from src.viz import world_map as wm
    x, y = wm.equal_earth(np.array([wm.LON0, wm.LON0 + 180.0 - 1e-9]), np.array([0.0, 0.0]))
    assert x[0] == pytest.approx(0) and y[0] == pytest.approx(0)
    assert abs(x[1]) == pytest.approx(2 * np.sqrt(3) * np.pi / (3 * wm.A1), rel=1e-6)
    _, ypole = wm.equal_earth(np.array([wm.LON0]), np.array([90.0]))
    th = np.arcsin(np.sqrt(3) / 2)
    assert ypole[0] == pytest.approx(th * (wm.A4 * th ** 8 + wm.A3 * th ** 6 + wm.A2 * th ** 2 + wm.A1))
