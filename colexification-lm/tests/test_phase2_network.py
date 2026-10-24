"""Phase 2 tests: colexification counting, coverage, evidence filter, matrices, validation."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import config
from src.network import build_matrices as bm
from src.network import build_network as bn

from tests.conftest import (CONCEPTS, EXPECTED_EDGES, EXPECTED_EDGES_MIN2,
                            EXPECTED_PAIR_COVERAGE_FAMILIES, FORMS, LANGUAGES)


@pytest.fixture
def tidy_forms():
    f = FORMS.rename(columns={"Language_ID": "variety_id", "Concept": "concept_id", "Form": "form"}).copy()
    f["segments"] = f["form"].map(lambda s: " ".join(s))
    f["local_id"] = [f"e{i}" for i in range(len(f))]
    f["shared_entry"] = False
    return f


@pytest.fixture
def tidy_languages():
    lg = LANGUAGES.rename(columns={"Language_ID": "variety_id"}).copy()
    lg["family_gc"] = lg["Family"].replace("", np.nan).fillna(lg["Glottocode"])   # isolate = own family
    lg["family_name"] = lg["Family_Name"].replace("", "isolate")
    lg["language_gc"] = lg["Glottocode"]
    lg["variety_name"] = lg["variety_id"]
    lg["include"] = True
    lg["contribution_id"] = "toy"
    return lg


def _counts(forms, languages):
    inst = bn.colex_instances(forms)
    units = languages[["variety_id", "variety_name", "language_gc", "family_gc", "family_name"]]
    return bn.aggregate_instances(inst, units), inst, units


def test_counts_match_ground_truth(tidy_forms, tidy_languages):
    counts, _, _ = _counts(tidy_forms, tidy_languages)
    got = {(r.concept_a, r.concept_b): {"languages": r.n_languages, "families": r.n_families}
           for r in counts.itertuples()}
    assert got == EXPECTED_EDGES


def test_identity_segments_vs_form(tidy_forms):
    f = tidy_forms.copy()
    # tone mark differs in Form but Segments are equal -> colexified only under 'segments'
    f.loc[(f.variety_id == "A3") & (f.concept_id == "TOMORROW"), ["form", "segments"]] = ["kàl", "k a l"]
    seg = bn.colex_instances(f, "segments")
    frm = bn.colex_instances(f, "form")
    key = (seg.concept_a == "TOMORROW") & (seg.concept_b == "YESTERDAY")
    assert key.any()
    assert not ((frm.concept_a == "TOMORROW") & (frm.concept_b == "YESTERDAY")).any()


def test_empty_keys_never_match(tidy_forms):
    f = tidy_forms.copy()
    f.loc[f.variety_id == "A3", "segments"] = ""
    inst = bn.colex_instances(f, "segments")
    assert not (inst.variety_id == "A3").any()


def test_evidence_filter_drops_split_entries(tidy_forms, tidy_languages):
    f = tidy_forms.copy()
    # mark C1's HAND/ARM as one source entry split into two concepts
    f.loc[(f.variety_id == "C1") & f.concept_id.isin(["HAND", "ARM"]), "shared_entry"] = True
    order = sorted(CONCEPTS)
    strict = bn.network_for(f, tidy_languages, order, "separate_entries", "segments")["pairs"]
    loose = bn.network_for(f, tidy_languages, order, "all", "segments")["pairs"]
    get = lambda p: p.set_index(["concept_a", "concept_b"]).loc[("ARM", "HAND")]   # noqa: E731
    assert get(loose)["n_families"] == 3 and get(strict)["n_families"] == 2
    assert get(strict)["cov_families"] == 2      # C1 no longer attests the pair either
    with pytest.raises(ValueError):
        bn.evidence_filter(f, "nonsense")


def test_coverage_and_pair_table(tidy_forms, tidy_languages):
    order = sorted(CONCEPTS)
    pairs = bn.network_for(tidy_forms, tidy_languages, order)["pairs"].set_index(["concept_a", "concept_b"])
    assert len(pairs) == len(order) * (len(order) - 1) // 2
    for k, cov in EXPECTED_PAIR_COVERAGE_FAMILIES.items():
        assert pairs.loc[k, "cov_families"] == cov
    assert pairs.loc[("ARM", "HAND"), "rate_families"] == pytest.approx(1.0)
    assert pairs.loc[("TREE", "WOOD"), "rate_families"] == pytest.approx(2 / 3)
    # A2 lacks TREE: TREE~WOOD covered by 5 of 6 languages, colexified in 3
    assert pairs.loc[("TREE", "WOOD"), "cov_languages"] == 5
    assert set(pairs.index[pairs["is_edge"]]) == EXPECTED_EDGES_MIN2
    assert (pairs["n_families"] <= pairs["cov_families"]).all()


def test_pair_table_guards_impossible_counts():
    counts = pd.DataFrame({"concept_a": ["A"], "concept_b": ["B"], "n_varieties": [1],
                           "n_languages": [1], "n_families": [1]})
    with pytest.raises(AssertionError, match="exceeds coverage"):
        bn.pair_table(counts, ["A", "B"], np.zeros((2, 2), int), np.zeros((2, 2), int))


def test_graph_and_examples(tidy_forms, tidy_languages):
    order = sorted(CONCEPTS)
    net = bn.network_for(tidy_forms, tidy_languages, order)
    concepts = pd.DataFrame({"concept_id": order, "gloss": [c.lower() for c in order],
                             "semantic_field": "x", "n_families": 3})
    g = bn.build_graph(net["pairs"], concepts)
    assert set(g.nodes) == set(order)
    assert {tuple(sorted(e)) for e in g.edges} == EXPECTED_EDGES_MIN2
    assert g.edges["ARM", "HAND"]["n_families"] == 3
    ex = bn.example_forms(net["inst"], net["units"]).set_index(["concept_a", "concept_b"])
    assert ex.loc[("ARM", "HAND"), "examples"].count("|") == 2   # one example per family
    assert "C1 (isolate): mara" in ex.loc[("ARM", "HAND"), "examples"]


def test_validation_against_shipped_table(tidy_forms, tidy_languages, tmp_path):
    v = tidy_languages.assign(glottocode=tidy_languages["Glottocode"], clics_family=tidy_languages["family_gc"])
    shipped = pd.DataFrame({"Source_Concept": ["HAND", "WOOD"], "Target_Concept": ["ARM", "TREE"],
                            "Variety_Count": [4, 3], "Language_Count": [4, 3], "Family_Count": [3, 2]})
    path = tmp_path / "colex.csv"
    shipped.to_csv(path, index=False)
    res = bn.validate_against_clics(bn.colex_instances(tidy_forms), v, path)
    assert res["share_family_count_equal"] == 1.0
    assert res["n_pairs_only_recount"] == 2          # the two single-family pairs absent from 'shipped'


def test_family_tables(tidy_forms, tidy_languages, monkeypatch):
    monkeypatch.setattr(config, "FAMILY_MIN_LANGUAGES", 3)
    monkeypatch.setattr(config, "FAMILY_MIN_CONCEPT_COVERAGE", 0.5)
    order = sorted(CONCEPTS)
    net = bn.network_for(tidy_forms, tidy_languages, order)
    fam, fp = bn.family_tables(net["inst"], net["forms_u"], net["units"], order)
    fam = fam.set_index("family_name")
    assert fam.loc["A", "qualifies"] and not fam.loc["B", "qualifies"]
    a = fp.set_index(["concept_a", "concept_b"])
    assert a.loc[("ARM", "HAND"), "n_languages"] == 2 and a.loc[("ARM", "HAND"), "is_edge"]
    assert a.loc[("SEE", "UNDERSTAND"), "is_edge"]               # 2 languages inside family A
    assert not a.loc[("TOMORROW", "YESTERDAY"), "is_edge"]      # 1 language only
    assert bn.slug("Nuclear Trans New Guinea") == "nuclear-trans-new-guinea"


# ------------------------------------------------------------- matrices ----
def test_profile_cosine_distance_properties():
    r = np.array([[1, .5, 0], [.5, 1, 0], [0, 0, 1.]])
    d = bm.profile_cosine_distance(r)
    assert np.allclose(np.diag(d), 0) and np.allclose(d, d.T)
    assert (d >= 0).all() and (d <= 1).all()
    assert d[0, 2] == pytest.approx(1.0) and d[0, 1] < 0.5
    dv = bm.profile_cosine_distance(r, valid=np.array([True, True, False]))
    assert np.isnan(dv[2]).all() and np.isnan(dv[:, 2]).all() and not np.isnan(dv[0, 1])


def test_shortest_path_known_values():
    order = ["A", "B", "C", "D"]
    pairs = pd.DataFrame({"concept_a": ["A", "B"], "concept_b": ["B", "C"],
                          "n_families": [4, 2], "is_edge": [True, True]})
    d, disc = bm.shortest_path_distance(pairs, order)
    assert d[0, 1] == pytest.approx(0.25) and d[1, 2] == pytest.approx(0.5)
    assert d[0, 2] == pytest.approx(0.75)                       # path A-B-C
    assert d[0, 3] == pytest.approx(0.75 + 1.0)                 # disconnected = max + 1
    assert disc == pytest.approx(3 / 6)
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0)


def test_rate_matrix_and_square():
    order = ["A", "B", "C"]
    pairs = pd.DataFrame({"concept_a": ["A", "A"], "concept_b": ["B", "C"], "rate_families": [0.5, 0.2],
                          "is_edge": [True, False]})
    r = bm.rate_matrix(pairs, order)
    assert r[0, 1] == r[1, 0] == 0.5 and r[0, 2] == 0.0 and np.allclose(np.diag(r), 1)


def test_family_distance_nan_for_unattested():
    order = ["A", "B", "C"]
    fp = pd.DataFrame({"concept_a": ["A"], "concept_b": ["B"], "rate_languages": [0.4], "is_edge": [True]})
    d = bm.family_distance(fp, order, {"A", "B"})
    assert d[0, 1] == pytest.approx(0.6) and np.isnan(d[2]).all() and d[0, 0] == 0


# ------------------------------------------------ real data (slow, cached) --
@pytest.mark.slow
def test_real_network_invariants():
    import json
    from src.utils.concept_order import load_concept_order
    from src.utils.matrices import load_matrix
    v = json.loads((config.RESULTS_DIR / "clics_validation.json").read_text())
    assert v["share_family_count_equal"] == 1.0 and v["n_pairs_only_clics"] == 0
    order = load_concept_order()
    m = load_matrix(config.MATRIX_DIR / "typological_distance.npz", order)
    assert m.kind == "distance" and m.ids == order and np.isfinite(m.values).all()
    g = nx.read_graphml(config.PROCESSED_DIR / "network.graphml")
    assert set(g.nodes) == set(order)
