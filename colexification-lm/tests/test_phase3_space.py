"""Phase 3 tests: community detection, stability statistics, freezing, MDS, map quality, HTML."""
from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import config
from src.network import communities as cm
from src.network import typological_space as ts


# ------------------------------------------------------------ communities --
@pytest.fixture
def two_cliques():
    """Two 5-cliques joined by one weak edge: an unambiguous 2-community graph."""
    g = nx.Graph()
    a, b = [f"A{i}" for i in range(5)], [f"B{i}" for i in range(5)]
    for grp in (a, b):
        for i in range(5):
            for j in range(i + 1, 5):
                g.add_edge(grp[i], grp[j], n_families=10.0)
    g.add_edge("A0", "B0", n_families=1.0)
    g.add_node("ISO")                                    # an isolated concept
    return g, sorted(g.nodes)


def test_louvain_recovers_planted_communities(two_cliques):
    g, order = two_cliques
    lab = cm.louvain_labels(g, order, seed=1)
    d = dict(zip(order, lab))
    assert len({d[f"A{i}"] for i in range(5)}) == 1 and len({d[f"B{i}"] for i in range(5)}) == 1
    assert d["A0"] != d["B0"] and d["ISO"] not in (d["A0"], d["B0"])
    np.testing.assert_array_equal(lab, cm.louvain_labels(g, order, seed=1))   # seeded = reproducible


def test_stability_helpers():
    p = np.array([[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1]])
    st = cm.pairwise_agreement(p)
    assert st["ari"][0] == pytest.approx(1.0)            # label permutation = same partition
    assert len(st["ari"]) == 3
    assert cm.medoid_run(p) in (0, 1)
    co = cm.coassignment(p)
    assert co[0, 1] == pytest.approx(2 / 3) and np.allclose(np.diag(co), 1)
    np.testing.assert_array_equal(cm.relabel_by_size(np.array([5, 5, 2, 7, 7, 7])), [1, 1, 2, 0, 0, 0])
    assert cm.describe(np.array([0.5, 1.0]))["median"] == pytest.approx(0.75)


def test_freeze_partition_refuses_silent_change(isolated_paths, monkeypatch):
    monkeypatch.setattr(cm, "REFERENCE_PATH", isolated_paths / "ref.json")
    order = ["A", "B", "C"]
    d1 = cm.freeze_partition(order, np.array([0, 0, 1]), {"t": 1})
    assert cm.freeze_partition(order, np.array([0, 0, 1]), {"t": 1}) == d1     # idempotent
    with pytest.raises(RuntimeError, match="different frozen partition"):
        cm.freeze_partition(order, np.array([0, 1, 1]), {"t": 1})
    np.testing.assert_array_equal(cm.load_reference_partition(order), [0, 0, 1])
    with pytest.raises(RuntimeError, match="different concept order"):
        cm.load_reference_partition(["A", "B", "D"])


# -------------------------------------------------------------------- MDS --
@pytest.fixture
def planar():
    g = np.random.default_rng(3)
    x = g.normal(size=(40, 2))
    d = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
    return x, d


def test_classical_mds_exact_for_euclidean(planar):
    x, d = planar
    y, ev = ts.classical_mds(d)
    dy = ts._pdist(y)
    np.testing.assert_allclose(dy, d, atol=1e-8)
    assert ev.sum() == pytest.approx(1.0)


def test_smacof_metric_and_nonmetric_recover_configuration(planar):
    x, d = planar
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=x.shape)
    y, stress, it = ts.smacof(d, x0, metric=True)
    assert stress < 1e-4 and it < 5000
    q = ts.quality(d, y)
    assert q["stress1"] < 0.02 and q["trustworthiness"] > 0.99
    y2, stress2, _ = ts.smacof(d ** 3, ts.classical_mds(d)[0], metric=False)   # monotone transform
    assert ts.quality(d, y2)["shepard_spearman"] > 0.99


def test_quality_and_choice_rule():
    q = pd.DataFrame({"method": ["a", "b", "c", "d"], "status": ["done", "done", "done", "PENDING USER RUN"],
                      "note": ["", "", "DEGENERATE: x", ""], "trustworthiness": [0.7, 0.8, 0.99, np.nan],
                      "stress1": [0.3, 0.3, 0.1, np.nan]})
    assert ts.choose(q) == "b"                           # degenerate and pending maps never chosen


# --------------------------------------------------------------- figures --
def test_label_deoverlap_separates():
    from src.viz.figures import _deoverlap
    pos = _deoverlap({"a": (0, 0), "b": (0.01, 0.0), "c": (5, 5)}, min_dy=1.0, min_dx=1.0)
    assert abs(pos["a"][1] - pos["b"][1]) >= 1.0 - 1e-6
    assert tuple(pos["c"]) == (5, 5)


def test_interactive_render_is_safe():
    from src.viz.interactive_network import render
    html = render({"nodes": [{"id": "A", "label": "</script><b>"}], "edges": [], "legend": [],
                   "n_nodes": 1, "n_edges": 0, "min_families": 2, "max_families": 2})
    assert html.count("</script>") == 2                 # library tag + page script; injected one escaped
    assert "<\\/script>" in html and "__DATA__" not in html


@pytest.mark.slow
def test_real_reference_partition_frozen():
    from src.utils.concept_order import load_concept_order
    order = load_concept_order()
    ref = cm.load_reference_partition(order)
    assert len(ref) == len(order)
    stab = json.loads((config.RESULTS_DIR / "community_stability.json").read_text())
    assert stab["n_runs"] == config.COMMUNITY_N_RUNS
