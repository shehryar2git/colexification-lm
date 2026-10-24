"""Phase 6 tests: H2 ranks, matching, balance, audit status, interaction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import h2_blindspot as h2


def test_mutual_rank_exact():
    s = np.array([[1.0, 0.9, 0.5, 0.1],
                  [0.9, 1.0, 0.2, 0.3],
                  [0.5, 0.2, 1.0, 0.8],
                  [0.1, 0.3, 0.8, 1.0]])
    # pair (0,1): each is the other's nearest -> 0
    assert h2.mutual_rank(s, np.array([0]), np.array([1]))[0] == 0.0
    # pair (0,3): from 0, concepts 1 and 2 are closer -> 2/2; from 3, concepts 1 and 2 are closer -> 2/2
    assert h2.mutual_rank(s, np.array([0]), np.array([3]))[0] == pytest.approx(1.0)
    # pair (2,3): each is the other's nearest neighbour -> 0
    assert h2.mutual_rank(s, np.array([2]), np.array([3]))[0] == pytest.approx(0.0)
    t = s.copy(); t[0, 2] = t[2, 0] = 0.9            # tie with the pair's own similarity counts 1/2
    assert h2.mutual_rank(t, np.array([0]), np.array([1]))[0] == pytest.approx((0.25 + 0.0) / 2)


def _pop(n_t=10, n_c=200, seed=0):
    g = np.random.default_rng(seed)
    t = pd.DataFrame({"group": "present", "log_families": g.normal(2.5, .5, n_t), "zipf_min": g.normal(5, .5, n_t),
                      "same_field": g.integers(0, 2, n_t)})
    c = pd.DataFrame({"group": "absent", "log_families": g.normal(1.0, .8, n_c), "zipf_min": g.normal(4.5, .7, n_c),
                      "same_field": g.integers(0, 2, n_c)})
    return pd.concat([t, c], ignore_index=True)


def test_matching_is_exact_within_caliper_and_without_replacement():
    pop = _pop()
    m = h2.match(pop, k=3, caliper_sd=0.2, rng=np.random.default_rng(1))
    ctrl = m[m["role"] == "absent"]
    assert ctrl["idx"].is_unique                                       # no control reused
    for mid, grp in m[m["role"].isin(["present", "absent"])].groupby("match_id"):
        assert grp["same_field"].nunique() == 1                        # exact on field
        t = grp[grp["role"] == "present"]["logit"].iloc[0]
        assert (grp.loc[grp["role"] == "absent", "logit"] - t).abs().max() <= m.attrs["caliper"] + 1e-12
        assert (grp["role"] == "absent").sum() <= 3
    m2 = h2.match(pop, k=3, caliper_sd=0.2, rng=np.random.default_rng(1))
    pd.testing.assert_frame_equal(m.reset_index(drop=True), m2.reset_index(drop=True))   # seeded = reproducible


def test_matching_improves_balance():
    pop = _pop(n_t=15, n_c=400, seed=2)
    m = h2.match(pop, k=3, caliper_sd=0.2, rng=np.random.default_rng(0))
    bal = h2.balance_table(pop.assign(zipf_max=pop["zipf_min"] + 1), m.assign(zipf_max=m["zipf_min"] + 1), "toy")
    lf = bal.set_index("covariate").loc["log_families"]
    assert abs(lf["smd_after"]) < abs(lf["smd_before"]) and abs(lf["smd_before"]) > 1


def test_audit_status(tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame({"concept_a": ["A", "B"], "concept_b": ["C", "D"], "suggested_decision": ["present", "absent"],
                  "decision": ["", ""]}).to_csv(p, index=False)
    a, status = h2.audit_decisions(p)
    assert status.startswith("PROVISIONAL") and set(a["final_source"]) == {"claude_suggestion"}
    pd.DataFrame({"concept_a": ["A", "B"], "concept_b": ["C", "D"], "suggested_decision": ["present", "absent"],
                  "decision": ["absent", ""]}).to_csv(p, index=False)
    a, status = h2.audit_decisions(p)
    assert status.startswith("PARTIAL") and a.set_index("concept_a").loc["A", "final"] == "absent"


def test_group_test_direction():
    g = np.random.default_rng(3)
    present, absent = g.uniform(0, 0.2, 40), g.uniform(0.1, 0.6, 120)
    r = h2.group_test(present, absent, g)
    assert r["cliffs_delta"] > 0.5 and r["p_one_sided"] < 0.001 and r["delta_ci_low"] > 0


def test_interaction_gap_and_contrast():
    g = np.random.default_rng(4)
    rows = []
    for m in range(30):                                     # 30 matched sets: 1 present + 2 absent
        for role, n in (("present", 1), ("absent", 2)):
            for j in range(n):
                pair = f"{m}-{role}-{j}"
                for cond, gap in (("w2v_gnews", 0.30), ("labse", 0.05)):
                    base = 0.1 if role == "present" else 0.1 + gap
                    rows.append({"condition": cond, "pair": pair, "concept_a": "x", "concept_b": "y",
                                 "group": role, "rank": base + g.normal(0, 0.02), "match_id": m,
                                 "weight": 1.0 if role == "present" else 0.5})
    out = h2.interaction(pd.DataFrame(rows), 300, np.random.default_rng(0)).set_index("condition")
    assert out.loc["w2v_gnews", "gap"] == pytest.approx(0.30, abs=0.02)
    assert out.loc["labse", "gap_minus_w2v"] == pytest.approx(-0.25, abs=0.02)
    assert out.loc["labse", "contrast_ci_high"] < 0 and out.loc["labse", "p_contrast_two_sided"] < 0.01


def test_logistic_mle_recovers_coefficients():
    g = np.random.default_rng(5)
    X = g.normal(size=(20000, 2))
    eta = -1.0 + 1.5 * X[:, 0] - 0.5 * X[:, 1]
    y = (g.uniform(size=len(eta)) < 1 / (1 + np.exp(-eta))).astype(float)
    np.testing.assert_allclose(h2.logistic_mle(X, y), [-1.0, 1.5, -0.5], atol=0.06)
