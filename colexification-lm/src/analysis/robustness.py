"""Phase 8 · Robustness: does each conclusion survive the alternatives? (Proposal Step 8)

Every alternative listed in DECISIONS.md is re-run here and the result is reported as
*conclusion survives* or *conclusion does not survive*, where the conclusion is the
pre-specified verdict of the primary analysis (H1: rho > 0 with one-sided Mantel
p < alpha; H2: Cliff's delta > 0 with one-sided p < alpha), not the size of the number.

Checks (the proposal's (i)-(v), plus every alternative recorded in DECISIONS.md):

(i)   edge rule / weighting: >= 1, >= 2 (primary), >= 3 families; binary profiles (D-07)
(ii)  alternative distance metrics: first-order rate, profile cosine, binary profile,
      language-weighted profile (D-08, D-39)
(iii) the shortest-path metric is the primary (D-39); (ii) supplies the alternatives
(iv)  concept-set threshold t = 100 instead of the knee t = 85; no anchor additions
      (D-05, D-05b); connected component only (D-42); per-model maximal set (D-16)
(v)   leave-one-family-out over the 16 qualifying families; and (v-b) the CLICS3
      published network instead of CLICS4 (D-01, D-60)
plus:  colexification evidence (all entries vs separate entries, D-37), identity on the
      exact form instead of segments (D-03), without the most family-diverse dataset
      (D-38), uncentred cosines (D-47), and for H2 the audit treatment of *unsure* pairs,
      1:1 matching, a tighter caliper, raw cosine instead of neighbour rank, and the
      human-audited-only gold standard when decisions exist.

Output: results/robustness.csv (one row per check x condition) and
results/robustness_summary.csv (one row per check, with the verdict).
"""
from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
from scipy import stats as ss

import config
from src.analysis.common import (CONDITION_LABELS, analysis_concepts, available_conditions,
                                 load_similarity, stream, typological_distance)
from src.analysis.h2_blindspot import (audit_decisions, build_population, cliffs_delta, group_test,
                                       match, mutual_rank)
from src.network.build_matrices import shortest_path_distance, with_edge_rule
from src.network.build_network import network_for
from src.network.typological_space import connected_mask
from src.utils.concept_order import load_concept_order
from src.utils.io import read_parquet, write_result_table
from src.utils.logging_utils import get_logger, log_run_header
from src.utils.matrices import load_matrix
from src.utils.seeds import set_all_seeds
from src.utils.stats import mantel, rsa_bootstrap_ci

log = get_logger(__name__, "robustness")
NPERM = config.N_PERMUTATIONS_ROBUSTNESS
NBOOT = config.N_BOOTSTRAP_ROBUSTNESS
TRANSLATED = ("xlmr_translated", "labse_translated")

# Checks that cannot run without the transformer stack; listed, never silently skipped.
PENDING_CHECKS = [
    ("h1_template_set", "H1", "Template sensitivity: each of the 10 templates alone, and leave-one-template-out "
     "(D-13). Needs the contextual models.", "python run.py embeddings  # then: python -m src.analysis.robustness"),
    ("h1_layer_sweep", "H1", "Layer sweep: hidden layers -1, -2, -4 and the mean of the last four (D-13). "
     "Needs the contextual models.", "python run.py embeddings  # then: python -m src.analysis.robustness"),
]


# The proposal's robustness axes (Step 8 (i)-(v)); every check is assigned to one.
AXES: dict[str, str] = {
    "i": "binary vs family-weighted colexification",
    "ii": "alternative distance metrics",
    "iii": "shortest-path typological distance",
    "iv": "a different concept-set threshold",
    "v": "leave-one-family-out",
    "extra": "further alternatives recorded in DECISIONS.md",
}
AXIS_OF: dict[str, str] = {
    "typ_minfam1": "i", "typ_minfam3": "i", "typ_binary_profile": "i",
    "h2_minfam1": "i", "h2_minfam3": "i",
    "typ_profile_cosine": "ii", "typ_language_profile": "ii", "typ_rate": "iii",
    "concepts_threshold_100": "iv", "concepts_no_anchor_additions": "iv",
    "concepts_connected_only": "iv", "concepts_model_maximal": "iv", "h2_threshold_100": "iv",
}


def axis_of(check: str) -> str:
    """Which of the proposal's robustness axes a check belongs to."""
    if check.startswith("lofo_") or check.startswith("h2_lofo_"):
        return "v"
    return AXIS_OF.get(check, "extra")


# ------------------------------------------------------------------ helpers --
def model_distance(cond: str, order: list[str], raw: bool = False) -> np.ndarray:
    if not raw:
        return 1.0 - load_similarity(cond, order)
    return 1.0 - load_matrix(config.MATRIX_DIR / f"model_{cond}__raw.npz", order).values


def finite_subset(*mats: np.ndarray) -> np.ndarray:
    """Largest set of concepts whose mutual distances are all finite in every matrix.
    Concepts are dropped greedily, most-missing first, so that one all-NaN row (an
    unattested concept) does not eliminate every concept through its column."""
    bad = np.zeros(mats[0].shape, dtype=bool)
    for m in mats:
        bad |= ~np.isfinite(m)
    np.fill_diagonal(bad, False)
    keep = np.ones(bad.shape[0], dtype=bool)
    while True:
        idx = np.flatnonzero(keep)
        counts = bad[np.ix_(idx, idx)].sum(axis=1)
        if counts.size == 0 or counts.max() == 0:
            return keep
        keep[idx[int(counts.argmax())]] = False


def rsa(d_typ: np.ndarray, d_mod: np.ndarray, seed_parts: tuple[str, ...]) -> dict:
    """RSA with a one-sided Mantel test and a concept-bootstrap CI."""
    ok = finite_subset(d_typ, d_mod)
    if ok.sum() < 10:
        return {"n_concepts": int(ok.sum()), "value": np.nan, "p": np.nan}
    a, b = d_typ[np.ix_(ok, ok)], d_mod[np.ix_(ok, ok)]
    mt = mantel(a, b, NPERM, stream(*seed_parts, "mantel"))
    lo, hi = rsa_bootstrap_ci(a, b, NBOOT, stream(*seed_parts, "boot"))
    n = int(ok.sum())
    return {"n_concepts": n, "n_pairs": n * (n - 1) // 2, "value": mt.r, "ci_low": lo, "ci_high": hi,
            "p": mt.p_greater}


def conclusion_of(value: float, p: float) -> str:
    """The verdict one variant on its own would support."""
    if not np.isfinite(value) or not np.isfinite(p):
        return "not computable"
    return "supported" if (value > 0 and p < config.ALPHA) else "not supported"


def verdict(value: float, p: float, primary_supported: bool) -> str:
    """A check survives when it reproduces the primary analysis's conclusion."""
    c = conclusion_of(value, p)
    if c == "not computable":
        return c
    return "survives" if (c == "supported") == primary_supported else "does not survive"


# --------------------------------------------------------- CLICS3 network ---
def parse_clics3_gml(path=None, member: str | None = None) -> pd.DataFrame:
    """Edge table (concept_a, concept_b, n_families, n_languages) of the CLICS3
    published network. Only the fields needed are parsed: node Gloss, and each edge's
    source/target and FamilyWeight/LanguageWeight (the `words`/`wofam` payloads, which
    are most of the file's 100+ MB, are skipped by a line-prefix test)."""
    path = path or config.CLICS3_GML_ZIP
    member = member or config.CLICS3_GML_MEMBER
    gloss: dict[str, str] = {}
    rows, cur, node_id = [], {}, None
    with zipfile.ZipFile(path) as z, z.open(member) as fh:
        mode = None
        for raw in fh:
            line = raw[:200].decode("utf-8", "replace").strip()
            if line == "node [":
                mode, node_id, cur = "node", None, {}
            elif line == "edge [":
                mode, cur = "edge", {}
            elif line == "]" and mode:
                if mode == "edge" and {"source", "target", "FamilyWeight"} <= cur.keys():
                    rows.append(cur)
                mode = None
            elif mode == "node":
                if line.startswith("id "):
                    node_id = line[3:].strip()
                elif line.startswith("Gloss ") and node_id is not None:
                    gloss[node_id] = line[6:].strip().strip('"')
            elif mode == "edge":
                for key in ("source", "target", "FamilyWeight", "LanguageWeight"):
                    if line.startswith(key + " "):
                        cur[key] = line[len(key):].strip()
    df = pd.DataFrame(rows)
    df["concept_a"] = df["source"].map(gloss)
    df["concept_b"] = df["target"].map(gloss)
    df = df.dropna(subset=["concept_a", "concept_b"])
    lo = np.minimum(df["concept_a"], df["concept_b"])
    hi = np.maximum(df["concept_a"], df["concept_b"])
    out = pd.DataFrame({"concept_a": lo, "concept_b": hi,
                        "n_families": df["FamilyWeight"].astype(int),
                        "n_languages": df["LanguageWeight"].astype(int)})
    out.attrs["n_nodes"] = len(gloss)
    return out.drop_duplicates(["concept_a", "concept_b"]).reset_index(drop=True)


def clics3_distance(order: list[str]) -> tuple[np.ndarray, dict]:
    e = parse_clics3_gml()
    full = pd.MultiIndex.from_arrays(np.triu_indices(len(order), 1)).to_frame(index=False)
    pairs = pd.DataFrame({"concept_a": np.array(order)[full[0]], "concept_b": np.array(order)[full[1]]})
    lo = pairs[["concept_a", "concept_b"]].min(axis=1)
    hi = pairs[["concept_a", "concept_b"]].max(axis=1)
    pairs = pairs.assign(concept_a=lo, concept_b=hi).merge(e, on=["concept_a", "concept_b"], how="left")
    pairs[["n_families", "n_languages"]] = pairs[["n_families", "n_languages"]].fillna(0).astype(int)
    pairs["is_edge"] = pairs["n_families"] > 0
    d, disc = shortest_path_distance(pairs, order)
    info = {"clics3_nodes": int(e.attrs["n_nodes"]), "clics3_edges": int(len(e)),
            "working_concepts_in_clics3": int(pd.Series(order).isin(
                set(e["concept_a"]) | set(e["concept_b"])).sum()),
            "edges_among_working_concepts": int(pairs["is_edge"].sum()),
            "share_pairs_disconnected": float(disc)}
    return d, info


# ------------------------------------------------- leave-one-family-out ---
def lofo_networks(order: list[str], families: pd.DataFrame) -> dict[str, tuple[np.ndarray, pd.DataFrame]]:
    """For each qualifying family: the typological distance and the pair table computed
    with that family's languages removed. Computed once and used by both H1 and H2."""
    forms = read_parquet(config.PROCESSED_DIR / "forms.parquet")
    languages = read_parquet(config.PROCESSED_DIR / "languages.parquet")
    out = {}
    for r in families.itertuples():
        keep = languages.assign(include=languages["include"] & (languages["family_gc"] != r.family_gc))
        pairs = network_for(forms, keep, order)["pairs"]
        d, _ = shortest_path_distance(pairs, order)
        out[r.family_name] = (d, pairs)
        log.info("  leave out %-26s %d edges remain", r.family_name, int(pairs["is_edge"].sum()))
    return out


# --------------------------------------------------------------------- H1 ---
def h1_checks(order: list[str], concepts: list[str], idx: np.ndarray, conds: list[str],
              primary: pd.DataFrame, lofo: dict[str, tuple[np.ndarray, pd.DataFrame]]) -> list[dict]:
    rows: list[dict] = []
    sub = np.ix_(idx, idx)
    d_primary = typological_distance(order)[sub]
    cs = read_parquet(config.PROCESSED_DIR / "concepts.parquet").set_index("concept_id")
    import json
    thr = json.loads((config.RESULTS_DIR / "concept_threshold.json").read_text())

    # concept-set masks over the analysis set (submatrix convention, as in the primary)
    nfam = cs.loc[concepts, "n_families"].to_numpy()
    masks = {
        "concepts_threshold_100": (nfam >= thr["robustness_threshold"],
                                   f"Concept set at the robustness threshold t = {thr['robustness_threshold']} "
                                   "families instead of the Kneedle knee (D-05, check iv)"),
        "concepts_no_anchor_additions": (nfam >= thr["threshold_used"],
                                         "Without the concepts kept only by the anchor rule (D-05b)"),
        "concepts_connected_only": (connected_mask(concepts),
                                    "Only concepts connected in the colexification network; the isolated "
                                    "ones enter the primary through the max + 1 convention (D-42)"),
    }
    typ_variants = {
        "typ_minfam1": ("minfam1", "Edge rule >= 1 family instead of >= 2 (D-07, check i)"),
        "typ_minfam3": ("minfam3", "Edge rule >= 3 families instead of >= 2 (D-07, check i)"),
        "typ_binary_profile": ("binary_profile", "Binary (unweighted) colexification profiles (check i/ii)"),
        "typ_rate": ("rate", "First-order distance 1 - family rate instead of shortest path (D-08, check ii)"),
        "typ_profile_cosine": ("profile_cosine", "Profile cosine on family rates (D-39, check ii)"),
        "typ_language_profile": ("language_profile", "Language-weighted profiles instead of family-weighted "
                                 "(D-04, check ii)"),
        "typ_evidence_all": ("evidence_all", "All CLICS4 entries, including forms whose source entry maps to "
                             "several concepts (D-37)"),
        "typ_identity_form": ("identity_form", "Colexification identity on the exact form string instead of "
                              "segments (D-03)"),
        "typ_no_johansson": ("no_johansson", "Without the single most family-diverse source dataset (D-38)"),
    }
    d_clics3, c3info = clics3_distance(order)
    log.info("CLICS3 network: %(clics3_edges)d edges among %(clics3_nodes)d concepts; "
             "%(edges_among_working_concepts)d edges among our working concepts", c3info)

    for cond in conds:
        pr = primary[primary["condition"] == cond]
        if pr.empty or pr["status"].iloc[0] != "done":
            continue
        p_supported = bool(pr["h1_supported"].iloc[0])
        p_value = float(pr["rho_raw"].iloc[0])
        M = model_distance(cond, order)[sub]
        base = {"hypothesis": "H1", "condition": cond, "label": CONDITION_LABELS[cond],
                "statistic": "Spearman rho (RSA)", "primary_value": p_value,
                "primary_conclusion": "supported" if p_supported else "not supported",
                "verdict_basis": "reproduces the primary analysis"}

        for cid, (name, desc) in typ_variants.items():
            r = rsa(typological_distance(order, name)[sub], M, ("rob", cond, cid))
            rows.append({**base, "check": cid, "description": desc, **r,
                         "verdict": verdict(r["value"], r["p"], p_supported),
                         "variant_conclusion": conclusion_of(r["value"], r["p"])})
        for cid, (mask, desc) in masks.items():
            m = np.ix_(mask, mask)
            r = rsa(d_primary[m], M[m], ("rob", cond, cid))
            rows.append({**base, "check": cid, "description": desc, **r,
                         "verdict": verdict(r["value"], r["p"], p_supported),
                         "variant_conclusion": conclusion_of(r["value"], r["p"])})

        # per-model maximal concept set (D-16): every concept this model can embed
        S_full = load_similarity(cond, order)
        full_ok = finite_subset(S_full)
        mm = np.ix_(full_ok, full_ok)
        r = rsa(typological_distance(order)[mm], (1.0 - S_full)[mm], ("rob", cond, "maximal"))
        rows.append({**base, "check": "concepts_model_maximal", "description":
                     "This model's maximal concept set instead of the six-condition intersection (D-16, check iv)",
                     **r, "verdict": verdict(r["value"], r["p"], p_supported),
                         "variant_conclusion": conclusion_of(r["value"], r["p"])})

        raw_path = config.MATRIX_DIR / f"model_{cond}__raw.npz"
        if raw_path.exists():
            r = rsa(d_primary, model_distance(cond, order, raw=True)[sub], ("rob", cond, "raw"))
            rows.append({**base, "check": "model_uncentred", "description":
                         "Uncentred cosine similarities instead of mean-centred vectors (D-47)",
                         **r, "verdict": verdict(r["value"], r["p"], p_supported),
                             "variant_conclusion": conclusion_of(r["value"], r["p"])})

        r = rsa(d_clics3[sub], M, ("rob", cond, "clics3"))
        rows.append({**base, "check": "data_clics3", "description":
                     "CLICS3's published family-weighted network (>= 3 families) instead of our CLICS4 "
                     "recount (D-01, check v-b); compare with typ_minfam3", **r,
                     "verdict": verdict(r["value"], r["p"], p_supported),
                     "variant_conclusion": conclusion_of(r["value"], r["p"]),
                     "note": "; ".join(f"{k}={v}" for k, v in c3info.items())})

        vals = []
        for name, (d, _) in lofo.items():
            r = rsa(d[sub], M, ("rob", cond, "lofo", name))
            vals.append(r)
            rows.append({**base, "check": "lofo_" + name.lower().replace(" ", "_"),
                         "description": f"Leave-one-family-out: all {name} languages removed (check v)",
                         **r, "verdict": verdict(r["value"], r["p"], p_supported), "lofo_family": name,
                         "variant_conclusion": conclusion_of(r["value"], r["p"])})
        v = np.array([x["value"] for x in vals])
        rows.append({**base, "check": "lofo_summary", "description":
                     f"Leave-one-family-out over {len(vals)} families: range of rho",
                     "n_concepts": vals[0]["n_concepts"], "value": float(np.min(v)), "ci_low": float(np.min(v)),
                     "ci_high": float(np.max(v)), "p": float(np.max([x["p"] for x in vals])),
                     "verdict": "survives" if all(verdict(x["value"], x["p"], p_supported) == "survives"
                                                  for x in vals) else "does not survive",
                     "variant_conclusion": ", ".join(sorted({conclusion_of(x["value"], x["p"]) for x in vals})),
                     "note": f"min rho {v.min():.3f}, max rho {v.max():.3f}"})
    return rows


# --------------------------------------------------------------------- H2 ---
def h2_rank_test(pop: pd.DataFrame, S: np.ndarray, posmap: dict, k: int, caliper: float,
                 seed_parts: tuple[str, ...], outcome: str = "rank") -> dict:
    m = match(pop, k, caliper, stream(*seed_parts, "match"))
    mm = m[m["role"].isin(["present", "absent"])].copy()
    ia = mm["concept_a"].map(posmap).to_numpy()
    ib = mm["concept_b"].map(posmap).to_numpy()
    if outcome == "rank":
        y = mutual_rank(S, ia, ib)
    else:                                  # negative cosine: lower = closer, same direction as rank
        y = -S[ia, ib]
    pr = y[(mm["role"] == "present").to_numpy()]
    ab = y[(mm["role"] == "absent").to_numpy()]
    res = group_test(pr, ab, stream(*seed_parts, "delta"))
    return {"n_present": len(pr), "n_absent": len(ab), "value": res["cliffs_delta"],
            "ci_low": res["delta_ci_low"], "ci_high": res["delta_ci_high"], "p": res["p_one_sided"]}


def h2_checks(order: list[str], concepts: list[str], idx: np.ndarray, conds: list[str],
              primary: pd.DataFrame, lofo: dict[str, tuple[np.ndarray, pd.DataFrame]]) -> list[dict]:
    rows: list[dict] = []
    posmap = {c: i for i, c in enumerate(concepts)}
    audit_raw = pd.read_csv(config.ENGLISH_AUDIT_PATH, dtype=str, keep_default_na=False)
    audit, audit_status = audit_decisions()
    pop_all = build_population(concepts, audit)

    # alternative audit treatments
    a_unsure_present = audit.copy()
    a_unsure_present.loc[a_unsure_present["final"] == "unsure", "final"] = "present"
    a_unsure_absent = audit.copy()
    a_unsure_absent.loc[a_unsure_absent["final"] == "unsure", "final"] = "absent"
    human = audit_raw[audit_raw["decision"].str.strip() != ""]
    variants = {
        "h2_unsure_as_present": (build_population(concepts, a_unsure_present), config.H2_MATCH_K,
                                 config.H2_MATCH_CALIPER_SD, "rank",
                                 "Audited *unsure* pairs counted as English-present instead of excluded (D-55)"),
        "h2_unsure_as_absent": (build_population(concepts, a_unsure_absent), config.H2_MATCH_K,
                                config.H2_MATCH_CALIPER_SD, "rank",
                                "Audited *unsure* pairs counted as English-absent instead of excluded (D-55)"),
        "h2_no_input_colexified": (pop_all[~pop_all["input_colexified"]].reset_index(drop=True),
                                   config.H2_MATCH_K, config.H2_MATCH_CALIPER_SD, "rank",
                                   "Without pairs that an input language itself colexifies, for every "
                                   "condition (D-49)"),
        "h2_matching_1to1": (pop_all, 1, config.H2_MATCH_CALIPER_SD, "rank",
                             "1:1 matching instead of 1:3 (D-53)"),
        "h2_caliper_010": (pop_all, config.H2_MATCH_K, 0.10, "rank",
                           "Caliper 0.10 SD instead of 0.20 SD (D-53)"),
        "h2_outcome_cosine": (pop_all, config.H2_MATCH_K, config.H2_MATCH_CALIPER_SD, "cosine",
                              "Raw cosine as the outcome instead of the mutual neighbour rank (D-24)"),
    }
    # the proposal's axes (i) and (iv) applied to H2: a different edge rule and a different
    # concept set change which pairs exist and how widespread they are, so the whole matched
    # design is rebuilt, not just filtered (D-61).
    import json

    all_pairs = read_parquet(config.PROCESSED_DIR / "pairs.parquet")
    thr = json.loads((config.RESULTS_DIR / "concept_threshold.json").read_text())
    wc = pd.read_csv(config.RESULTS_DIR / "working_concepts.csv").set_index("concept_id")
    robust_concepts = [c for c in concepts if bool(wc.loc[c, "in_robustness_set"])]
    for k in (1, 3):
        variants[f"h2_minfam{k}"] = (build_population(concepts, audit, with_edge_rule(all_pairs, k)),
                                     config.H2_MATCH_K, config.H2_MATCH_CALIPER_SD, "rank",
                                     f"Edge rule >= {k} family instead of >= 2, population rebuilt "
                                     "(D-07, check i)")
    variants["h2_threshold_100"] = (build_population(robust_concepts, audit), config.H2_MATCH_K,
                                    config.H2_MATCH_CALIPER_SD, "rank",
                                    f"Concept set at the robustness threshold t = {thr['robustness_threshold']} "
                                    f"({len(robust_concepts)} of {len(concepts)} analysis concepts) (D-05, check iv)")
    for name, (_, pairs) in lofo.items():
        variants["h2_lofo_" + name.lower().replace(" ", "_")] = (
            build_population(concepts, audit, pairs), config.H2_MATCH_K, config.H2_MATCH_CALIPER_SD, "rank",
            f"Leave-one-family-out: all {name} languages removed, population rebuilt (check v)")

    for cond in conds:
        pr = primary[(primary["condition"] == cond) & (primary.get("design") == "matched")
                     & (primary["primary"] == True)]      # noqa: E712
        if pr.empty:
            continue
        p_value = float(pr["cliffs_delta"].iloc[0])
        p_supported = bool(p_value > 0 and float(pr["p_one_sided"].iloc[0]) < config.ALPHA)
        S = load_similarity(cond, order)[np.ix_(idx, idx)]
        base = {"hypothesis": "H2", "condition": cond, "label": CONDITION_LABELS[cond],
                "statistic": "Cliff's delta", "primary_value": p_value,
                "primary_conclusion": "supported" if p_supported else "not supported",
                "verdict_basis": "reproduces the primary analysis", "audit_status": audit_status}
        for cid, (pop, k, cal, outcome, desc) in variants.items():
            if pop.empty or (pop["group"] == "present").sum() < 5:
                rows.append({**base, "check": cid, "description": desc, "verdict": "not computable",
                             "note": "fewer than 5 English-present pairs"})
                continue
            r = h2_rank_test(pop, S, posmap, k, cal, ("rob", cond, cid), outcome)
            rows.append({**base, "check": cid, "description": desc, **r,
                         "verdict": verdict(r["value"], r["p"], p_supported),
                         "variant_conclusion": conclusion_of(r["value"], r["p"])})
        lof = [r for r in rows if r["check"].startswith("h2_lofo_") and r.get("condition") == cond
               and np.isfinite(r.get("value", np.nan))]
        if lof:
            v = np.array([r["value"] for r in lof])
            rows.append({**base, "check": "h2_lofo_summary", "description":
                         f"Leave-one-family-out over {len(lof)} families: range of Cliff's delta",
                         "value": float(v.min()), "ci_low": float(v.min()), "ci_high": float(v.max()),
                         "p": float(max(r["p"] for r in lof)),
                         "verdict": "survives" if all(r["verdict"] == "survives" for r in lof)
                         else "does not survive",
                         "variant_conclusion": ", ".join(sorted({r["variant_conclusion"] for r in lof})),
                         "note": f"delta {v.min():.2f}-{v.max():.2f}"})
        # unmatched (always reported, confounded by construction)
        y = mutual_rank(S, pop_all["concept_a"].map(posmap).to_numpy(),
                        pop_all["concept_b"].map(posmap).to_numpy())
        g = (pop_all["group"] == "present").to_numpy()
        res = group_test(y[g], y[~g], stream("rob", cond, "unmatched"))
        rows.append({**base, "check": "h2_unmatched", "description":
                     "No matching at all (confounded; reported for comparison, D-24)",
                     "n_present": int(g.sum()), "n_absent": int((~g).sum()), "value": res["cliffs_delta"],
                     "ci_low": res["delta_ci_low"], "ci_high": res["delta_ci_high"], "p": res["p_one_sided"],
                     "verdict": verdict(res["cliffs_delta"], res["p_one_sided"], p_supported),
                         "variant_conclusion": conclusion_of(res["cliffs_delta"], res["p_one_sided"])})
        # human-audited gold standard only
        if human.empty:
            rows.append({**base, "check": "h2_human_audited_only", "description":
                         "English gold standard restricted to rows a person has decided (D-55)",
                         "verdict": "PENDING USER RUN",
                         "note": "no human decisions in audit/english_colexification_manual_check.csv; "
                                 "fill the `decision` column, then: python -m src.analysis.robustness"})
        else:
            ha = audit[audit["final_source"] == "human"]
            pop_h = build_population(concepts, ha)
            pop_h = pop_h[pop_h["final_source"] == "human"].reset_index(drop=True)
            if (pop_h["group"] == "present").sum() < 5:
                rows.append({**base, "check": "h2_human_audited_only", "verdict": "not computable",
                             "description": "English gold standard restricted to human-decided rows (D-55)",
                             "note": f"only {(pop_h['group'] == 'present').sum()} audited present pairs"})
            else:
                r = h2_rank_test(pop_h, S, posmap, config.H2_MATCH_K, config.H2_MATCH_CALIPER_SD,
                                 ("rob", cond, "human"))
                rows.append({**base, "check": "h2_human_audited_only", **r,
                             "description": "English gold standard restricted to human-decided rows (D-55)",
                             "verdict": verdict(r["value"], r["p"], p_supported),
                         "variant_conclusion": conclusion_of(r["value"], r["p"])})
    return rows


# --------------------------------------------------------------------- H3 ---
def h3_checks() -> list[dict]:
    """H3's own alternatives were computed in Phase 7; they are restated here so that the
    robustness table covers all three hypotheses."""
    fit = pd.read_csv(config.RESULTS_DIR / "h3_family_fit.csv")
    grad = pd.read_csv(config.RESULTS_DIR / "h3_typology.csv")
    rows = []
    for cond, g in fit.groupby("condition"):
        # H3 has no separate primary analysis to reproduce: each row *is* a test of H3,
        # so "survives" here means "supports H3" (controls are marked in their own verdict).
        base = {"hypothesis": "H3", "condition": cond, "label": CONDITION_LABELS.get(cond, cond),
                "verdict_basis": "tests H3 directly"}
        allsig = bool((g["p_auc_bh"] < config.ALPHA).all())
        rows.append({**base, "check": "h3_all_families_above_chance", "statistic": "AUC range",
                     "description": "Every qualifying family's fit, full-family AUC with BH-adjusted "
                                    "permutation p (D-09 amendment)",
                     "n_concepts": int(g["n_concepts"].min()), "value": float(g["auc"].min()),
                     "ci_low": float(g["auc"].min()), "ci_high": float(g["auc"].max()),
                     "p": float(g["p_auc_bh"].max()),
                     "verdict": "survives" if allsig else "does not survive",
                     "variant_conclusion": "every family above chance" if allsig else "not every family",
                     "note": f"{len(g)} families, AUC {g['auc'].min():.2f}-{g['auc'].max():.2f}"})
        rows.append({**base, "check": "h3_spearman_instead_of_auc", "statistic": "Spearman rho",
                     "description": "Spearman correlation with the within-family rate matrix instead of the "
                                    "pair-level AUC, as the proposal asks (D-09 amendment)",
                     "value": float(g["spearman"].min()), "ci_low": float(g["spearman"].min()),
                     "ci_high": float(g["spearman"].max()), "p": float(g["p_spearman_bh"].max()),
                     "verdict": "survives" if bool((g["p_spearman_bh"] < config.ALPHA).all())
                     else "does not survive",
                     "variant_conclusion": "every family above chance"
                     if bool((g["p_spearman_bh"] < config.ALPHA).all()) else "not every family",
                     "note": "the matrix is >99% ties, so rho is small by construction"})
        for kind in ("full", "rarefied"):
            for proxy in ("wikipedia_log10_articles", "joshi_class", "log10_n_languages_clics"):
                r = grad[(grad["condition"] == cond) & (grad["proxy"] == proxy) & (grad["auc_kind"] == kind)]
                if r.empty:
                    continue
                r = r.iloc[0]
                pend = str(r["status"]).startswith("PENDING")
                control = proxy == "log10_n_languages_clics"
                sig = bool(pd.notna(r.get("rho")) and r.get("p_two_sided", 1) < config.ALPHA)
                if pend:
                    verd = "PENDING USER RUN"
                elif control:       # we want NO relation here: a null result is the pass
                    verd = "survives (control)" if not sig else "does not survive (control)"
                else:
                    verd = ("survives" if (pd.notna(r.get("rho")) and r["rho"] > 0
                                           and r.get("p_one_sided", 1) < config.ALPHA)
                            else "does not survive")
                rows.append({**base, "check": f"h3_gradient_{proxy}_{kind}", "statistic": "Spearman rho",
                             "description": ("Control: family fit must *not* track how densely CLICS4 samples "
                                             f"the family ({kind} AUC)" if control else
                                             f"Family gradient: {kind} AUC vs {r['proxy_label']}"),
                             "value": r.get("rho"), "p": r.get("p_one_sided"),
                             "n_concepts": r.get("n_families"),
                             "verdict": verd,
                             "variant_conclusion": "no relation" if (control and not sig) else
                                                   conclusion_of(r.get("rho", np.nan),
                                                                 r.get("p_one_sided", np.nan)),
                             "note": str(r["status"])[:120]})
    return rows


# -------------------------------------------------------------------- main ---
def main() -> None:
    set_all_seeds()
    log_run_header(log)
    order = load_concept_order()
    concepts, idx = analysis_concepts(order, write=False)
    conds = available_conditions()
    if not conds:
        raise SystemExit("no model similarity matrices: run `python run.py embeddings` first")
    h1 = pd.read_csv(config.RESULTS_DIR / "h1_alignment.csv")
    h2 = pd.read_csv(config.RESULTS_DIR / "h2_blindspot.csv")
    log.info("robustness: %d concepts, conditions %s, %d permutations per check",
             len(concepts), conds, NPERM)

    fam = pd.read_csv(config.RESULTS_DIR / "family_subnetworks.csv")
    fam = fam[fam["qualifies"]].sort_values("n_languages", ascending=False)
    log.info("leave-one-family-out networks for %d families (used by H1 and H2)", len(fam))
    lofo = lofo_networks(order, fam)
    rows = h1_checks(order, concepts, idx, conds, h1, lofo)
    rows += h2_checks(order, concepts, idx, conds, h2, lofo)
    rows += h3_checks()
    for cid, hyp, desc, cmd in PENDING_CHECKS:
        rows.append({"hypothesis": hyp, "check": cid, "description": desc, "verdict": "PENDING USER RUN",
                     "note": cmd})
    for cond in config.MODEL_CONDITIONS:
        if cond not in conds:
            rows.append({"hypothesis": "H1/H2/H3", "check": "condition_pending", "condition": cond,
                         "label": CONDITION_LABELS[cond], "verdict": "PENDING USER RUN",
                         "description": "Every check above, for this condition",
                         "note": "python run.py embeddings"})
    tab = pd.DataFrame(rows)
    tab["axis"] = tab["check"].map(axis_of)
    tab["axis_label"] = tab["axis"].map(AXES)
    cols = ["hypothesis", "axis", "axis_label", "check", "condition", "label", "description", "statistic", "n_concepts", "n_pairs",
            "n_present", "n_absent", "value", "ci_low", "ci_high", "p", "primary_value", "primary_conclusion",
            "variant_conclusion", "verdict", "verdict_basis", "note", "lofo_family", "audit_status"]
    tab = tab.reindex(columns=[c for c in cols if c in tab.columns])
    write_result_table(tab, "robustness", producer=__name__,
                       extra={"n_permutations": NPERM, "n_bootstrap": NBOOT,
                              "verdict_rule": "a check survives when it reproduces the primary conclusion "
                                              "(sign and significance at alpha)"})

    detail = tab["check"].str.startswith(("lofo_", "h2_lofo_"), na=False)
    keep = ~tab["check"].isin(["condition_pending"]) & ~detail
    sub = tab[keep | tab["check"].isin(["lofo_summary", "h2_lofo_summary"])]
    axis_summ = (sub[sub["verdict"].notna()]
                 .groupby(["hypothesis", "axis", "axis_label"], as_index=False)
                 .agg(n_checks=("check", "nunique"),
                      n_survive=("verdict", lambda v: int(v.str.startswith("survives").sum())),
                      n_pending=("verdict", lambda v: int((v == "PENDING USER RUN").sum())),
                      all_survive=("verdict", lambda v: bool(v.str.startswith("survives").all()))))
    write_result_table(axis_summ, "robustness_by_axis", producer=__name__)
    summ = (sub
            .groupby(["hypothesis", "check"], as_index=False)
            .agg(description=("description", "first"),
                 n_conditions=("condition", "nunique"),
                 value_min=("value", "min"), value_max=("value", "max"),
                 p_max=("p", "max"),
                 verdicts=("verdict", lambda s: ", ".join(sorted(set(s.dropna())))),
                 variant_conclusion=("variant_conclusion",
                                     lambda s: ", ".join(sorted(set(s.dropna()))) or "")))
    summ["conclusion"] = np.where(summ["verdicts"].str.startswith("survives"), summ["verdicts"],
                                  np.where(summ["verdicts"].str.contains("PENDING"), "PENDING USER RUN",
                                           summ["verdicts"]))
    write_result_table(summ, "robustness_summary", producer=__name__)
    n_sur = int(summ["conclusion"].str.startswith("survives").sum())
    log.info("robustness: %d checks, %d survive, %d pending, %d do not",
             len(summ), n_sur, int((summ["conclusion"] == "PENDING USER RUN").sum()),
             len(summ) - n_sur - int((summ["conclusion"] == "PENDING USER RUN").sum()))


if __name__ == "__main__":
    main()
