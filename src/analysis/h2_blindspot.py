"""Phase 6 · H2: the English blind spot (Proposal Step 6) — pre-specified design.

Population (DECISIONS.md D-24, D-35, D-49, D-53): colexification edges (>= 2
families) between concepts of the analysis set, excluding pairs inside one
gloss-collision group. Groups, from the audit sheet (human ``decision`` if
filled, otherwise Claude's ``suggested_decision``, flagged PROVISIONAL):
  (a) English-present: audited "present";
  (b) English-absent: audited "absent" or not a WordNet candidate;
  audited "unsure" pairs are excluded.

Outcome: mutual nearest-neighbour rank of the pair in a model's similarity
space, normalised to [0, 1] (0 = each is the other's nearest neighbour,
~0.5 = chance), averaged over both directions.

Matched design: exact on same-semantic-field; propensity score (logistic
regression on log family count and the pair's lower English Zipf frequency)
within strata; 1:H2_MATCH_K nearest-neighbour matching without replacement,
caliper H2_MATCH_CALIPER_SD x SD(logit propensity) (Austin 2011). Balance:
standardised mean differences before and after (pooled pre-matching SD).

Tests per condition: one-sided Mann-Whitney U (present ranks lower), Cliff's
delta (positive = English-present pairs closer) with BCa 95% CI, BH across
conditions. Across conditions: the group x model interaction — per-model gaps
with pair-cluster bootstrap CIs, contrasts against word2vec, and (with
statsmodels) a mixed model rank ~ group * model with variance components for
pair, concept a and concept b. The multilingual-input conditions use the
population without input-colexified pairs (D-49); the interaction uses that
common population for all conditions.

Outputs: results/h2_blindspot.csv, h2_balance.csv, h2_matched_pairs.csv,
h2_ranks.csv, h2_interaction.csv, h2_case_study.csv, h2_design.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as ss

import config
from src.analysis.common import (CONDITION_LABELS, analysis_concepts, available_conditions, concept_frequencies,
                                 load_similarity, stream)
from src.utils.concept_order import load_concept_order
from src.utils.io import read_parquet, write_result_json, write_result_table
from src.utils.logging_utils import get_logger, log_run_header
from src.utils.seeds import set_all_seeds
from src.utils.stats import bh_adjust, cliffs_delta, cliffs_delta_ci

log = get_logger(__name__, "phase6_h2")
TRANSLATED = ("xlmr_translated", "labse_translated")
H2_MATCH_SENSITIVITY_RUNS = 200
COVARIATES = ["log_families", "zipf_min", "same_field"]
CASE_PAIRS = [("TOMORROW", "YESTERDAY"), ("ARM", "HAND"), ("TREE", "WOOD"), ("FINGER", "TOE"),
              ("SEE", "UNDERSTAND"), ("LANGUAGE", "TONGUE")]


# --------------------------------------------------------------- population --
def audit_decisions(path=None) -> tuple[pd.DataFrame, str]:
    a = pd.read_csv(path or config.ENGLISH_AUDIT_PATH, dtype=str, keep_default_na=False)
    human = a["decision"].str.strip() != ""
    a["final"] = np.where(human, a["decision"].str.strip(), a["suggested_decision"].str.strip())
    a["final_source"] = np.where(human, "human", "claude_suggestion")
    if (a["final"] == "").any():
        raise ValueError("audit rows without any decision or suggestion")
    status = "human" if human.all() else ("PROVISIONAL (Claude suggestions)" if not human.any()
                                          else f"PARTIAL ({int(human.sum())}/{len(a)} human-audited)")
    return a[["concept_a", "concept_b", "final", "final_source"]], status


def build_population(concepts: list[str], audit: pd.DataFrame,
                     edges: pd.DataFrame | None = None) -> pd.DataFrame:
    """The H2 pair population (D-53). ``edges`` overrides the stored edge table so that
    Phase 8 can rebuild the population under an alternative edge rule, concept set or
    leave-one-family-out network; it needs concept_a, concept_b, n_families, is_edge."""
    keep = set(concepts)
    e = read_parquet(config.PROCESSED_DIR / "edges.parquet") if edges is None else edges
    e = e[e["is_edge"] & e["concept_a"].isin(keep) & e["concept_b"].isin(keep)]
    c = read_parquet(config.PROCESSED_DIR / "concepts.parquet").set_index("concept_id")
    cg = c["collision_group"].fillna("")
    same_group = (e["concept_a"].map(cg) != "") & (e["concept_a"].map(cg) == e["concept_b"].map(cg))
    e = e[~same_group].merge(audit, on=["concept_a", "concept_b"], how="left")
    e["final"] = e["final"].fillna("not_candidate")
    e["final_source"] = e["final_source"].fillna("not_candidate")
    e = e[e["final"] != "unsure"].copy()
    e["group"] = np.where(e["final"] == "present", "present", "absent")
    field = c["semantic_field"].fillna("")
    e["same_field"] = (e["concept_a"].map(field) == e["concept_b"].map(field)).astype(int)
    order = load_concept_order()
    z = dict(zip(order, concept_frequencies(order)))
    za, zb = e["concept_a"].map(z), e["concept_b"].map(z)
    e["zipf_min"], e["zipf_max"] = np.minimum(za, zb), np.maximum(za, zb)
    e["log_families"] = np.log(e["n_families"])
    icp = pd.read_csv(config.RESULTS_DIR / "translated_input_colexified_pairs.csv")
    e = e.merge(icp[["concept_a", "concept_b", "input_isos"]], on=["concept_a", "concept_b"], how="left")
    e["input_colexified"] = e["input_isos"].notna()
    e["pair"] = e["concept_a"] + "~" + e["concept_b"]
    return e.reset_index(drop=True)


# ----------------------------------------------------------------- matching --
def logistic_mle(X: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-10) -> np.ndarray:
    """Unpenalised logistic regression by Newton-Raphson (intercept prepended).
    Implemented here so the propensity model is identical across scikit-learn
    versions, whose penalty arguments changed between 1.5 and 1.9."""
    D = np.column_stack([np.ones(len(y)), X])
    beta = np.zeros(D.shape[1])
    for _ in range(max_iter):
        eta = np.clip(D @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        grad = D.T @ (y - p)
        hess = D.T @ (D * (p * (1 - p))[:, None])
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.abs(step).max() < tol:
            break
    return beta


def propensity(pop: pd.DataFrame) -> np.ndarray:
    """Logit of the estimated propensity of being an English-present pair."""
    X = pop[["log_families", "zipf_min"]].to_numpy(dtype=float)
    X = (X - X.mean(0)) / X.std(0)
    y = (pop["group"] == "present").to_numpy().astype(float)
    beta = logistic_mle(X, y)
    return np.column_stack([np.ones(len(y)), X]) @ beta


def match(pop: pd.DataFrame, k: int, caliper_sd: float, rng: np.random.Generator) -> pd.DataFrame:
    """Greedy 1:k nearest-neighbour matching on the logit propensity, without
    replacement, exact on same_field, within a caliper. Treated pairs are taken in
    a seeded random order (so the result does not depend on row order)."""
    pop = pop.copy()
    pop["logit"] = propensity(pop)
    cal = caliper_sd * pop["logit"].std()
    used: set[int] = set()
    rows = []
    treated = pop.index[pop["group"] == "present"].to_numpy()
    for t in rng.permutation(treated):
        cand = pop[(pop["group"] == "absent") & (pop["same_field"] == pop.at[t, "same_field"])
                   & ~pop.index.isin(used)]
        dist = (cand["logit"] - pop.at[t, "logit"]).abs()
        dist = dist[dist <= cal].sort_values(kind="mergesort")
        chosen = dist.index[:k].tolist()
        if not chosen:
            rows.append({"idx": t, "match_id": t, "role": "unmatched_present"})
            continue
        used.update(chosen)
        rows.append({"idx": t, "match_id": t, "role": "present"})
        rows += [{"idx": c, "match_id": t, "role": "absent"} for c in chosen]
    m = pd.DataFrame(rows).merge(pop, left_on="idx", right_index=True)
    m.attrs["caliper"] = cal
    return m


def smd(x_t: np.ndarray, x_c: np.ndarray, sd_pooled: float) -> float:
    return float((np.mean(x_t) - np.mean(x_c)) / sd_pooled) if sd_pooled > 0 else 0.0


def balance_table(pop: pd.DataFrame, matched: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for cov in COVARIATES + ["zipf_max"]:
        t0, c0 = pop.loc[pop["group"] == "present", cov], pop.loc[pop["group"] == "absent", cov]
        sd = np.sqrt((t0.var() + c0.var()) / 2)
        t1 = matched.loc[matched["role"] == "present", cov]
        c1 = matched.loc[matched["role"] == "absent", cov]
        rows.append({"population": label, "covariate": cov, "matched_on": cov in COVARIATES,
                     "mean_present_before": t0.mean(), "mean_absent_before": c0.mean(),
                     "smd_before": smd(t0, c0, sd),
                     "mean_present_after": t1.mean(), "mean_absent_after": c1.mean(),
                     "smd_after": smd(t1, c1, sd),
                     "balanced_after": abs(smd(t1, c1, sd)) < config.H2_SMD_THRESHOLD})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- ranks --
def mutual_rank(sim: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    """Normalised mutual nearest-neighbour rank for pairs (ia[k], ib[k]).
    rank(b | a) = #{c != a, b : S[a,c] > S[a,b]} + 0.5 * #{ties}, divided by N - 2."""
    n = sim.shape[0]
    out = np.empty(len(ia))
    for k, (a, b) in enumerate(zip(ia, ib)):
        r = []
        for x, y in ((a, b), (b, a)):
            row = np.delete(sim[x], [x, y])
            s = sim[x, y]
            r.append(((row > s).sum() + 0.5 * (row == s).sum()) / (n - 2))
        out[k] = np.mean(r)
    return out


def neighbours(sim: np.ndarray, i: int, concepts: list[str], gloss: dict, k: int = 5) -> str:
    row = sim[i].copy()
    row[i] = -np.inf
    top = np.argsort(-row, kind="mergesort")[:k]
    return ", ".join(gloss[concepts[j]] for j in top)


# -------------------------------------------------------------------- tests --
def group_test(ranks_present: np.ndarray, ranks_absent: np.ndarray, rg: np.random.Generator) -> dict:
    u = ss.mannwhitneyu(ranks_present, ranks_absent, alternative="less")
    u2 = ss.mannwhitneyu(ranks_present, ranks_absent, alternative="two-sided")
    delta = cliffs_delta(ranks_absent, ranks_present)          # > 0: English-present pairs are closer
    lo, hi = cliffs_delta_ci(ranks_absent, ranks_present, config.N_BOOTSTRAP, rg)
    return {"n_present": len(ranks_present), "n_absent": len(ranks_absent),
            "median_rank_present": float(np.median(ranks_present)),
            "median_rank_absent": float(np.median(ranks_absent)),
            "mean_rank_gap": float(np.mean(ranks_absent) - np.mean(ranks_present)),
            "U": float(u.statistic), "p_one_sided": float(u.pvalue), "p_two_sided": float(u2.pvalue),
            "cliffs_delta": delta, "delta_ci_low": lo, "delta_ci_high": hi}


def interaction(long: pd.DataFrame, n_boot: int, rg: np.random.Generator) -> pd.DataFrame:
    """Per-model gap (mean absent rank - mean present rank) with pair-cluster bootstrap
    CIs, and each model's gap minus word2vec's (the group x model interaction contrasts)."""
    models = list(dict.fromkeys(long["condition"]))
    pairs = long["pair"].unique()
    wide = long.pivot_table(index="pair", columns="condition", values="rank").loc[pairs, models]
    grp = long.drop_duplicates("pair").set_index("pair").loc[pairs, "group"].to_numpy()
    w = long.drop_duplicates("pair").set_index("pair").loc[pairs, "weight"].to_numpy()

    def gaps(idx: np.ndarray) -> np.ndarray:
        g, ww, v = grp[idx], w[idx], wide.to_numpy()[idx]
        pa, ab = g == "present", g == "absent"
        return np.array([np.average(v[ab, j], weights=ww[ab]) - np.average(v[pa, j], weights=ww[pa])
                         for j in range(len(models))])

    obs = gaps(np.arange(len(pairs)))
    # resample matched sets (clusters) so each present pair keeps its controls
    mid = long.drop_duplicates("pair").set_index("pair").loc[pairs, "match_id"].to_numpy()
    sets = [np.flatnonzero(mid == m) for m in np.unique(mid)]
    draws = np.array([gaps(np.concatenate([sets[i] for i in rg.integers(0, len(sets), len(sets))]))
                      for _ in range(n_boot)])
    rows = []
    for j, m in enumerate(models):
        lo, hi = np.percentile(draws[:, j], [2.5, 97.5])
        d = draws[:, j] - draws[:, 0]
        clo, chi = np.percentile(d, [2.5, 97.5])
        rows.append({"condition": m, "gap": obs[j], "gap_ci_low": lo, "gap_ci_high": hi,
                     "gap_minus_w2v": obs[j] - obs[0], "contrast_ci_low": clo, "contrast_ci_high": chi,
                     "p_contrast_two_sided": float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))
                     if j else np.nan, "n_matched_sets": len(sets)})
    return pd.DataFrame(rows)


def mixed_model(long: pd.DataFrame) -> dict:
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {"status": "PENDING USER RUN: statsmodels not installed"}
    d = long.assign(one=1, present=(long["group"] == "present").astype(int))
    full = smf.mixedlm("rank ~ present * C(condition)", d, groups="one",
                       vc_formula={"pair": "0 + C(pair)", "ca": "0 + C(concept_a)", "cb": "0 + C(concept_b)"})
    red = smf.mixedlm("rank ~ present + C(condition)", d, groups="one",
                      vc_formula={"pair": "0 + C(pair)", "ca": "0 + C(concept_a)", "cb": "0 + C(concept_b)"})
    f, r = full.fit(reml=False), red.fit(reml=False)
    lr = 2 * (f.llf - r.llf)
    dfree = len(f.fe_params) - len(r.fe_params)
    return {"status": "done", "lr_stat_interaction": float(lr), "df": int(dfree),
            "p_interaction": float(ss.chi2.sf(lr, dfree)),
            "fixed_effects": {k: float(v) for k, v in f.fe_params.items()}}


def matching_sensitivity(pop: pd.DataFrame, sim: np.ndarray, pos: dict, n_runs: int, cond: str,
                         pname: str) -> dict:
    """Repeat the greedy matching under n_runs different treated orders (D-54) and
    summarise the spread of Cliff's delta and of the one-sided p-value."""
    deltas, ps, smds = [], [], []
    for r in range(n_runs):
        m = match(pop, config.H2_MATCH_K, config.H2_MATCH_CALIPER_SD, stream("h2", "match_sens", pname, str(r)))
        mm = m[m["role"].isin(["present", "absent"])]
        rk = mutual_rank(sim, mm["concept_a"].map(pos).to_numpy(), mm["concept_b"].map(pos).to_numpy())
        pr, ab = rk[(mm["role"] == "present").to_numpy()], rk[(mm["role"] == "absent").to_numpy()]
        deltas.append(cliffs_delta(ab, pr))
        ps.append(ss.mannwhitneyu(pr, ab, alternative="less").pvalue)
        sd = np.sqrt((pop.loc[pop["group"] == "present", "log_families"].var()
                      + pop.loc[pop["group"] == "absent", "log_families"].var()) / 2)
        smds.append(smd(mm.loc[mm["role"] == "present", "log_families"], mm.loc[mm["role"] == "absent", "log_families"], sd))
    d, p, sm = np.array(deltas), np.array(ps), np.abs(np.array(smds))
    return {"condition": cond, "population": pname, "n_runs": n_runs,
            "delta_median": float(np.median(d)), "delta_q025": float(np.quantile(d, .025)),
            "delta_q975": float(np.quantile(d, .975)), "share_p_below_05": float((p < 0.05).mean()),
            "p_median": float(np.median(p)), "share_balanced_log_families": float((sm < config.H2_SMD_THRESHOLD).mean())}


# --------------------------------------------------------------------- main --
def main() -> None:
    set_all_seeds()
    log_run_header(log)
    order = load_concept_order()
    concepts, idx = analysis_concepts(order, write=False)
    pos = {c: i for i, c in enumerate(concepts)}
    gloss = read_parquet(config.PROCESSED_DIR / "concepts.parquet").set_index("concept_id")["gloss"].to_dict()
    audit, audit_status = audit_decisions()
    pop_all = build_population(concepts, audit)
    pops = {"all_pairs": pop_all, "no_input_colexified": pop_all[~pop_all["input_colexified"]].reset_index(drop=True)}
    k, cal = config.H2_MATCH_K, config.H2_MATCH_CALIPER_SD
    matched = {name: match(p, k, cal, stream("h2", "match", name)) for name, p in pops.items()}
    bal = pd.concat([balance_table(pops[n], matched[n], n) for n in pops])
    write_result_table(bal, "h2_balance", producer=__name__)
    mp = pd.concat([m.assign(population=n) for n, m in matched.items()])
    write_result_table(mp[["population", "match_id", "role", "concept_a", "concept_b", "gloss_a", "gloss_b",
                           "n_families", "zipf_min", "same_field", "logit", "final", "final_source",
                           "input_colexified"]], "h2_matched_pairs", producer=__name__)

    rows, rank_rows, long_parts, case_rows, sens_rows = [], [], [], [], []
    avail = available_conditions()
    for cond in config.MODEL_CONDITIONS:
        if cond not in avail:
            rows.append({"condition": cond, "label": CONDITION_LABELS[cond], "status": "PENDING USER RUN"})
            continue
        S = load_similarity(cond, order)[np.ix_(idx, idx)]
        for pname, m in matched.items():
            mm = m[m["role"].isin(["present", "absent"])].copy()
            mm["rank"] = mutual_rank(S, mm["concept_a"].map(pos).to_numpy(), mm["concept_b"].map(pos).to_numpy())
            primary = (pname == "no_input_colexified") == (cond in TRANSLATED)
            res = group_test(mm.loc[mm["role"] == "present", "rank"].to_numpy(),
                             mm.loc[mm["role"] == "absent", "rank"].to_numpy(), stream("h2", cond, pname, "delta"))
            rows.append({"condition": cond, "label": CONDITION_LABELS[cond], "status": "done", "design": "matched",
                         "population": pname, "primary": primary, "audit_status": audit_status, **res})
            rank_rows.append(mm.assign(condition=cond, population=pname)[["condition", "population", "role", "match_id", "pair",
                                                        "rank"]])
            if pname == "no_input_colexified":
                cnt = mm.groupby("match_id")["role"].transform(lambda s: (s == "absent").sum())
                long_parts.append(mm.assign(condition=cond, group=mm["role"],
                                            weight=np.where(mm["role"] == "absent", 1.0 / cnt.clip(lower=1), 1.0))
                                  [["condition", "pair", "concept_a", "concept_b", "group", "rank", "match_id",
                                    "weight"]])
        for pname in pops:
            sens_rows.append(matching_sensitivity(pops[pname], S, pos, H2_MATCH_SENSITIVITY_RUNS, cond, pname))
        # unmatched analysis, reported alongside and labelled (D-24)
        pa = pop_all.copy()
        pa["rank"] = mutual_rank(S, pa["concept_a"].map(pos).to_numpy(), pa["concept_b"].map(pos).to_numpy())
        res = group_test(pa.loc[pa["group"] == "present", "rank"].to_numpy(),
                         pa.loc[pa["group"] == "absent", "rank"].to_numpy(), stream("h2", cond, "unmatched"))
        rows.append({"condition": cond, "label": CONDITION_LABELS[cond], "status": "done", "design": "unmatched",
                     "population": "all_pairs", "primary": False, "audit_status": audit_status, **res})
        for a, b in CASE_PAIRS:
            ia, ib = pos[a], pos[b]
            r = mutual_rank(S, np.array([ia]), np.array([ib]))[0]
            case_rows.append({"condition": cond, "pair": f"{gloss[a]} ~ {gloss[b]}",
                              "english": pa.loc[pa["pair"] == f"{a}~{b}", "group"].squeeze()
                              if (pa["pair"] == f"{a}~{b}").any() else "not in population",
                              "families": int(pa.loc[pa["pair"] == f"{a}~{b}", "n_families"].squeeze())
                              if (pa["pair"] == f"{a}~{b}").any() else np.nan,
                              "cosine": S[ia, ib], "mutual_rank": r,
                              "input_colexified": bool(pa.loc[pa["pair"] == f"{a}~{b}", "input_colexified"].squeeze())
                              if (pa["pair"] == f"{a}~{b}").any() else False,
                              f"neighbours_{gloss[a]}": neighbours(S, ia, concepts, gloss),
                              f"neighbours_{gloss[b]}": neighbours(S, ib, concepts, gloss)})
        log.info("%s: %s", cond, "; ".join(f"{r['population']}/{r['design']}: delta={r['cliffs_delta']:.3f} "
                                          f"[{r['delta_ci_low']:.3f}, {r['delta_ci_high']:.3f}] p={r['p_one_sided']:.3g}"
                                          for r in rows if r.get("condition") == cond and r.get("status") == "done"))

    res = pd.DataFrame(rows)
    if "primary" in res:
        prim = res["primary"].fillna(False).astype(bool)
        res["p_bh_primary"] = np.nan
        res.loc[prim, "p_bh_primary"] = bh_adjust(res.loc[prim, "p_one_sided"].to_numpy(dtype=float))
    write_result_table(res, "h2_blindspot", producer=__name__,
                       extra={"audit_status": audit_status, "match_k": k, "caliper_sd": cal})
    if rank_rows:
        write_result_table(pd.concat(rank_rows), "h2_ranks", producer=__name__)
    write_result_table(pd.DataFrame(case_rows), "h2_case_study", producer=__name__)
    if sens_rows:
        write_result_table(pd.DataFrame(sens_rows), "h2_matching_sensitivity", producer=__name__)

    design = {"audit_status": audit_status, "match_k": k, "caliper_sd": cal,
              "caliper_logit": {n: m.attrs["caliper"] for n, m in matched.items()},
              "populations": {n: {"n_present": int((p["group"] == "present").sum()),
                                  "n_absent": int((p["group"] == "absent").sum()),
                                  "n_matched_present": int((matched[n]["role"] == "present").sum()),
                                  "n_unmatched_present": int((matched[n]["role"] == "unmatched_present").sum()),
                                  "n_matched_absent": int((matched[n]["role"] == "absent").sum())}
                              for n, p in pops.items()},
            "balanced_after_matching": {n: bool(bal.loc[bal["matched_on"] & (bal["population"] == n),
                                                        "balanced_after"].all()) for n in pops}}
    if len(long_parts) >= 2:
        long = pd.concat(long_parts)
        inter = interaction(long, config.N_BOOTSTRAP, stream("h2", "interaction"))
        write_result_table(inter, "h2_interaction", producer=__name__)
        design["mixed_model"] = mixed_model(long)
    else:
        design["interaction"] = "PENDING USER RUN: needs at least two model conditions"
    write_result_json(design, "h2_design", producer=__name__)
    log.info("Audit: %s; matched present pairs: %s; balanced (|SMD| < 0.1 on matched covariates): %s", audit_status,
             {n: v["n_matched_present"] for n, v in design["populations"].items()}, design["balanced_after_matching"])


if __name__ == "__main__":
    main()
