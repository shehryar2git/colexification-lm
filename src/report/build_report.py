"""Phase 8 · Build report/report.md from report/report_template.md and results/.

Every number in the report is a ``{{placeholder}}`` resolved here from a file in
results/, and every resolved value is written with its source file to
results/report_values.json. Nothing in the template is a literal statistic, so
re-running this after `python run.py analyse` regenerates a correct report.

A few interpretive sentences depend on what the data say (H1 supported or not,
how many conditions have been run). Those are chosen here from pre-written
alternatives by a condition on the results, never written by hand into the
template; the alternatives are in the ``verdict_*`` values below.

    python -m src.report.build_report      # writes report/report.md + appendix/
"""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.utils.io import write_result_json
from src.utils.logging_utils import get_logger, log_run_header

log = get_logger(__name__, "build_report")
TEMPLATE = config.REPORT_DIR / "report_template.md"
OUT = config.REPORT_DIR / "report.md"
APPENDIX = config.REPORT_DIR / "appendix"
R = config.RESULTS_DIR


# ------------------------------------------------------------- formatting --
def num(x, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{d}f}"


def thousands(x) -> str:
    return f"{int(x):,}"


def pval(p) -> str:
    """APA-style p. The permutation floor 1/(B+1) is reported as '< .001' only when it is."""
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "n/a"
    if p < 0.001:
        return "*p* < .001"
    return f"*p* = {p:.3f}".replace("0.", ".", 1)


def pval_bare(p) -> str:
    """p for a table cell: '< .001' or '.065'."""
    t = pval(p)
    return t.replace("*p* = ", "").replace("*p* ", "")


def ci(lo, hi, d: int = 2) -> str:
    return f"[{num(lo, d)}, {num(hi, d)}]"


def pct(x, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


class Values:
    """Placeholder store: every value carries the results file it came from."""

    def __init__(self) -> None:
        self.v: dict[str, str] = {}
        self.src: dict[str, str] = {}

    def add(self, key: str, value, source: str) -> None:
        self.v[key] = value if isinstance(value, str) else str(value)
        self.src[key] = source

    def many(self, source: str, **kw) -> None:
        for k, val in kw.items():
            self.add(k, val, source)


def read_json(name: str) -> dict:
    return json.loads((R / f"{name}.json").read_text())


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(R / f"{name}.csv")


# ----------------------------------------------------------------- tables --
def md_table(df: pd.DataFrame, align: dict[str, str] | None = None) -> str:
    align = align or {}
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join(("---:" if align.get(c) == "r" else " --- ") for c in cols) + "|"
    body = ["| " + " | ".join("" if pd.isna(x) else str(x) for x in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def table_h1(h1: pd.DataFrame) -> str:
    rows = []
    for _, r in h1.iterrows():
        if r["status"] != "done":
            rows.append({"Model condition": r["label"], "ρ (RSA)": "PENDING USER RUN", "95% CI": "",
                         "Mantel *p*": "", "Partial ρ": "", "ARI": "", "Procrustes *m*²": ""})
            continue
        rows.append({"Model condition": r["label"], "ρ (RSA)": num(r["rho_raw"], 3),
                     "95% CI": ci(r["rho_ci_low"], r["rho_ci_high"], 3),
                     "Mantel *p*": pval_bare(r["p_mantel_bh"]),
                     "Partial ρ": num(r["rho_partial"], 3), "ARI": num(r["ari_primary"], 2),
                     "Procrustes *m*²": num(r["procrustes_m2"], 2)})
    return md_table(pd.DataFrame(rows))


def table_h2(h2: pd.DataFrame) -> str:
    rows = []
    for cond in config.MODEL_CONDITIONS:
        m = h2[(h2["condition"] == cond) & (h2.get("design") == "matched") & (h2["primary"] == True)]  # noqa: E712
        u = h2[(h2["condition"] == cond) & (h2.get("design") == "unmatched")]
        lab = h2.loc[h2["condition"] == cond, "label"].iloc[0]
        if m.empty:
            rows.append({"Model condition": lab, "*n* present / absent": "PENDING USER RUN",
                         "δ matched": "", "95% CI": "", "*p* (BH)": "", "δ unmatched": ""})
            continue
        r = m.iloc[0]
        rows.append({"Model condition": lab,
                     "*n* present / absent": f"{int(r['n_present'])} / {int(r['n_absent'])}",
                     "δ matched": num(r["cliffs_delta"], 2),
                     "95% CI": ci(r["delta_ci_low"], r["delta_ci_high"], 2),
                     "*p* (BH)": pval_bare(r.get("p_bh_primary", r["p_one_sided"])),
                     "δ unmatched": num(u.iloc[0]["cliffs_delta"], 2) if len(u) else ""})
    return md_table(pd.DataFrame(rows))


def table_h3(fit: pd.DataFrame, prox: pd.DataFrame, best: str) -> str:
    f = fit[fit["condition"] == best].merge(prox, on="family_gc", how="left", suffixes=("", "_p"))
    f = f.sort_values("auc", ascending=False)
    rows = []
    for r in f.itertuples():
        wiki = getattr(r, "wikipedia_log10_articles", np.nan)
        rows.append({"Family": r.family_name, "Languages": int(r.n_languages),
                     "Colex. pairs": int(r.n_pos), "AUC": num(r.auc, 2),
                     "95% CI": ci(r.auc_ci_low, r.auc_ci_high, 2),
                     "*p* (BH)": pval_bare(r.p_auc_bh),
                     "AUC rarefied": num(r.auc_rarefied_mean, 2),
                     "Joshi class": "–" if pd.isna(r.joshi_class) else f"≥{int(r.joshi_class)}",
                     "log₁₀ Wikipedia": "pending" if pd.isna(wiki) else num(wiki, 1)})
    return md_table(pd.DataFrame(rows))


def table_robustness(summ: pd.DataFrame) -> str:
    order = {"H1": 0, "H2": 1, "H3": 2}
    s = summ.copy()
    s["o"] = s["hypothesis"].map(order).fillna(9)
    s = s.sort_values(["o", "check"])
    rows = []
    for r in s.itertuples():
        if r.conclusion == "PENDING USER RUN":
            val = "pending"
        elif pd.isna(r.value_min):
            val = "–"
        elif abs(r.value_max - r.value_min) < 1e-9:
            val = num(r.value_min, 3)
        else:
            val = f"{num(r.value_min, 3)} – {num(r.value_max, 3)}"
        short = re.sub(r"\s*\([^)]*\)\s*$", "", str(r.description).split(";")[0]).strip()
        if len(short) > 54:
            short = short[:51].rsplit(" ", 1)[0] + "…"
        rows.append({"H": r.hypothesis, "Check": short, "Statistic": val,
                     "Variant says": getattr(r, "variant_conclusion", ""), "Conclusion": r.conclusion})
    return md_table(pd.DataFrame(rows))


def table_case_study(cs: pd.DataFrame) -> str:
    if cs.empty:
        return "_PENDING USER RUN: no model condition has been embedded._"
    rows = []
    for r in cs.itertuples():
        rows.append({"Pair": r.pair, "English": r.english, "Families": "" if pd.isna(r.families) else int(r.families),
                     "Model": r.condition, "cos": num(r.cosine, 2), "Mutual rank": num(r.mutual_rank, 3),
                     "Input-colexified": "yes" if r.input_colexified else "no"})
    return md_table(pd.DataFrame(rows))


# ------------------------------------------------------------------ values --
def build_values() -> Values:
    V = Values()
    lang, net = read_json("language_summary"), read_json("network_summary")
    thr, val = read_json("concept_threshold"), read_json("clics_validation")
    aset, ops = read_json("analysis_concept_set"), read_json("colex_operationalisations")
    eng, tr = read_json("english_colex_summary"), read_json("translated_coverage")
    stab, drc = read_json("community_stability"), read_json("dr_choice")
    h2d, h3d = read_json("h2_design"), read_json("h3_design")
    prox_status = read_json("resource_proxies_status")
    h1, h2 = read_csv("h1_alignment"), read_csv("h2_blindspot")
    reg, clu = read_csv("h1_regression"), read_csv("h1_cluster_agreement")
    fit, grad = read_csv("h3_family_fit"), read_csv("h3_typology")
    prox, macro = read_csv("resource_proxies"), read_csv("h3_macroarea")
    langagr, cs = read_csv("h3_language_agreement"), read_csv("h2_case_study")
    bal, sens = read_csv("h2_balance"), read_csv("h2_matching_sensitivity")
    cov, dq = read_csv("embedding_coverage"), read_csv("dr_quality")
    fam, ties = read_csv("family_subnetworks"), read_csv("typological_matrix_ties")
    summ = read_csv("robustness_summary")
    sim = read_csv("model_similarity_summary")

    done = h1[h1["status"] == "done"].sort_values("rho_raw", ascending=False)
    best = str(done["condition"].iloc[0]) if len(done) else None
    best_label = str(done["label"].iloc[0]) if len(done) else "none"
    n_done, n_cond = len(done), len(config.MODEL_CONDITIONS)

    # --- data and network -------------------------------------------------
    V.many("results/language_summary.json", n_varieties=thousands(lang["n_varieties"]),
           n_varieties_included=thousands(lang["n_varieties_included"]),
           n_languages=thousands(lang["n_languages_included"]),
           n_families=lang["n_families_included"], n_isolates=lang["n_isolate_families"],
           n_family_disagreements=lang["n_family_disagreements_with_clics"])
    V.many("results/clics_validation.json", n_pairs_clics=thousands(val["n_pairs_clics"]),
           validation_share=pct(val["share_family_count_equal"], 0))
    V.many("results/concept_threshold.json", knee=thr["threshold_used"],
           knee_n=thr["knee_n_concepts"], n_working=thr["n_working"],
           n_robustness_set=thr["n_robustness_set"], n_overrides=thr["n_gloss_overrides"],
           n_excluded=thr["n_excluded_by_override"], n_anchors=len(thr["anchors_added_below_threshold"]),
           n_concepts_total=thousands(thr["n_concepts_total"]),
           n_fields=len(thr["semantic_fields"]), concept_order_sha=thr["concept_order_sha256"][:12])
    V.many("results/network_summary.json", n_pairs_total=thousands(net["n_pairs"]),
           n_colex_any=thousands(net["n_pairs_colexified_any"]),
           n_single_family=thousands(net["n_colexified_pairs_single_family"]),
           share_single_family=pct(net["n_colexified_pairs_single_family"] / net["n_pairs_colexified_any"], 0),
           n_edges=thousands(net["n_edges_min_families"]),
           edge_q90=int(net["edge_family_count_quantiles"]["0.9"]),
           edge_max=int(net["edge_family_count_quantiles"]["1.0"]),
           largest_component=net["largest_component"],
           n_isolated=len(net["isolated_concepts"]), degree_median=int(net["degree_median"]),
           degree_max=net["degree_max"], top_hub=list(net["highest_degree"])[0],
           n_families_qualifying=net["n_families_qualifying"])
    V.many("results/colex_operationalisations.json",
           n_edges_evidence_all=thousands(ops["comparisons"][0]["n_edges_alternative"]),
           spearman_evidence=num(ops["comparisons"][0]["spearman_rate_all_pairs"], 2))
    V.many("results/typological_matrix_ties.csv",
           ties_shortest_path=pct(float(ties.loc[ties["matrix"] == "shortest_path", "share_pairs_at_max"].iloc[0])),
           ties_profile_cosine=pct(float(ties.loc[ties["matrix"] == "profile_cosine",
                                                  "share_pairs_at_max"].iloc[0]), 0))
    V.many("results/english_colex_summary.json", n_in_wordnet=eng["n_concepts_in_wordnet"],
           n_candidate_pairs=thousands(eng["n_candidate_pairs"]))
    V.many("results/translated_coverage.json", n_translated_concepts=tr["n_concepts_meeting_min_units"],
           translated_min_units=tr["min_units"], n_curated_forms=tr["n_forms_from_curated"])
    V.many("results/analysis_concept_set.json", n_analysis=aset["n_concepts"],
           n_analysis_pairs=thousands(aset["n_pairs"]))
    V.many("results/community_stability.json", louvain_ari=num(stab["louvain"]["ari"]["median"], 2),
           louvain_ari_q25=num(stab["louvain"]["ari"]["q25"], 2),
           louvain_ari_q75=num(stab["louvain"]["ari"]["q75"], 2),
           n_communities=int(stab["louvain"]["n_communities"]["median"]),
           modularity=num(stab["louvain"]["modularity"]["median"], 2),
           stability_verdict=stab["louvain_stability_verdict"], infomap_status=str(stab["infomap"])[:60])
    V.many("results/dr_quality.csv", dr_chosen=drc["chosen"].replace("_", " "),
           dr_trust=num(float(dq.loc[dq["chosen"], "trustworthiness"].iloc[0]), 2),
           dr_stress=num(float(dq.loc[dq["chosen"], "stress1"].iloc[0]), 2),
           n_mds_concepts=drc["n_concepts_embedded"])
    V.many("results/embedding_coverage.csv",
           w2v_embedded=int(cov.loc[cov["condition"] == "w2v_gnews", "n_embedded"].iloc[0]),
           w2v_missing=str(cov.loc[cov["condition"] == "w2v_gnews", "missing_concepts"].iloc[0]).lower())
    V.many("results/model_similarity_summary.csv",
           mean_cos_centred=num(float(sim.loc[sim["condition"] == "w2v_gnews", "mean_cosine"].iloc[0]), 3),
           mean_cos_raw=num(float(sim.loc[sim["condition"] == "w2v_gnews__raw", "mean_cosine"].iloc[0]), 2))
    ftop = fam.sort_values("n_languages", ascending=False).iloc[0]
    V.many("results/family_subnetworks.csv", n_families_total=len(fam),
           family_largest=ftop["family_name"], family_largest_n=int(ftop["n_languages"]),
           family_ie_n=int(fam.loc[fam["family_name"] == "Indo-European", "n_languages"].iloc[0]))

    # --- run status -------------------------------------------------------
    pending = [c for c in config.MODEL_CONDITIONS if c not in set(done["condition"])]
    V.many("results/h1_alignment.csv", n_conditions_done=n_done, n_conditions=n_cond,
           conditions_done=", ".join(done["label"]),
           conditions_pending=", ".join(h1.loc[h1["condition"].isin(pending), "label"]) or "none",
           n_conditions_pending=len(pending),
           best_label=best_label)
    V.add("generated_on", dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"), "build time")
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                                    # noqa: BLE001
        sha = "unknown"
    V.add("git_commit", sha, "git")
    import platform
    V.add("python_version", platform.python_version(), "environment")

    # --- H1 ---------------------------------------------------------------
    V.add("table_h1", table_h1(h1), "results/h1_alignment.csv")
    if best:
        b = done.iloc[0]
        V.many("results/h1_alignment.csv", h1_rho=num(b["rho_raw"], 3),
               h1_ci=ci(b["rho_ci_low"], b["rho_ci_high"], 3), h1_p=pval(b["p_mantel_one_sided"]),
               h1_null_sd=num(b["null_sd"], 3), h1_partial=num(b["rho_partial"], 3),
               h1_partial_freq=num(b["rho_partial_freq"], 3), h1_partial_field=num(b["rho_partial_field"], 3),
               h1_drop=num(b["rho_drop_raw_to_partial"], 3),
               h1_drop_share=pct(b["rho_drop_raw_to_partial"] / b["rho_raw"], 0),
               h1_ari=num(b["ari_primary"], 2), h1_nmi=num(b["nmi_primary"], 2),
               h1_k=int(b["k_reference"]), h1_m2=num(b["procrustes_m2"], 2),
               h1_n_procrustes=int(b["n_concepts_procrustes"]))
        rb = reg[reg["condition"] == best].set_index("term")
        V.many("results/h1_regression.csv",
               **{f"beta_{t}": num(rb.loc[t, "beta_std"], 2) for t in rb.index},
               **{f"beta_{t}_ci": ci(rb.loc[t, "ci_low"], rb.loc[t, "ci_high"], 2) for t in rb.index},
               p_log_families=pval(rb.loc["log_families", "p_mrqap"]))
        cl = clu[(clu["condition"] == best) & (clu["subset"] == "all")]
        V.many("results/h1_cluster_agreement.csv", ari_min=num(cl["ari"].min(), 2), ari_max=num(cl["ari"].max(), 2))
        cl_nv = clu[(clu["condition"] == best) & (clu["subset"] != "all")]
        V.add("ari_no_verbs", num(cl_nv["ari"].max(), 2) if len(cl_nv) else "n/a",
              "results/h1_cluster_agreement.csv")
    if len(hm := h2[(h2["design"] == "matched") & (h2["primary"] == True)]):   # noqa: E712
        any_sig = bool(((hm["cliffs_delta"] > 0) & (hm["p_one_sided"] < config.ALPHA)).any())
        all_sig = bool(((hm["cliffs_delta"] > 0) & (hm["p_one_sided"] < config.ALPHA)).all())
        V.add("abstract_h2", (
            "The English blind spot is visible in every computed condition under the matched design"
            if all_sig else
            ("The English blind spot appears in some conditions but not all under the matched design"
             if any_sig else
             "The English blind spot does not survive a matched design")), "results/h2_blindspot.csv")
    fb0 = fit[fit["condition"] == best] if best else fit.iloc[:0]
    if len(fb0):
        n_sig = int((fb0["p_auc_bh"] < config.ALPHA).sum())
        V.add("abstract_h3", (f"Within every one of the {len(fb0)} qualifying families" if n_sig == len(fb0)
                              else f"Within {n_sig} of the {len(fb0)} qualifying families"),
              "results/h3_family_fit.csv")
        V.add("h3_all_above", (f"within every one of the {len(fb0)} families tested" if n_sig == len(fb0)
                               else f"within {n_sig} of the {len(fb0)} families tested"),
              "results/h3_family_fit.csv")
    if len(cs):
        k = cs.sort_values("mutual_rank").iloc[0]
        V.many("results/h2_case_study.csv", case_closest_pair=str(k["pair"]),
               case_closest_rank=num(k["mutual_rank"], 3),
               kal_closer=int(round(float(cs.loc[cs["pair"].str.contains("tomorrow"), "mutual_rank"].iloc[0])
                                    * (int(aset["n_concepts"]) - 2))) if
               cs["pair"].str.contains("tomorrow").any() else "n/a")
    supported = bool(len(done) and done["h1_supported"].all())
    V.add("verdict_h1", ("H1 is supported for every condition computed so far" if supported else
                         "H1 is not supported for every condition computed so far"),
          "results/h1_alignment.csv")

    # --- H2 ---------------------------------------------------------------
    V.add("table_h2", table_h2(h2), "results/h2_blindspot.csv")
    V.add("audit_status", h2d["audit_status"], "results/h2_design.json")
    p_all = h2d["populations"]["all_pairs"]
    V.many("results/h2_design.json", h2_n_present=p_all["n_present"], h2_n_absent=thousands(p_all["n_absent"]),
           h2_matched_present=p_all["n_matched_present"], h2_matched_absent=p_all["n_matched_absent"],
           h2_unmatched_present=p_all["n_unmatched_present"],
           h2_n_present_noic=h2d["populations"]["no_input_colexified"]["n_present"],
           h2_caliper=num(h2d["caliper_logit"]["all_pairs"], 2), h2_k=h2d["match_k"],
           h2_interaction=str(h2d["interaction"])[:70])
    ba = bal[bal["population"] == "all_pairs"].set_index("covariate")
    V.many("results/h2_balance.csv",
           gm_present=num(float(np.exp(ba.loc["log_families", "mean_present_before"])), 0),
           gm_absent=num(float(np.exp(ba.loc["log_families", "mean_absent_before"])), 0),
           smd_before=num(ba.loc["log_families", "smd_before"], 2),
           smd_after=num(abs(ba.loc["log_families", "smd_after"]), 2),
           smd_after_max=num(ba["smd_after"].abs().max(), 2))
    hm = h2[(h2["design"] == "matched") & (h2["primary"] == True)]              # noqa: E712
    if len(hm):
        r = hm.iloc[0]
        hu = h2[(h2["condition"] == r["condition"]) & (h2["design"] == "unmatched")].iloc[0]
        V.many("results/h2_blindspot.csv", h2_delta=num(r["cliffs_delta"], 2),
               h2_ci=ci(r["delta_ci_low"], r["delta_ci_high"], 2), h2_p=pval(r["p_one_sided"]),
               h2_median_present=num(r["median_rank_present"], 3), h2_median_absent=num(r["median_rank_absent"], 3),
               h2_delta_unmatched=num(hu["cliffs_delta"], 2),
               h2_ci_unmatched=ci(hu["delta_ci_low"], hu["delta_ci_high"], 2),
               h2_p_unmatched=pval(hu["p_one_sided"]))
        s0 = sens[(sens["condition"] == r["condition"]) & (sens["population"] == "all_pairs")].iloc[0]
        V.many("results/h2_matching_sensitivity.csv", sens_delta=num(s0["delta_median"], 2),
               sens_lo=num(s0["delta_q025"], 2), sens_hi=num(s0["delta_q975"], 2),
               sens_share=pct(s0["share_p_below_05"], 0), sens_p=pval(s0["p_median"]),
               sens_runs=int(s0["n_runs"]))
        s1 = sens[(sens["condition"] == r["condition"]) & (sens["population"] == "no_input_colexified")].iloc[0]
        V.many("results/h2_matching_sensitivity.csv", sens_delta_noic=num(s1["delta_median"], 2),
               sens_share_noic=pct(s1["share_p_below_05"], 0))
        blind = bool(r["cliffs_delta"] > 0 and r["p_one_sided"] < config.ALPHA)
        V.add("verdict_h2", ("H2 is supported: English-present colexifications are closer in model space than "
                             "matched English-absent ones" if blind else
                             "H2 is not supported at α = .05: after matching, English-present and English-absent "
                             "colexifications are not reliably different in model space"),
              "results/h2_blindspot.csv")
    V.add("table_case_study", table_case_study(cs), "results/h2_case_study.csv")

    # --- H3 ---------------------------------------------------------------
    V.add("table_h3", table_h3(fit, prox, best) if best else "_PENDING USER RUN._",
          "results/h3_family_fit.csv")
    fb = fit[fit["condition"] == best] if best else fit.iloc[:0]
    if len(fb):
        V.many("results/h3_family_fit.csv", h3_auc_min=num(fb["auc"].min(), 2), h3_auc_max=num(fb["auc"].max(), 2),
               h3_best_family=fb.sort_values("auc").iloc[-1]["family_name"],
               h3_worst_family=fb.sort_values("auc").iloc[0]["family_name"],
               h3_p_max=pval(fb["p_auc_bh"].max()), h3_n_families=len(fb),
               h3_rare_min=num(fb["auc_rarefied_mean"].min(), 2), h3_rare_max=num(fb["auc_rarefied_mean"].max(), 2),
               h3_ie_auc=num(fb.loc[fb["family_name"] == "Indo-European", "auc"].iloc[0], 2),
               h3_ie_ci=ci(fb.loc[fb["family_name"] == "Indo-European", "auc_ci_low"].iloc[0],
                           fb.loc[fb["family_name"] == "Indo-European", "auc_ci_high"].iloc[0], 2))
        g = grad[grad["condition"] == best].set_index(["proxy", "auc_kind"])
        for proxy, tag in (("wikipedia_log10_articles", "wiki"), ("joshi_class", "joshi"),
                           ("log10_n_languages_clics", "density")):
            for kind in ("rarefied", "full"):
                key = (proxy, kind)
                if key in g.index:
                    r = g.loc[key]
                    pend = str(r["status"]).startswith("PENDING")
                    V.add(f"h3_{tag}_{kind}", "PENDING USER RUN" if pend else
                          f"ρ = {num(r['rho'], 2)}, {pval(r['p_one_sided'])}, "
                          f"{int(r['n_families'])} families", "results/h3_typology.csv")
        V.add("h3_density_two_sided",
              pval(float(g.loc[("log10_n_languages_clics", "full"), "p_two_sided"])), "results/h3_typology.csv")
        V.add("h3_density_rho", num(float(g.loc[("log10_n_languages_clics", "full"), "rho"]), 2),
              "results/h3_typology.csv")
        col = f"auc_{best}"
        m = langagr[langagr["mapped"] & langagr[col].notna()]
        V.many("results/h3_language_agreement.csv", n_languages_mapped=thousands(len(m)),
               n_languages_total=thousands(len(langagr)), lang_auc_median=num(m[col].median(), 2),
               lang_above_chance=pct((m[col] > 0.5).mean(), 1),
               macro_min=num(macro[f"median_auc_{best}"].min(), 2),
               macro_max=num(macro[f"median_auc_{best}"].max(), 2),
               macro_min_name=macro.loc[macro[f"median_auc_{best}"].idxmin(), "macroarea"],
               macro_max_name=macro.loc[macro[f"median_auc_{best}"].idxmax(), "macroarea"])
    V.many("results/resource_proxies_status.json", wiki_status=prox_status["wikipedia_log10_articles_status"],
           joshi_status=prox_status["joshi_class_status"],
           n_joshi_families=int(prox["joshi_class"].notna().sum()))
    wiki_pending = prox_status["wikipedia_log10_articles_status"].startswith("PENDING")
    V.add("verdict_h3", ("H3 cannot be decided yet: the confirmatory proxy (Wikipedia article counts) is "
                         "PENDING USER RUN" if wiki_pending else
                         ("H3 is supported: better-resourced families are fitted better" if
                          bool(grad[(grad["proxy"] == "wikipedia_log10_articles")
                                    & (grad["auc_kind"] == "rarefied")]["h3_supported"].any()) else
                          "H3 is not supported: family fit does not track the resource proxy")),
          "results/h3_typology.csv")

    # --- robustness -------------------------------------------------------
    V.add("table_robustness", table_robustness(summ), "results/robustness_summary.csv")
    V.add("references", references_block(), "report/references.bib")
    V.add("n_references", str(len(parse_bib(config.REPORT_DIR / "references.bib"))), "report/references.bib")
    V.many("results/robustness_summary.csv", n_checks=len(summ),
           n_survive=int(summ["conclusion"].str.startswith("survives").sum()),
           n_pending_checks=int((summ["conclusion"] == "PENDING USER RUN").sum()),
           n_fail=int((~(summ["conclusion"].str.startswith("survives")
                         | (summ["conclusion"] == "PENDING USER RUN"))).sum()))
    rob = read_csv("robustness")
    is_lofo = rob["check"].str.startswith("h2_lofo_", na=False)
    h2v = rob[(rob["hypothesis"] == "H2") & (rob["check"] != "h2_unmatched") & ~is_lofo]
    sup = h2v[h2v["variant_conclusion"] == "supported"]
    lofo_rows = rob[is_lofo & (rob["check"] != "h2_lofo_summary")]
    V.many("results/robustness.csv", n_h2_variants=len(h2v), n_h2_variants_supported=len(sup),
           n_h2_lofo=len(lofo_rows),
           n_h2_lofo_supported=int((lofo_rows["variant_conclusion"] == "supported").sum()),
           h2_lofo_supported_families=", ".join(
               sorted(lofo_rows.loc[lofo_rows["variant_conclusion"] == "supported", "check"]
                      .str.replace("h2_lofo_", "", regex=False).str.replace("_", " ").str.title())) or "none",
           h2_fragility_list="; ".join(sorted({str(d).split("(")[0].strip().rstrip(",")
                                               for d in sup["description"]})) or "none")
    lofo = rob[rob["check"] == "lofo_summary"]
    if len(lofo):
        V.add("lofo_range", str(lofo.iloc[0]["note"]), "results/robustness.csv")
    c3 = rob[rob["check"] == "data_clics3"]
    if len(c3):
        d = dict(kv.split("=") for kv in str(c3.iloc[0]["note"]).split("; ") if "=" in kv)
        V.many("results/robustness.csv", clics3_rho=num(c3.iloc[0]["value"], 3),
               clics3_note=(f"CLICS³'s network has {thousands(d['clics3_edges'])} edges over "
                            f"{thousands(d['clics3_nodes'])} concepts, of which "
                            f"{thousands(d['working_concepts_in_clics3'])} are in our working set and "
                            f"{thousands(d['edges_among_working_concepts'])} edges fall inside it"))
    return V


# --------------------------------------------------------------- appendix --
def write_appendix(V: Values) -> list[Path]:
    APPENDIX.mkdir(parents=True, exist_ok=True)
    out = []
    wc = read_csv("working_concepts")
    a = wc[["concept_id", "gloss", "semantic_field", "n_families", "n_languages", "is_anchor",
            "in_robustness_set"]].rename(columns={
                "concept_id": "Concepticon gloss", "gloss": "English gloss used", "semantic_field": "Field",
                "n_families": "Families", "n_languages": "Languages", "is_anchor": "Named in proposal",
                "in_robustness_set": "In t = 100 set"})
    p = APPENDIX / "A_concepts.md"
    p.write_text(f"# Appendix A · The {len(wc)} working concepts\n\n"
                 "Source: `results/working_concepts.csv`. Families and languages are CLICS⁴ attestation counts "
                 "for the concept itself (not for any colexification).\n\n" + md_table(a) + "\n", encoding="utf-8")
    out.append(p)

    au = pd.read_csv(config.ENGLISH_AUDIT_PATH, keep_default_na=False)
    show = au[au["suggested_decision"].isin(["present", "unsure"])]
    cols = [c for c in ("gloss_a", "gloss_b", "n_families", "wordnet_route", "wordnet_evidence",
                        "suggested_decision", "decision", "notes") if c in show.columns]
    p = APPENDIX / "B_english_audit.md"
    p.write_text("# Appendix B · English colexification audit\n\n"
                 f"Source: `audit/english_colexification_manual_check.csv` ({len(au)} candidate pairs). "
                 f"Status: **{V.v['audit_status']}**. The table lists every pair suggested *present* or "
                 "*unsure*; the full sheet, including the suggested-absent pairs, is the CSV.\n\n"
                 + md_table(show[cols].head(200)) + "\n", encoding="utf-8")
    out.append(p)

    rob = read_csv("robustness")
    keep = rob[["hypothesis", "check", "label", "description", "statistic", "value", "ci_low", "ci_high",
                "p", "verdict", "note"]].copy()
    for c in ("value", "ci_low", "ci_high"):
        keep[c] = keep[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    keep["p"] = keep["p"].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    p = APPENDIX / "C_robustness.md"
    p.write_text("# Appendix C · Every robustness check\n\nSource: `results/robustness.csv`. A check "
                 "*survives* when it reproduces the primary analysis's conclusion (sign and significance at "
                 f"α = {config.ALPHA}), not when it reproduces the number.\n\n"
                 + md_table(keep) + "\n", encoding="utf-8")
    out.append(p)

    figs = sorted(config.FIGURES_DIR.glob("*.pdf"))
    rows = [{"File": f"`figures/{f.name}`", "PNG": f"`figures/{f.stem}.png`"} for f in figs]
    html = sorted(config.FIGURES_DIR.glob("*.html"))
    rows += [{"File": f"`figures/{f.name}`", "PNG": "interactive"} for f in html]
    p = APPENDIX / "D_figures.md"
    p.write_text("# Appendix D · Figure files\n\nEvery figure is written by `src/viz/figures.py` or "
                 "`src/viz/world_map.py` at 300 dpi in PDF and PNG.\n\n" + md_table(pd.DataFrame(rows)) + "\n",
                 encoding="utf-8")
    out.append(p)

    pend = []
    cov = read_csv("embedding_coverage")
    for r in cov[cov["status"].astype(str).str.startswith("PENDING")].itertuples():
        pend.append({"Item": f"Model condition `{r.condition}`", "What is missing": "embeddings and every "
                     "statistic that depends on them (H1, H2, H3, robustness)",
                     "Command": str(r.command), "Source": "results/embedding_coverage.csv"})
    rob = read_csv("robustness")
    for r in rob[rob["verdict"].astype(str).str.startswith("PENDING")].itertuples():
        if r.check == "condition_pending":
            continue                                    # already listed once per condition above
        pend.append({"Item": f"Robustness check `{r.check}`", "What is missing": str(r.description),
                     "Command": str(r.note), "Source": "results/robustness.csv"})
    h3 = read_csv("h3_typology")
    for r in h3[h3["status"].astype(str).str.startswith("PENDING")].itertuples():
        pend.append({"Item": f"H3 gradient vs {r.proxy} ({r.auc_kind})", "What is missing": str(r.status),
                     "Command": "python run.py analyse   (needs access to query.wikidata.org and *.wikipedia.org)",
                     "Source": "results/h3_typology.csv"})
    stab = read_json("community_stability")
    if str(stab.get("infomap", "")).startswith("PENDING"):
        pend.append({"Item": "Infomap communities", "What is missing": "the second community algorithm "
                     "(Louvain is computed; the reference partition does not depend on Infomap)",
                     "Command": str(stab["infomap"]).split(": ", 1)[-1], "Source": "results/community_stability.json"})
    drc = read_json("dr_choice")
    if str(drc.get("umap_status", "")).startswith("PENDING"):
        pend.append({"Item": "UMAP 2-D map", "What is missing": "one of the four dimensionality reductions; the "
                     f"display map ({drc['chosen']}) is chosen from those that ran and is re-chosen automatically",
                     "Command": "pip install umap-learn==0.5.6; python run.py network",
                     "Source": "results/dr_choice.json"})
    au = pd.read_csv(config.ENGLISH_AUDIT_PATH, keep_default_na=False)
    if (au["decision"].astype(str).str.strip() == "").any():
        pend.append({"Item": "Human audit of the English gold standard",
                     "What is missing": f"{int((au['decision'].astype(str).str.strip() == '').sum())} of {len(au)} "
                     "rows still carry Claude's suggested decision; every H2 number is labelled PROVISIONAL",
                     "Command": "edit audit/english_colexification_manual_check.csv, then python run.py analyse report",
                     "Source": "audit/english_colexification_manual_check.csv"})
    p = APPENDIX / "E_pending.md"
    p.write_text("# Appendix E · Everything marked PENDING USER RUN\n\nNothing in this report is estimated or "
                 "imputed. Each row below is a result that the build environment could not produce, with the "
                 "command that produces it.\n\n" + md_table(pd.DataFrame(pend)) + "\n", encoding="utf-8")
    out.append(p)
    return out


# ------------------------------------------------------------ references ---
LATEX = {r"\c{c}": "ç", r"\v{c}": "č", r"\v{C}": "Č", r"\v{s}": "š", r"\v{S}": "Š",
         r"\v{r}": "ř", r"\'{a}": "á", r"\'a": "á", r"\'{e}": "é", r"\'e": "é",
         r"\'{o}": "ó", r"\'o": "ó", r'\"{o}': "ö", r'\"o': "ö", r'\"{u}': "ü", r'\"u': "ü",
         r'\"{a}': "ä", r'\"a': "ä", r"\~{n}": "ñ", r"\~n": "ñ", r"\i": "i", r"\&": "&",
         r"\url": "", r"\v{z}": "ž"}


def delatex(t: str) -> str:
    for a, b in LATEX.items():
        t = t.replace(a.replace("\\", "\\"), b)
    return t.replace("{", "").replace("}", "").replace("--", "–").replace("\\", "").strip()


def parse_bib(path) -> list[dict]:
    """Minimal BibTeX reader: entry type, key and brace-delimited fields."""
    text = path.read_text(encoding="utf-8")
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{([^,]+),", text):
        start = m.end()
        depth, i = 1, text.index("{", m.start())
        i += 1
        while depth and i < len(text):                       # find the entry's closing brace
            depth += (text[i] == "{") - (text[i] == "}")
            i += 1
        body = text[start:i - 1]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            j = fm.end()
            if body[j] == "{":
                d, k = 1, j + 1
                while d and k < len(body):
                    d += (body[k] == "{") - (body[k] == "}")
                    k += 1
                fields[fm.group(1).lower()] = body[j + 1:k - 1]
            else:
                k = body.find(",", j)
                fields[fm.group(1).lower()] = body[j:k if k > 0 else len(body)].strip()
        entries.append({"type": m.group(1).lower(), "key": m.group(2).strip(), **fields})
    return entries


def apa_authors(raw: str) -> str:
    people = [a.strip() for a in re.split(r"\s+and\s+", delatex(raw)) if a.strip()]
    out = []
    for p in people:
        if p.startswith("{") or (p.isupper() and " " not in p):
            out.append(p)
            continue
        if "," in p:
            last, first = [x.strip() for x in p.split(",", 1)]
        else:
            bits = p.split()
            last, first = bits[-1], " ".join(bits[:-1])
        initials = " ".join(f"{b[0]}." for b in first.replace(".", " ").split() if b)
        out.append(f"{last}, {initials}".strip().rstrip(","))
    if len(out) == 1:
        return out[0]
    if len(out) == 2:
        return f"{out[0]}, & {out[1]}"
    return ", ".join(out[:-1]) + f", & {out[-1]}"


def apa_entry(e: dict) -> str:
    a = apa_authors(e.get("author", e.get("editor", "")))
    year = delatex(e.get("year", "n.d."))
    title = delatex(e.get("title", ""))
    bits = [f"{a} ({year}). {title}."]
    if e.get("journal"):
        j = f" *{delatex(e['journal'])}*"
        if e.get("volume"):
            j += f", *{delatex(e['volume'])}*"
            if e.get("number"):
                j += f"({delatex(e['number'])})"
        if e.get("pages"):
            j += f", {delatex(e['pages'])}"
        bits.append(j + ".")
    elif e.get("booktitle"):
        b = f" In *{delatex(e['booktitle'])}*"
        if e.get("pages"):
            b += f" (pp. {delatex(e['pages'])})"
        bits.append(b + ".")
    elif e.get("howpublished"):
        bits.append(f" {delatex(e['howpublished'])}.")
    if e.get("publisher"):
        bits.append(f" {delatex(e['publisher'])}.")
    if e.get("note"):
        bits.append(f" {delatex(e['note'])}.")
    if e.get("doi"):
        bits.append(f" https://doi.org/{delatex(e['doi'])}")
    return "".join(bits).replace("..", ".")


def references_block() -> str:
    entries = parse_bib(config.REPORT_DIR / "references.bib")
    lines = sorted((apa_entry(e) for e in entries), key=lambda t: t.lower())
    return "\n".join(f"{t}\n" for t in lines)


# --------------------------------------------------------------- README ----
README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"


def readme_summary(V: Values) -> str:
    """The README's results block, regenerated from results/ on every build."""
    v = V.v
    return f"""{README_START}
*Generated by `python -m src.report.build_report` on {v['generated_on']} from `results/`; do not edit by hand.*

**Conditions computed: {v['n_conditions_done']} of {v['n_conditions']}** ({v['conditions_done']}).
Pending: {v['conditions_pending']}. English gold standard: **{v['audit_status']}**.

| | Headline | Where |
|---|---|---|
| **H1 · alignment** | ρ = {v['h1_rho']} {v['h1_ci']}, {v['h1_p'].replace('*', '')} for {v['best_label']}; partial ρ = {v['h1_partial']} after controlling English frequency and semantic field | `results/h1_alignment.csv` |
| **H2 · English blind spot** | matched δ = {v['h2_delta']} {v['h2_ci']}, {v['h2_p'].replace('*', '')} ({v['h2_matched_present']} vs {v['h2_matched_absent']} pairs); unmatched δ = {v['h2_delta_unmatched']} — most of the apparent blind spot is confounding | `results/h2_blindspot.csv` |
| **H3 · typological unevenness** | all {v['h3_n_families']} qualifying families above chance (AUC {v['h3_auc_min']}–{v['h3_auc_max']}); resource gradient {v['h3_joshi_rarefied']} under the provisional proxy | `results/h3_family_fit.csv` |
| **Robustness** | {v['n_survive']} of {v['n_checks']} checks reproduce the primary conclusion, {v['n_pending_checks']} pending; of the {v['n_fail']} that do not, {v['n_h2_variants_supported']} are H2 variants that would have called the blind spot *significant* — the H2 null is boundary-sensitive (report §6) | `results/robustness_summary.csv` |

**The three figures to look at first**

1. `figures/fig11_synthesis.pdf` — the three hypotheses on one panel.
2. `figures/fig1_network.pdf` — the colexification network the whole study is built on
   (interactive: `figures/interactive_network.html`).
3. `figures/fig9_h3_families.pdf` — per-family fit, full and rarefied
   (world map: `figures/fig10_world_map.pdf`, interactive `figures/world_map.html`).

Read `report/report.md` for the full paper and `DECISIONS.md` for every methodological choice.
{README_END}"""


def update_readme(V: Values) -> bool:
    p = config.ROOT / "README.md"
    text = p.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        log.warning("README has no RESULTS markers; summary not written")
        return False
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    p.write_text(head + readme_summary(V) + tail, encoding="utf-8")
    return True


# -------------------------------------------------------------------- main --
def substitute(template: str, V: Values) -> tuple[str, set[str], set[str]]:
    """Fill every {{key}}; returns (text, keys used, keys with no value)."""
    used: set[str] = set()
    missing: set[str] = set()

    def sub(m: re.Match) -> str:
        key = m.group(1).strip()
        if key not in V.v:
            missing.add(key)
            return m.group(0)
        used.add(key)
        return V.v[key]

    return re.sub(r"\{\{([^}]+)\}\}", sub, template), used, missing


def main() -> None:
    log_run_header(log)
    V = build_values()
    text, used, missing = substitute(TEMPLATE.read_text(encoding="utf-8"), V)
    if missing:
        raise SystemExit(f"template placeholders with no value: {sorted(missing)}")
    OUT.write_text(text, encoding="utf-8")
    paths = write_appendix(V)
    update_readme(V)
    body = text.split("\n## References", 1)[0]
    words = len(re.findall(r"\b[\w'’-]+\b", re.sub(r"\|.*\|", "", body)))
    write_result_json({"generated": V.v["generated_on"], "git_commit": V.v["git_commit"],
                       "n_placeholders": len(V.v), "n_used": len(used),
                       "unused": sorted(set(V.v) - used), "word_count_body_excluding_tables_and_references": words,
                       "values": {k: {"value": V.v[k], "source": V.src[k]} for k in sorted(V.v)
                                  if "\n" not in V.v[k]}},
                      "report_values", producer=__name__)
    log.info("wrote %s (%d words of body text), %d appendix files, %d/%d placeholders used",
             OUT.name, words, len(paths), len(used), len(V.v))
    if set(V.v) - used:
        log.info("unused values: %s", ", ".join(sorted(set(V.v) - used)))


if __name__ == "__main__":
    main()
