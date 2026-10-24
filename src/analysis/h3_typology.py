"""H3 — Is the model fit uneven across language families? (Proposal Step 7)

Pre-specified design (DECISIONS.md D-09 amendment, D-25, D-26, D-27, D-28, D-56, D-57):

* Statistic per family x condition: the pair-level **AUC** — the probability that a
  random within-family colexified pair (>= 2 languages of the family) has higher
  model similarity than a random non-colexified pair, among analysis-set pairs whose
  two concepts the family attests. The Spearman correlation between model similarity
  and the stored within-family colexification rate (``family_<slug>.npz``) is
  reported alongside, as the proposal asks.
* Inference: concept-label permutation within the family's attested concepts
  (rows and columns permuted jointly; 10,000 permutations, one-sided, AUC > 0.5),
  concept-bootstrap 95% CI, Benjamini-Hochberg across conditions x families.
* Confound control (D-25): every family is rarefied to 20 languages, 200 times;
  the confirmatory H3 test uses the rarefied mean AUC.
* Confirmatory H3 test (D-28, D-57): per condition, the Spearman correlation across
  families between rarefied AUC and the resource proxy (primary: log10 Wikipedia
  articles of the family's largest edition; secondary: Joshi et al. 2020 class),
  with an exact (n <= 9) or Monte-Carlo permutation p, BH across conditions.
  Reported as descriptive given the small number of families (D-26).
* Exploratory: per-language AUC for the world map; macroarea summary.

Outputs: results/h3_family_fit.csv, results/h3_typology.csv,
results/h3_language_agreement.csv, results/h3_macroarea.csv, results/h3_design.json,
data/processed/h3_rarefaction.parquet.
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pandas as pd
from scipy import stats as ss
from scipy.stats import rankdata

import config
from src.analysis.common import (CONDITION_LABELS, analysis_concepts, available_conditions,
                                 load_similarity, stream)
from src.network.build_network import network_for, slug
from src.utils.concept_order import load_concept_order
from src.utils.io import read_parquet, write_parquet, write_result_json, write_result_table
from src.utils.logging_utils import get_logger, log_run_header
from src.utils.matrices import load_matrix
from src.utils.seeds import set_all_seeds
from src.utils.stats import bh_adjust, bootstrap_ci

log = get_logger(__name__, "h3_typology")


# ------------------------------------------------------------------ AUC core --
def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score_pos > score_neg) + 0.5 P(tie). NaN if a class is empty."""
    labels = np.asarray(labels, dtype=bool)
    n1 = int(labels.sum())
    n0 = labels.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def square_ranks(sim: np.ndarray) -> np.ndarray:
    """Symmetric matrix of average ranks of the upper-triangle entries (diag = 0)."""
    n = sim.shape[0]
    iu = np.triu_indices(n, 1)
    r = np.zeros((n, n))
    r[iu] = rankdata(sim[iu])
    return r + r.T


def permutation_auc_spearman(sim: np.ndarray, rate: np.ndarray, n_perm: int,
                             rng: np.random.Generator) -> dict[str, float]:
    """AUC (rate > 0 = positive) and Spearman(sim, rate) with a joint concept-label
    permutation null. ``sim`` and ``rate`` are square over the same concepts.

    Permuting concepts maps the full set of pairs onto itself, so the ranks of the
    similarity values are permuted rather than recomputed: the AUC under a
    permutation pi is (sum_{(i,j) positive} R[pi_i, pi_j] - n1(n1+1)/2) / (n1 n0),
    and the Spearman numerator only needs the pairs with non-zero rate (all zero-rate
    pairs share one tied rank). Both are exact, not approximations.
    """
    n = sim.shape[0]
    iu = np.triu_indices(n, 1)
    R = square_ranks(sim)
    rv = rate[iu]
    pos = rv > 0
    n1, n_pairs = int(pos.sum()), rv.size
    n0 = n_pairs - n1
    pi, pj = iu[0][pos], iu[1][pos]
    q = rankdata(rv)
    qc = q - q.mean()
    rc_norm = math.sqrt(((R[iu] - R[iu].mean()) ** 2).sum())
    q_norm = math.sqrt((qc ** 2).sum())
    qc_pos, qc_zero = qc[pos], (qc[~pos][0] if n0 else 0.0)
    r_total = R[iu].sum()
    r_mean = R[iu].mean()

    def stats_for(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rp = R[rows, cols]                                   # (..., n1) ranks of positive pairs
        a = (rp.sum(axis=-1) - n1 * (n1 + 1) / 2) / (n1 * n0)
        # sum_all Rc*qc = sum_pos Rc*(qc_pos - qc_zero) + qc_zero*sum_all Rc ; sum_all Rc = 0
        num = ((rp - r_mean) * (qc_pos - qc_zero)).sum(axis=-1)
        return a, num / (rc_norm * q_norm)

    a_obs, s_obs = stats_for(pi, pj)
    assert abs(r_total / n_pairs - r_mean) < 1e-9
    ge_a = ge_s = 0
    done = 0
    chunk = max(1, int(4e6 // max(n1, 1)))
    while done < n_perm:
        m = min(chunk, n_perm - done)
        perms = np.argsort(rng.random((m, n)), axis=1)
        a_p, s_p = stats_for(perms[:, pi], perms[:, pj])
        ge_a += int((a_p >= a_obs - 1e-12).sum())
        ge_s += int((s_p >= s_obs - 1e-12).sum())
        done += m
    return {"auc": float(a_obs), "spearman": float(s_obs), "n_pos": n1, "n_neg": n0,
            "p_auc_one_sided": (ge_a + 1) / (n_perm + 1), "p_spearman_one_sided": (ge_s + 1) / (n_perm + 1)}


def auc_bootstrap_ci(sim: np.ndarray, pos: np.ndarray, n_boot: int, rng: np.random.Generator
                     ) -> tuple[float, float]:
    """Concept-bootstrap 95% CI of the AUC (pairs of resampled concepts)."""
    lo, hi, _ = bootstrap_ci(lambda r, c: auc(sim[r, c], pos[r, c]), sim.shape[0], n_boot, rng)
    return lo, hi


# ------------------------------------------------------------ family data --
class FamilyData:
    """Language-level attestation and colexification incidence for one family,
    restricted to the analysis concept set (indices into ``concepts``)."""

    def __init__(self, family_gc: str, languages: list[str], attest: np.ndarray,
                 colex_lang: np.ndarray, colex_i: np.ndarray, colex_j: np.ndarray):
        self.family_gc = family_gc
        self.languages = languages
        self.attest = attest                  # (n_lang, n_concepts) bool
        self.colex_lang = colex_lang          # language index of each (language, pair) instance
        self.colex_i, self.colex_j = colex_i, colex_j   # concept indices, i < j

    def subsample(self, lang_idx: np.ndarray, n_concepts: int, min_languages: int
                  ) -> tuple[np.ndarray, np.ndarray]:
        """(attested concept mask, square bool matrix of within-sample edges)."""
        att = self.attest[lang_idx].any(axis=0)
        keep = np.isin(self.colex_lang, lang_idx)
        cnt = np.zeros((n_concepts, n_concepts), dtype=np.int32)
        np.add.at(cnt, (self.colex_i[keep], self.colex_j[keep]), 1)
        edge = cnt >= min_languages
        return att, edge | edge.T


def family_data(net: dict, concepts: list[str], families: list[str]) -> dict[str, FamilyData]:
    ci = {c: k for k, c in enumerate(concepts)}
    units = net["units"]
    fu = net["forms_u"]
    fu = fu[fu["concept_id"].isin(ci)]
    inst = net["inst"]
    inst = inst[inst["concept_a"].isin(ci) & inst["concept_b"].isin(ci)].merge(
        units[["variety_id", "language_gc", "family_gc"]], on="variety_id")
    inst = inst.drop_duplicates(["language_gc", "concept_a", "concept_b"])
    out = {}
    for fgc in families:
        langs = sorted(units.loc[units["family_gc"] == fgc, "language_gc"].unique())
        li = {lg: k for k, lg in enumerate(langs)}
        a = np.zeros((len(langs), len(concepts)), dtype=bool)
        f = fu[fu["family_gc"] == fgc]
        a[f["language_gc"].map(li).to_numpy(), f["concept_id"].map(ci).to_numpy()] = True
        x = inst[inst["family_gc"] == fgc]
        i = x["concept_a"].map(ci).to_numpy()
        j = x["concept_b"].map(ci).to_numpy()
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        out[fgc] = FamilyData(fgc, langs, a, x["language_gc"].map(li).to_numpy(), lo, hi)
    return out


def rarefied_auc(fd: FamilyData, sims: dict[str, np.ndarray], n_lang: int, repeats: int,
                 rng: np.random.Generator) -> pd.DataFrame:
    """AUC for every condition in ``repeats`` subsamples of ``n_lang`` languages."""
    n = fd.attest.shape[1]
    iu = np.triu_indices(n, 1)
    rows = []
    for r in range(repeats):
        idx = np.sort(rng.choice(len(fd.languages), size=n_lang, replace=False))
        att, edge = fd.subsample(idx, n, config.EDGE_MIN_FAMILIES)
        elig = att[iu[0]] & att[iu[1]]
        lab = edge[iu][elig]
        for cond, s in sims.items():
            rows.append({"family_gc": fd.family_gc, "repeat": r, "condition": cond,
                         "n_concepts": int(att.sum()), "n_pos": int(lab.sum()),
                         "auc": auc(s[iu][elig], lab)})
    return pd.DataFrame(rows)


# ------------------------------------------------------ language agreement --
def language_agreement(net: dict, concepts: list[str], sims: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-language AUC: do the language's own colexified pairs have higher model
    similarity than its non-colexified pairs (both concepts attested in it)?"""
    ci = {c: k for k, c in enumerate(concepts)}
    n = len(concepts)
    iu = np.triu_indices(n, 1)
    units = net["units"]
    fu = net["forms_u"]
    fu = fu[fu["concept_id"].isin(ci)]
    inst = net["inst"]
    inst = inst[inst["concept_a"].isin(ci) & inst["concept_b"].isin(ci)].merge(
        units[["variety_id", "language_gc"]], on="variety_id")
    att_by = {lg: np.array(sorted({ci[c] for c in g})) for lg, g in fu.groupby("language_gc")["concept_id"]}
    col_by = {lg: g for lg, g in inst.groupby("language_gc")}
    uppers = {c: s[iu] for c, s in sims.items()}
    flat = lambda i, j: i * n + j  # noqa: E731
    pair_index = np.full(n * n, -1, dtype=np.int64)
    pair_index[flat(*iu)] = np.arange(iu[0].size)
    rows = []
    for lg, att in att_by.items():
        mask = np.zeros(n, dtype=bool)
        mask[att] = True
        elig = mask[iu[0]] & mask[iu[1]]
        lab = np.zeros(iu[0].size, dtype=bool)
        if lg in col_by:
            g = col_by[lg]
            a = g["concept_a"].map(ci).to_numpy()
            b = g["concept_b"].map(ci).to_numpy()
            lab[pair_index[flat(np.minimum(a, b), np.maximum(a, b))]] = True
        rec = {"language_gc": lg, "n_concepts": int(mask.sum()), "n_pairs": int(elig.sum()),
               "n_colex_pairs": int(lab[elig].sum())}
        for c, u in uppers.items():
            rec[f"auc_{c}"] = auc(u[elig], lab[elig]) if rec["n_colex_pairs"] else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------- family gradient --
def permutation_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                         exact_max_n: int = config.H3_EXACT_PERMUTATION_MAX_N,
                         n_mc: int = config.H3_MC_PERMUTATIONS) -> dict[str, float]:
    """Spearman rho across families; one-sided (rho > 0) and two-sided permutation p.
    Exact enumeration for n <= exact_max_n, else Monte-Carlo."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 4:
        return {"n_families": n, "rho": float("nan"), "p_one_sided": float("nan"),
                "p_two_sided": float("nan"), "p_method": "too few families"}
    rx, ry = rankdata(x), rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if n <= exact_max_n:
        null = np.array([np.corrcoef(rx, ry[list(p)])[0, 1] for p in itertools.permutations(range(n))])
        method = f"exact ({math.factorial(n)} permutations)"
        p1 = float(np.mean(null >= rho - 1e-12))
        p2 = float(np.mean(np.abs(null) >= abs(rho) - 1e-12))
    else:
        perms = np.argsort(rng.random((n_mc, n)), axis=1)
        ryc = ry - ry.mean()
        rxc = rx - rx.mean()
        null = (ryc[perms] @ rxc) / math.sqrt((rxc ** 2).sum() * (ryc ** 2).sum())
        method = f"Monte-Carlo ({n_mc} permutations)"
        p1 = (int((null >= rho - 1e-12).sum()) + 1) / (n_mc + 1)
        p2 = (int((np.abs(null) >= abs(rho) - 1e-12).sum()) + 1) / (n_mc + 1)
    return {"n_families": n, "rho": rho, "p_one_sided": p1, "p_two_sided": p2, "p_method": method}


def gradient_table(fit: pd.DataFrame, proxies: pd.DataFrame, conditions: list[str],
                   rng: np.random.Generator) -> pd.DataFrame:
    """One row per condition x proxy x AUC kind (rarefied = confirmatory, full = descriptive)
    plus the sampling-density check (full AUC vs number of languages)."""
    rows = []
    proxy_cols = [("wikipedia_log10_articles", "Wikipedia (log10 articles, largest edition)"),
                  ("joshi_class", "Joshi et al. (2020) class, max over family")]
    for cond in conditions:
        pcols = ["family_gc"] + [c for c, _ in proxy_cols if c in proxies]
        f = fit[fit["condition"] == cond].merge(proxies[pcols], on="family_gc", how="left")
        for col, label in proxy_cols:
            status = proxies.attrs.get(f"{col}_status", "PENDING USER RUN")
            for kind, auc_col in (("rarefied", "auc_rarefied_mean"), ("full", "auc")):
                if col not in f or f[col].notna().sum() < 4:
                    rows.append({"condition": cond, "proxy": col, "proxy_label": label, "auc_kind": kind,
                                 "status": status, "n_families": int(f[col].notna().sum()) if col in f else 0})
                    continue
                r = permutation_spearman(f[auc_col].to_numpy(float), f[col].to_numpy(float), rng)
                rows.append({"condition": cond, "proxy": col, "proxy_label": label, "auc_kind": kind,
                             "status": status, **r})
        r = permutation_spearman(f["auc"].to_numpy(float), np.log10(f["n_languages"].to_numpy(float)), rng)
        rows.append({"condition": cond, "proxy": "log10_n_languages_clics", "auc_kind": "full",
                     "proxy_label": "sampling density (log10 CLICS languages)", "status": "done", **r})
        r = permutation_spearman(f["auc_rarefied_mean"].to_numpy(float),
                                 np.log10(f["n_languages"].to_numpy(float)), rng)
        rows.append({"condition": cond, "proxy": "log10_n_languages_clics", "auc_kind": "rarefied",
                     "proxy_label": "sampling density (log10 CLICS languages)", "status": "done", **r})
    out = pd.DataFrame(rows)
    conf = (out["auc_kind"] == "rarefied") & (out["proxy"] == "wikipedia_log10_articles")
    out["confirmatory"] = conf
    out["p_bh"] = np.nan
    if "p_one_sided" in out:
        for proxy in out["proxy"].unique():
            m = (out["proxy"] == proxy) & (out["auc_kind"] == "rarefied")
            out.loc[m, "p_bh"] = bh_adjust(out.loc[m, "p_one_sided"].to_numpy(float))
    out["h3_supported"] = conf & (out.get("rho", np.nan) > 0) & (out["p_bh"] < config.ALPHA)
    return out


# ----------------------------------------------------------------- main ---
def main() -> None:
    set_all_seeds()
    log_run_header(log)
    order = load_concept_order()
    concepts, idx = analysis_concepts(order, write=False)
    conds = available_conditions()
    if not conds:
        raise SystemExit("no model similarity matrices: run `python run.py embeddings` first")
    sims = {c: load_similarity(c, order)[np.ix_(idx, idx)] for c in conds}
    pending = [c for c in config.MODEL_CONDITIONS if c not in conds]
    log.info("H3: %d concepts, conditions %s (pending: %s)", len(concepts), conds, pending)

    fam = pd.read_csv(config.RESULTS_DIR / "family_subnetworks.csv")
    fam = fam[fam["qualifies"]].sort_values("n_languages", ascending=False)
    forms = read_parquet(config.PROCESSED_DIR / "forms.parquet")
    languages = read_parquet(config.PROCESSED_DIR / "languages.parquet")
    net = network_for(forms, languages, order)
    fdata = family_data(net, concepts, fam["family_gc"].tolist())

    # ---- full-family fit with permutation and bootstrap --------------------------
    rows, draws = [], []
    for _, fr in fam.iterrows():
        fgc, name = fr["family_gc"], fr["family_name"]
        dist = load_matrix(config.MATRIX_DIR / f"family_{slug(name)}.npz", order).values[np.ix_(idx, idx)]
        att = np.isfinite(np.diag(dist))
        rate = 1.0 - dist[np.ix_(att, att)]
        np.fill_diagonal(rate, 0.0)
        # consistency: the stored matrix's edges equal the recount from language incidence
        fd = fdata[fgc]
        _, edge = fd.subsample(np.arange(len(fd.languages)), len(concepts), config.EDGE_MIN_FAMILIES)
        stored_edges = rate > 0
        if not np.array_equal(edge[np.ix_(att, att)], stored_edges):
            raise RuntimeError(f"{name}: family matrix and language incidence disagree")
        for cond in conds:
            s = sims[cond][np.ix_(att, att)]
            res = permutation_auc_spearman(s, rate, config.N_PERMUTATIONS, stream("h3", fgc, cond, "perm"))
            lo, hi = auc_bootstrap_ci(s, stored_edges, config.N_BOOTSTRAP, stream("h3", fgc, cond, "boot"))
            rows.append({"family_gc": fgc, "family_name": name, "condition": cond,
                         "label": CONDITION_LABELS[cond], "n_languages": int(fr["n_languages"]),
                         "named_in_proposal": bool(fr["named_in_proposal"]), "n_concepts": int(att.sum()),
                         "n_pairs": res["n_pos"] + res["n_neg"], **res, "auc_ci_low": lo, "auc_ci_high": hi})
        rd = rarefied_auc(fd, sims, config.FAMILY_RAREFACTION_N_LANGUAGES, config.FAMILY_RAREFACTION_REPEATS,
                          stream("h3", fgc, "rarefaction"))
        draws.append(rd)
        log.info("%-26s %3d languages: %s", name, fr["n_languages"],
                 ", ".join(f"{r['condition']} AUC {r['auc']:.3f}" for r in rows if r["family_gc"] == fgc))
    fit = pd.DataFrame(rows)
    rare = pd.concat(draws, ignore_index=True)
    write_parquet(rare, config.PROCESSED_DIR / "h3_rarefaction.parquet")
    summ = rare.groupby(["family_gc", "condition"]).agg(
        auc_rarefied_mean=("auc", "mean"), auc_rarefied_low=("auc", lambda v: np.nanpercentile(v, 2.5)),
        auc_rarefied_high=("auc", lambda v: np.nanpercentile(v, 97.5)),
        n_pos_rarefied_mean=("n_pos", "mean"), n_concepts_rarefied_mean=("n_concepts", "mean")).reset_index()
    fit = fit.merge(summ, on=["family_gc", "condition"], how="left")
    fit["p_auc_bh"] = bh_adjust(fit["p_auc_one_sided"])        # D-27: conditions x families
    fit["p_spearman_bh"] = bh_adjust(fit["p_spearman_one_sided"])
    fit["bh_family"] = f"{len(conds)} conditions x {fam.shape[0]} families"
    write_result_table(fit, "h3_family_fit", producer=__name__)

    # ---- family-level gradient vs resource proxies ---------------------------------
    proxy_path = config.RESULTS_DIR / "resource_proxies.csv"
    if proxy_path.exists():
        proxies = pd.read_csv(proxy_path)
        meta = json.loads((config.RESULTS_DIR / "resource_proxies_status.json").read_text())
        for k, v in meta.items():
            proxies.attrs[k] = v
    else:
        proxies = pd.DataFrame({"family_gc": fam["family_gc"]})
    grad = gradient_table(fit, proxies, conds, stream("h3", "gradient"))
    grad.insert(1, "label", grad["condition"].map(CONDITION_LABELS))
    write_result_table(grad, "h3_typology", producer=__name__)

    # ---- exploratory: language-level agreement and macroareas --------------------
    lang = language_agreement(net, concepts, sims)
    meta = languages[languages["include"]].drop_duplicates("language_gc").set_index("language_gc")
    lang = lang.join(meta[["family_gc", "family_name", "macroarea", "latitude", "longitude"]], on="language_gc")
    names = languages[languages["include"]].groupby("language_gc")["variety_name"].first()
    lang.insert(1, "language_name", lang["language_gc"].map(names))
    lang["mapped"] = lang["n_colex_pairs"] >= config.H3_MIN_LANGUAGE_COLEX
    write_result_table(lang, "h3_language_agreement", producer=__name__)
    m = lang[lang["mapped"]]
    macro = (m.groupby("macroarea")
              .agg(n_languages=("language_gc", "size"), n_families=("family_gc", "nunique"),
                   **{f"median_auc_{c}": (f"auc_{c}", "median") for c in conds})
              .reset_index())
    write_result_table(macro, "h3_macroarea", producer=__name__)

    write_result_json({
        "statistic": "pair-level AUC (D-09 amendment); Spearman with family_<slug>.npz alongside",
        "n_concepts_analysis_set": len(concepts), "conditions": conds, "pending_conditions": pending,
        "families": fam[["family_gc", "family_name", "n_languages"]].to_dict("records"),
        "n_permutations": config.N_PERMUTATIONS, "n_bootstrap": config.N_BOOTSTRAP,
        "rarefaction": {"n_languages": config.FAMILY_RAREFACTION_N_LANGUAGES,
                        "repeats": config.FAMILY_RAREFACTION_REPEATS},
        "edge_rule_within_family": f">= {config.EDGE_MIN_FAMILIES} languages",
        "language_map_rule": f">= {config.H3_MIN_LANGUAGE_COLEX} colexified analysis pairs",
        "n_languages_total": int(lang.shape[0]), "n_languages_mapped": int(lang["mapped"].sum()),
        "proxy_status": {k: v for k, v in proxies.attrs.items()},
        "confirmatory": "rarefied AUC vs Wikipedia proxy, one-sided Spearman, BH across conditions (D-57)",
    }, "h3_design", producer=__name__)
    log.info("H3 written: %d family x condition rows, %d languages mapped", fit.shape[0], int(lang["mapped"].sum()))


if __name__ == "__main__":
    main()
