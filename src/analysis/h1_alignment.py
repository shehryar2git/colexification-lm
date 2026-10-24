"""Phase 5 · H1: does model space mirror typological space? (Proposal Step 5)

Pre-specified analyses (DECISIONS.md D-19 to D-23, D-28, D-41, D-42):

1. **RSA** (confirmatory): Spearman rho between the upper triangles of the
   typological distance (weighted shortest path) and the model distance
   (1 - centred cosine) over the analysis concept set; Mantel test with 10,000
   joint row/column permutations, one-sided (rho > 0) as the confirmatory test;
   concept-level bootstrap 95% CI; Benjamini-Hochberg across conditions.
2. **Partial Mantel**: the same correlation controlling for English frequency
   difference and semantic-field difference (jointly and separately).
3. **Pair regression**: cos ~ z(log1p n_families) + z(zipf_min) + z(zipf_max) +
   z(same_field), standardised; MRQAP-DSP p-values, dyadic cluster-robust SEs,
   concept-bootstrap CIs.
4. **Cluster agreement** (exploratory): reference Louvain partition vs
   agglomerative (average linkage, model distance) and k-means (centred,
   L2-normalised vectors) clusters, k in CLUSTER_K_RANGE plus k_ref
   (pre-specified primary: agglomerative at k_ref); ARI/NMI with a 10,000-
   permutation null; with and without the large verb community (D-41).
5. **Procrustes** (exploratory): classical MDS of both spaces over the connected
   concepts; PROTEST m² with 10,000 permutations.

Conditions whose embeddings are missing are written as PENDING USER RUN rows.
Outputs: results/h1_alignment.csv, results/h1_regression.csv,
results/h1_cluster_agreement.csv, results/h1_matrix_correlations.csv,
data/processed/h1_procrustes_coords.parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as ss
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import procrustes
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import config
from src.analysis.common import (CONDITION_LABELS, analysis_concepts, available_conditions, control_matrices,
                                 load_similarity, pair_frame, stream, typological_distance)
from src.embeddings.store import load_embeddings
from src.network.communities import load_reference_partition
from src.network.typological_space import classical_mds, connected_mask
from src.utils.concept_order import load_concept_order
from src.utils.io import write_parquet, write_result_table
from src.utils.logging_utils import get_logger, log_run_header
from src.utils.seeds import set_all_seeds
from src.utils.stats import (bh_adjust, bootstrap_pairs, dyadic_cluster_se, mantel, mrqap_dsp, ols,
                             partial_mantel, protest, rsa_bootstrap_ci, upper)

log = get_logger(__name__, "phase5_h1")
TERMS = ["log_families", "zipf_min", "zipf_max", "same_field"]


def zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / v.std()


# ----------------------------------------------------------- regression ---
def regression(cond: str, sim: np.ndarray, pf: pd.DataFrame, zipf_sub: np.ndarray, field_same_sq: np.ndarray,
               n_perm: int, n_boot: int) -> pd.DataFrame:
    n = sim.shape[0]
    iu = np.triu_indices(n, 1)
    lf_sq = np.zeros((n, n))
    lf_sq[pf["i"].to_numpy(), pf["j"].to_numpy()] = np.log1p(pf["n_families"].to_numpy())
    lf_sq = lf_sq + lf_sq.T
    zmin = np.minimum(zipf_sub[:, None], zipf_sub[None, :])
    zmax = np.maximum(zipf_sub[:, None], zipf_sub[None, :])
    preds_sq = [lf_sq, zmin, zmax, field_same_sq]
    y = zscore(sim[iu])
    X = np.column_stack([zscore(p[iu]) for p in preds_sq])
    beta, resid, D = ols(y, X)
    se_dy = dyadic_cluster_se(D, resid, iu[0], iu[1], n)
    p_dy = 2 * ss.norm.sf(np.abs(beta / se_dy))
    # MRQAP on the standardised square matrices
    def sq(v: np.ndarray) -> np.ndarray:
        m = np.zeros((n, n))
        m[iu] = v
        return m + m.T
    p_qap = mrqap_dsp(sq(y), [sq(X[:, k]) for k in range(X.shape[1])], n_perm, stream("h1", cond, "mrqap"))
    # concept bootstrap of the standardised coefficients
    rg = stream("h1", cond, "reg_boot")
    draws = []
    for _ in range(n_boot):
        _, r, c = bootstrap_pairs(n, rg)
        yy = zscore(sim[r, c])
        XX = np.column_stack([zscore(p[r, c]) for p in preds_sq])
        b_, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(yy)), XX]), yy, rcond=None)
        draws.append(b_)
    draws = np.array(draws)
    rows = []
    for k, term in enumerate(TERMS):
        lo, hi = np.percentile(draws[:, k + 1], [2.5, 97.5])
        rows.append({"condition": cond, "term": term, "beta_std": beta[k + 1], "ci_low": lo, "ci_high": hi,
                     "se_dyadic": se_dy[k + 1], "p_dyadic": p_dy[k + 1], "p_mrqap": p_qap[k],
                     "n_pairs": len(y), "n_concepts": n})
    rows.append({"condition": cond, "term": "R2", "beta_std": 1 - resid @ resid / (y @ y), "ci_low": np.nan,
                 "ci_high": np.nan, "se_dyadic": np.nan, "p_dyadic": np.nan, "p_mrqap": np.nan,
                 "n_pairs": len(y), "n_concepts": n})
    return pd.DataFrame(rows)


# ------------------------------------------------------ cluster agreement --
def agglomerative_labels(dist: np.ndarray, k: int) -> np.ndarray:
    z = linkage(squareform(dist, checks=False), method="average")
    return fcluster(z, t=k, criterion="maxclust")


def kmeans_labels(vectors: np.ndarray, k: int, seed: int) -> np.ndarray:
    v = vectors - vectors.mean(axis=0)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(v)


def perm_p(ref: np.ndarray, lab: np.ndarray, obs: float, n_perm: int, rg: np.random.Generator) -> float:
    null = np.array([adjusted_rand_score(ref, rg.permutation(lab)) for _ in range(n_perm)])
    return float((np.sum(null >= obs) + 1) / (n_perm + 1))


def cluster_agreement(cond: str, dist: np.ndarray, vectors: np.ndarray | None, ref: np.ndarray,
                      exclude_mask: np.ndarray, n_perm: int) -> pd.DataFrame:
    rows = []
    k_ref = len(np.unique(ref))
    ks = sorted(set(config.CLUSTER_K_RANGE) | {k_ref})
    for subset, keep in (("all", np.ones(len(ref), bool)), ("without_largest_community", ~exclude_mask)):
        r = ref[keep]
        d = dist[np.ix_(keep, keep)]
        kref_sub = len(np.unique(r))
        for method in ("agglomerative", "kmeans"):
            if method == "kmeans" and vectors is None:
                continue
            for k in sorted(k_ for k_ in set(ks) | {kref_sub} if 2 <= k_ < int(keep.sum())):
                lab = agglomerative_labels(d, k) if method == "agglomerative" else kmeans_labels(vectors[keep], k, config.SEED)
                ari = adjusted_rand_score(r, lab)
                primary = method == "agglomerative" and k == kref_sub
                p = perm_p(r, lab, ari, n_perm if primary else min(n_perm, 2000),
                           stream("h1", cond, "clust", subset, method, str(k)))
                rows.append({"condition": cond, "subset": subset, "method": method, "k": k,
                             "k_is_reference": k == kref_sub, "primary": primary and subset == "all",
                             "ari": ari, "nmi": normalized_mutual_info_score(r, lab), "p_ari": p,
                             "n_perm": n_perm if primary else min(n_perm, 2000), "n_concepts": int(keep.sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- main --
def main() -> None:
    set_all_seeds()
    log_run_header(log)
    order = load_concept_order()
    concepts, idx = analysis_concepts(order)
    n = len(concepts)
    D_full = typological_distance(order)
    D = D_full[np.ix_(idx, idx)]
    ctrl = control_matrices(order)
    F = ctrl["freq_diff"][np.ix_(idx, idx)]
    Fld = ctrl["diff_field"][np.ix_(idx, idx)]
    zipf_sub = ctrl["zipf"][idx]
    pf = pair_frame(order, idx)
    ref_full = load_reference_partition(order)
    ref = ref_full[idx]
    largest = np.bincount(ref_full).argmax()
    conn = connected_mask(order)[idx]
    log.info("Analysis set: %d concepts, %d pairs; available conditions: %s", n, n * (n - 1) // 2,
             available_conditions())

    rows, reg_tabs, clu_tabs, coords = [], [], [], []
    xd, _ = classical_mds(D[np.ix_(conn, conn)])
    coords.append(pd.DataFrame({"concept_id": np.array(concepts)[conn], "space": "typology",
                                "x": xd[:, 0], "y": xd[:, 1]}))
    mats = {"typology": D}
    for cond in config.MODEL_CONDITIONS:
        if cond not in available_conditions():
            rows.append({"condition": cond, "label": CONDITION_LABELS[cond], "status": "PENDING USER RUN"})
            continue
        S = load_similarity(cond, order)[np.ix_(idx, idx)]
        M = 1.0 - S
        np.fill_diagonal(M, 0.0)
        mats[cond] = M
        mt = mantel(D, M, config.N_PERMUTATIONS, stream("h1", cond, "mantel"))
        lo, hi = rsa_bootstrap_ci(D, M, config.N_BOOTSTRAP, stream("h1", cond, "boot"))
        pm = partial_mantel(D, M, [F, Fld], config.N_PERMUTATIONS, stream("h1", cond, "pmantel"))
        pm_f = partial_mantel(D, M, [F], config.N_PERMUTATIONS, stream("h1", cond, "pmantel_f"))
        pm_s = partial_mantel(D, M, [Fld], config.N_PERMUTATIONS, stream("h1", cond, "pmantel_s"))
        xm = classical_mds(M[np.ix_(conn, conn)])[0]
        m2, p_pro = protest(xd, xm, config.N_PERMUTATIONS, stream("h1", cond, "protest"))
        xd_std, xm_aligned, _ = procrustes(xd, xm)
        if not coords[0].attrs.get("standardised"):
            coords[0] = coords[0].assign(x=xd_std[:, 0], y=xd_std[:, 1])
            coords[0].attrs["standardised"] = True
        coords.append(pd.DataFrame({"concept_id": np.array(concepts)[conn], "space": cond,
                                    "x": xm_aligned[:, 0], "y": xm_aligned[:, 1]}))
        try:
            vecs, _, _, _ = load_embeddings(cond)
            vecs = vecs[idx]
        except FileNotFoundError:
            vecs = None
        clu = cluster_agreement(cond, M, vecs, ref, ref == largest, config.N_PERMUTATIONS)
        clu_tabs.append(clu)
        prim = clu[clu["primary"]].iloc[0]
        reg_tabs.append(regression(cond, S, pf, zipf_sub, 1.0 - Fld, config.N_PERMUTATIONS, config.N_BOOTSTRAP))
        rows.append({"condition": cond, "label": CONDITION_LABELS[cond], "status": "done",
                     "n_concepts": n, "n_pairs": n * (n - 1) // 2,
                     "rho_raw": mt.r, "rho_ci_low": lo, "rho_ci_high": hi,
                     "p_mantel_one_sided": mt.p_greater, "p_mantel_two_sided": mt.p_two_sided,
                     "null_mean": mt.null_mean, "null_sd": mt.null_sd,
                     "rho_partial": pm.r, "p_partial_one_sided": pm.p_greater,
                     "rho_partial_freq": pm_f.r, "rho_partial_field": pm_s.r,
                     "rho_drop_raw_to_partial": mt.r - pm.r,
                     "ari_primary": prim["ari"], "nmi_primary": prim["nmi"], "p_ari_primary": prim["p_ari"],
                     "k_reference": int(prim["k"]),
                     "procrustes_m2": m2, "p_protest": p_pro, "n_concepts_procrustes": int(conn.sum())})
        log.info("%s: rho=%.3f [%.3f, %.3f], p(one-sided)=%.4g; partial rho=%.3f; ARI=%.3f; m2=%.3f",
                 cond, mt.r, lo, hi, mt.p_greater, pm.r, prim["ari"], m2)

    res = pd.DataFrame(rows)
    if "p_mantel_one_sided" in res:
        res["p_mantel_bh"] = bh_adjust(res["p_mantel_one_sided"].to_numpy(dtype=float))
        res["p_partial_bh"] = bh_adjust(res["p_partial_one_sided"].to_numpy(dtype=float))
        res["h1_supported"] = (res["p_mantel_bh"] < config.ALPHA) & (res["rho_raw"] > 0)
    res["bh_family"] = f"{len(available_conditions())} available conditions"
    write_result_table(res, "h1_alignment", producer=__name__,
                       extra={"n_permutations": config.N_PERMUTATIONS, "n_bootstrap": config.N_BOOTSTRAP,
                              "typological_metric": config.TYPOLOGICAL_DISTANCE_PRIMARY})
    if reg_tabs:
        write_result_table(pd.concat(reg_tabs), "h1_regression", producer=__name__)
        write_result_table(pd.concat(clu_tabs), "h1_cluster_agreement", producer=__name__)
    names = list(mats)
    corr = pd.DataFrame([[ss.spearmanr(upper(mats[a]), upper(mats[b])).statistic for b in names] for a in names],
                        index=names, columns=names)
    write_result_table(corr.reset_index().rename(columns={"index": "matrix"}), "h1_matrix_correlations",
                       producer=__name__)
    write_parquet(pd.concat(coords, ignore_index=True), config.PROCESSED_DIR / "h1_procrustes_coords.parquet")


if __name__ == "__main__":
    main()
