"""Phase 3 · Two-dimensional maps of the typological space (Proposal Step 3).

All methods embed the **same** primary typological distance matrix D, so their
quality scores are comparable (DECISIONS.md D-42):

* ``pca``            classical (Torgerson) MDS = PCA of the double-centred D² (Gower 1966)
* ``mds_metric``     SMACOF stress majorisation (de Leeuw 1977), init = classical MDS
* ``mds_nonmetric``  SMACOF with Kruskal's monotone regression of disparities (Kruskal 1964)
* ``umap``           UMAP on the precomputed metric (when ``umap-learn`` is installed)

SMACOF is implemented here (not via scikit-learn) because scikit-learn changed its
MDS API between the pinned 1.5 and later releases; the implementation is short,
deterministic and unit-tested against a known configuration.

Quality (all computed against D): Kruskal stress-1 after optimal scaling, the
Shepard rank correlation between D and 2-D distances, and trustworthiness
(k = TRUSTWORTHINESS_K; Venna & Kaski 2001). Pre-specified rule: the display map
is the method with the highest trustworthiness (ties -> lower stress).

Exploratory replication of the Oct 10 practice session: PCA of the binary
concept x lexeme matrix (concept x language-specific word form), the analogue of
the session's synset x lemma matrix; variance explained and agreement with D reported.

Outputs: data/processed/typological_2d.parquet, results/dr_quality.csv,
results/dr_choice.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.manifold import trustworthiness

import config
from src.utils.concept_order import load_concept_order
from src.utils.io import write_parquet, write_result_json, write_result_table
from src.utils.logging_utils import get_logger, log_run_header
from src.utils.matrices import load_matrix
from src.utils.seeds import set_all_seeds

log = get_logger(__name__, "phase3_typological_space")


# ------------------------------------------------------------- embeddings ---
def classical_mds(d: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Torgerson MDS. Returns coordinates and the share of positive eigenvalue
    mass captured by each of the k axes. Sign of each axis fixed (largest |loading| positive)."""
    n = d.shape[0]
    j = np.eye(n) - 1.0 / n
    b = -0.5 * j @ (d ** 2) @ j
    w, v = np.linalg.eigh(b)
    idx = np.argsort(w)[::-1]
    w, v = w[idx], v[:, idx]
    pos = np.clip(w, 0, None)
    x = v[:, :k] * np.sqrt(pos[:k])
    for c in range(k):                                   # deterministic orientation
        if x[np.argmax(np.abs(x[:, c])), c] < 0:
            x[:, c] *= -1
    return x, pos[:k] / pos.sum()


def _pdist(x: np.ndarray) -> np.ndarray:
    return np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))


def smacof(d: np.ndarray, x0: np.ndarray, metric: bool = True, max_iter: int = 5000,
           eps: float = 1e-7) -> tuple[np.ndarray, float, int]:
    """Unweighted SMACOF. Returns (X, raw stress / sum of squared disparities, iterations)."""
    n = d.shape[0]
    iu = np.triu_indices(n, 1)
    x = x0.copy()
    target = d.copy()
    prev = np.inf
    stress = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        dist = _pdist(x)
        if not metric:
            iso = IsotonicRegression(increasing=True)
            disp = iso.fit_transform(d[iu], dist[iu])
            disp *= np.sqrt(len(disp) / (disp ** 2).sum())       # Kruskal normalisation
            target = np.zeros_like(d)
            target[iu] = disp
            target = target + target.T
        stress = float(((target[iu] - dist[iu]) ** 2).sum() / (target[iu] ** 2).sum())
        if prev - stress < eps * max(prev, 1e-12) and it > 1:
            break
        prev = stress
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(dist > 0, target / dist, 0.0)
        bmat = -ratio
        np.fill_diagonal(bmat, 0.0)
        np.fill_diagonal(bmat, -bmat.sum(axis=1))
        x = bmat @ x / n                                          # Guttman transform
    return x, stress, it


def umap_available() -> bool:
    try:
        import umap  # noqa: F401
    except ImportError:
        return False
    return True


def umap_embed(d: np.ndarray) -> np.ndarray:
    import umap

    reducer = umap.UMAP(n_components=2, metric="precomputed", n_neighbors=config.UMAP_N_NEIGHBORS,
                        min_dist=config.UMAP_MIN_DIST, random_state=config.SEED, n_jobs=config.N_JOBS,
                        init="random")   # spectral init is unavailable for precomputed metrics in some versions
    return reducer.fit_transform(d)


# ---------------------------------------------------------------- quality ---
def quality(d: np.ndarray, x: np.ndarray) -> dict[str, float]:
    iu = np.triu_indices(d.shape[0], 1)
    e = _pdist(x)[iu]
    t = d[iu]
    b = (t * e).sum() / (e * e).sum()                 # optimal scaling of the map onto D
    stress1 = float(np.sqrt(((t - b * e) ** 2).sum() / (t ** 2).sum()))
    return {
        "stress1": stress1,
        "shepard_spearman": float(spearmanr(t, e).statistic),
        "trustworthiness": float(trustworthiness(d, x, n_neighbors=config.TRUSTWORTHINESS_K,
                                                 metric="precomputed")),
    }


def choose(q: pd.DataFrame) -> str:
    ok = q[(q["status"] == "done") & ~q["note"].fillna("").str.contains("DEGENERATE")]
    ok = ok.sort_values(["trustworthiness", "stress1"], ascending=[False, True])
    return str(ok["method"].iloc[0])


def concept_lexeme_pca(order: list[str]) -> dict[str, object]:
    """Oct 10 practice replication: PCA of the binary concept x lexeme matrix
    (a lexeme = one language-specific word form shared by >= 2 working concepts),
    the CLICS-scale analogue of the session's synset x lemma matrix. Columns are
    weighted equally, so large, well-sampled families dominate — one reason the
    project's primary analysis uses the family-weighted network instead."""
    z = np.load(config.MATRIX_DIR / "concept_lexeme.npz")
    assert [str(c) for c in z["ids"]] == order
    pca = PCA(n_components=2, random_state=config.SEED, svd_solver="full")
    xy = pca.fit_transform(z["values"].astype(float))
    for c in range(2):
        if xy[np.argmax(np.abs(xy[:, c])), c] < 0:
            xy[:, c] *= -1
    return {"coords": xy, "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "n_lexemes": int(z["values"].shape[1])}


# ------------------------------------------------------------------- main ---
def connected_mask(order: list[str]) -> np.ndarray:
    """Concepts in the largest connected component of the network. Isolated concepts
    sit at the conventional distance max + 1 from everything; that convention makes
    a rank correlation computable but is not geometry, and it dominates any 2-D map
    (D-42), so maps embed the connected component only."""
    import networkx as nx

    g = nx.read_graphml(config.PROCESSED_DIR / "network.graphml")
    core = max(nx.connected_components(g), key=len)
    return np.array([c in core for c in order])


def main() -> None:
    set_all_seeds()
    log_run_header(log)
    order_all = load_concept_order()
    d_all = load_matrix(config.MATRIX_DIR / "typological_distance.npz", order_all).values
    keep = connected_mask(order_all)
    order = [c for c, k in zip(order_all, keep) if k]
    d = d_all[np.ix_(keep, keep)]
    omitted = [c for c, k in zip(order_all, keep) if not k]
    log.info("Embedding the connected component: %d concepts (%d isolated omitted)", len(order), len(omitted))

    coords: dict[str, np.ndarray] = {}
    rows = []
    x_cmds, ev = classical_mds(d)
    coords["pca"] = x_cmds
    rows.append({"method": "pca", "status": "done", "iterations": 0, "note": f"axes explain {ev[0]:.3f}, {ev[1]:.3f}",
                 **quality(d, x_cmds)})
    for name, metric in (("mds_metric", True), ("mds_nonmetric", False)):
        x, raw, it = smacof(d, x_cmds, metric=metric)
        coords[name] = x
        q_ = quality(d, x)
        note = f"final raw stress {raw:.4f}; {'converged' if it < 5000 else 'NOT converged (5000 iterations)'}"
        if not metric and raw < 0.01 and q_["shepard_spearman"] < 0.5:
            note += ("; DEGENERATE: near-zero stress against its own disparities but poor rank agreement with D "
                     "(tied block of disconnected pairs; the degeneracy described by Kruskal 1964)")
        rows.append({"method": name, "status": "done", "iterations": it, "note": note, **q_})
    if umap_available():
        x = umap_embed(d)
        coords["umap"] = x
        rows.append({"method": "umap", "status": "done", "iterations": 0, "note": "", **quality(d, x)})
    else:
        rows.append({"method": "umap", "status": "PENDING USER RUN", "iterations": 0,
                     "note": "pip install umap-learn==0.5.6; python run.py network",
                     "stress1": np.nan, "shepard_spearman": np.nan, "trustworthiness": np.nan})
        log.warning("umap-learn not installed: UMAP map is PENDING USER RUN")

    q = pd.DataFrame(rows)
    chosen = choose(q)
    q["chosen"] = q["method"] == chosen
    write_result_table(q, "dr_quality", producer=__name__)

    cl = concept_lexeme_pca(order_all)
    cl["coords"] = cl["coords"][keep]
    iu = np.triu_indices(len(order), 1)
    cl_q = quality(d, cl["coords"])
    long = pd.concat([pd.DataFrame({"concept_id": order, "method": m, "x": xy[:, 0], "y": xy[:, 1]})
                      for m, xy in {**coords, "pca_concept_lexeme": cl["coords"]}.items()], ignore_index=True)
    write_parquet(long, config.PROCESSED_DIR / "typological_2d.parquet")
    write_result_json({
        "rule": "highest trustworthiness (k=%d), ties -> lower stress-1" % config.TRUSTWORTHINESS_K,
        "chosen": chosen,
        "umap_status": "done" if "umap" in coords else "PENDING USER RUN",
        "note": ("If UMAP is installed the choice is re-made on the full set of methods; "
                 "results/dr_quality.csv is authoritative."),
        "concept_lexeme_pca": {"n_lexemes": cl["n_lexemes"],
                               "explained_variance_ratio": cl["explained_variance_ratio"],
                               "agreement_with_typological_distance": cl_q},
        "n_concepts_embedded": len(order),
        "n_pairs": int(len(iu[0])),
        "isolated_concepts_omitted": omitted,
    }, "dr_choice", producer=__name__)
    log.info("2-D maps: %s; chosen = %s", "; ".join(
        f"{r.method} T={r.trustworthiness:.3f} S1={r.stress1:.3f}" for r in q.itertuples() if r.status == "done"),
        chosen)


if __name__ == "__main__":
    main()
