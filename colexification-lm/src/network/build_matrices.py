"""Phase 2 · Concept x language matrix and concept x concept typological distances.

All concept x concept matrices are saved through ``src.utils.matrices`` and so
carry the canonical-order hash (D-31). Definitions (DECISIONS.md D-08):

primary  ``shortest_path``  weighted shortest path over edges (>= EDGE_MIN_FAMILIES),
                            edge length 1 / n_families; disconnected pairs = max + 1   (D-39)
alternative metrics (same network)
  ``profile_cosine``    R_ij = n_families / cov_families on edges, R_ii = 1; 1 - cos(R_i, R_j)
  ``rate``              1 - R_ij  (first-order)
  ``binary_profile``    profile cosine on 0/1 edges
  ``language_profile``  profile cosine on language rates (n_languages / cov_languages)
operationalisation variants (primary metric)
  ``minfam1``, ``minfam3``   edge thresholds of 1 and 3 families
  ``evidence_all``      CLICS4 as published (shared source entries kept, D-37)
  ``identity_form``     exact-Form identity instead of Segments (D-03)
  ``no_johansson``      without the johanssonsoundsymbolic dataset (D-38)

Per-family matrices (H3): within-family first-order distance 1 - R^f_ij with
R^f = n_languages / cov_languages on within-family edges (>= 2 languages), NaN
for concepts the family does not attest. These are >98% ties by construction,
so H3's pre-specified statistic is the pair-level AUC (D-25 amendment).

Outputs: data/processed/matrices/typological_distance.npz (primary),
typological_<name>.npz, concept_language.npz (attestation), concept_lexeme.npz
(concept x colexifying word form: the Oct 10 practice structure), family_<slug>.npz;
results/typological_matrix_correlations.csv, results/family_matrices.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

import config
from src.network.build_network import network_for, slug
from src.utils.concept_order import load_concept_order
from src.utils.io import read_parquet, write_result_table
from src.utils.logging_utils import get_logger, log_run_header
from src.utils.matrices import ConceptMatrix, save_matrix
from src.utils.seeds import set_all_seeds

log = get_logger(__name__, "phase2_build_matrices")
EXCLUDED_DATASET_ROBUSTNESS = "johanssonsoundsymbolic"


# ------------------------------------------------------------- primitives ---
def square(pairs: pd.DataFrame, order: list[str], value: str, diag: float) -> np.ndarray:
    """Symmetric (n, n) array from a long pair table; missing pairs = 0."""
    idx = {c: i for i, c in enumerate(order)}
    m = np.zeros((len(order), len(order)))
    i = pairs["concept_a"].map(idx).to_numpy()
    j = pairs["concept_b"].map(idx).to_numpy()
    v = pairs[value].to_numpy(dtype=float)
    m[i, j] = v
    m[j, i] = v
    np.fill_diagonal(m, diag)
    return m


def profile_cosine_distance(r: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """1 - cosine between rows of a non-negative profile matrix. Rows marked
    invalid (e.g. concepts unattested in a family) become NaN."""
    norms = np.linalg.norm(r, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = r / norms[:, None]
    d = 1.0 - unit @ unit.T
    d = np.clip(d, 0.0, 1.0)
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    if valid is not None:
        d[~valid, :] = np.nan
        d[:, ~valid] = np.nan
    return d


def rate_matrix(pairs: pd.DataFrame, order: list[str], rate_col: str = "rate_families",
                edge_col: str = "is_edge") -> np.ndarray:
    p = pairs.assign(_r=np.where(pairs[edge_col], pairs[rate_col], 0.0))
    return square(p, order, "_r", diag=1.0)


def shortest_path_distance(pairs: pd.DataFrame, order: list[str]) -> tuple[np.ndarray, float]:
    e = pairs[pairs["is_edge"]]
    idx = {c: i for i, c in enumerate(order)}
    n = len(order)
    w = 1.0 / e["n_families"].to_numpy(dtype=float)
    i = e["concept_a"].map(idx).to_numpy()
    j = e["concept_b"].map(idx).to_numpy()
    g = csr_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(n, n))
    d = shortest_path(g, method="D", directed=False)
    finite = np.isfinite(d)
    share_disconnected = float((~finite[np.triu_indices(n, 1)]).mean())
    d[~finite] = d[finite].max() + 1.0
    np.fill_diagonal(d, 0.0)
    return d, share_disconnected


def with_edge_rule(pairs: pd.DataFrame, min_families: int) -> pd.DataFrame:
    return pairs.assign(is_edge=pairs["n_families"] >= min_families)


def upper(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices(m.shape[0], 1)]


# ----------------------------------------------------------- per family ---
def family_distance(fp: pd.DataFrame, order: list[str], attested: set[str]) -> np.ndarray:
    """Within-family first-order distance 1 - R^f (language rates on within-family
    edges); NaN rows/columns for concepts the family does not attest (D-09)."""
    d = 1.0 - rate_matrix(fp, order, "rate_languages", "is_edge")
    np.fill_diagonal(d, 0.0)
    valid = np.array([c in attested for c in order])
    d[~valid, :] = np.nan
    d[:, ~valid] = np.nan
    return d


# -------------------------------------------------------------------- main ---
def main() -> None:
    set_all_seeds()
    log_run_header(log)
    order = load_concept_order()
    md = config.MATRIX_DIR
    pairs = read_parquet(config.PROCESSED_DIR / "pairs.parquet")

    # --- concept x language binary matrix (primary evidence) ----------------
    forms = read_parquet(config.PROCESSED_DIR / "forms.parquet")
    languages = read_parquet(config.PROCESSED_DIR / "languages.parquet")
    inc = languages.loc[languages["include"], ["variety_id", "language_gc", "family_gc"]]
    f = forms[~forms["shared_entry"]] if config.COLEX_EVIDENCE == "separate_entries" else forms
    f = f[f["concept_id"].isin(order)].merge(inc, on="variety_id")
    langs = sorted(inc["language_gc"].unique())
    li = {lg: k for k, lg in enumerate(langs)}
    ci = {c: k for k, c in enumerate(order)}
    cl = np.zeros((len(order), len(langs)), dtype=bool)
    cl[f["concept_id"].map(ci).to_numpy(), f["language_gc"].map(li).to_numpy()] = True
    from src.utils.concept_order import order_hash
    md.mkdir(parents=True, exist_ok=True)
    fam_of = inc.drop_duplicates("language_gc").set_index("language_gc").loc[langs, "family_gc"].to_numpy()
    np.savez_compressed(md / "concept_language.npz", values=cl, ids=np.array(order),
                        languages=np.array(langs), language_family=fam_of.astype(str),
                        order_sha256=np.array(order_hash(order)))
    log.info("concept x language matrix: %d x %d, density %.3f", *cl.shape, cl.mean())

    # --- concept x lexeme matrix (the Oct 10 practice structure, D-43) ------
    # a lexeme = one word form (Segments) in one variety; kept if it covers >= 2 working concepts
    key = f["variety_id"].astype(str) + "\t" + f["segments"].astype(str)
    lex = f.assign(lexeme=key)[f["segments"].fillna("") != ""].drop_duplicates(["lexeme", "concept_id"])
    multi = lex.groupby("lexeme")["concept_id"].transform("nunique") >= 2
    lex = lex[multi]
    lexemes = sorted(lex["lexeme"].unique())
    xi = {k: i for i, k in enumerate(lexemes)}
    clx = np.zeros((len(order), len(lexemes)), dtype=bool)
    clx[lex["concept_id"].map(ci).to_numpy(), lex["lexeme"].map(xi).to_numpy()] = True
    np.savez_compressed(md / "concept_lexeme.npz", values=clx, ids=np.array(order),
                        lexemes=np.array(lexemes), order_sha256=np.array(order_hash(order)))
    log.info("concept x lexeme matrix: %d x %d colexifying word forms", *clx.shape)

    # --- typological distances ----------------------------------------------
    mats: dict[str, np.ndarray] = {}
    meta: dict[str, dict[str, str]] = {}

    def sp(q: pd.DataFrame, name: str) -> None:
        mats[name], disc = shortest_path_distance(q, order)
        meta[name] = {"share_pairs_disconnected": f"{disc:.6f}"}

    sp(pairs, "shortest_path")
    mats["profile_cosine"] = profile_cosine_distance(rate_matrix(pairs, order))
    mats["rate"] = 1.0 - rate_matrix(pairs, order)
    np.fill_diagonal(mats["rate"], 0.0)
    mats["binary_profile"] = profile_cosine_distance(
        square(pairs.assign(_b=pairs["is_edge"].astype(float)), order, "_b", diag=1.0))
    mats["language_profile"] = profile_cosine_distance(rate_matrix(pairs, order, "rate_languages"))
    sp(with_edge_rule(pairs, 1), "minfam1")
    sp(with_edge_rule(pairs, 3), "minfam3")
    sp(read_parquet(config.PROCESSED_DIR / "pairs__all__segments.parquet"), "evidence_all")
    sp(read_parquet(config.PROCESSED_DIR / f"pairs__{config.COLEX_EVIDENCE}__form.parquet"), "identity_form")
    keep = languages.assign(include=languages["include"]
                            & (languages["contribution_id"] != EXCLUDED_DATASET_ROBUSTNESS))
    sp(network_for(forms, keep, order)["pairs"], "no_johansson")
    expected = {config.TYPOLOGICAL_DISTANCE_PRIMARY, *config.TYPOLOGICAL_DISTANCE_ALTERNATIVES,
                *config.TYPOLOGICAL_VARIANTS}
    assert set(mats) == expected, f"matrix set {sorted(mats)} != config {sorted(expected)}"

    for name, d in mats.items():
        m = ConceptMatrix(d, order, "distance", {"metric": name, "colex_evidence": config.COLEX_EVIDENCE,
                                                  **meta.get(name, {})})
        save_matrix(md / f"typological_{name}.npz", m, canonical=order)
    save_matrix(md / "typological_distance.npz",
                ConceptMatrix(mats[config.TYPOLOGICAL_DISTANCE_PRIMARY], order, "distance",
                              {"metric": config.TYPOLOGICAL_DISTANCE_PRIMARY}), canonical=order)

    names = list(mats)
    corr = pd.DataFrame([{"matrix_a": a, "matrix_b": b,
                          "spearman": spearmanr(upper(mats[a]), upper(mats[b])).statistic}
                         for k, a in enumerate(names) for b in names[k + 1:]])
    write_result_table(corr, "typological_matrix_correlations", producer=__name__)
    prim = config.TYPOLOGICAL_DISTANCE_PRIMARY
    vs_primary = corr[(corr["matrix_a"] == prim) | (corr["matrix_b"] == prim)]
    log.info("Spearman vs primary (%s): %s", prim, ", ".join(
        f"{(r.matrix_b if r.matrix_a == prim else r.matrix_a)}={r.spearman:.3f}"
        for r in vs_primary.itertuples()))
    ties = pd.DataFrame([{"matrix": n, "share_pairs_at_max": float(np.isclose(upper(d), upper(d).max()).mean()),
                          "n_unique_values": int(len(np.unique(upper(d).round(9))))} for n, d in mats.items()])
    write_result_table(ties, "typological_matrix_ties", producer=__name__)

    # --- per-family matrices (H3) ------------------------------------------
    fam = pd.read_csv(config.RESULTS_DIR / "family_subnetworks.csv")
    fam_pairs = read_parquet(config.PROCESSED_DIR / "family_pairs.parquet")
    rows = []
    for r in fam[fam["qualifies"]].itertuples():
        fp = fam_pairs[fam_pairs["family_gc"] == r.family_gc]
        fams_forms = f[f["family_gc"] == r.family_gc]
        attested = set(fams_forms["concept_id"])
        d = family_distance(fp, order, attested)
        name = f"family_{slug(r.family_name)}"
        save_matrix(md / f"{name}.npz",
                    ConceptMatrix(d, order, "distance", {"family": r.family_name, "family_gc": r.family_gc,
                                                         "n_languages": str(r.n_languages)}), canonical=order)
        valid = ~np.isnan(upper(d))
        rows.append({"family_name": r.family_name, "family_gc": r.family_gc, "file": f"{name}.npz",
                     "n_languages": r.n_languages, "n_concepts_attested": len(attested),
                     "n_within_family_edges": int(fp["is_edge"].sum()),
                     "share_pairs_tied_at_1": float(np.isclose(upper(d)[valid], 1).mean()),
                     "spearman_with_global_rate": spearmanr(upper(d)[valid],
                                                            upper(mats["rate"])[valid]).statistic})
    write_result_table(pd.DataFrame(rows), "family_matrices", producer=__name__)
    log.info("Wrote %d typological matrices and %d family matrices", len(mats) + 1, len(rows))


if __name__ == "__main__":
    main()
