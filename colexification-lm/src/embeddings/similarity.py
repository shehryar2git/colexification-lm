"""Phase 4 · Concept x concept cosine-similarity matrices for every model condition.

For each condition in ``config.MODEL_CONDITIONS`` whose embeddings are cached:

1. load the vectors and **assert they follow the canonical concept order**
   (``load_embeddings`` checks the stored order hash; this module asserts the
   ids again before computing anything, and ``save_matrix`` asserts on write);
2. centre: subtract the condition's mean vector over the concepts it embeds
   (D-47; removes the shared direction that inflates all cosines, Ethayarajh 2019);
3. cosine similarity; concepts the condition could not embed are NaN rows/columns.

Robustness variants written alongside: ``model_<cond>__raw.npz`` (no centring)
and, for contextual conditions, ``model_<base>_bare_en.npz`` (English bare word,
template 0) — the baseline for the multilingual-input condition (D-17).

Outputs: data/processed/matrices/model_<condition>.npz,
results/embedding_coverage.csv (one row per condition, including PENDING ones),
results/model_similarity_summary.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.embeddings.store import exists, load_embeddings
from src.utils.concept_order import assert_canonical, load_concept_order
from src.utils.io import write_result_table
from src.utils.logging_utils import get_logger, log_run_header
from src.utils.matrices import ConceptMatrix, save_matrix
from src.utils.seeds import set_all_seeds

log = get_logger(__name__, "phase4_similarity")
BARE_TEMPLATE_INDEX = 0
PENDING_HINT = {
    "w2v_gnews": "python -m src.embeddings.static_w2v",
    "bert_base": "pip install -r requirements.txt; python -m src.embeddings.contextual",
    "xlmr_base": "pip install -r requirements.txt; python -m src.embeddings.contextual",
    "labse": "pip install -r requirements.txt; python -m src.embeddings.contextual",
    "xlmr_translated": "pip install -r requirements.txt; python -m src.embeddings.translated_input",
    "labse_translated": "pip install -r requirements.txt; python -m src.embeddings.translated_input",
}


def cosine_matrix(vectors: np.ndarray, center: bool = config.CENTER_EMBEDDINGS) -> np.ndarray:
    """Cosine similarity between rows; rows with any NaN stay NaN. With ``center``,
    the mean of the valid rows is subtracted first."""
    v = np.asarray(vectors, dtype=np.float64).copy()
    ok = np.isfinite(v).all(axis=1)
    if center and ok.any():
        v[ok] -= v[ok].mean(axis=0)
    norms = np.linalg.norm(v, axis=1)
    bad = ~ok | (norms == 0)
    unit = np.where(bad[:, None], 0.0, v / np.where(norms == 0, 1, norms)[:, None])
    s = np.clip(unit @ unit.T, -1.0, 1.0)
    s = (s + s.T) / 2
    np.fill_diagonal(s, 1.0)
    s[bad, :] = np.nan
    s[:, bad] = np.nan
    return s


def summarise(name: str, s: np.ndarray) -> dict[str, float]:
    iu = np.triu_indices(s.shape[0], 1)
    u = s[iu]
    u = u[np.isfinite(u)]
    return {"condition": name, "n_pairs": int(len(u)), "mean_cosine": float(u.mean()),
            "sd_cosine": float(u.std()), "q05": float(np.quantile(u, .05)), "q95": float(np.quantile(u, .95))}


def main() -> None:
    set_all_seeds()
    log_run_header(log)
    order = load_concept_order()
    cov_rows, sum_rows = [], []
    for cond in config.MODEL_CONDITIONS:
        if not exists(cond):
            cov_rows.append({"condition": cond, "status": "PENDING USER RUN", "n_embedded": np.nan,
                             "n_missing": np.nan, "missing_concepts": "", "command": PENDING_HINT[cond]})
            log.warning("PENDING USER RUN: %s (%s)", cond, PENDING_HINT[cond])
            continue
        vecs, ids, meta, extras = load_embeddings(cond)
        assert_canonical(ids, order, what=f"embedding rows of {cond}")       # the canonical-order guarantee
        variants = {cond: cosine_matrix(vecs), f"{cond}__raw": cosine_matrix(vecs, center=False)}
        if "per_template" in extras and cond in config.TRANSLATED_INPUT_MODELS + ("bert_base",):
            bare = extras["per_template"][BARE_TEMPLATE_INDEX]
            variants[f"{cond}_bare_en"] = cosine_matrix(bare)
        for name, s in variants.items():
            save_matrix(config.MATRIX_DIR / f"model_{name}.npz",
                        ConceptMatrix(s, order, "similarity", {"condition": name, "model": meta.get("model", ""),
                                                                "centered": str("__raw" not in name)}),
                        canonical=order)
            sum_rows.append(summarise(name, s))
        ok = np.isfinite(vecs).all(axis=1)
        cov_rows.append({"condition": cond, "status": "done", "n_embedded": int(ok.sum()),
                         "n_missing": int((~ok).sum()),
                         "missing_concepts": "; ".join(c for c, k in zip(order, ok) if not k), "command": ""})
        log.info("%s: %d/%d concepts; %d matrices written", cond, ok.sum(), len(order), len(variants))
    write_result_table(pd.DataFrame(cov_rows), "embedding_coverage", producer=__name__)
    if sum_rows:
        write_result_table(pd.DataFrame(sum_rows), "model_similarity_summary", producer=__name__)
    done = sum(r["status"] == "done" for r in cov_rows)
    log.info("%d of %d model conditions available; others PENDING USER RUN", done, len(config.MODEL_CONDITIONS))


if __name__ == "__main__":
    main()
