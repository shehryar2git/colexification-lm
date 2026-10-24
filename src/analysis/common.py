"""Shared helpers for H1-H3: the analysis concept set, condition matrices, controls.

Analysis concept set (DECISIONS.md D-16, pre-specified): the concepts that
*every* model condition can embed. Because the multilingual-input conditions
need forms in >= TRANSLATED_MIN_UNITS units, the set is fixed by data, not by
which models happen to have been run: concepts with enough multilingual forms
AND a finite vector in every condition whose matrix exists. On a machine where
all six conditions are embedded this is exactly D-16's intersection.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

import config
from src.data.frequency import zipf
from src.utils.concept_order import load_concept_order
from src.utils.io import read_parquet, write_result_json
from src.utils.logging_utils import get_logger
from src.utils.matrices import load_matrix
from src.utils.seeds import rng

log = get_logger(__name__, "analysis_common")

CONDITION_LABELS = {k: v["label"] for k, v in config.MODELS.items()} | {
    "xlmr_translated": "XLM-R (multi input)", "labse_translated": "LaBSE (multi input)"}


def stream(*parts: str) -> np.random.Generator:
    """Independent, reproducible random stream for one named analysis."""
    return rng(zlib.crc32(":".join(parts).encode()))


def model_matrix_path(condition: str):
    return config.MATRIX_DIR / f"model_{condition}.npz"


def available_conditions() -> list[str]:
    return [c for c in config.MODEL_CONDITIONS if model_matrix_path(c).exists()]


def load_similarity(condition: str, order: list[str]) -> np.ndarray:
    return load_matrix(model_matrix_path(condition), order).values


def translated_coverage_set(order: list[str]) -> set[str]:
    f = pd.read_csv(config.RESULTS_DIR / "translated_forms.csv")
    n = f.groupby("concept_id")["unit"].nunique()
    return {c for c in order if n.get(c, 0) >= config.TRANSLATED_MIN_UNITS}


def analysis_concepts(order: list[str] | None = None, write: bool = True) -> tuple[list[str], np.ndarray]:
    """Return the analysis concept list (canonical order) and its index array."""
    order = order or load_concept_order()
    keep = translated_coverage_set(order)
    dropped_by: dict[str, list[str]] = {"translated_coverage": sorted(set(order) - keep)}
    for cond in available_conditions():
        s = load_similarity(cond, order)
        finite = {c for c, ok in zip(order, np.isfinite(np.diag(s))) if ok}
        lost = sorted(keep - finite)
        if lost:
            dropped_by[cond] = lost
        keep &= finite
    concepts = [c for c in order if c in keep]
    idx = np.array([i for i, c in enumerate(order) if c in keep])
    if write:
        write_result_json({"rule": "D-16: >= %d multilingual units AND embedded by every available condition"
                                   % config.TRANSLATED_MIN_UNITS,
                           "n_concepts": len(concepts), "n_pairs": len(concepts) * (len(concepts) - 1) // 2,
                           "available_conditions": available_conditions(),
                           "dropped": {k: v for k, v in dropped_by.items()},
                           "concepts": concepts}, "analysis_concept_set", producer=__name__)
    return concepts, idx


def typological_distance(order: list[str], name: str | None = None) -> np.ndarray:
    fname = "typological_distance.npz" if name is None else f"typological_{name}.npz"
    return load_matrix(config.MATRIX_DIR / fname, order).values


def concept_frequencies(order: list[str]) -> np.ndarray:
    gl = read_parquet(config.PROCESSED_DIR / "concepts.parquet").set_index("concept_id")["gloss"]
    return np.array([zipf(gl[c]) for c in order])


def control_matrices(order: list[str]) -> dict[str, np.ndarray]:
    """Pair-level controls (D-20): |difference in English Zipf frequency| and
    'different semantic field' (1 = different), both as distance-like matrices."""
    z = concept_frequencies(order)
    field = read_parquet(config.PROCESSED_DIR / "concepts.parquet").set_index("concept_id").loc[order, "semantic_field"]
    f = field.fillna("").to_numpy()
    return {"freq_diff": np.abs(z[:, None] - z[None, :]),
            "diff_field": (f[:, None] != f[None, :]).astype(float),
            "zipf": z}


def pair_frame(order: list[str], idx: np.ndarray) -> pd.DataFrame:
    """Pair table restricted to the analysis set, in upper-triangle order of idx."""
    pairs = read_parquet(config.PROCESSED_DIR / "pairs.parquet")
    sub = [order[i] for i in idx]
    a, b = np.triu_indices(len(sub), 1)
    df = pd.DataFrame({"i": a, "j": b, "concept_a": np.array(sub)[a], "concept_b": np.array(sub)[b]})
    lo = df[["concept_a", "concept_b"]].min(axis=1)
    hi = df[["concept_a", "concept_b"]].max(axis=1)
    df = df.assign(key_a=lo, key_b=hi).merge(
        pairs.rename(columns={"concept_a": "key_a", "concept_b": "key_b"}), on=["key_a", "key_b"], how="left")
    if df["n_families"].isna().any():
        raise RuntimeError("pairs.parquet does not cover the analysis set")
    return df.drop(columns=["key_a", "key_b"])
