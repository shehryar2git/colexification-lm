"""Embedding cache: one ``data/processed/embeddings/<condition>.npz`` per model condition.

Embedding is the expensive step, so every condition is written once and reused
(``exists`` + ``--force``). Each cache file holds:

* ``vectors``  float32 (n_concepts, dim) in the **canonical concept order**,
               NaN rows for concepts the condition could not embed;
* ``ids``      the concept IDs (checked against concept_order.json on load);
* ``order_sha256``, ``condition``, and a JSON ``meta`` string (model id,
  resolved revision, coverage, the rule used per concept, library versions);
* optional extra arrays (e.g. per-template or per-layer vectors) for robustness.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import config
from src.utils.concept_order import assert_canonical, load_concept_order, order_hash


def cache_path(condition: str) -> Path:
    return config.EMBEDDING_DIR / f"{condition}.npz"


def exists(condition: str) -> bool:
    return cache_path(condition).exists()


def save_embeddings(condition: str, vectors: np.ndarray, ids: list[str], meta: dict[str, Any],
                    extras: dict[str, np.ndarray] | None = None) -> Path:
    canonical = load_concept_order()
    assert_canonical(ids, canonical, what=f"embeddings for {condition}")
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape[0] != len(ids):
        raise ValueError(f"{condition}: {vectors.shape[0]} vectors for {len(ids)} concepts")
    path = cache_path(condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vectors=vectors, ids=np.array(ids), order_sha256=np.array(order_hash(canonical)),
                        condition=np.array(condition), meta=np.array(json.dumps(meta, default=str)),
                        **(extras or {}))
    return path


def load_embeddings(condition: str) -> tuple[np.ndarray, list[str], dict[str, Any], dict[str, np.ndarray]]:
    canonical = load_concept_order()
    with np.load(cache_path(condition), allow_pickle=False) as z:
        ids = [str(x) for x in z["ids"]]
        if str(z["order_sha256"]) != order_hash(canonical):
            raise RuntimeError(f"Embeddings for {condition} were built for another concept order; rerun with --force")
        assert_canonical(ids, canonical, what=f"embeddings for {condition}")
        core = {"vectors", "ids", "order_sha256", "condition", "meta"}
        extras = {k: z[k] for k in z.files if k not in core}
        return z["vectors"].astype(np.float64), ids, json.loads(str(z["meta"])), extras


def coverage(vectors: np.ndarray) -> dict[str, int]:
    ok = np.isfinite(vectors).all(axis=1)
    return {"n_concepts": int(len(ok)), "n_embedded": int(ok.sum()), "n_missing": int((~ok).sum())}
