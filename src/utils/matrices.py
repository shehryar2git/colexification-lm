"""Save / load concept x concept matrices with the canonical-order guarantee.

Every ``.npz`` written by this project contains:

* ``values``  — float64 (n, n) matrix
* ``ids``     — the concept IDs labelling rows and columns
* ``order_sha256`` — hash of the canonical order it was built against
* ``kind``    — "similarity" or "distance"
* ``mask``    — boolean (n, n): False where a value is undefined
                (e.g. a concept a model could not embed); NaN in ``values``

Loading checks all of this against ``concept_order.json``. A matrix that was
built against an older concept set therefore fails loudly instead of being
silently correlated against the wrong rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from src.utils.concept_order import (
    ConceptOrderError,
    assert_canonical,
    load_concept_order,
    order_hash,
)

VALID_KINDS = ("similarity", "distance")


@dataclass
class ConceptMatrix:
    values: np.ndarray
    ids: list[str]
    kind: str
    meta: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        n = len(self.ids)
        if self.values.shape != (n, n):
            raise ValueError(f"values shape {self.values.shape} does not match {n} ids")
        if self.kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {VALID_KINDS}, got {self.kind!r}")
        finite = np.isfinite(self.values)
        sym = np.allclose(
            np.where(finite, self.values, 0.0), np.where(finite.T, self.values.T, 0.0), atol=1e-8
        )
        if not sym:
            raise ValueError("concept x concept matrices must be symmetric")

    @property
    def mask(self) -> np.ndarray:
        return np.isfinite(self.values)

    def to_distance(self) -> "ConceptMatrix":
        """Cosine-style similarity in [-1, 1] -> distance 1 - s (diagonal 0)."""
        if self.kind == "distance":
            return self
        d = 1.0 - self.values
        np.fill_diagonal(d, 0.0)
        return ConceptMatrix(d, list(self.ids), "distance", dict(self.meta))

    def upper(self, k: int = 1) -> np.ndarray:
        """Upper-triangle vector (row-major), the unit of analysis for RSA."""
        iu = np.triu_indices(len(self.ids), k=k)
        return self.values[iu]


def save_matrix(path: Path, matrix: ConceptMatrix, canonical: Sequence[str] | None = None) -> None:
    canonical = list(canonical) if canonical is not None else load_concept_order()
    assert_canonical(matrix.ids, canonical, what=f"matrix for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_keys = np.array(list(matrix.meta.keys()), dtype=str)
    meta_vals = np.array([str(v) for v in matrix.meta.values()], dtype=str)
    np.savez_compressed(
        path,
        values=matrix.values,
        ids=np.array(matrix.ids, dtype=str),
        order_sha256=np.array(order_hash(canonical)),
        kind=np.array(matrix.kind),
        mask=matrix.mask,
        meta_keys=meta_keys,
        meta_vals=meta_vals,
    )


def load_matrix(path: Path, canonical: Sequence[str] | None = None) -> ConceptMatrix:
    canonical = list(canonical) if canonical is not None else load_concept_order()
    with np.load(path, allow_pickle=False) as z:
        ids = [str(x) for x in z["ids"]]
        stored = str(z["order_sha256"])
        if stored != order_hash(canonical):
            raise ConceptOrderError(
                f"{path.name} was built against concept order {stored[:12]}, "
                f"current order is {order_hash(canonical)[:12]}. Rebuild it."
            )
        assert_canonical(ids, canonical, what=f"matrix {path.name}")
        meta = dict(zip((str(k) for k in z["meta_keys"]), (str(v) for v in z["meta_vals"])))
        return ConceptMatrix(z["values"], ids, str(z["kind"]), meta)
