"""The canonical concept ordering — the single source of truth for every matrix.

Rule: every concept x concept matrix in this project (typological or model)
has its rows and columns in exactly the order stored in
``data/processed/concept_order.json``. That file is written once, in Phase 1,
by ``freeze_concept_order`` and never silently overwritten. Its SHA-256 is
stored inside every saved matrix (see ``src/utils/matrices.py``), so a matrix
built against a different concept set cannot be loaded by mistake.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import config


class ConceptOrderError(RuntimeError):
    """Raised when a matrix or table does not follow the canonical order."""


def order_hash(concepts: Sequence[str]) -> str:
    """SHA-256 of the newline-joined concept IDs (order-sensitive)."""
    return hashlib.sha256("\n".join(concepts).encode("utf-8")).hexdigest()


def freeze_concept_order(
    concepts: Sequence[str],
    path: Path | None = None,
    provenance: dict[str, object] | None = None,
    overwrite: bool = False,
) -> str:
    """Write the canonical order. Refuses to overwrite a different frozen order.

    Re-freezing the *identical* list is a no-op, so reruns are idempotent.
    Changing the concept set requires ``overwrite=True`` (``make refreeze``),
    which invalidates every downstream matrix — by design.
    """
    path = Path(path) if path is not None else config.CONCEPT_ORDER_PATH   # resolved at call time
    concepts = list(concepts)
    if len(concepts) != len(set(concepts)):
        dupes = sorted({c for c in concepts if concepts.count(c) > 1})
        raise ConceptOrderError(f"Duplicate concept IDs: {dupes[:10]}")
    if not all(isinstance(c, str) and c for c in concepts):
        raise ConceptOrderError("Concept IDs must be non-empty strings.")
    digest = order_hash(concepts)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing["sha256"] == digest:
            return digest
        if not overwrite:
            raise ConceptOrderError(
                f"{path} already holds a different frozen order "
                f"(sha256 {existing['sha256'][:12]} vs {digest[:12]}). "
                "Pass overwrite=True (make refreeze) and rebuild all matrices."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_concepts": len(concepts),
        "sha256": digest,
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": provenance or {},
        "concepts": concepts,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return digest


def load_concept_order(path: Path | None = None) -> list[str]:
    """Load the canonical order and verify its stored hash."""
    path = Path(path) if path is not None else config.CONCEPT_ORDER_PATH   # resolved at call time
    if not path.exists():
        raise ConceptOrderError(f"{path} not found — run `make data` (Phase 1) first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    concepts = payload["concepts"]
    if order_hash(concepts) != payload["sha256"]:
        raise ConceptOrderError(f"{path} has been edited by hand: hash mismatch.")
    return concepts


def assert_canonical(ids: Sequence[str], canonical: Sequence[str], what: str = "matrix") -> None:
    """Raise unless ``ids`` equals ``canonical`` element for element.

    The error message reports the first mismatching position and set
    differences, which is what you need to debug an ordering bug.
    """
    ids, canonical = list(ids), list(canonical)
    if ids == canonical:
        return
    missing = sorted(set(canonical) - set(ids))
    extra = sorted(set(ids) - set(canonical))
    first = next(
        (i for i, (a, b) in enumerate(zip(ids, canonical)) if a != b),
        min(len(ids), len(canonical)),
    )
    raise ConceptOrderError(
        f"{what} does not follow the canonical concept order: "
        f"len {len(ids)} vs {len(canonical)}; first mismatch at index {first}; "
        f"missing={missing[:5]}{'…' if len(missing) > 5 else ''}; "
        f"extra={extra[:5]}{'…' if len(extra) > 5 else ''}"
    )


def reindex_to_canonical(ids: Sequence[str], canonical: Sequence[str]) -> list[int]:
    """Positions that reorder ``ids`` into ``canonical`` (all must be present)."""
    pos = {c: i for i, c in enumerate(ids)}
    missing = [c for c in canonical if c not in pos]
    if missing:
        raise ConceptOrderError(f"Cannot reindex: {len(missing)} canonical concepts absent, e.g. {missing[:5]}")
    return [pos[c] for c in canonical]
