"""Small I/O helpers: Parquet tables, JSON results, and result-file provenance.

Results discipline: every statistic reported anywhere is written by a named
script through ``write_result_table`` / ``write_result_json``. Each file gets
a sidecar ``<name>.provenance.json`` recording the producing script, seed,
git commit and package versions, so the report can cite it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from src.utils.logging_utils import run_header


def parquet_engine_available() -> bool:
    for mod in ("pyarrow", "fastparquet"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def _fallback_path(path: Path) -> Path:
    return path.with_name(path.name + ".pkl")


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write Parquet. If no Parquet engine is installed AND the environment
    variable COLEX_ALLOW_PICKLE_FALLBACK=1 is set, write ``<path>.pkl`` instead
    (DECISIONS.md D-33: used only in the build sandbox, never in `make all`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_engine_available():
        df.to_parquet(path, index=False)
        return path
    if os.environ.get("COLEX_ALLOW_PICKLE_FALLBACK") == "1":
        fb = _fallback_path(path)
        df.to_pickle(fb)
        return fb
    raise ImportError("No Parquet engine: `pip install pyarrow` (pinned in requirements.txt).")


def read_parquet(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    fb = _fallback_path(path)
    if fb.exists() and os.environ.get("COLEX_ALLOW_PICKLE_FALLBACK") == "1":
        return pd.read_pickle(fb)
    raise FileNotFoundError(f"{path} missing — run the producing pipeline step first.")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj.relative_to(config.ROOT)) if obj.is_relative_to(config.ROOT) else str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def _write_provenance(path: Path, producer: str, extra: dict[str, Any] | None) -> None:
    prov = {"file": path.name, "producer": producer, **run_header(), **(extra or {})}
    sidecar = path.with_name(path.stem + ".provenance.json")
    sidecar.write_text(json.dumps(prov, indent=2, default=_json_default) + "\n", encoding="utf-8")


def write_result_table(
    df: pd.DataFrame, name: str, producer: str, extra: dict[str, Any] | None = None
) -> Path:
    """Write ``results/<name>.csv`` plus provenance. ``producer`` = module path."""
    path = config.RESULTS_DIR / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6g")
    _write_provenance(path, producer, extra)
    return path


def write_result_json(
    obj: dict[str, Any], name: str, producer: str, extra: dict[str, Any] | None = None
) -> Path:
    path = config.RESULTS_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_provenance(path, producer, extra)
    return path
