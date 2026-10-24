"""Project-wide logging: console + one rotating-free file per entry point.

Every entry point does::

    from src.utils.logging_utils import get_logger, log_run_header
    log = get_logger(__name__)
    log_run_header(log)

so each log file begins with the seed, git commit, Python and key package
versions — enough to tell whether two runs are comparable.
"""
from __future__ import annotations

import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata

import config

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_KEY_PACKAGES = (
    "numpy", "scipy", "pandas", "networkx", "scikit-learn", "statsmodels",
    "pycldf", "gensim", "torch", "transformers", "sentence-transformers", "umap-learn",
)


def get_logger(name: str, logfile: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to stderr and to ``logs/<logfile or name>.log``."""
    logger = logging.getLogger(name)
    if logger.handlers:          # already configured in this process
        return logger
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter(_FMT, datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    logger.addHandler(console)

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    stem = (logfile or name).replace("/", "_").replace(".", "_")
    fileh = logging.FileHandler(config.LOG_DIR / f"{stem}.log", mode="a", encoding="utf-8")
    fileh.setFormatter(fmt)
    logger.addHandler(fileh)
    return logger


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "not-a-git-repo"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def package_versions() -> dict[str, str]:
    """Installed versions of the packages that can change numerical results."""
    versions: dict[str, str] = {}
    for pkg in _KEY_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def run_header() -> dict[str, object]:
    return {
        "utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": config.SEED,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
    }


def log_run_header(logger: logging.Logger) -> dict[str, object]:
    header = run_header()
    logger.info("=" * 70)
    for key, value in header.items():
        if key == "packages":
            for pkg, ver in value.items():  # type: ignore[union-attr]
                logger.info("  %-22s %s", pkg, ver)
        else:
            logger.info("%-10s %s", key, value)
    logger.info("=" * 70)
    return header
