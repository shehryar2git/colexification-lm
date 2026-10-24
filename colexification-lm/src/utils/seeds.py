"""Global seeding. Call ``set_all_seeds()`` at the top of every entry point."""
from __future__ import annotations

import os
import random

import numpy as np

import config


def set_all_seeds(seed: int = config.SEED, deterministic_torch: bool = True) -> dict[str, object]:
    """Seed python, numpy and (if installed) torch; return a record for the log.

    ``PYTHONHASHSEED`` only affects interpreters started after it is set, so the
    Makefile exports it as well; here it is set for any subprocesses.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    record: dict[str, object] = {"seed": seed, "python_random": True, "numpy": True, "torch": False}
    try:
        import torch
    except ImportError:
        return record
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    record["torch"] = True
    record["torch_version"] = torch.__version__
    return record


def rng(offset: int = 0) -> np.random.Generator:
    """Return an independent, reproducible numpy Generator.

    Use a distinct ``offset`` per analysis (e.g. one per permutation test) so
    that adding a new analysis never shifts the random stream of an old one.
    """
    return np.random.default_rng(np.random.SeedSequence([config.SEED, offset]))
