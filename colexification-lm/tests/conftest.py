"""Shared pytest fixtures: tiny synthetic data with hand-computable answers.

The synthetic wordlist is designed so every downstream quantity (edge family
counts, language counts, typological distances, communities) can be checked
by hand. Its ground truth lives in ``EXPECTED_*`` constants used by the tests
of Phases 1-2.

Synthetic world
---------------
Families: A (3 languages), B (2), C (1 isolate, treated as its own family).
Concepts: HAND, ARM, TREE, WOOD, YESTERDAY, TOMORROW, SEE, UNDERSTAND.

Colexifications planted (word shared by both concepts in a language):
    HAND~ARM            A1, A2, B1, C1      -> 4 languages, 3 families
    TREE~WOOD           A1, B1, B2          -> 3 languages, 2 families
    YESTERDAY~TOMORROW  A3                  -> 1 language,  1 family
    SEE~UNDERSTAND      A1, A2              -> 2 languages, 1 family
A2 has no form for TREE (missing data, tests coverage normalisation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config

CONCEPTS = ["HAND", "ARM", "TREE", "WOOD", "YESTERDAY", "TOMORROW", "SEE", "UNDERSTAND"]

LANGUAGES = pd.DataFrame(
    {
        "Language_ID": ["A1", "A2", "A3", "B1", "B2", "C1"],
        "Glottocode": ["aaaa1111", "aaaa2222", "aaaa3333", "bbbb1111", "bbbb2222", "cccc1111"],
        "Family": ["fama0001", "fama0001", "fama0001", "famb0001", "famb0001", ""],
        "Family_Name": ["A", "A", "A", "B", "B", ""],
        "Macroarea": ["Eurasia", "Eurasia", "Africa", "Papunesia", "Papunesia", "Australia"],
        "Latitude": [50.0, 51.0, 10.0, -5.0, -6.0, -25.0],
        "Longitude": [10.0, 11.0, 20.0, 140.0, 141.0, 135.0],
    }
)

# (language, concept, form). Identical form within a language = colexification.
_FORMS = [
    # A1: HAND~ARM, TREE~WOOD, SEE~UNDERSTAND
    ("A1", "HAND", "ma"), ("A1", "ARM", "ma"), ("A1", "TREE", "ki"), ("A1", "WOOD", "ki"),
    ("A1", "YESTERDAY", "ne"), ("A1", "TOMORROW", "su"), ("A1", "SEE", "lo"), ("A1", "UNDERSTAND", "lo"),
    # A2: HAND~ARM, SEE~UNDERSTAND; TREE missing
    ("A2", "HAND", "pa"), ("A2", "ARM", "pa"), ("A2", "WOOD", "ta"),
    ("A2", "YESTERDAY", "ri"), ("A2", "TOMORROW", "so"), ("A2", "SEE", "vi"), ("A2", "UNDERSTAND", "vi"),
    # A3: YESTERDAY~TOMORROW (the "kal" language)
    ("A3", "HAND", "ha"), ("A3", "ARM", "bo"), ("A3", "TREE", "dr"), ("A3", "WOOD", "hu"),
    ("A3", "YESTERDAY", "kal"), ("A3", "TOMORROW", "kal"), ("A3", "SEE", "de"), ("A3", "UNDERSTAND", "sa"),
    # B1: HAND~ARM, TREE~WOOD
    ("B1", "HAND", "lima"), ("B1", "ARM", "lima"), ("B1", "TREE", "kayu"), ("B1", "WOOD", "kayu"),
    ("B1", "YESTERDAY", "kemarin"), ("B1", "TOMORROW", "besok"), ("B1", "SEE", "lihat"), ("B1", "UNDERSTAND", "erti"),
    # B2: TREE~WOOD
    ("B2", "HAND", "nima"), ("B2", "ARM", "rima"), ("B2", "TREE", "ai"), ("B2", "WOOD", "ai"),
    ("B2", "YESTERDAY", "nanafi"), ("B2", "TOMORROW", "apopo"), ("B2", "SEE", "kite"), ("B2", "UNDERSTAND", "mohio"),
    # C1 (isolate): HAND~ARM
    ("C1", "HAND", "mara"), ("C1", "ARM", "mara"), ("C1", "TREE", "yuku"), ("C1", "WOOD", "warra"),
    ("C1", "YESTERDAY", "ngulu"), ("C1", "TOMORROW", "pirri"), ("C1", "SEE", "nhan"), ("C1", "UNDERSTAND", "pina"),
]

FORMS = pd.DataFrame(_FORMS, columns=["Language_ID", "Concept", "Form"])

# Ground truth (unordered pairs, alphabetical within pair)
EXPECTED_EDGES = {
    ("ARM", "HAND"): {"languages": 4, "families": 3},
    ("TREE", "WOOD"): {"languages": 3, "families": 2},
    ("TOMORROW", "YESTERDAY"): {"languages": 1, "families": 1},
    ("SEE", "UNDERSTAND"): {"languages": 2, "families": 1},
}
# Families in which BOTH concepts of a pair are attested (opportunity / coverage)
EXPECTED_PAIR_COVERAGE_FAMILIES = {("ARM", "HAND"): 3, ("TREE", "WOOD"): 3}
# With EDGE_MIN_FAMILIES = 2 only these survive
EXPECTED_EDGES_MIN2 = {("ARM", "HAND"), ("TREE", "WOOD")}


@pytest.fixture
def concepts() -> list[str]:
    return list(CONCEPTS)


@pytest.fixture
def languages() -> pd.DataFrame:
    return LANGUAGES.copy()


@pytest.fixture
def forms() -> pd.DataFrame:
    return FORMS.copy()


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Redirect every config path into a temp dir so tests never touch real data."""
    for attr in ("DATA_DIR", "RAW_DIR", "PROCESSED_DIR", "MATRIX_DIR", "EMBEDDING_DIR",
                 "RESULTS_DIR", "FIGURES_DIR", "AUDIT_DIR", "LOG_DIR"):
        monkeypatch.setattr(config, attr, tmp_path / attr.lower())
    monkeypatch.setattr(config, "CONCEPT_ORDER_PATH", tmp_path / "processed" / "concept_order.json")
    return tmp_path


@pytest.fixture
def random_similarity(concepts):
    """A reproducible symmetric cosine-similarity matrix over the toy concepts."""
    g = np.random.default_rng(0)
    x = g.normal(size=(len(concepts), 16))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    s = x @ x.T
    np.fill_diagonal(s, 1.0)
    return s


@pytest.fixture
def correlated_distance_pair():
    """Two distance matrices whose upper triangles have a known strong
    monotone relation (d2 = d1**2 + small noise), plus an independent one.
    Used in Phase 5 to validate the Mantel implementation."""
    g = np.random.default_rng(1)
    pts = g.normal(size=(30, 3))
    d1 = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    noise = g.normal(scale=0.01, size=d1.shape)
    d2 = d1**2 + (noise + noise.T) / 2
    np.fill_diagonal(d2, 0.0)
    other = g.normal(size=(30, 3))
    d3 = np.linalg.norm(other[:, None] - other[None], axis=-1)
    return d1, d2, d3
