"""Phase 0 tests: config sanity, seeding, canonical order, matrix I/O, results I/O."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import config
from src.utils import concept_order as co
from src.utils.io import write_parquet, read_parquet, write_result_json, write_result_table
from src.utils.matrices import ConceptMatrix, load_matrix, save_matrix
from src.utils.seeds import rng, set_all_seeds

from tests.conftest import CONCEPTS, EXPECTED_EDGES, FORMS, LANGUAGES


# ---------------------------------------------------------------- config ----
def test_config_six_conditions():
    assert len(config.MODEL_CONDITIONS) == 6
    base = [c for c in config.MODEL_CONDITIONS if not c.endswith("_translated")]
    assert set(base) == set(config.MODELS)
    for m in config.TRANSLATED_INPUT_MODELS:
        assert f"{m.split('_')[0]}_translated" in config.MODEL_CONDITIONS


def test_config_templates_and_stats():
    assert len(config.TEMPLATES) >= 10
    assert all("{w}" in t for t in config.TEMPLATES)
    assert config.N_PERMUTATIONS >= 10_000
    assert config.N_BOOTSTRAP >= 1_000
    assert config.FDR_METHOD == "fdr_bh"
    assert len(config.OKABE_ITO) == 8
    assert config.CONCEPT_SET_TARGET == (300, 500)


# ----------------------------------------------------------------- seeds ----
def test_seeds_reproducible():
    set_all_seeds()
    a = np.random.rand(5)
    set_all_seeds()
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_rng_streams_independent_and_stable():
    np.testing.assert_array_equal(rng(3).random(4), rng(3).random(4))
    assert not np.allclose(rng(3).random(4), rng(4).random(4))


# --------------------------------------------------------- concept order ----
def test_freeze_and_load_roundtrip(isolated_paths, concepts):
    digest = co.freeze_concept_order(concepts, path=config.CONCEPT_ORDER_PATH)
    assert co.load_concept_order(config.CONCEPT_ORDER_PATH) == concepts
    assert digest == co.order_hash(concepts)


def test_refreeze_identical_is_noop(isolated_paths, concepts):
    d1 = co.freeze_concept_order(concepts, path=config.CONCEPT_ORDER_PATH)
    d2 = co.freeze_concept_order(concepts, path=config.CONCEPT_ORDER_PATH)
    assert d1 == d2


def test_refreeze_different_refused(isolated_paths, concepts):
    co.freeze_concept_order(concepts, path=config.CONCEPT_ORDER_PATH)
    with pytest.raises(co.ConceptOrderError, match="different frozen order"):
        co.freeze_concept_order(list(reversed(concepts)), path=config.CONCEPT_ORDER_PATH)
    co.freeze_concept_order(list(reversed(concepts)), path=config.CONCEPT_ORDER_PATH, overwrite=True)
    assert co.load_concept_order(config.CONCEPT_ORDER_PATH)[0] == concepts[-1]


def test_duplicates_rejected(isolated_paths):
    with pytest.raises(co.ConceptOrderError, match="Duplicate"):
        co.freeze_concept_order(["A", "B", "A"], path=config.CONCEPT_ORDER_PATH)


def test_hand_edit_detected(isolated_paths, concepts):
    co.freeze_concept_order(concepts, path=config.CONCEPT_ORDER_PATH)
    payload = json.loads(config.CONCEPT_ORDER_PATH.read_text())
    payload["concepts"][0], payload["concepts"][1] = payload["concepts"][1], payload["concepts"][0]
    config.CONCEPT_ORDER_PATH.write_text(json.dumps(payload))
    with pytest.raises(co.ConceptOrderError, match="hash mismatch"):
        co.load_concept_order(config.CONCEPT_ORDER_PATH)


def test_assert_canonical_reports_position(concepts):
    swapped = concepts.copy()
    swapped[2], swapped[3] = swapped[3], swapped[2]
    with pytest.raises(co.ConceptOrderError, match="first mismatch at index 2"):
        co.assert_canonical(swapped, concepts)
    co.assert_canonical(concepts, concepts)


def test_reindex_to_canonical(concepts):
    shuffled = concepts[::-1]
    idx = co.reindex_to_canonical(shuffled, concepts)
    assert [shuffled[i] for i in idx] == concepts
    with pytest.raises(co.ConceptOrderError):
        co.reindex_to_canonical(shuffled[:-1], concepts)


# --------------------------------------------------------------- matrices ---
def test_matrix_roundtrip(isolated_paths, concepts, random_similarity):
    co.freeze_concept_order(concepts, path=config.CONCEPT_ORDER_PATH)
    m = ConceptMatrix(random_similarity, concepts, "similarity", {"model": "toy"})
    path = config.MATRIX_DIR / "model_toy.npz"
    save_matrix(path, m, canonical=concepts)
    back = load_matrix(path, canonical=concepts)
    np.testing.assert_allclose(back.values, random_similarity)
    assert back.meta == {"model": "toy"} and back.kind == "similarity"


def test_matrix_wrong_order_rejected_on_save(isolated_paths, concepts, random_similarity):
    m = ConceptMatrix(random_similarity, concepts[::-1], "similarity")
    with pytest.raises(co.ConceptOrderError):
        save_matrix(config.MATRIX_DIR / "bad.npz", m, canonical=concepts)


def test_matrix_stale_order_rejected_on_load(isolated_paths, concepts, random_similarity):
    save_matrix(config.MATRIX_DIR / "m.npz", ConceptMatrix(random_similarity, concepts, "similarity"),
                canonical=concepts)
    with pytest.raises(co.ConceptOrderError, match="Rebuild"):
        load_matrix(config.MATRIX_DIR / "m.npz", canonical=concepts[::-1])


def test_matrix_rejects_asymmetric_and_bad_shape(concepts):
    a = np.arange(len(concepts) ** 2, dtype=float).reshape(len(concepts), -1)
    with pytest.raises(ValueError, match="symmetric"):
        ConceptMatrix(a, concepts, "distance")
    with pytest.raises(ValueError, match="shape"):
        ConceptMatrix(np.zeros((3, 3)), concepts, "distance")


def test_matrix_nan_mask_and_distance(concepts, random_similarity):
    s = random_similarity.copy()
    s[0, :] = s[:, 0] = np.nan       # concept 0 not embeddable
    m = ConceptMatrix(s, concepts, "similarity")
    assert not m.mask[0].any() and m.mask[1:, 1:].all()
    d = m.to_distance()
    assert d.kind == "distance" and np.allclose(np.diag(d.values), 0.0)
    assert len(d.upper()) == len(concepts) * (len(concepts) - 1) // 2


# ------------------------------------------------------------ results I/O ---
def test_result_table_has_provenance(isolated_paths):
    path = write_result_table(pd.DataFrame({"model": ["toy"], "rho": [0.3]}), "toy", "tests")
    prov = json.loads(path.with_name("toy.provenance.json").read_text())
    assert prov["producer"] == "tests" and prov["seed"] == config.SEED
    j = write_result_json({"x": np.float64(1.5), "n": np.int64(3)}, "toyj", "tests")
    assert json.loads(j.read_text()) == {"x": 1.5, "n": 3}


def test_parquet_roundtrip(isolated_paths):
    pytest.importorskip("pyarrow")
    p = write_parquet(FORMS, config.PROCESSED_DIR / "forms.parquet")
    pd.testing.assert_frame_equal(read_parquet(p), FORMS)


# --------------------------------------------- synthetic fixture integrity --
def test_fixture_ground_truth_is_self_consistent():
    """Recompute planted colexifications naively; guards the fixture itself,
    which later phases treat as ground truth."""
    fam = LANGUAGES.set_index("Language_ID")["Family"].replace("", np.nan)
    fam = fam.fillna(LANGUAGES.set_index("Language_ID")["Glottocode"])  # isolate = own family
    found: dict[tuple[str, str], dict[str, set]] = {}
    for lang, grp in FORMS.groupby("Language_ID"):
        for _, sub in grp.groupby("Form"):
            cs = sorted(sub["Concept"])
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    rec = found.setdefault((cs[i], cs[j]), {"languages": set(), "families": set()})
                    rec["languages"].add(lang)
                    rec["families"].add(fam[lang])
    got = {k: {"languages": len(v["languages"]), "families": len(v["families"])} for k, v in found.items()}
    assert got == EXPECTED_EDGES
    assert set(FORMS["Concept"]) == set(CONCEPTS)
