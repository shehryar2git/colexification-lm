"""Phase 4 tests: word2vec reader and lookup, contextual pooling, multilingual forms, similarity."""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

import config
from src.embeddings import contextual as cx
from src.embeddings import similarity as sim
from src.embeddings import static_w2v as w2v
from src.embeddings import translated_input as ti
from src.embeddings.store import load_embeddings, save_embeddings
from src.utils.concept_order import ConceptOrderError, freeze_concept_order


# --------------------------------------------------------------- word2vec ---
def _write_w2v(path, words, dim=4, newline=False, gz=True):
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(len(words), dim)).astype("<f4")
    body = f"{len(words)} {dim}\n".encode()
    for w, v in zip(words, vecs):
        body += w.encode("utf-8") + b" " + v.tobytes() + (b"\n" if newline else b"")
    (gzip.open if gz else open)(path, "wb").write(body)
    return dict(zip(words, vecs))


@pytest.mark.parametrize("newline,gz", [(False, True), (True, True), (False, False)])
def test_stream_vectors_reads_binary_format(tmp_path, newline, gz):
    words = ["</s>", "hand", "Hand", "upper_arm", "gray", "über", "I"]
    truth = _write_w2v(tmp_path / ("v.bin.gz" if gz else "v.bin"), words, newline=newline, gz=gz)
    found, rank, vocab = w2v.stream_vectors(tmp_path / ("v.bin.gz" if gz else "v.bin"),
                                            {"hand", "über", "I", "absent"}, chunk=64)
    assert vocab == len(words) and set(found) == {"hand", "über", "I"}
    np.testing.assert_array_equal(found["über"], truth["über"])
    assert rank == {"hand": 1, "über": 5, "I": 6}


def test_lookup_rule_order():
    v = {k: np.full(2, i, dtype=np.float32) for i, k in enumerate(["Hand", "hand", "gray", "upper_arm", "old", "man"])}
    r = {k: i for i, k in enumerate(v)}
    assert w2v.lookup("Hand", v, r)[1:3] == ("exact", "Hand")
    assert w2v.lookup("HAND", v, r)[1:3] == ("lower", "hand")
    assert w2v.lookup("grey", v, r)[1:3] == ("spelling_variant", "gray")
    assert w2v.lookup("upper arm", v, r)[1:3] == ("phrase", "upper_arm")
    vec, rule, key, rank = w2v.lookup("old man", v, r)
    assert rule == "token_mean" and key == "old+man" and vec[0] == pytest.approx(4.5) and rank == 4.5
    assert w2v.lookup("in front of", {}, {})[1] == "missing"
    assert w2v.candidate_keys("in front of")["token_mean"] == ["front"]     # function words dropped


@pytest.mark.slow
def test_reader_matches_gensim_on_real_file():
    gensim = pytest.importorskip("gensim")
    if not config.W2V_PATH.exists():
        pytest.skip("word2vec file not downloaded")
    kv = gensim.models.KeyedVectors.load_word2vec_format(str(config.W2V_PATH), binary=True, limit=5000)
    probe = [w for w in ("hand", "tree", "water", "go") if w in kv.key_to_index]
    found, rank, _ = w2v.stream_vectors(config.W2V_PATH, set(probe))
    for w in probe:
        np.testing.assert_allclose(found[w], kv[w], rtol=0, atol=0)
        assert rank[w] == kv.key_to_index[w]


# -------------------------------------------------------------- contextual --
def test_render_and_span():
    text, span = cx.render("the word '{w}'", "upper arm")
    assert text == "the word 'upper arm'" and text[span[0]:span[1]] == "upper arm"
    with pytest.raises(ValueError):
        cx.render("no slot", "x")


def test_pool_target_selects_overlapping_tokens_only():
    hidden = np.arange(2 * 5 * 3, dtype=float).reshape(2, 5, 3)       # 2 layers, 5 tokens, dim 3
    offsets = [(0, 0), (0, 3), (4, 6), (6, 9), (0, 0)]                # CLS, "the", "fo", "rearm", SEP
    pooled = cx.pool_target(hidden, offsets, (4, 11))
    np.testing.assert_allclose(pooled, hidden[:, [2, 3], :].mean(axis=1))
    with pytest.raises(ValueError):
        cx.pool_target(hidden, offsets, (20, 25))


class FakeEncoder:
    """Deterministic stand-in for a transformer: one token per whitespace word;
    each token's vector at layer l is (hash-derived word vector) * (l + 1)."""
    layers, dim = 3, 4

    def vec(self, word: str) -> np.ndarray:
        import zlib
        g = np.random.default_rng(zlib.crc32(word.encode()))
        return g.normal(size=self.dim)

    def __call__(self, texts):
        out = []
        for t in texts:
            words, offs, pos = t.split(" "), [(0, 0)], 0
            for w in words:
                offs.append((pos, pos + len(w)))
                pos += len(w) + 1
            offs.append((0, 0))
            tok = np.stack([np.zeros(self.dim)] + [self.vec(w) for w in words] + [np.zeros(self.dim)])
            out.append((np.stack([tok * (l + 1) for l in range(self.layers)]), offs))
        return out


def test_embed_contextual_template_and_layer_averages():
    enc = FakeEncoder()
    res = cx.embed_contextual(["hand", "tree"], ["{w}", "a {w} here", "{w} again"], enc, layer=-1, batch_size=2)
    assert res["vectors"].shape == (2, 4) and res["per_template"].shape == (3, 2, 4)
    assert res["per_layer"].shape == (3, 2, 4)
    # the target token's vector does not depend on the template in the fake model
    np.testing.assert_allclose(res["vectors"][0], enc.vec("hand") * 3)
    np.testing.assert_allclose(res["per_layer"][0, 1], enc.vec("tree") * 1)


def test_embed_sentences_averages_templates():
    enc = lambda texts: np.array([[len(t), 1.0] for t in texts])        # noqa: E731
    res = cx.embed_sentences(["ab"], ["{w}", "x {w}"], enc)
    np.testing.assert_allclose(res["vectors"], [[3.0, 1.0]])            # mean of lengths 2 and 4


# ----------------------------------------------------- multilingual input --
def test_clean_and_usable():
    assert ti.clean_value("мост (1), мостик") == "мост"
    assert ti.clean_value("bata; batu") == "bata"
    assert ti.usable("दूध", "hin") and ti.usable("остров", "rus") and ti.usable("yarı", "tur")
    assert not ti.usable("ɡəɾd̪ən", "urd")          # IPA transcription, wrong script
    assert not ti.usable("takɔ", "swh")             # IPA letter
    assert not ti.usable("kumi_na_moja", "swh")     # underscore notation
    assert not ti.usable("tʃai", "swh")
    assert not ti.usable("ostrov", "rus")           # Latin transliteration of Russian


def test_select_forms_prefers_orthographic_dataset():
    forms = pd.DataFrame({
        "form_id": ["1", "2", "3", "4", "5"], "variety_id": ["r1", "r1", "r2", "r2", "s1"],
        "concept_id": ["HAND", "TREE", "HAND", "TREE", "HAND"],
        "value": ["рука", "дерево", "ruka", "derevo", "mkono"]})
    languages = pd.DataFrame({"variety_id": ["r1", "r2", "s1"], "variety_name": ["Russian"] * 2 + ["Swahili"],
                              "iso639_3": ["rus", "rus", "swh"], "contribution_id": ["nel", "dunn", "wold"]})
    sel, choice = ti.select_forms(forms, languages, ["HAND", "TREE"])
    assert set(sel.loc[sel["iso"] == "rus", "form_used"]) == {"рука", "дерево"}
    assert choice.loc[choice["chosen"], "contribution_id"].tolist() == ["nel", "wold"]
    assert set(sel["unit"]) == {"russian", "swahili"}
    cur = pd.DataFrame({"concept_id": ["TREE"], "iso": ["swh"], "form": ["mti"], "source": ["TUKI 2001, p. 1"]})
    sel2, _ = ti.select_forms(forms, languages, ["HAND", "TREE"], cur)
    assert sel2.set_index(["concept_id", "iso"]).loc[("TREE", "swh"), "source"] == "TUKI 2001, p. 1"


def test_combine_units_requires_min_units():
    nan = np.nan
    units = {"a": np.array([[1.0, 1.0], [nan, nan]]), "b": np.array([[3.0, 3.0], [2.0, 2.0]]),
             "c": np.array([[5.0, 5.0], [4.0, 4.0]])}
    out = ti.combine_units(units, min_units=3)
    np.testing.assert_allclose(out[0], [3.0, 3.0])
    assert np.isnan(out[1]).all()                     # only 2 units available
    np.testing.assert_allclose(ti.combine_units(units, min_units=2)[1], [3.0, 3.0])


def test_embed_units_averages_hindi_and_urdu(monkeypatch):
    monkeypatch.setattr(config, "TRANSLATED_UNITS", {"hindustani": ("hin", "urd"), "russian": ("rus",)})
    sel = pd.DataFrame({"concept_id": ["A", "A", "B"], "iso": ["hin", "urd", "rus"],
                        "form_used": ["x", "yy", "zzz"]})
    out = ti.embed_units(sel, ["A", "B"], lambda words: np.array([[len(w), 0.0] for w in words]))
    np.testing.assert_allclose(out["hindustani"][0], [1.5, 0.0])
    assert np.isnan(out["hindustani"][1]).all() and out["russian"][1][0] == 3


# -------------------------------------------------------------- similarity --
def test_cosine_matrix_centering_and_nan():
    v = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [np.nan, 0.0]])
    raw = sim.cosine_matrix(v, center=False)
    assert raw[0, 1] == pytest.approx(0.0) and raw[0, 2] == pytest.approx(1 / np.sqrt(2))
    assert np.isnan(raw[3]).all() and np.isnan(raw[:, 3]).all() and raw[0, 0] == 1.0
    c = sim.cosine_matrix(v, center=True)            # mean of valid rows (2/3, 2/3) removed
    # (1/3, -2/3) vs (-2/3, 1/3): cosine = (-2/9 - 2/9) / (5/9) = -0.8
    assert c[0, 1] == pytest.approx(-0.8)
    assert np.allclose(c[:3, :3], c[:3, :3].T)


def test_store_roundtrip_and_order_guard(isolated_paths):
    order = ["A", "B", "C"]
    freeze_concept_order(order, path=config.CONCEPT_ORDER_PATH)
    save_embeddings("toy", np.eye(3), order, {"model": "toy"}, extras={"per_template": np.zeros((2, 3, 3))})
    v, ids, meta, ex = load_embeddings("toy")
    assert ids == order and meta["model"] == "toy" and ex["per_template"].shape == (2, 3, 3)
    with pytest.raises(ConceptOrderError):
        save_embeddings("bad", np.eye(3), ["B", "A", "C"], {})


def test_input_colexified_pairs():
    sel = pd.DataFrame({"concept_id": ["TOMORROW", "YESTERDAY", "TREE", "WOOD", "TREE"],
                        "iso": ["hin", "hin", "swh", "swh", "tur"], "form_used": ["कल", "कल", "mti", "mti", "ağaç"]})
    out = ti.input_colexified_pairs(sel).set_index(["concept_a", "concept_b"])
    assert out.loc[("TOMORROW", "YESTERDAY"), "input_isos"] == "hin"
    assert out.loc[("TREE", "WOOD"), "forms"] == "mti" and len(out) == 2
