"""Phase 1 tests: CLDF loading, Glottolog attachment, concept selection, WordNet, audit sheet."""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

import config
from src.data import attach_glottolog as ag
from src.data import english_colex as ec
from src.data import load_clics as lc
from src.data import select_concepts as sc
from src.data.wordnet_db import WordNet

CLDF = lc.CLDF


# ------------------------------------------------------------ CLDF loading --
def _write_mini_cldf(d, zipped_forms=True):
    d.mkdir(parents=True, exist_ok=True)
    col = lambda name, prop=None: {"name": name, **({"propertyUrl": CLDF + prop} if prop else {})}  # noqa: E731
    meta = {"dc:conformsTo": CLDF + "Wordlist", "tables": [
        {"url": "forms.csv", "dc:conformsTo": CLDF + "FormTable", "tableSchema": {"columns": [
            col("FID", "id"), col("Lang", "languageReference"), col("Param", "parameterReference"),
            col("TheForm", "form"), col("Seg", "segments"), col("Orig", "value"),
            {"name": "LID", "propertyUrl": "http://purl.org/dc/terms/identifier"}]}},
        {"url": "languages.csv", "dc:conformsTo": CLDF + "LanguageTable", "tableSchema": {"columns": [
            col("ID", "id"), col("Name", "name"), col("Glottocode", "glottocode"),
            col("ISO", "iso639P3code"), col("Macroarea", "macroarea"), col("Lat", "latitude"),
            col("Lon", "longitude"), col("Contribution_ID", "contributionReference"),
            col("Family"), col("Family_Name")]}},
        {"url": "concepts.csv", "dc:conformsTo": CLDF + "ParameterTable", "tableSchema": {"columns": [
            col("ID", "id"), col("Concepticon_ID", "concepticonReference"), col("Concepticon_Gloss"),
            col("Language_Count"), col("Family_Count")]}},
    ]}
    (d / "Wordlist-metadata.json").write_text(json.dumps(meta))
    forms = pd.DataFrame({"FID": ["1", "2", "3", "4"], "Lang": ["L1", "L1", "L2", "L2"],
                          "Param": ["hand", "arm", "hand", "arm"], "TheForm": ["ma", "ma", "te", ""],
                          "Seg": ["m a", "m a", "t e", ""], "Orig": ["Ma", "Ma", "Te", ""],
                          # L1's two forms come from ONE source entry split across concepts
                          "LID": ["src-1", "src-1", "src-7", "src-8"]})
    if zipped_forms:
        with zipfile.ZipFile(d / "forms.csv.zip", "w") as zf:
            zf.writestr("forms.csv", forms.to_csv(index=False))
    else:
        forms.to_csv(d / "forms.csv", index=False)
    pd.DataFrame({"ID": ["L1", "L2"], "Name": ["One", "Two"], "Glottocode": ["aaaa1111", "bbbb1111"],
                  "ISO": ["aaa", ""], "Macroarea": ["Eurasia", "Africa"], "Lat": ["1.5", "2"],
                  "Lon": ["3", "4"], "Contribution_ID": ["ds", "ds"], "Family": ["fama1", "famb1"],
                  "Family_Name": ["A", "B"]}).to_csv(d / "languages.csv", index=False)
    pd.DataFrame({"ID": ["hand", "arm"], "Concepticon_ID": ["1277", "1673"],
                  "Concepticon_Gloss": ["HAND", "ARM"], "Language_Count": ["2", "1"],
                  "Family_Count": ["2", "1"]}).to_csv(d / "concepts.csv", index=False)
    return d / "Wordlist-metadata.json"


@pytest.mark.parametrize("zipped", [True, False])
def test_metadata_reader_resolves_by_property(tmp_path, zipped):
    meta = _write_mini_cldf(tmp_path / "cldf", zipped_forms=zipped)
    tables = lc.read_with_metadata(meta)
    assert list(tables["FormTable"].columns) == list(lc.FORM_FIELDS)
    forms, varieties, concepts = lc.tidy(tables)
    assert len(forms) == 3                       # empty form dropped
    assert set(forms["concept_id"]) == {"HAND", "ARM"}
    assert forms.set_index("form_id")["shared_entry"].to_dict() == {"1": True, "2": True, "3": False}
    assert forms.set_index("form_id").loc["1", "segments"] == "m a"
    assert varieties["clics_latitude"].tolist() == [1.5, 2.0]
    assert concepts.set_index("concept_id").loc["HAND", "clics_family_count"] == 2


def test_missing_required_column_fails_loudly(tmp_path):
    meta = _write_mini_cldf(tmp_path / "cldf")
    m = json.loads(meta.read_text())
    m["tables"][0]["tableSchema"]["columns"][3].pop("propertyUrl")   # no #form column any more
    meta.write_text(json.dumps(m))
    with pytest.raises(KeyError, match="form"):
        lc.resolve_schema(meta)


def test_tidy_rejects_unknown_parameter(tmp_path):
    tables = lc.read_with_metadata(_write_mini_cldf(tmp_path / "cldf"))
    tables["FormTable"].loc[0, "parameter_id"] = "nonexistent"
    with pytest.raises(ValueError, match="unknown parameters"):
        lc.tidy(tables)


def test_pycldf_and_metadata_reader_agree(tmp_path):
    pytest.importorskip("pycldf")
    meta = _write_mini_cldf(tmp_path / "cldf", zipped_forms=False)
    a, b = lc.read_with_metadata(meta), lc.read_with_pycldf(meta)
    for comp in a:
        pd.testing.assert_frame_equal(a[comp].fillna("").astype(str).reset_index(drop=True),
                                      b[comp].fillna("").astype(str).reset_index(drop=True))


# ------------------------------------------------------- Glottolog attach --
@pytest.fixture
def mini_glottolog():
    rows = [
        # ID, Name, Level, Family_ID, Language_ID, Is_Isolate, Macroarea, Lat, Lon
        ("fama1234", "Alpha", "family", None, None, None, None, np.nan, np.nan),
        ("suba1234", "Alpha-Sub", "family", "fama1234", None, None, None, np.nan, np.nan),
        ("lana1234", "Lang A", "language", "fama1234", None, "False", "Eurasia", 10.0, 20.0),
        ("diaa1234", "Dial A", "dialect", "fama1234", "lana1234", None, None, np.nan, np.nan),
        ("isol1234", "Isolate", "language", None, None, "True", "Africa;Eurasia", 1.0, 2.0),
        ("book1242", "Bookkeeping", "family", None, None, None, None, np.nan, np.nan),
        ("bkla1234", "Bk Lang", "language", "book1242", None, "False", "Eurasia", 5.0, 5.0),
    ]
    return pd.DataFrame(rows, columns=ag.GLOTTOLOG_COLS).set_index("ID")


def test_resolve_languoid(mini_glottolog):
    g = mini_glottolog
    assert ag.resolve_languoid("lana1234", g) == {"glottolog_level": "language", "language_gc": "lana1234", "family_gc": "fama1234"}
    assert ag.resolve_languoid("diaa1234", g)["language_gc"] == "lana1234"
    assert ag.resolve_languoid("isol1234", g)["family_gc"] == "isol1234"        # isolate = own family
    sub = ag.resolve_languoid("suba1234", g)
    assert sub["language_gc"] == "suba1234" and sub["family_gc"] == "fama1234"
    assert ag.resolve_languoid("zzzz9999", g)["glottolog_level"] is None


def test_attach_flags_and_geography(mini_glottolog):
    v = pd.DataFrame({
        "variety_id": ["1", "2", "3", "4", "5"],
        "glottocode": ["diaa1234", "lana1234", "isol1234", "bkla1234", "zzzz9999"],
        "clics_macroarea": ["Eurasia"] * 5, "clics_latitude": [0.0] * 4 + [7.0],
        "clics_longitude": [0.0] * 4 + [8.0], "clics_family": [""] * 5, "clics_family_name": [""] * 5,
    })
    out = ag.attach(v, mini_glottolog).set_index("variety_id")
    assert out.loc["1", "latitude"] == 10.0 and out.loc["1", "coord_source"] == "glottolog_language"
    assert out.loc["3", "is_isolate"] and out.loc["3", "macroarea"] == "Africa" and out.loc["3", "macroarea_multi"]
    assert not out.loc["4", "include"] and "Bookkeeping" in out.loc["4", "exclude_reason"]
    assert not out.loc["5", "include"] and out.loc["5", "coord_source"] == "clics"
    assert out.loc["1", "include"] and out.loc["2", "include"]


# ------------------------------------------------------- concept selection --
def test_kneedle_finds_planted_knee():
    x = np.arange(0, 101, 5)
    y = np.where(x <= 40, 1000 - 20 * x, 200 - 1 * (x - 40))   # steep then flat, knee at 40
    idx, diff = sc.kneedle(x, y)
    assert x[idx] == 40
    assert diff[0] == pytest.approx(0) and diff[-1] == pytest.approx(0)


def test_kneedle_rejects_degenerate():
    with pytest.raises(ValueError):
        sc.kneedle([1, 2, 3], [5, 5, 5])


def test_choose_threshold_constrains_to_target():
    curve = pd.DataFrame({"min_families": [10, 20, 30, 40, 50],
                          "n_concepts": [2000, 900, 600, 450, 400]})
    out = sc.choose_threshold(curve, (300, 500))
    assert out["knee_threshold"] == 20 and out["constrained_to_target"]
    assert out["chosen_threshold"] == 40     # nearest admissible to the knee


def test_clean_gloss():
    assert sc.clean_gloss("BLOW (OF WIND)") == "blow"
    assert sc.clean_gloss("UPPER LEG (THIGH)") == "upper leg"
    assert sc.clean_gloss("IN FRONT OF") == "in front of"


def test_apply_glosses_and_collision_check(tmp_path):
    sel = pd.DataFrame({"concept_id": ["FLY (INSECT)", "FLY (MOVE THROUGH AIR)", "WE (EXCLUSIVE)", "I"]})
    ovp = tmp_path / "ov.csv"
    pd.DataFrame({"concept_id": ["FLY (INSECT)", "WE (EXCLUSIVE)", "I"], "action": ["gloss", "exclude", "gloss"],
                  "gloss": ["housefly", "", "I"], "collision_group": ["fly", "", ""],
                  "reason": ["r", "r", "r"]}).to_csv(ovp, index=False)
    out = sc.apply_glosses(sel, sc.load_overrides(ovp)).set_index("concept_id")
    assert out.loc["FLY (INSECT)", "gloss"] == "housefly"
    assert out.loc["FLY (MOVE THROUGH AIR)", "gloss"] == "fly"
    assert out.loc["I", "gloss"] == "I"                       # override keeps case
    assert out.loc["WE (EXCLUSIVE)", "excluded_reason"] == "r"
    collide = pd.DataFrame({"concept_id": ["A", "B"], "gloss": ["x", "x"]})
    with pytest.raises(ValueError, match="share English glosses"):
        sc.check_unique_glosses(collide)


def test_real_overrides_file_is_valid():
    ov = sc.load_overrides()
    assert (ov["action"] == "exclude").sum() >= 1
    groups = ov[ov["collision_group"] != ""].groupby("collision_group")["gloss"]
    assert all(g.is_unique for _, g in groups), "glosses inside a collision group must differ"


# ------------------------------------------------------------- WordNet ------
def _mini_wordnet_zip(path):
    """Two senses for 'tongue', one for 'language', 'see'/'understand' sharing a synset."""
    files = {
        "index.noun": "  licence header line\n"
                      "tongue n 2 0 2 0 00000001 00000002\n"
                      "language n 1 0 1 0 00000003\n"
                      "wood n 2 0 2 0 00000004 00000005\n"
                      "tree n 1 0 1 0 00000006\n",
        "data.noun": "  licence header line\n"
                     "00000001 08 n 01 tongue 0 000 | a mobile mass of muscular tissue\n"
                     "00000002 10 n 02 natural_language 0 tongue 0 000 | a human written or spoken language used by a community\n"
                     "00000003 10 n 01 language 0 000 | a systematic means of communicating\n"
                     "00000004 27 n 01 wood 0 000 | the hard fibrous substance under the bark of trees\n"
                     "00000005 18 n 01 Natalie_Wood 0 000 | United States film actress\n"
                     "00000006 20 n 01 tree 0 000 | a tall perennial woody plant\n",
        "noun.exc": "trees tree\n",
        "index.verb": "see v 2 0 2 0 00000010 00000011\nunderstand v 1 0 1 0 00000011\n",
        "data.verb": "00000010 39 v 01 see 0 000 | perceive by sight\n"
                     "00000011 31 v 02 understand 0 see 0 000 | know and comprehend the nature or meaning of\n",
        "verb.exc": "",
        "index.adj": "", "data.adj": "", "adj.exc": "",
        "index.adv": "", "data.adv": "", "adv.exc": "",
    }
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in files.items():
            zf.writestr(f"wordnet/{name}", text)
    return path


@pytest.fixture
def mini_wn(tmp_path):
    return WordNet(_mini_wordnet_zip(tmp_path / "wn.zip"))


def test_wordnet_reader(mini_wn):
    assert {s.id for s in mini_wn.synsets_for("tongue")} == {"00000001-n", "00000002-n"}
    assert mini_wn.synsets["00000002-n"].lemmas == ("natural_language", "tongue")
    assert mini_wn.morphy("trees", "n") == ["tree"]               # exception list
    assert mini_wn.morphy("tongues", "n") == ["tongue"]           # detachment rule
    assert [s.id for s in mini_wn.synsets_for("wood")] == ["00000004-n"]   # proper noun filtered
    assert len(mini_wn.synsets_for("wood", include_proper=True)) == 2


def test_candidate_pairs_both_routes(mini_wn):
    glosses = pd.Series({"TONGUE": "tongue", "LANGUAGE": "language", "SEE": "see",
                         "UNDERSTAND": "understand", "TREE": "tree", "WOOD": "wood"})
    cands = ec.candidate_pairs(ec.concept_senses(glosses, mini_wn), mini_wn).set_index(["concept_a", "concept_b"])
    assert cands.loc[("SEE", "UNDERSTAND"), "evidence_route"] == "shared_lemma"
    assert cands.loc[("SEE", "UNDERSTAND"), "gloss_is_shared"]
    assert cands.loc[("LANGUAGE", "TONGUE"), "evidence_route"] == "definition"
    assert ("TREE", "WOOD") in cands.index        # over-recall by design: the audit decides
    assert ("LANGUAGE", "SEE") not in cands.index


def test_audit_sheet_preserves_human_decisions(isolated_paths, mini_wn, monkeypatch):
    monkeypatch.setattr(config, "ENGLISH_AUDIT_PATH", isolated_paths / "audit" / "check.csv")
    glosses = pd.Series({"TONGUE": "tongue", "LANGUAGE": "language", "SEE": "see", "UNDERSTAND": "understand"})
    cands = ec.candidate_pairs(ec.concept_senses(glosses, mini_wn), mini_wn)
    edges = pd.DataFrame({"concept_a": ["LANGUAGE", "SEE"], "concept_b": ["TONGUE", "UNDERSTAND"],
                          "n_families": [13, 3]})
    sheet = ec.write_audit_sheet(cands, edges)
    assert len(sheet) == 2 and (sheet["decision"] == "").all()
    df = pd.read_csv(config.ENGLISH_AUDIT_PATH, dtype=str, keep_default_na=False)
    df.loc[df["concept_a"] == "SEE", ["decision", "annotator"]] = ["present", "human"]
    df.to_csv(config.ENGLISH_AUDIT_PATH, index=False)
    sheet2 = ec.write_audit_sheet(cands, edges).set_index("concept_a")
    assert sheet2.loc["SEE", "decision"] == "present" and sheet2.loc["SEE", "annotator"] == "human"
    df.loc[df["concept_a"] == "SEE", "decision"] = "maybe"
    df.to_csv(config.ENGLISH_AUDIT_PATH, index=False)
    with pytest.raises(ValueError, match="Invalid audit decisions"):
        ec.write_audit_sheet(cands, edges)


# ------------------------------------------------- real data (slow, cached) --
@pytest.mark.slow
def test_real_working_set_invariants():
    from src.utils.concept_order import load_concept_order
    from src.utils.io import read_parquet
    order = load_concept_order()
    c = read_parquet(config.PROCESSED_DIR / "concepts.parquet")
    w = c[c["working"]]
    assert sorted(order) == order == sorted(w["concept_id"])
    lo, hi = config.CONCEPT_SET_TARGET
    assert lo <= len(order) <= hi
    assert w["gloss"].is_unique
    assert set(config.ANCHOR_CONCEPTS) <= set(order)
