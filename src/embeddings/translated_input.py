"""Phase 4 · Multilingual-input condition: XLM-R and LaBSE fed non-English word forms.

Forms (DECISIONS.md D-17, D-48): for each language unit in
``config.TRANSLATED_UNITS`` (Hindustani = Hindi + Urdu, Russian, Swahili,
Indonesian, Turkish) the word forms come from CLICS⁴ ``Value`` (the source
orthography), which is grounded data rather than translation.

* A value is usable only if it is written in the language's standard script
  (first letter's Unicode script, ``config.TRANSLATED_SCRIPTS``) and contains no
  phonetic notation (IPA letters, modifier letters, combining diacritics after
  NFC, digits, underscores). Several CLICS sources give transcriptions, not
  orthography; models are trained on orthography.
* Per ISO code, the variety (dataset) with the most usable values is chosen;
  ties go to the alphabetically first dataset (deterministic).
* Values are cleaned: bracketed notes removed, first alternative kept.
* ``translations_curated.csv`` may fill gaps (concept_id, iso, form, source),
  each with a citable source; it ships empty, so every form used is CLICS data.

Embedding: each form is embedded bare (no English template) — XLM-R with
target pooling over the whole string, LaBSE with its sentence embedding. Hindi
and Urdu vectors are averaged into one Hindustani vector; a concept's vector is
the mean over units, required in >= TRANSLATED_MIN_UNITS of the 5 units.

Outputs: results/translated_forms.csv (every form used, with its source),
results/translated_input_colexified_pairs.csv (pairs sharing an input string, D-49),
results/translated_coverage.json, and (with torch) embeddings/{xlmr,labse}_translated.npz.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd

import config
from src.embeddings.contextual import (HFTokenEncoder, STSentenceEncoder, embed_contextual, embed_sentences,
                                       torch_available, update_manifest)
from src.embeddings.store import coverage, exists, save_embeddings
from src.utils.concept_order import load_concept_order
from src.utils.io import read_parquet, write_result_json, write_result_table
from src.utils.logging_utils import get_logger, log_run_header
from src.utils.seeds import set_all_seeds

log = get_logger(__name__, "phase4_translated_input")
CONDITIONS = {"xlmr_translated": "xlmr_base", "labse_translated": "labse"}


# --------------------------------------------------------------- cleaning ---
def clean_value(value: str | None) -> str:
    if not isinstance(value, str):
        return ""
    v = unicodedata.normalize("NFC", value)
    v = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", v)
    v = re.split(r"[,;/~|]", v)[0]
    return re.sub(r"\s+", " ", v).strip().strip("'\"-‑ ").strip()


def first_letter_script(text: str) -> str:
    for ch in text:
        if ch.isalpha():
            return unicodedata.name(ch, "UNKNOWN").split()[0]
    return ""


def looks_phonetic(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if 0x0250 <= cp <= 0x02FF or 0x1D00 <= cp <= 0x1DBF:          # IPA, modifier letters, phonetic ext.
            return True
        if 0x0300 <= cp <= 0x036F:                                      # combining diacritics left after NFC
            return True
        if ch.isdigit() or ch in "_:ːˑ*=":
            return True
    return False


def usable(form: str, iso: str) -> bool:
    return bool(form) and first_letter_script(form) == config.TRANSLATED_SCRIPTS[iso] and not looks_phonetic(form)


# -------------------------------------------------------------- selection ---
def select_forms(forms: pd.DataFrame, languages: pd.DataFrame, order: list[str],
                 curated: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One form per (concept, iso). Returns (forms table, dataset choice table)."""
    isos = [iso for unit in config.TRANSLATED_UNITS.values() for iso in unit]
    lg = languages[languages["iso639_3"].isin(isos)][["variety_id", "variety_name", "iso639_3", "contribution_id"]]
    f = forms[forms["concept_id"].isin(order)].merge(lg, on="variety_id")
    f = f.assign(clean=f["value"].map(clean_value))
    f["usable"] = [usable(c, iso) for c, iso in zip(f["clean"], f["iso639_3"])]
    counts = (f[f["usable"]].groupby(["iso639_3", "contribution_id", "variety_id"])["concept_id"].nunique()
              .rename("n_usable").reset_index())
    tot = f.groupby(["iso639_3", "contribution_id", "variety_id"])["concept_id"].nunique().rename("n_total").reset_index()
    choice = tot.merge(counts, how="left").fillna({"n_usable": 0})
    choice["n_usable"] = choice["n_usable"].astype(int)
    choice = choice.sort_values(["iso639_3", "n_usable", "contribution_id"], ascending=[True, False, True])
    choice["chosen"] = ~choice.duplicated("iso639_3") & (choice["n_usable"] > 0)
    keep = choice[choice["chosen"]][["iso639_3", "variety_id"]]
    sel = f[f["usable"]].merge(keep, on=["iso639_3", "variety_id"])
    sel = sel.sort_values(["iso639_3", "concept_id", "form_id"]).drop_duplicates(["iso639_3", "concept_id"])
    out = sel.rename(columns={"iso639_3": "iso", "clean": "form_used", "value": "clics_value"})[
        ["concept_id", "iso", "form_used", "clics_value", "variety_name", "contribution_id", "form_id"]]
    out["source"] = "CLICS4:" + out["contribution_id"]
    if curated is not None and len(curated):
        have = set(zip(out["concept_id"], out["iso"]))
        add = curated[[(c, i) not in have for c, i in zip(curated["concept_id"], curated["iso"])]]
        add = add.assign(form_used=add["form"], clics_value="", variety_name="", contribution_id="", form_id="")
        out = pd.concat([out, add[out.columns.drop("source").tolist() + ["source"]]], ignore_index=True)
    unit_of = {iso: unit for unit, isos_ in config.TRANSLATED_UNITS.items() for iso in isos_}
    out["unit"] = out["iso"].map(unit_of)
    return out.sort_values(["concept_id", "iso"]).reset_index(drop=True), choice


def load_curated() -> pd.DataFrame:
    p = config.TRANSLATED_CURATED_PATH
    if not p.exists():
        return pd.DataFrame(columns=["concept_id", "iso", "form", "source"])
    c = pd.read_csv(p, dtype=str, keep_default_na=False)
    if (c["source"].str.strip() == "").any():
        raise ValueError("Every curated translation needs a citable source")
    return c


def unit_coverage(sel: pd.DataFrame, order: list[str]) -> dict:
    per = sel.groupby("concept_id")["unit"].nunique().reindex(order, fill_value=0)
    return {
        "forms_per_unit": sel.groupby("unit")["concept_id"].nunique().to_dict(),
        "concepts_by_n_units": per.value_counts().sort_index().to_dict(),
        "n_concepts_meeting_min_units": int((per >= config.TRANSLATED_MIN_UNITS).sum()),
        "min_units": config.TRANSLATED_MIN_UNITS,
        "n_forms_from_curated": int((~sel["source"].str.startswith("CLICS4:")).sum()),
    }


def input_colexified_pairs(sel: pd.DataFrame) -> pd.DataFrame:
    """Concept pairs given an identical input string in at least one language
    (e.g. Hindi कल for YESTERDAY and TOMORROW). In the multilingual-input
    condition such pairs share a unit vector by construction, so H2 reports them
    separately (D-49)."""
    m = sel.merge(sel, on=["iso", "form_used"], suffixes=("_a", "_b"))
    m = m[m["concept_id_a"] < m["concept_id_b"]]
    out = (m.groupby(["concept_id_a", "concept_id_b"])
            .agg(input_isos=("iso", lambda s: ",".join(sorted(set(s)))),
                 forms=("form_used", lambda s: " | ".join(sorted(set(s)))))
            .reset_index().rename(columns={"concept_id_a": "concept_a", "concept_id_b": "concept_b"}))
    return out


# -------------------------------------------------------------- combining ---
def combine_units(unit_vectors: dict[str, np.ndarray], min_units: int = config.TRANSLATED_MIN_UNITS) -> np.ndarray:
    """Mean over the units available for each concept; NaN if fewer than min_units."""
    stack = np.stack(list(unit_vectors.values()))           # U, N, D (NaN where missing)
    ok = np.isfinite(stack).all(axis=2)                      # U, N
    n = ok.sum(axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nansum(np.where(ok[..., None], stack, 0.0), axis=0) / np.maximum(n, 1)[:, None]
    mean[n < min_units] = np.nan
    return mean


def embed_units(sel: pd.DataFrame, order: list[str], embed_words) -> dict[str, np.ndarray]:
    """Per unit: (N, D) array, NaN rows where the unit has no form. Within a unit
    with several ISO codes (Hindustani), available ISO vectors are averaged."""
    idx = {c: i for i, c in enumerate(order)}
    out: dict[str, np.ndarray] = {}
    for unit, isos in config.TRANSLATED_UNITS.items():
        per_iso = []
        for iso in isos:
            s = sel[sel["iso"] == iso]
            if s.empty:
                continue
            vec = embed_words(s["form_used"].tolist())
            arr = np.full((len(order), vec.shape[1]), np.nan)
            arr[s["concept_id"].map(idx).to_numpy()] = vec
            per_iso.append(arr)
        if per_iso:
            stack = np.stack(per_iso)
            with warnings.catch_warnings():        # all-NaN rows (no form in any ISO) stay NaN by design
                warnings.simplefilter("ignore", RuntimeWarning)
                out[unit] = np.nanmean(stack, axis=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    set_all_seeds()
    log_run_header(log)
    order = load_concept_order()
    forms = read_parquet(config.PROCESSED_DIR / "forms.parquet")
    languages = read_parquet(config.PROCESSED_DIR / "languages.parquet")
    sel, choice = select_forms(forms, languages, order, load_curated())
    write_result_table(sel, "translated_forms", producer=__name__)
    icp = input_colexified_pairs(sel)
    write_result_table(icp, "translated_input_colexified_pairs", producer=__name__)
    cov = unit_coverage(sel, order)
    cov["dataset_choice"] = choice.to_dict("records")
    cov["n_input_colexified_pairs"] = int(len(icp))
    write_result_json(cov, "translated_coverage", producer=__name__)
    log.info("Translated forms: %s per unit; %d/%d concepts have >= %d units",
             cov["forms_per_unit"], cov["n_concepts_meeting_min_units"], len(order), config.TRANSLATED_MIN_UNITS)

    todo = [c for c in CONDITIONS if args.force or not exists(c)]
    if not todo:
        return
    if not torch_available():
        log.warning("PENDING USER RUN: torch/transformers not installed; %s not embedded.", ", ".join(todo))
        return
    for cond in todo:
        base = config.MODELS[CONDITIONS[cond]]
        if base["kind"] == "sentence":
            enc = STSentenceEncoder(base["name"], base.get("revision", "main"))
            embed_words = lambda words, e=enc: embed_sentences(words, ("{w}",), e)["vectors"]   # noqa: E731
        else:
            enc = HFTokenEncoder(base["name"], base.get("revision", "main"))
            embed_words = lambda words, e=enc: embed_contextual(words, ("{w}",), e)["vectors"]  # noqa: E731
        units = embed_units(sel, order, embed_words)
        vec = combine_units(units)
        meta = {"model": base["name"], "resolved_revision": enc.revision, "input": "bare CLICS4 word forms",
                "units": list(units), **coverage(vec)}
        save_embeddings(cond, vec, order, meta, extras={f"unit_{u}": v.astype(np.float32) for u, v in units.items()})
        update_manifest(cond, meta)
        log.info("%s: %d/%d concepts embedded", cond, meta["n_embedded"], len(order))


if __name__ == "__main__":
    main()
