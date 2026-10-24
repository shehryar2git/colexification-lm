"""Phase 1 · CLICS⁴ CLDF  →  tidy forms / varieties / concepts tables.

Columns are resolved by **CLDF property URL**, not by literal header, so a
renamed column in a future release fails loudly instead of silently loading
the wrong field (DECISIONS.md D-03). Two readers are provided:

* ``read_with_pycldf`` — the reference reader (``pycldf``), used when installed;
* ``read_with_metadata`` — a dependency-free reader that parses the same
  ``Wordlist-metadata.json`` and reads (optionally zipped) CSVs with pandas.

``tests/test_phase1_data.py`` asserts both give identical tables when pycldf
is available. Outputs (data/processed/):

* ``forms.parquet``          form_id, variety_id, concept_id, form, segments, value, local_id, shared_entry
* ``varieties_raw.parquet``  variety-level metadata exactly as shipped by CLICS⁴
* ``concepts_raw.parquet``   concept_id (= Concepticon gloss), concepticon_id, CLICS counts
* ``results/clics_schema.json`` the resolved table/column map (documentation)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config
from src.utils.io import write_parquet, write_result_json
from src.utils.logging_utils import get_logger, log_run_header
from src.utils.seeds import set_all_seeds

CLDF = "http://cldf.clld.org/v1.0/terms.rdf#"

# Properties we need from each component: output column -> CLDF property (or literal name)
FORM_FIELDS = {
    "form_id": CLDF + "id",
    "variety_id": CLDF + "languageReference",
    "parameter_id": CLDF + "parameterReference",
    "form": CLDF + "form",
    "segments": CLDF + "segments",   # CLICS identity criterion (D-03 amendment)
    "value": CLDF + "value",
    "local_id": "http://purl.org/dc/terms/identifier",   # the source's own entry ID (D-37)
}
LANGUAGE_FIELDS = {
    "variety_id": CLDF + "id",
    "variety_name": CLDF + "name",
    "glottocode": CLDF + "glottocode",
    "iso639_3": CLDF + "iso639P3code",
    "clics_macroarea": CLDF + "macroarea",
    "clics_latitude": CLDF + "latitude",
    "clics_longitude": CLDF + "longitude",
    "contribution_id": CLDF + "contributionReference",
    "clics_family": "Family",            # non-standard column: matched by name
    "clics_family_name": "Family_Name",
}
PARAMETER_FIELDS = {
    "parameter_id": CLDF + "id",
    "concepticon_id": CLDF + "concepticonReference",
    "concepticon_gloss": "Concepticon_Gloss",
    "clics_language_count": "Language_Count",
    "clics_family_count": "Family_Count",
}
COMPONENTS = {"FormTable": FORM_FIELDS, "LanguageTable": LANGUAGE_FIELDS, "ParameterTable": PARAMETER_FIELDS}

log = get_logger(__name__, "phase1_load_clics")


# ----------------------------------------------------------- schema resolution
def resolve_schema(metadata_path: Path) -> dict[str, dict]:
    """Map each needed CLDF component to its file and {out_col: header}."""
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    schema: dict[str, dict] = {}
    for table in meta["tables"]:
        conforms = str(table.get("dc:conformsTo", ""))
        component = conforms.rsplit("#", 1)[-1] if conforms.startswith(CLDF) else None
        if component not in COMPONENTS:
            continue
        cols = table["tableSchema"]["columns"]
        by_prop = {c.get("propertyUrl"): c["name"] for c in cols if c.get("propertyUrl")}
        by_name = {c["name"]: c["name"] for c in cols}
        mapping = {}
        for out, key in COMPONENTS[component].items():
            header = by_prop.get(key) if key.startswith("http") else by_name.get(key)
            if header is None:
                raise KeyError(f"{component}: required column {key!r} not found in {metadata_path.name}")
            mapping[out] = header
        schema[component] = {"url": table["url"], "columns": mapping}
    missing = set(COMPONENTS) - set(schema)
    if missing:
        raise KeyError(f"CLDF components missing from {metadata_path.name}: {sorted(missing)}")
    return schema


def _table_path(cldf_dir: Path, url: str) -> Path:
    """CLDF datasets may ship tables zipped as ``<url>.zip``."""
    plain = cldf_dir / url
    if plain.exists():
        return plain
    zipped = cldf_dir / (url + ".zip")
    if zipped.exists():
        return zipped
    raise FileNotFoundError(f"Neither {plain} nor {zipped} exists")


# --------------------------------------------------------------------- readers
def read_with_metadata(metadata_path: Path) -> dict[str, pd.DataFrame]:
    schema = resolve_schema(metadata_path)
    out: dict[str, pd.DataFrame] = {}
    for component, spec in schema.items():
        path = _table_path(metadata_path.parent, spec["url"])
        cols = spec["columns"]
        df = pd.read_csv(path, usecols=list(cols.values()), dtype=str, keep_default_na=False,
                         na_values=[""], encoding="utf-8")
        out[component] = df.rename(columns={v: k for k, v in cols.items()})[list(cols)]
    return out


def read_with_pycldf(metadata_path: Path) -> dict[str, pd.DataFrame]:
    from pycldf import Dataset  # imported lazily: optional at import time

    ds = Dataset.from_metadata(metadata_path)
    schema = resolve_schema(metadata_path)
    out: dict[str, pd.DataFrame] = {}
    for component, spec in schema.items():
        cols = spec["columns"]
        rows = [{k: r.get(v) for k, v in cols.items()} for r in ds[component]]
        df = pd.DataFrame(rows, columns=list(cols))
        for c in df.columns:  # normalise to the string dtype the metadata reader produces
            df[c] = df[c].map(lambda x: None if x is None or x == "" else
                              (";".join(map(str, x)) if isinstance(x, list) else str(x)))
        out[component] = df
    return out


def load(metadata_path: Path) -> tuple[dict[str, pd.DataFrame], str]:
    try:
        tables = read_with_pycldf(metadata_path)
        return tables, "pycldf"
    except ImportError:
        log.warning("pycldf not installed: using the metadata-driven pandas reader (D-03).")
    except Exception as exc:  # pragma: no cover - depends on pycldf version
        log.warning("pycldf failed (%s: %s): using the metadata-driven pandas reader.",
                    type(exc).__name__, exc)
    return read_with_metadata(metadata_path), "metadata-pandas"


# --------------------------------------------------------------------- tidying
def tidy(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params = tables["ParameterTable"].copy()
    if params["concepticon_gloss"].isna().any():
        raise ValueError("Some CLICS concepts lack a Concepticon gloss")
    if not params["concepticon_gloss"].is_unique:
        raise ValueError("Concepticon glosses are not unique; cannot use them as concept IDs")
    params["concept_id"] = params["concepticon_gloss"]
    for c in ("clics_language_count", "clics_family_count"):
        params[c] = pd.to_numeric(params[c]).astype("int64")
    params["concepticon_id"] = params["concepticon_id"].astype("int64")

    forms = tables["FormTable"].copy()
    n0 = len(forms)
    forms = forms.dropna(subset=["form"])
    forms["form"] = forms["form"].str.strip()
    forms = forms[forms["form"] != ""]
    log.info("Forms: %d rows, %d dropped for empty Form", n0, n0 - len(forms))
    pid2cid = params.set_index("parameter_id")["concept_id"]
    unknown = ~forms["parameter_id"].isin(pid2cid.index)
    if unknown.any():
        raise ValueError(f"{unknown.sum()} forms reference unknown parameters")
    forms["concept_id"] = forms["parameter_id"].map(pid2cid)
    forms["segments"] = forms["segments"].fillna("").str.strip()
    no_seg = forms["segments"] == ""
    if no_seg.any():
        log.warning("%d forms have no Segments; they cannot take part in colexifications", no_seg.sum())
    # D-37: a source entry mapped to several concepts (e.g. one item "ARM OR HAND")
    n_lid = forms.groupby(["variety_id", "local_id"])["concept_id"].transform("nunique")
    forms["shared_entry"] = forms["local_id"].notna() & (n_lid > 1)
    log.info("%.2f%% of forms share their source entry with another concept", 100 * forms["shared_entry"].mean())
    forms = forms[["form_id", "variety_id", "concept_id", "form", "segments", "value", "local_id",
                   "shared_entry"]].reset_index(drop=True)

    varieties = tables["LanguageTable"].copy()
    for c in ("clics_latitude", "clics_longitude"):
        varieties[c] = pd.to_numeric(varieties[c])
    if not varieties["variety_id"].is_unique:
        raise ValueError("Variety IDs are not unique")
    orphan = ~forms["variety_id"].isin(varieties["variety_id"])
    if orphan.any():
        raise ValueError(f"{orphan.sum()} forms reference unknown varieties")

    concepts = params[["concept_id", "parameter_id", "concepticon_id", "concepticon_gloss",
                       "clics_language_count", "clics_family_count"]].reset_index(drop=True)
    return forms, varieties.reset_index(drop=True), concepts


def main() -> None:
    set_all_seeds()
    log_run_header(log)
    metadata = config.CLICS_DIR / "cldf" / "Wordlist-metadata.json"
    if not metadata.exists():
        raise FileNotFoundError(f"{metadata} missing — run `python run.py fetch` first.")
    schema = resolve_schema(metadata)
    tables, reader = load(metadata)
    forms, varieties, concepts = tidy(tables)

    write_parquet(forms, config.PROCESSED_DIR / "forms.parquet")
    write_parquet(varieties, config.PROCESSED_DIR / "varieties_raw.parquet")
    write_parquet(concepts, config.PROCESSED_DIR / "concepts_raw.parquet")
    summary = {
        "source": f"{config.CLICS_REPO} @ {config.CLICS_TAG} ({config.CLICS_COMMIT})",
        "reader": reader,
        "schema": schema,
        "n_forms": len(forms),
        "n_varieties": len(varieties),
        "n_varieties_with_forms": int(forms["variety_id"].nunique()),
        "n_concepts": len(concepts),
        "n_contributions": int(varieties["contribution_id"].nunique()),
    }
    write_result_json(summary, "clics_schema", producer=__name__)
    log.info("Wrote %d forms, %d varieties, %d concepts (reader: %s)",
             len(forms), len(varieties), len(concepts), reader)


if __name__ == "__main__":
    main()
