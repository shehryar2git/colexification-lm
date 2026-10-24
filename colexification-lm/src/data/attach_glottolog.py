"""Phase 1 · Attach Glottolog v5.3 genealogy and geography to every CLICS variety.

For each variety (by its Glottocode) we derive (DECISIONS.md D-02, D-04, D-06):

* ``language_gc``  language-level Glottocode = the counting unit for "languages".
  Dialects map to their language; a variety coded with a subgroup (family-level)
  node keeps that node as its own unit (flagged ``glottolog_level = family``).
* ``family_gc`` / ``family_name``  top-level Glottolog family; isolates are
  their own one-language family.
* ``include``  False for pseudo-families (Bookkeeping, Sign Language, ...).
* ``macroarea``, ``latitude``, ``longitude`` with ``coord_source``.

Outputs: data/processed/languages.parquet, results/glottolog_reconciliation.csv
(every variety whose CLICS family differs from Glottolog v5.3), and
results/language_summary.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.utils.io import read_parquet, write_parquet, write_result_json, write_result_table
from src.utils.logging_utils import get_logger, log_run_header
from src.utils.seeds import set_all_seeds

log = get_logger(__name__, "phase1_attach_glottolog")

GLOTTOLOG_COLS = ["ID", "Name", "Level", "Family_ID", "Language_ID", "Is_Isolate",
                  "Macroarea", "Latitude", "Longitude"]


def load_glottolog(path=None) -> pd.DataFrame:
    path = path or (config.GLOTTOLOG_DIR / "cldf" / "languages.csv")
    g = pd.read_csv(path, usecols=GLOTTOLOG_COLS, dtype=str, keep_default_na=False, na_values=[""])
    g["Latitude"] = pd.to_numeric(g["Latitude"])
    g["Longitude"] = pd.to_numeric(g["Longitude"])
    return g.set_index("ID")


def resolve_languoid(gc: str, g: pd.DataFrame) -> dict[str, object]:
    """Language-level unit and top-level family for one Glottocode."""
    if gc not in g.index:
        return {"glottolog_level": None, "language_gc": gc, "family_gc": None}
    row = g.loc[gc]
    level = row["Level"]
    if level == "dialect":
        language_gc = row["Language_ID"] if isinstance(row["Language_ID"], str) else gc
    else:  # 'language', or a subgroup/family node used as a variety code
        language_gc = gc
    fam = row["Family_ID"]
    if not isinstance(fam, str):
        # No parent family: an isolate (language), a dialect of an isolate,
        # or a variety coded directly with a top-level family node.
        fam = language_gc if level in ("language", "dialect") else gc
    return {"glottolog_level": level, "language_gc": language_gc, "family_gc": fam}


def attach(varieties: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    res = pd.DataFrame([resolve_languoid(gc, g) for gc in varieties["glottocode"]],
                       index=varieties.index)
    out = pd.concat([varieties, res], axis=1)
    out["family_name"] = out["family_gc"].map(lambda f: g.at[f, "Name"] if f in g.index else None)
    out["is_isolate"] = out["family_gc"] == out["language_gc"]

    # Geography: language-level Glottolog row first, then the variety's own row, then CLICS.
    def geo(row: pd.Series) -> tuple[object, float, float, str]:
        for code, src in ((row["language_gc"], "glottolog_language"), (row["glottocode"], "glottolog_variety")):
            if code in g.index and np.isfinite(g.at[code, "Latitude"]):
                return g.at[code, "Macroarea"], g.at[code, "Latitude"], g.at[code, "Longitude"], src
        if np.isfinite(row["clics_latitude"]):
            return row["clics_macroarea"], row["clics_latitude"], row["clics_longitude"], "clics"
        return row["clics_macroarea"], np.nan, np.nan, "missing"

    geo_df = pd.DataFrame([geo(r) for _, r in out.iterrows()],
                          columns=["macroarea", "latitude", "longitude", "coord_source"], index=out.index)
    out = pd.concat([out, geo_df], axis=1)
    # Glottolog can list several macroareas ("Eurasia;Africa"): keep the first, flag it
    out["macroarea_multi"] = out["macroarea"].fillna("").str.contains(";")
    out["macroarea"] = out["macroarea"].fillna("").str.split(";").str[0].replace("", None)

    out["include"] = True
    out["exclude_reason"] = None
    unmatched = out["glottolog_level"].isna()
    out.loc[unmatched, ["include", "exclude_reason"]] = [False, "glottocode not in Glottolog 5.3"]
    pseudo = out["family_name"].isin(config.NON_GENEALOGICAL_FAMILIES)
    out.loc[pseudo, "include"] = False
    out.loc[pseudo, "exclude_reason"] = "pseudo-family: " + out.loc[pseudo, "family_name"]
    return out


def main() -> None:
    set_all_seeds()
    log_run_header(log)
    varieties = read_parquet(config.PROCESSED_DIR / "varieties_raw.parquet")
    g = load_glottolog()
    langs = attach(varieties, g)

    disagree = langs[langs["clics_family"].fillna("") != langs["family_gc"].fillna("")]
    write_result_table(
        disagree[["variety_id", "variety_name", "glottocode", "contribution_id", "clics_family",
                  "clics_family_name", "family_gc", "family_name", "glottolog_level"]],
        "glottolog_reconciliation", producer=__name__,
    )
    write_parquet(langs, config.PROCESSED_DIR / "languages.parquet")

    inc = langs[langs["include"]]
    summary = {
        "n_varieties": len(langs),
        "n_varieties_included": int(len(inc)),
        "excluded_by_reason": langs.loc[~langs["include"], "exclude_reason"].value_counts().to_dict(),
        "n_languages_included": int(inc["language_gc"].nunique()),
        "n_families_included": int(inc["family_gc"].nunique()),
        "n_isolate_families": int(inc.loc[inc["is_isolate"], "family_gc"].nunique()),
        "glottolog_level_counts": langs["glottolog_level"].value_counts(dropna=False).to_dict(),
        "n_family_disagreements_with_clics": int(len(disagree)),
        "coord_source_counts": inc["coord_source"].value_counts().to_dict(),
        "n_multi_macroarea": int(inc["macroarea_multi"].sum()),
        "languages_per_family_top15": inc.groupby("family_name")["language_gc"].nunique()
                                         .sort_values(ascending=False).head(15).to_dict(),
    }
    write_result_json(summary, "language_summary", producer=__name__)
    log.info("Included %d varieties = %d languages in %d families (%d isolates); "
             "%d family disagreements with CLICS",
             summary["n_varieties_included"], summary["n_languages_included"],
             summary["n_families_included"], summary["n_isolate_families"], len(disagree))


if __name__ == "__main__":
    main()
