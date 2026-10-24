"""Phase 1 · Concept selection, English glosses, semantic fields, canonical order.

Procedure (DECISIONS.md D-05, D-05b, D-11, D-12, D-35):

1. Count, for every CLICS concept, the number of Glottolog families (included
   languages only, D-06) in which at least one language attests it.
2. Coverage curve n(t) = #concepts attested in >= t families, for the
   pre-specified candidate thresholds in config. The threshold is the curve's
   knee (Kneedle; Satopää et al. 2011), constrained to give 300-500 concepts.
3. Add anchor concepts named in the proposal if attested in >= ANCHOR_MIN_FAMILIES.
4. Clean Concepticon glosses, apply ``gloss_overrides.csv`` (glosses and
   exclusions), and verify no two working concepts share an English gloss.
5. Attach Concepticon semantic field and ontological category.
6. Freeze ``concept_order.json`` (alphabetical by concept ID).

Outputs: data/processed/concepts.parquet (all 1,730 concepts with flags),
data/processed/concept_order.json, results/concept_threshold.json,
results/concept_coverage_curve.csv, results/working_concepts.csv.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

import config
from src.utils.concept_order import freeze_concept_order
from src.utils.io import read_parquet, write_parquet, write_result_json, write_result_table
from src.utils.logging_utils import get_logger, log_run_header
from src.utils.seeds import set_all_seeds

log = get_logger(__name__, "phase1_select_concepts")


# ------------------------------------------------------------------- coverage
def attach_units(forms: pd.DataFrame, languages: pd.DataFrame) -> pd.DataFrame:
    """Forms of included languages, with language and family units attached."""
    inc = languages.loc[languages["include"], ["variety_id", "language_gc", "family_gc"]]
    return forms.merge(inc, on="variety_id", how="inner")


def concept_coverage(forms_u: pd.DataFrame) -> pd.DataFrame:
    g = forms_u.groupby("concept_id")
    return pd.DataFrame({
        "n_families": g["family_gc"].nunique(),
        "n_languages": g["language_gc"].nunique(),
        "n_varieties": g["variety_id"].nunique(),
    })


def coverage_curve(n_families: pd.Series, thresholds) -> pd.DataFrame:
    return pd.DataFrame({
        "min_families": list(thresholds),
        "n_concepts": [int((n_families >= t).sum()) for t in thresholds],
    })


def kneedle(x: np.ndarray, y: np.ndarray) -> tuple[int, np.ndarray]:
    """Knee of a decreasing curve: the point farthest below the chord joining
    its end points after min-max normalisation (Satopää et al. 2011's difference
    curve for a decreasing curve). Returns (index, difference curve)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError("kneedle needs >= 3 points with non-constant x and y")
    xn = (x - x.min()) / np.ptp(x)
    yn = (y - y.min()) / np.ptp(y)
    chord = yn[0] + (yn[-1] - yn[0]) * xn
    diff = chord - yn
    return int(np.argmax(diff)), diff


def choose_threshold(curve: pd.DataFrame, target: tuple[int, int]) -> dict[str, object]:
    idx, diff = kneedle(curve["min_families"].to_numpy(), curve["n_concepts"].to_numpy())
    knee_t = int(curve["min_families"].iloc[idx])
    knee_n = int(curve["n_concepts"].iloc[idx])
    lo, hi = target
    admissible = curve[(curve["n_concepts"] >= lo) & (curve["n_concepts"] <= hi)]
    if admissible.empty:
        raise ValueError(f"No candidate threshold gives {lo}-{hi} concepts")
    if lo <= knee_n <= hi:
        chosen, constrained = knee_t, False
    else:  # nearest admissible threshold to the knee
        j = (admissible["min_families"] - knee_t).abs().idxmin()
        chosen, constrained = int(admissible.loc[j, "min_families"]), True
    return {"knee_threshold": knee_t, "knee_n_concepts": knee_n, "chosen_threshold": chosen,
            "constrained_to_target": constrained, "difference_curve": diff.round(4).tolist()}


def knee_sensitivity(n_families: pd.Series) -> list[dict[str, int]]:
    """Where the knee lands if the candidate range is changed (reported, not used)."""
    out = []
    for lo, hi in [(5, 150), (10, 150), (10, 200), (10, 250), (20, 200), (30, 200)]:
        ts = list(range(lo, hi + 1, 5))
        c = coverage_curve(n_families, ts)
        i, _ = kneedle(c["min_families"].to_numpy(), c["n_concepts"].to_numpy())
        out.append({"range": f"{lo}-{hi}", "knee_threshold": int(c["min_families"].iloc[i]),
                    "n_concepts": int(c["n_concepts"].iloc[i])})
    return out


# --------------------------------------------------------------------- glosses
def clean_gloss(concepticon_gloss: str) -> str:
    """'BLOW (OF WIND)' -> 'blow'; 'IN FRONT OF' -> 'in front of'."""
    g = re.sub(r"\s*\([^)]*\)", "", concepticon_gloss)
    return re.sub(r"\s+", " ", g).strip().lower()


def load_overrides(path=None) -> pd.DataFrame:
    path = path or config.GLOSS_OVERRIDES_PATH
    ov = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not ov["concept_id"].is_unique:
        raise ValueError("gloss_overrides.csv has duplicate concept_ids")
    bad = ~ov["action"].isin(["gloss", "exclude"])
    if bad.any():
        raise ValueError(f"Unknown actions in gloss_overrides.csv: {ov.loc[bad, 'action'].tolist()}")
    if ((ov["action"] == "gloss") & (ov["gloss"].str.strip() == "")).any():
        raise ValueError("A 'gloss' override has an empty gloss")
    return ov.set_index("concept_id")


def apply_glosses(selected: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    df = selected.copy()
    df["gloss_default"] = df["concept_id"].map(clean_gloss)
    df["gloss"] = df["gloss_default"]
    df["gloss_source"] = "concepticon"
    df["collision_group"] = ""
    df["excluded_reason"] = ""
    for cid, row in overrides.iterrows():
        m = df["concept_id"] == cid
        if not m.any():
            log.warning("Override for %s ignored: concept not in the candidate set", cid)
            continue
        if row["action"] == "exclude":
            df.loc[m, "excluded_reason"] = row["reason"]
        else:
            df.loc[m, "gloss"] = row["gloss"].strip()  # overrides keep their case (e.g. "I")
            df.loc[m, "gloss_source"] = "override"
        df.loc[m, "collision_group"] = row.get("collision_group", "")
    df["gloss_n_tokens"] = df["gloss"].str.split().str.len()
    return df


def check_unique_glosses(working: pd.DataFrame) -> None:
    dup = working[working["gloss"].duplicated(keep=False)]
    if not dup.empty:
        pairs = dup.sort_values("gloss")[["concept_id", "gloss"]].to_string(index=False)
        raise ValueError("Working concepts share English glosses; add overrides:\n" + pairs)


def load_concepticon(path=None) -> pd.DataFrame:
    path = path or (config.CONCEPTICON_DIR / "concepticondata" / "concepticon.tsv")
    c = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    c["ID"] = c["ID"].astype("int64")
    return c.rename(columns={"ID": "concepticon_id", "SEMANTICFIELD": "semantic_field",
                             "ONTOLOGICAL_CATEGORY": "ontological_category",
                             "DEFINITION": "concepticon_definition"})[
        ["concepticon_id", "semantic_field", "ontological_category", "concepticon_definition"]]


# ------------------------------------------------------------------------ main
def build(forms: pd.DataFrame, languages: pd.DataFrame, concepts_raw: pd.DataFrame,
          concepticon: pd.DataFrame, overrides: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cov = concept_coverage(attach_units(forms, languages))
    curve = coverage_curve(cov["n_families"], config.CONCEPT_MIN_FAMILIES_CANDIDATES)
    thr = choose_threshold(curve, config.CONCEPT_SET_TARGET)
    t = config.CONCEPT_MIN_FAMILIES if config.CONCEPT_MIN_FAMILIES is not None else thr["chosen_threshold"]
    t_robust = t + config.CONCEPT_MIN_FAMILIES_ROBUSTNESS_OFFSET

    df = concepts_raw.merge(cov, left_on="concept_id", right_index=True, how="left")
    df[["n_families", "n_languages", "n_varieties"]] = (
        df[["n_families", "n_languages", "n_varieties"]].fillna(0).astype("int64"))
    df = df.merge(concepticon, on="concepticon_id", how="left")

    df["passes_threshold"] = df["n_families"] >= t
    df["is_anchor"] = df["concept_id"].isin(config.ANCHOR_CONCEPTS)
    missing_anchor = set(config.ANCHOR_CONCEPTS) - set(df["concept_id"])
    if missing_anchor:
        raise ValueError(f"Anchor concepts not in CLICS: {sorted(missing_anchor)}")
    df["anchor_added"] = df["is_anchor"] & ~df["passes_threshold"] & (df["n_families"] >= config.ANCHOR_MIN_FAMILIES)
    anchor_dropped = df.loc[df["is_anchor"] & ~df["passes_threshold"] & ~df["anchor_added"], "concept_id"].tolist()
    df["candidate"] = df["passes_threshold"] | df["anchor_added"]

    cand = apply_glosses(df[df["candidate"]], overrides)
    df = df.merge(cand[["concept_id", "gloss_default", "gloss", "gloss_source", "collision_group",
                        "excluded_reason", "gloss_n_tokens"]], on="concept_id", how="left")
    df["working"] = df["candidate"] & (df["excluded_reason"].fillna("") == "")
    df["in_robustness_set"] = df["working"] & (df["n_families"] >= t_robust)
    working = df[df["working"]]
    check_unique_glosses(working)

    lo, hi = config.CONCEPT_SET_TARGET
    if not lo <= len(working) <= hi:
        raise ValueError(f"Working set has {len(working)} concepts, outside {lo}-{hi}")

    info = {
        **{k: v for k, v in thr.items() if k != "difference_curve"},
        "threshold_used": t,
        "threshold_source": "config override" if config.CONCEPT_MIN_FAMILIES is not None else "kneedle",
        "robustness_threshold": t_robust,
        "n_concepts_total": len(df),
        "n_passing_threshold": int(df["passes_threshold"].sum()),
        "anchors_added_below_threshold": df.loc[df["anchor_added"], ["concept_id", "n_families"]]
                                          .to_dict("records"),
        "anchors_not_available": anchor_dropped,
        "n_excluded_by_override": int((df["excluded_reason"].fillna("") != "").sum()),
        "n_gloss_overrides": int((df["gloss_source"] == "override").sum()),
        "n_working": len(working),
        "n_robustness_set": int(df["in_robustness_set"].sum()),
        "n_multiword_glosses": int((working["gloss_n_tokens"] > 1).sum()),
        "semantic_fields": working["semantic_field"].value_counts().to_dict(),
        "ontological_categories": working["ontological_category"].value_counts().to_dict(),
        "knee_sensitivity_to_candidate_range": knee_sensitivity(cov["n_families"]),
    }
    curve["difference"] = thr["difference_curve"]
    return df, {"info": info, "curve": curve}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refreeze", action="store_true",
                    help="overwrite a different frozen concept order (invalidates all matrices)")
    args = ap.parse_args()
    set_all_seeds()
    log_run_header(log)

    forms = read_parquet(config.PROCESSED_DIR / "forms.parquet")
    languages = read_parquet(config.PROCESSED_DIR / "languages.parquet")
    concepts_raw = read_parquet(config.PROCESSED_DIR / "concepts_raw.parquet")
    df, out = build(forms, languages, concepts_raw, load_concepticon(), load_overrides())

    working = df[df["working"]].sort_values("concept_id")
    order = working["concept_id"].tolist()
    digest = freeze_concept_order(order, provenance={
        "threshold_min_families": out["info"]["threshold_used"],
        "anchors_added": [a["concept_id"] for a in out["info"]["anchors_added_below_threshold"]],
        "clics_commit": config.CLICS_COMMIT, "producer": __name__,
    }, overwrite=args.refreeze)
    out["info"]["concept_order_sha256"] = digest

    write_parquet(df, config.PROCESSED_DIR / "concepts.parquet")
    write_result_table(out["curve"], "concept_coverage_curve", producer=__name__)
    write_result_json(out["info"], "concept_threshold", producer=__name__)
    write_result_table(
        working[["concept_id", "gloss", "gloss_source", "semantic_field", "ontological_category",
                 "n_families", "n_languages", "is_anchor", "anchor_added", "in_robustness_set",
                 "collision_group"]],
        "working_concepts", producer=__name__)
    i = out["info"]
    log.info("Knee at t=%d (%d concepts); using t=%d. Working set: %d concepts "
             "(%d anchors added, %d excluded by override, %d gloss overrides). Order sha256 %s",
             i["knee_threshold"], i["knee_n_concepts"], i["threshold_used"], i["n_working"],
             len(i["anchors_added_below_threshold"]), i["n_excluded_by_override"],
             i["n_gloss_overrides"], digest[:12])


if __name__ == "__main__":
    main()
