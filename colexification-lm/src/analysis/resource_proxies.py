"""Digital-resource proxies for each qualifying language family (H3; D-26, D-56, D-57).

1. **Wikipedia (primary).** For every Wikipedia language edition: its article count
   (Wikimedia API, ``siteinfo`` statistics) and its language's Glottocode / ISO 639-3
   code (Wikidata P424 Wikimedia language code, P1394 Glottocode, P220 ISO 639-3).
   Each edition is assigned to the Glottolog family of its language; the family proxy
   is log10(1 + articles of its largest edition). Matching on Glottocodes at any
   level lets macrolanguage editions (Chinese, Arabic, Malay...) reach their family
   without an ISO macrolanguage table. Needs internet access; the raw responses are
   cached in data/raw/wikipedia/ with the retrieval time, so a rerun is offline.
2. **Joshi et al. (2020) class (secondary).** If the authors' ``lang2tax.txt`` is
   placed at data/raw/joshi/lang2tax.txt it is used in full (family = max class over
   its languages; unlisted languages count as class 0). Otherwise only the 37
   languages whose class the paper itself states are used
   (src/analysis/joshi_classes_verified.csv, with page locations); the family value is
   then a *lower bound* and families with no listed language are missing.

Outputs: results/resource_proxies.csv, results/resource_proxies_status.json,
results/wikipedia_editions_unmatched.csv (when Wikipedia data are available).
"""
from __future__ import annotations

import datetime as dt
import json
import time

import numpy as np
import pandas as pd

import config
from src.utils.io import write_result_json, write_result_table
from src.utils.logging_utils import get_logger, log_run_header

log = get_logger(__name__, "resource_proxies")

USER_AGENT = "colexification-lm/1.0 (research reproduction; https://github.com/) python-requests"
EDITIONS_CSV = config.WIKIPEDIA_DIR / "wikipedia_editions.csv"
WIKIDATA_CSV = config.WIKIPEDIA_DIR / "wikidata_language_codes.csv"
PENDING_WIKI = ("PENDING USER RUN: `python run.py analyse` with internet access "
                "(query.wikidata.org, meta.wikimedia.org, *.wikipedia.org)")
EXCLUDED_EDITIONS = {"simple"}   # Simple English is not a separate language


# ----------------------------------------------------------------- Glottolog --
def glottolog_languoids() -> pd.DataFrame:
    """All Glottolog languoids with their top-level family (a top-level family or
    isolate is its own family)."""
    g = pd.read_csv(config.GLOTTOLOG_DIR / "cldf" / "languages.csv", dtype=str, keep_default_na=False)
    g["family_gc"] = np.where(g["Family_ID"] != "", g["Family_ID"], g["ID"])
    return g[["ID", "Name", "Level", "ISO639P3code", "family_gc"]].rename(
        columns={"ID": "glottocode", "Name": "name", "Level": "level", "ISO639P3code": "iso639_3"})


# -------------------------------------------------------------------- Joshi --
def parse_lang2tax(text: str) -> pd.DataFrame:
    """Parse lines of ``<language name><sep><class>`` (sep = comma or tab)."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sep = "\t" if "\t" in line else ","
        name, _, cls = line.rpartition(sep)
        cls = cls.strip()
        if name and cls.isdigit() and 0 <= int(cls) <= 5:
            rows.append({"language_paper": name.strip(), "class_low": int(cls), "class_high": int(cls)})
    return pd.DataFrame(rows)


def joshi_languages(langs: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Joshi classes with Glottolog family attached, and a status string."""
    lv = langs[langs["level"] == "language"]
    if config.JOSHI_LANG2TAX_PATH.exists():
        t = parse_lang2tax(config.JOSHI_LANG2TAX_PATH.read_text(encoding="utf-8"))
        by_name = lv.assign(key=lv["name"].str.lower()).drop_duplicates("key").set_index("key")
        t["family_gc"] = t["language_paper"].str.lower().map(by_name["family_gc"])
        matched = t["family_gc"].notna().mean() if len(t) else 0.0
        return t, (f"done: full lang2tax.txt, {len(t)} languages, "
                   f"{matched:.1%} matched to Glottolog by name; unlisted languages = class 0")
    t = pd.read_csv(config.JOSHI_VERIFIED_PATH)
    by_iso = lv[lv["iso639_3"] != ""].drop_duplicates("iso639_3").set_index("iso639_3")
    t["family_gc"] = t["iso639_3"].map(by_iso["family_gc"])
    missing = t.loc[t["family_gc"].isna(), "language_paper"].tolist()
    if missing:
        raise RuntimeError(f"verified Joshi languages not in Glottolog by ISO: {missing}")
    return t, (f"PROVISIONAL: {len(t)} languages whose class the paper states (lower bounds per family); "
               f"for the full taxonomy place lang2tax.txt at {config.JOSHI_LANG2TAX_PATH.relative_to(config.ROOT)}")


def family_joshi(t: pd.DataFrame, families: list[str], full_table: bool) -> pd.DataFrame:
    g = t.dropna(subset=["family_gc"]).groupby("family_gc")
    best = g["class_low"].max()
    top = g.apply(lambda d: ", ".join(d.loc[d["class_low"] == d["class_low"].max(), "language_paper"]),
                  include_groups=False)
    out = pd.DataFrame({"family_gc": families})
    out["joshi_class"] = out["family_gc"].map(best)
    if full_table:
        out["joshi_class"] = out["joshi_class"].fillna(0)
    out["joshi_top_languages"] = out["family_gc"].map(top)
    out["joshi_is_lower_bound"] = not full_table
    return out


# ---------------------------------------------------------------- Wikipedia --
def _get_json(session, url: str, params: dict, retries: int = 4) -> dict:
    for k in range(retries):
        try:
            r = session.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception:                                            # noqa: BLE001
            if k == retries - 1:
                raise
            time.sleep(2 ** k)
    raise RuntimeError("unreachable")


def fetch_wikipedia() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (once) Wikipedia edition sizes and Wikidata language codes."""
    if EDITIONS_CSV.exists() and WIKIDATA_CSV.exists():
        return pd.read_csv(EDITIONS_CSV, keep_default_na=False), pd.read_csv(WIKIDATA_CSV, keep_default_na=False)
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sm = _get_json(s, "https://meta.wikimedia.org/w/api.php",
                   {"action": "sitematrix", "format": "json", "smtype": "language", "smlimit": "max"})
    eds = []
    for key, lang in sm["sitematrix"].items():
        if key == "count":
            continue
        for site in lang.get("site", []):
            if site.get("code") == "wiki":
                eds.append({"code": lang["code"], "dbname": site["dbname"], "url": site["url"],
                            "closed": "closed" in site, "language_name": lang.get("localname", "")})
    log.info("sitematrix: %d Wikipedia editions", len(eds))
    for k, e in enumerate(eds):
        if e["closed"]:
            e["articles"] = 0
            continue
        st = _get_json(s, e["url"] + "/w/api.php",
                       {"action": "query", "meta": "siteinfo", "siprop": "statistics", "format": "json"})
        e["articles"] = int(st["query"]["statistics"]["articles"])
        time.sleep(0.05)
        if k % 50 == 0:
            log.info("  %d / %d editions", k, len(eds))
    editions = pd.DataFrame(eds).assign(retrieved_utc=stamp)
    q = """SELECT ?item ?wm ?glottocode ?iso3 WHERE {
             ?item wdt:P424 ?wm .
             OPTIONAL { ?item wdt:P1394 ?glottocode }
             OPTIONAL { ?item wdt:P220 ?iso3 } }"""
    res = _get_json(s, config.WIKIDATA_SPARQL_URL, {"query": q, "format": "json"})
    wd = pd.DataFrame([{"item": b["item"]["value"].rsplit("/", 1)[-1], "wm_code": b["wm"]["value"],
                        "glottocode": b.get("glottocode", {}).get("value", ""),
                        "iso639_3": b.get("iso3", {}).get("value", "")} for b in res["results"]["bindings"]])
    wd = wd.assign(retrieved_utc=stamp)
    config.WIKIPEDIA_DIR.mkdir(parents=True, exist_ok=True)
    editions.to_csv(EDITIONS_CSV, index=False)
    wd.to_csv(WIKIDATA_CSV, index=False)
    return editions, wd


def edition_families(editions: pd.DataFrame, wd: pd.DataFrame, langs: pd.DataFrame) -> pd.DataFrame:
    """Glottolog family of each open edition (via Glottocode, else ISO 639-3)."""
    fam_by_gc = langs.set_index("glottocode")["family_gc"]
    by_iso = langs[(langs["iso639_3"] != "") & (langs["level"] == "language")].drop_duplicates("iso639_3")
    fam_by_iso = by_iso.set_index("iso639_3")["family_gc"]
    wd = wd.assign(family_gc=wd["glottocode"].map(fam_by_gc))
    wd["family_gc"] = wd["family_gc"].fillna(wd["iso639_3"].map(fam_by_iso))
    m = wd.dropna(subset=["family_gc"]).groupby("wm_code")["family_gc"].agg(lambda s: sorted(set(s)))
    e = editions[(~editions["closed"].astype(str).str.lower().isin(["true", "1"]))
                 & (~editions["code"].isin(EXCLUDED_EDITIONS))].copy()
    e["families"] = e["code"].map(m)
    e["family_gc"] = e["families"].map(lambda v: v[0] if isinstance(v, list) and len(v) == 1 else None)
    e["mapping"] = e["families"].map(lambda v: "unmatched" if not isinstance(v, list)
                                     else ("ambiguous" if len(v) > 1 else "ok"))
    return e


def family_wikipedia(e: pd.DataFrame, families: list[str]) -> pd.DataFrame:
    ok = e[e["mapping"] == "ok"]
    g = ok.sort_values("articles", ascending=False).groupby("family_gc")
    out = pd.DataFrame({"family_gc": families})
    out["wikipedia_max_articles"] = out["family_gc"].map(g["articles"].max()).fillna(0).astype(int)
    out["wikipedia_top_edition"] = out["family_gc"].map(g["code"].first())
    out["wikipedia_n_editions"] = out["family_gc"].map(g.size()).fillna(0).astype(int)
    out["wikipedia_total_articles"] = out["family_gc"].map(g["articles"].sum()).fillna(0).astype(int)
    out["wikipedia_log10_articles"] = np.log10(1 + out["wikipedia_max_articles"])
    return out


# --------------------------------------------------------------------- main ---
def main() -> None:
    log_run_header(log)
    fam = pd.read_csv(config.RESULTS_DIR / "family_subnetworks.csv")
    fam = fam[fam["qualifies"]][["family_gc", "family_name", "n_languages"]]
    families = fam["family_gc"].tolist()
    langs = glottolog_languoids()

    t, joshi_status = joshi_languages(langs)
    out = fam.merge(family_joshi(t, families, config.JOSHI_LANG2TAX_PATH.exists()), on="family_gc")

    status = {"joshi_class_status": joshi_status}
    try:
        editions, wd = fetch_wikipedia()
        e = edition_families(editions, wd, langs)
        out = out.merge(family_wikipedia(e, families), on="family_gc")
        n_ok = int((e["mapping"] == "ok").sum())
        share = e.loc[e["mapping"] == "ok", "articles"].sum() / max(e["articles"].sum(), 1)
        status["wikipedia_log10_articles_status"] = (
            f"done: {n_ok} of {len(e)} open editions mapped to a Glottolog family "
            f"({share:.1%} of all articles); retrieved {editions['retrieved_utc'].iloc[0]}")
        write_result_table(e[e["mapping"] != "ok"].drop(columns=["families"]),
                           "wikipedia_editions_unmatched", producer=__name__)
    except Exception as exc:                                         # noqa: BLE001
        log.warning("Wikipedia proxy unavailable (%s: %s)", type(exc).__name__, exc)
        out["wikipedia_log10_articles"] = np.nan
        status["wikipedia_log10_articles_status"] = PENDING_WIKI
    write_result_table(out, "resource_proxies", producer=__name__)
    write_result_json(status, "resource_proxies_status", producer=__name__)
    for k, v in status.items():
        log.info("%s: %s", k, v)


if __name__ == "__main__":
    main()
