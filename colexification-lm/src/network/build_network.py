"""Phase 2 · Family-weighted colexification network (Proposal Step 2).

Definitions (DECISIONS.md D-03, D-04, D-06, D-07, D-08, D-09):

* Two concepts are **colexified in a variety** when that variety has identical
  ``Segments`` (CLICS's normalised, tokenised IPA) for both — the CLICS⁴
  criterion, verified against its shipped table — never across varieties or
  datasets. ``identity="form"`` (exact Form string) is a robustness check.
* A **language** (Glottolog language-level code) colexifies them if any of its
  varieties does; a **family** if any of its languages does.
* **Coverage** C_ij counts the units (families / languages) in which some single
  variety attests *both* concepts — the opportunity to colexify them.
* Edge support: ``n_families`` (primary weight) and ``n_languages`` (stored for
  robustness). The network keeps edges with n_families >= EDGE_MIN_FAMILIES.

Outputs
  data/processed/pairs.parquet       all concept pairs of the working set (incl. zeros):
                                      counts, coverage, rates — the unit of H1's regression
  data/processed/edges.parquet       pairs with >= 1 family, plus example forms
  data/processed/network.graphml     weighted graph (edges >= EDGE_MIN_FAMILIES)
  data/processed/family_pairs.parquet  within-family counts for qualifying families (H3)
  results/clics_validation.json      recount vs the colexification table shipped with CLICS⁴
  results/network_summary.json, results/family_subnetworks.csv, results/top_edges.csv
"""
from __future__ import annotations

import re

import networkx as nx
import numpy as np
import pandas as pd

import config
from src.utils.concept_order import load_concept_order
from src.utils.io import read_parquet, write_parquet, write_result_json, write_result_table
from src.utils.logging_utils import get_logger, log_run_header
from src.utils.seeds import set_all_seeds

log = get_logger(__name__, "phase2_build_network")
N_EXAMPLES = 6


# ------------------------------------------------------------ colexification
def colex_instances(forms: pd.DataFrame, identity: str = "segments") -> pd.DataFrame:
    """One row per (variety, unordered concept pair) sharing an identical form.

    ``identity`` is the column that defines "same word": ``segments`` (CLICS's
    criterion, primary) or ``form`` (exact string, robustness). Empty keys never
    match. Several shared forms for one pair in one variety collapse to one
    instance; the ``form`` column keeps a displayable form for examples.
    """
    cols = ["variety_id", "concept_id", identity] + (["form"] if identity != "form" else [])
    f = forms[cols].rename(columns={identity: "key"})
    if identity == "form":
        f = f.assign(form=f["key"])
    f = f[f["key"].fillna("") != ""].drop_duplicates(["variety_id", "concept_id", "key"])
    f = f[f.duplicated(["variety_id", "key"], keep=False)]
    m = f.merge(f, on=["variety_id", "key"], suffixes=("_a", "_b"))
    m = m[m["concept_id_a"] < m["concept_id_b"]]
    m = m.rename(columns={"concept_id_a": "concept_a", "concept_id_b": "concept_b", "form_a": "form"})
    return (m.sort_values(["variety_id", "concept_a", "concept_b", "key"])
             .drop_duplicates(["variety_id", "concept_a", "concept_b"])
             .reset_index(drop=True)[["variety_id", "concept_a", "concept_b", "form"]])


def aggregate_instances(inst: pd.DataFrame, units: pd.DataFrame,
                        language_col: str = "language_gc", family_col: str = "family_gc") -> pd.DataFrame:
    """Variety / language / family counts per pair. ``units`` maps variety_id
    to the counting units (rows absent from ``units`` are dropped)."""
    x = inst.merge(units, on="variety_id", how="inner")
    g = x.groupby(["concept_a", "concept_b"])
    return pd.DataFrame({
        "n_varieties": g["variety_id"].nunique(),
        "n_languages": g[language_col].nunique(),
        "n_families": g[family_col].nunique(),
    }).reset_index()


def example_forms(inst: pd.DataFrame, units: pd.DataFrame, k: int = N_EXAMPLES) -> pd.DataFrame:
    """Up to k examples per pair, one per family first (deterministic order),
    formatted 'Language (Family): form' for the interactive network."""
    x = inst.merge(units[["variety_id", "variety_name", "family_name", "family_gc"]], on="variety_id")
    x = x.sort_values(["concept_a", "concept_b", "family_gc", "variety_name", "variety_id"])
    first = x.drop_duplicates(["concept_a", "concept_b", "family_gc"])
    first = first.groupby(["concept_a", "concept_b"]).head(k)
    first = first.assign(example=first["variety_name"] + " (" + first["family_name"].fillna("?") + "): "
                         + first["form"])
    out = first.groupby(["concept_a", "concept_b"]).agg(
        examples=("example", lambda s: " | ".join(s)),
        families=("family_name", lambda s: "; ".join(sorted(set(s.dropna())))),
    ).reset_index()
    return out


# --------------------------------------------------------------- coverage ---
def attestation_matrix(forms: pd.DataFrame, concepts: list[str], varieties: list[str]) -> np.ndarray:
    """Boolean (n_varieties, n_concepts): variety attests concept."""
    ci = {c: i for i, c in enumerate(concepts)}
    vi = {v: i for i, v in enumerate(varieties)}
    f = forms[forms["concept_id"].isin(ci) & forms["variety_id"].isin(vi)]
    a = np.zeros((len(varieties), len(concepts)), dtype=bool)
    a[f["variety_id"].map(vi).to_numpy(), f["concept_id"].map(ci).to_numpy()] = True
    return a


def coverage_counts(a: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Number of groups (e.g. families) in which some single variety attests
    both concepts. ``a`` is (n_varieties, n_concepts), ``groups`` labels varieties."""
    n = a.shape[1]
    cov = np.zeros((n, n), dtype=np.int64)
    af = a.astype(np.float32)
    for g in np.unique(groups):
        sub = af[groups == g]
        cov += (sub.T @ sub > 0)
    return cov


# ------------------------------------------------------------------- pairs ---
def pair_table(counts: pd.DataFrame, concepts: list[str], cov_fam: np.ndarray,
               cov_lang: np.ndarray) -> pd.DataFrame:
    """All unordered pairs of the working set (alphabetical a < b), zeros included."""
    order = sorted(concepts)
    idx = {c: i for i, c in enumerate(concepts)}
    iu = np.triu_indices(len(order), k=1)
    a = np.array(order)[iu[0]]
    b = np.array(order)[iu[1]]
    df = pd.DataFrame({"concept_a": a, "concept_b": b})
    df = df.merge(counts, on=["concept_a", "concept_b"], how="left")
    for c in ("n_varieties", "n_languages", "n_families"):
        df[c] = df[c].fillna(0).astype("int64")
    ia = df["concept_a"].map(idx).to_numpy()
    ib = df["concept_b"].map(idx).to_numpy()
    df["cov_families"] = cov_fam[ia, ib]
    df["cov_languages"] = cov_lang[ia, ib]
    if (df["n_families"] > df["cov_families"]).any() or (df["n_languages"] > df["cov_languages"]).any():
        raise AssertionError("colexification count exceeds coverage — counting bug")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["rate_families"] = np.where(df["cov_families"] > 0, df["n_families"] / df["cov_families"], 0.0)
        df["rate_languages"] = np.where(df["cov_languages"] > 0, df["n_languages"] / df["cov_languages"], 0.0)
    df["is_edge"] = df["n_families"] >= config.EDGE_MIN_FAMILIES
    return df


def build_graph(pairs: pd.DataFrame, concepts: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    meta = concepts.set_index("concept_id")
    for cid in sorted(meta.index):
        r = meta.loc[cid]
        g.add_node(cid, gloss=r["gloss"], semantic_field=r["semantic_field"] or "",
                   n_families=int(r["n_families"]))
    for r in pairs[pairs["is_edge"]].itertuples():
        g.add_edge(r.concept_a, r.concept_b, n_families=int(r.n_families), n_languages=int(r.n_languages),
                   rate_families=float(r.rate_families))
    return g


# -------------------------------------------------------------- validation --
def validate_against_clics(inst_all: pd.DataFrame, varieties: pd.DataFrame, clics_path) -> dict:
    """Recount with CLICS's own conventions (all varieties, variety Glottocode
    as 'language', CLICS family) and compare with its shipped table."""
    units = varieties[["variety_id", "glottocode", "clics_family"]]
    ours = aggregate_instances(inst_all, units, "glottocode", "clics_family")
    shipped = pd.read_csv(clics_path, usecols=["Source_Concept", "Target_Concept", "Variety_Count",
                                               "Language_Count", "Family_Count"])
    a = shipped[["Source_Concept", "Target_Concept"]].min(axis=1)
    b = shipped[["Source_Concept", "Target_Concept"]].max(axis=1)
    shipped = shipped.assign(concept_a=a, concept_b=b).drop(columns=["Source_Concept", "Target_Concept"])
    shipped = shipped.groupby(["concept_a", "concept_b"], as_index=False).max()   # guard against dup orderings
    m = ours.merge(shipped, on=["concept_a", "concept_b"], how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    res = {
        "n_pairs_recount": int(len(ours)),
        "n_pairs_clics": int(len(shipped)),
        "n_pairs_only_recount": int((m["_merge"] == "left_only").sum()),
        "n_pairs_only_clics": int((m["_merge"] == "right_only").sum()),
        "share_variety_count_equal": float((both["n_varieties"] == both["Variety_Count"]).mean()),
        "share_language_count_equal": float((both["n_languages"] == both["Language_Count"]).mean()),
        "share_family_count_equal": float((both["n_families"] == both["Family_Count"]).mean()),
        "max_abs_family_count_diff": int((both["n_families"] - both["Family_Count"]).abs().max()),
    }
    diff = both[(both["n_families"] != both["Family_Count"]) | (both["n_languages"] != both["Language_Count"])]
    res["examples_of_disagreement"] = diff.head(10)[["concept_a", "concept_b", "n_languages", "Language_Count",
                                                     "n_families", "Family_Count"]].to_dict("records")
    return res


# --------------------------------------------------------- family subnets --
def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def family_tables(inst: pd.DataFrame, forms_u: pd.DataFrame, units: pd.DataFrame,
                  concepts: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Qualification table for every family, and within-family pair counts
    (languages as the unit) for the qualifying ones."""
    cset = set(concepts)
    fu = forms_u[forms_u["concept_id"].isin(cset)]
    att = fu.groupby("family_gc")["concept_id"].nunique()
    n_lang = units.groupby("family_gc")["language_gc"].nunique()
    n_var = units.groupby("family_gc")["variety_id"].nunique()
    names = units.drop_duplicates("family_gc").set_index("family_gc")["family_name"]
    fam = pd.DataFrame({"family_name": names, "n_languages": n_lang, "n_varieties": n_var,
                        "n_concepts_attested": att}).fillna({"n_concepts_attested": 0})
    fam["n_concepts_attested"] = fam["n_concepts_attested"].astype("int64")
    fam["concept_coverage"] = fam["n_concepts_attested"] / len(concepts)
    fam["qualifies"] = (fam["n_languages"] >= config.FAMILY_MIN_LANGUAGES) & \
                       (fam["concept_coverage"] >= config.FAMILY_MIN_CONCEPT_COVERAGE)
    fam["named_in_proposal"] = fam["family_name"].isin(config.FAMILIES_OF_INTEREST)
    fam["reason"] = np.where(fam["qualifies"], "",
                             np.where(fam["n_languages"] < config.FAMILY_MIN_LANGUAGES,
                                      "fewer than " + str(config.FAMILY_MIN_LANGUAGES) + " languages",
                                      "concept coverage below " + str(config.FAMILY_MIN_CONCEPT_COVERAGE)))
    fam = fam.reset_index().rename(columns={"index": "family_gc"})

    rows = []
    for fgc in fam.loc[fam["qualifies"], "family_gc"]:
        u = units[units["family_gc"] == fgc]
        vids = sorted(u["variety_id"])
        a = attestation_matrix(fu, concepts, vids)
        lang_labels = u.set_index("variety_id").loc[vids, "language_gc"].to_numpy()
        cov = coverage_counts(a, lang_labels)
        cnt = aggregate_instances(inst[inst["variety_id"].isin(vids)], u)
        attested = a.any(axis=0)
        # inside one family, family-level coverage is 1 wherever any language covers the pair
        p = pair_table(cnt, concepts, (cov > 0).astype(np.int64), cov)
        p = p.drop(columns=["cov_families", "rate_families", "n_families", "is_edge"])
        p["is_edge"] = p["n_languages"] >= config.EDGE_MIN_FAMILIES   # >= 2 languages inside a family (D-09)
        idx = {c: i for i, c in enumerate(concepts)}
        p["both_attested"] = attested[p["concept_a"].map(idx).to_numpy()] & attested[p["concept_b"].map(idx).to_numpy()]
        p.insert(0, "family_gc", fgc)
        rows.append(p)
    fam_pairs = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return fam, fam_pairs


# ------------------------------------------------------------------- main ---
def evidence_filter(forms: pd.DataFrame, evidence: str) -> pd.DataFrame:
    """D-37. 'all' = CLICS4 as published; 'separate_entries' drops rows whose
    source entry was mapped to more than one concept."""
    if evidence == "all":
        return forms
    if evidence == "separate_entries":
        return forms[~forms["shared_entry"]]
    raise ValueError(f"unknown evidence mode {evidence!r}")


def network_for(forms: pd.DataFrame, languages: pd.DataFrame, order: list[str],
                evidence: str = config.COLEX_EVIDENCE, identity: str = config.COLEX_IDENTITY) -> dict:
    """Pair table (all working-set pairs) under one operationalisation."""
    units = languages.loc[languages["include"], ["variety_id", "variety_name", "language_gc", "family_gc",
                                                 "family_name"]]
    f = evidence_filter(forms, evidence)
    f = f[f["variety_id"].isin(units["variety_id"]) & f["concept_id"].isin(order)]
    forms_u = f.merge(units[["variety_id", "language_gc", "family_gc"]], on="variety_id")
    inst = colex_instances(f, identity)
    counts = aggregate_instances(inst, units)
    vids = sorted(units["variety_id"])
    a = attestation_matrix(forms_u, order, vids)
    u_idx = units.set_index("variety_id").loc[vids]
    cov_fam = coverage_counts(a, u_idx["family_gc"].to_numpy())
    cov_lang = coverage_counts(a, u_idx["language_gc"].to_numpy())
    pairs = pair_table(counts, order, cov_fam, cov_lang)
    return {"pairs": pairs, "inst": inst, "units": units, "forms_u": forms_u}


def compare_operationalisations(primary: pd.DataFrame, alt: pd.DataFrame, label: str) -> dict:
    from scipy.stats import spearmanr
    m = primary.merge(alt, on=["concept_a", "concept_b"], suffixes=("", "_alt"))
    either = m[(m["n_families"] > 0) | (m["n_families_alt"] > 0)]
    rho_all = spearmanr(m["rate_families"], m["rate_families_alt"]).statistic
    rho_col = spearmanr(either["n_families"], either["n_families_alt"]).statistic
    lost = m[m["is_edge_alt"] & ~m["is_edge"]].sort_values("n_families_alt", ascending=False)
    return {
        "alternative": label,
        "n_edges_primary": int(m["is_edge"].sum()),
        "n_edges_alternative": int(m["is_edge_alt"].sum()),
        "n_edges_in_both": int((m["is_edge"] & m["is_edge_alt"]).sum()),
        "spearman_rate_all_pairs": float(rho_all),
        "spearman_family_count_colexified_pairs": float(rho_col),
        "largest_family_count_drops": (m.assign(drop=m["n_families_alt"] - m["n_families"])
                                        .sort_values("drop", ascending=False).head(15)
                                        [["concept_a", "concept_b", "n_families_alt", "n_families"]]
                                        .rename(columns={"n_families_alt": f"n_families_{label}",
                                                         "n_families": "n_families_primary"})
                                        .to_dict("records")),
        "n_edges_lost": int(len(lost)),
    }


def summarise(pairs: pd.DataFrame, g: nx.Graph, fam: pd.DataFrame, order: list[str]) -> dict:
    e = pairs[pairs["is_edge"]]
    comps = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
    deg = pd.Series(dict(g.degree()))
    return {
        "colex_evidence": config.COLEX_EVIDENCE,
        "colex_identity": config.COLEX_IDENTITY,
        "n_concepts": len(order),
        "n_pairs": len(pairs),
        "n_pairs_colexified_any": int((pairs["n_families"] >= 1).sum()),
        "n_colexified_pairs_single_family": int((pairs["n_families"] == 1).sum()),
        "n_edges_min_families": int(len(e)),
        "edge_min_families": config.EDGE_MIN_FAMILIES,
        "share_pairs_rate_zero": float((pairs["rate_families"] == 0).mean()),
        "share_pairs_not_edges": float((~pairs["is_edge"]).mean()),
        "edge_family_count_quantiles": e["n_families"].quantile([0, .25, .5, .75, .9, 1]).to_dict(),
        "n_components": len(comps),
        "component_sizes": comps,
        "largest_component": comps[0] if comps else 0,
        "isolated_concepts": sorted(n for n in g.nodes if g.degree(n) == 0),
        "degree_median": float(deg.median()),
        "degree_max": int(deg.max()),
        "highest_degree": deg.sort_values(ascending=False).head(10).to_dict(),
        "n_families_qualifying": int(fam["qualifies"].sum()),
        "qualifying_families": fam.loc[fam["qualifies"], "family_name"].tolist(),
        "proposal_families_not_qualifying": fam.loc[fam["named_in_proposal"] & ~fam["qualifies"],
                                                    ["family_name", "reason"]].to_dict("records"),
    }


def main() -> None:
    set_all_seeds()
    log_run_header(log)
    order = load_concept_order()
    forms = read_parquet(config.PROCESSED_DIR / "forms.parquet")
    languages = read_parquet(config.PROCESSED_DIR / "languages.parquet")
    concepts = read_parquet(config.PROCESSED_DIR / "concepts.parquet")
    working = concepts[concepts["working"]]

    # 1. validate the recount on ALL concepts and varieties under CLICS's conventions
    val = validate_against_clics(colex_instances(forms, "segments"), languages,
                                 config.CLICS_DIR / "cldf" / "colexifications.csv.zip")
    write_result_json(val, "clics_validation", producer=__name__)
    log.info("Validation vs CLICS4 table: %d vs %d pairs; family counts equal for %.4f of shared pairs",
             val["n_pairs_recount"], val["n_pairs_clics"], val["share_family_count_equal"])

    # 2. primary network (project conventions)
    net = network_for(forms, languages, order)
    pairs, inst, units, forms_u = net["pairs"], net["inst"], net["units"], net["forms_u"]
    write_parquet(pairs, config.PROCESSED_DIR / "pairs.parquet")

    ex = example_forms(inst, units)
    edges = pairs[pairs["n_families"] >= 1].merge(ex, on=["concept_a", "concept_b"], how="left")
    gl = working.set_index("concept_id")["gloss"]
    edges.insert(2, "gloss_a", edges["concept_a"].map(gl))
    edges.insert(3, "gloss_b", edges["concept_b"].map(gl))
    write_parquet(edges, config.PROCESSED_DIR / "edges.parquet")
    g = build_graph(pairs, working)
    nx.write_graphml(g, config.PROCESSED_DIR / "network.graphml")

    # 3. alternative operationalisations, stored for robustness (D-37, D-03)
    comparisons = []
    alts = [(ev, config.COLEX_IDENTITY) for ev in config.COLEX_EVIDENCE_ALTERNATIVES]
    alts.append((config.COLEX_EVIDENCE, "form"))
    for ev, ident in alts:
        label = f"{ev}__{ident}"
        alt = network_for(forms, languages, order, ev, ident)["pairs"]
        write_parquet(alt, config.PROCESSED_DIR / f"pairs__{label}.parquet")
        comparisons.append(compare_operationalisations(pairs, alt, label))
    write_result_json({"primary": f"{config.COLEX_EVIDENCE}__{config.COLEX_IDENTITY}",
                       "comparisons": comparisons}, "colex_operationalisations", producer=__name__)

    # 4. per-family subnetworks (H3)
    fam, fam_pairs = family_tables(inst, forms_u, units, order)
    write_parquet(fam_pairs, config.PROCESSED_DIR / "family_pairs.parquet")
    write_result_table(fam.sort_values("n_languages", ascending=False), "family_subnetworks", producer=__name__)

    # 5. summaries
    summary = summarise(pairs, g, fam, order)
    write_result_json(summary, "network_summary", producer=__name__)
    top = edges[edges["is_edge"]].sort_values(["n_families", "n_languages"], ascending=False).head(50)
    write_result_table(top[["concept_a", "concept_b", "gloss_a", "gloss_b", "n_families", "n_languages",
                            "cov_families", "rate_families", "families"]], "top_edges", producer=__name__)
    log.info("%d edges (>= %d families) among %d concepts; %d components (largest %d, %d isolated); "
             "%d qualifying families", summary["n_edges_min_families"], config.EDGE_MIN_FAMILIES, len(order),
             summary["n_components"], summary["largest_component"], len(summary["isolated_concepts"]),
             summary["n_families_qualifying"])


if __name__ == "__main__":
    main()
