"""Phase 1 (+ Phase 6) · WordNet-based English colexification flags and audit sheet.

English-present candidate (DECISIONS.md D-24): concepts a and b are a
candidate if some English lemma has a WordNet sense among the synsets of a's
gloss AND a sense among the synsets of b's gloss — i.e. one English word could
cover both meanings (e.g. 'see' is in a synset of 'understand'; 'tongue' is in
a synset of 'language'). This is deliberately high-recall: WordNet also links
slang, rare and technical senses ('testicle' ~ 'egg'), so **every candidate
is audited by hand** before it enters H2.

Phase 1 output (always):
  data/processed/concept_wordnet.parquet   per-concept WordNet coverage and polysemy
  data/processed/english_colex_candidates.parquet   every candidate pair + evidence
  results/english_colex_summary.json

Phase 6 output (when data/processed/edges.parquet exists):
  audit/english_colexification_manual_check.csv — candidates that are also
  CLICS edges. Existing human decisions are preserved on every rerun.
"""
from __future__ import annotations

import re
from itertools import combinations

import pandas as pd

import config
from src.data.wordnet_db import WordNet, load_wordnet
from src.utils.io import read_parquet, write_parquet, write_result_json
from src.utils.logging_utils import get_logger, log_run_header
from src.utils.seeds import set_all_seeds

log = get_logger(__name__, "phase1_english_colex")

AUDIT_COLUMNS = ["concept_a", "concept_b", "gloss_a", "gloss_b", "shared_lemma",
                 "evidence_route", "wordnet_evidence", "n_families", "suggested_decision", "decision",
                 "annotator", "notes"]
VALID_DECISIONS = {"present", "absent", "unsure", ""}
AUDIT_CRITERIA = (
    "present = one ordinary present-day English word is used for both concepts in "
    "everyday speech (metaphorical extensions listed in a standard dictionary count); "
    "absent = the shared lemma is slang, technical, archaic, a proper name, a numeral "
    "or letter, or an artefact of WordNet grouping; unsure = annotator cannot decide "
    "(excluded from H2)."
)


def concept_senses(glosses: pd.Series, wn: WordNet) -> pd.DataFrame:
    rows = []
    for cid, gloss in glosses.items():
        ss = wn.synsets_for(gloss)
        pos_counts = pd.Series([s.pos for s in ss]).value_counts()
        rows.append({
            "concept_id": cid,
            "gloss": gloss,
            "wn_n_synsets": len(ss),
            "wn_primary_pos": pos_counts.index[0] if len(ss) else None,
            "wn_in_wordnet": bool(ss),
            "wn_lemmas": sorted({lem.replace("_", " ") for s in ss for lem in s.lemmas}),
            "wn_synset_ids": [s.id for s in ss],
        })
    return pd.DataFrame(rows)


def _word_re(gloss: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(gloss.lower()) + r"s?\b")


def definition_hits(a_gloss: str, a_synsets: list[str], b_gloss: str, wn: WordNet) -> list[str]:
    """Senses of the word a_gloss whose WordNet definition names b_gloss
    (e.g. 'tongue' -> 'a human written or spoken language used by a community')."""
    key = a_gloss.lower().replace(" ", "_")
    pat = _word_re(b_gloss)
    return [s for s in a_synsets if key in wn.synsets[s].lemmas and pat.search(wn.synsets[s].definition)]


def _fmt(sids: list[str], wn: WordNet) -> str:
    return "; ".join(f"{s} ({wn.synsets[s].definition[:70]})" for s in sids[:2])


def candidate_pairs(senses: pd.DataFrame, wn: WordNet) -> pd.DataFrame:
    """All unordered pairs (a < b alphabetically) with WordNet evidence that one
    English word covers both concepts. Two evidence routes, both recorded:

    * shared_lemma — a lemma occurs in a synset of each gloss;
    * definition   — a sense of one gloss word is defined using the other gloss
                     (catches 'tongue' = a language, which WordNet files under
                     natural_language, not language).
    """
    lemmas = {r.concept_id: set(r.wn_lemmas) for r in senses.itertuples()}
    synset_ids = {r.concept_id: r.wn_synset_ids for r in senses.itertuples()}
    gloss = senses.set_index("concept_id")["gloss"]
    rows = []
    for a, b in combinations(sorted(lemmas), 2):
        shared = lemmas[a] & lemmas[b]
        def_ab = definition_hits(gloss[a], synset_ids[a], gloss[b], wn)
        def_ba = definition_hits(gloss[b], synset_ids[b], gloss[a], wn)
        if not (shared or def_ab or def_ba):
            continue
        evidence, routes = [], []
        if shared:
            routes.append("shared_lemma")
        for w in sorted(shared):
            wkey = w.replace(" ", "_")
            sa = [s for s in synset_ids[a] if wkey in wn.synsets[s].lemmas]
            sb = [s for s in synset_ids[b] if wkey in wn.synsets[s].lemmas]
            evidence.append(f"lemma '{w}': {a} via {_fmt(sa, wn)} | {b} via {_fmt(sb, wn)}")
        if def_ab or def_ba:
            routes.append("definition")
        if def_ab:
            evidence.append(f"'{gloss[a]}' defined via '{gloss[b]}': {_fmt(def_ab, wn)}")
        if def_ba:
            evidence.append(f"'{gloss[b]}' defined via '{gloss[a]}': {_fmt(def_ba, wn)}")
        rows.append({
            "concept_a": a, "concept_b": b, "gloss_a": gloss[a], "gloss_b": gloss[b],
            "shared_lemma": "; ".join(sorted(shared)),
            "n_shared_lemmas": len(shared),
            "evidence_route": "+".join(routes),
            # strongest evidence: one concept's own gloss names a sense of the other
            "gloss_is_shared": gloss[a].lower() in shared or gloss[b].lower() in shared,
            "wordnet_evidence": " || ".join(evidence),
        })
    return pd.DataFrame(rows, columns=["concept_a", "concept_b", "gloss_a", "gloss_b", "shared_lemma",
                                       "n_shared_lemmas", "evidence_route", "gloss_is_shared",
                                       "wordnet_evidence"])


def write_audit_sheet(candidates: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Candidates that are CLICS edges, merged with any existing human decisions."""
    e = edges[["concept_a", "concept_b", "n_families"]]
    sheet = candidates.merge(e, on=["concept_a", "concept_b"], how="inner")
    sheet = sheet.assign(suggested_decision="", decision="", annotator="", notes="")[AUDIT_COLUMNS]
    path = config.ENGLISH_AUDIT_PATH
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        if not old.empty:
            keep = old.set_index(["concept_a", "concept_b"])[["suggested_decision", "decision",
                                                              "annotator", "notes"]]
            sheet = sheet.set_index(["concept_a", "concept_b"])
            sheet.update(keep)
            sheet = sheet.reset_index()[AUDIT_COLUMNS]
            bad = set(sheet["decision"]) - VALID_DECISIONS
            if bad:
                raise ValueError(f"Invalid audit decisions {bad}; allowed: {sorted(VALID_DECISIONS)}")
            dropped = len(old) - len(old.merge(sheet[["concept_a", "concept_b"]]))
            if dropped:
                log.warning("%d audited pairs are no longer candidates/edges and were dropped", dropped)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.sort_values(["n_families", "concept_a", "concept_b"], ascending=[False, True, True]) \
         .to_csv(path, index=False)
    return sheet


def main() -> None:
    set_all_seeds()
    log_run_header(log)
    concepts = read_parquet(config.PROCESSED_DIR / "concepts.parquet")
    working = concepts[concepts["working"]].set_index("concept_id")["gloss"]
    wn = load_wordnet()

    senses = concept_senses(working, wn)
    cands = candidate_pairs(senses, wn)
    write_parquet(senses, config.PROCESSED_DIR / "concept_wordnet.parquet")
    write_parquet(cands, config.PROCESSED_DIR / "english_colex_candidates.parquet")

    n = len(working)
    summary = {
        "wordnet_source": f"{config.WORDNET_REPO} @ {config.WORDNET_COMMIT} ({config.WORDNET_REPO_PATH})",
        "n_working_concepts": n,
        "n_concepts_in_wordnet": int(senses["wn_in_wordnet"].sum()),
        "concepts_not_in_wordnet": senses.loc[~senses["wn_in_wordnet"], "gloss"].tolist(),
        "wn_n_synsets_median": float(senses["wn_n_synsets"].median()),
        "n_pairs_total": n * (n - 1) // 2,
        "n_candidate_pairs": len(cands),
        "n_candidates_gloss_is_shared": int(cands["gloss_is_shared"].sum()),
        "n_candidates_by_route": cands["evidence_route"].value_counts().to_dict(),
        "audit_criteria": AUDIT_CRITERIA,
    }
    edges_path = config.PROCESSED_DIR / "edges.parquet"
    try:
        edges = read_parquet(edges_path)
    except FileNotFoundError:
        log.info("edges.parquet not built yet: audit sheet is written in Phase 2/6.")
    else:
        # H2 population: network edges, minus pairs inside one gloss-collision group (D-35)
        cg = concepts.set_index("concept_id")["collision_group"].fillna("")
        same_group = (edges["concept_a"].map(cg) != "") & (edges["concept_a"].map(cg) == edges["concept_b"].map(cg))
        eligible = edges[edges["is_edge"] & ~same_group]
        summary["n_h2_population_pairs"] = int(len(eligible))
        summary["n_edges_excluded_same_collision_group"] = int((edges["is_edge"] & same_group).sum())
        sheet = write_audit_sheet(cands, eligible)
        summary["n_audit_rows"] = len(sheet)
        summary["audit_decisions"] = sheet["decision"].replace("", "pending").value_counts().to_dict()
    write_result_json(summary, "english_colex_summary", producer=__name__)
    log.info("%d/%d concepts in WordNet; %d candidate English-colexification pairs "
             "(%d where a gloss itself is the shared word)",
             summary["n_concepts_in_wordnet"], n, len(cands), summary["n_candidates_gloss_is_shared"])


if __name__ == "__main__":
    main()
