"""OPTIONAL Streamlit dashboard: explore the network, query a concept pair, see the model gap live.

    pip install streamlit==1.39.0
    streamlit run src/viz/dashboard.py

Reads only cached outputs (data/processed/, results/); it never recomputes a
statistic. The query logic lives in plain functions (``pair_report``,
``neighbours``) so it is unit-tested without Streamlit (tests/test_phase7_h3.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # `streamlit run` does not set the path

import config  # noqa: E402
from src.analysis.common import CONDITION_LABELS, available_conditions, load_similarity  # noqa: E402
from src.analysis.h2_blindspot import mutual_rank  # noqa: E402
from src.utils.concept_order import load_concept_order  # noqa: E402
from src.utils.io import read_parquet  # noqa: E402


def mutual_rank_one(sim: np.ndarray, i: int, j: int) -> float:
    """Normalised mutual neighbour rank of one pair (the H2 statistic, D-24):
    0 = each concept is the other's nearest neighbour, 1 = farthest."""
    return float(mutual_rank(sim, np.array([i]), np.array([j]))[0])


def neighbours(sim: np.ndarray, order: list[str], gloss: dict[str, str], concept: str, k: int = 8) -> list[str]:
    i = order.index(concept)
    s = sim[i].copy()
    s[i] = -np.inf
    s[~np.isfinite(s)] = -np.inf
    return [gloss.get(order[j], order[j]) for j in np.argsort(-s)[:k]]


def pair_report(a: str, b: str, order: list[str], sims: dict[str, np.ndarray], pairs: pd.DataFrame,
                audit: pd.DataFrame | None = None) -> dict:
    """Everything the dashboard shows for one concept pair."""
    lo, hi = sorted((a, b))
    row = pairs[(pairs["concept_a"] == lo) & (pairs["concept_b"] == hi)]
    i, j = order.index(a), order.index(b)
    models = []
    for c, s in sims.items():
        ok = np.isfinite(s[i, j])
        models.append({"condition": c, "label": CONDITION_LABELS.get(c, c),
                       "cosine": float(s[i, j]) if ok else np.nan,
                       "mutual_rank": mutual_rank_one(s, i, j) if ok else np.nan})
    english = None
    if audit is not None:
        m = audit[((audit["concept_a"] == lo) & (audit["concept_b"] == hi))]
        if len(m):
            r = m.iloc[0]
            english = str(r["decision"]) if isinstance(r.get("decision"), str) and r["decision"] else \
                f"{r['suggested_decision']} (suggested, not yet audited)"
    return {"n_families": int(row["n_families"].iloc[0]) if len(row) else 0,
            "n_languages": int(row["n_languages"].iloc[0]) if len(row) else 0,
            "cov_families": int(row["cov_families"].iloc[0]) if len(row) else 0,
            "english": english or "not a WordNet candidate (English-absent)",
            "models": pd.DataFrame(models)}


def main() -> None:                                                # pragma: no cover - UI
    import streamlit as st

    st.set_page_config(page_title="Colexification and language models", layout="wide")
    order = load_concept_order()
    concepts = read_parquet(config.PROCESSED_DIR / "concepts.parquet")
    gloss = concepts.set_index("concept_id")["gloss"].to_dict()
    pairs = read_parquet(config.PROCESSED_DIR / "pairs.parquet")
    edges = read_parquet(config.PROCESSED_DIR / "edges.parquet")
    conds = available_conditions()
    sims = {c: load_similarity(c, order) for c in conds}
    audit = pd.read_csv(config.ENGLISH_AUDIT_PATH, keep_default_na=False) if config.ENGLISH_AUDIT_PATH.exists() else None

    st.title("Do language models reproduce the world's colexifications?")
    st.caption(f"{len(order)} concepts · CLICS⁴ · conditions computed: "
               + ", ".join(CONDITION_LABELS[c] for c in conds)
               + (" · pending: " + ", ".join(CONDITION_LABELS[c] for c in config.MODEL_CONDITIONS if c not in conds)
                  if len(conds) < len(config.MODEL_CONDITIONS) else ""))
    tab1, tab2, tab3 = st.tabs(["Concept pair", "Concept neighbourhood", "Results"])
    labels = sorted(order, key=lambda c: gloss.get(c, c))
    fmt = lambda c: f"{gloss.get(c, c)}  [{c}]"  # noqa: E731
    with tab1:
        c1, c2 = st.columns(2)
        a = c1.selectbox("Concept A", labels, index=labels.index("YESTERDAY") if "YESTERDAY" in labels else 0,
                         format_func=fmt)
        b = c2.selectbox("Concept B", labels, index=labels.index("TOMORROW") if "TOMORROW" in labels else 1,
                         format_func=fmt)
        if a != b:
            rep = pair_report(a, b, order, sims, pairs, audit)
            m1, m2, m3 = st.columns(3)
            m1.metric("Families colexifying", rep["n_families"])
            m2.metric("Languages colexifying", rep["n_languages"])
            m3.metric("Families attesting both", rep["cov_families"])
            st.write(f"**English:** {rep['english']}")
            st.dataframe(rep["models"].drop(columns="condition"), hide_index=True)
            st.caption("Mutual rank: 0 = each concept is the other's nearest neighbour, 1 = farthest.")
            lo, hi = sorted((a, b))
            ex = edges[(edges["concept_a"] == lo) & (edges["concept_b"] == hi)]
            if len(ex) and isinstance(ex["examples"].iloc[0], str):
                st.write("**Example forms:**")
                st.write("\n".join(f"- {e}" for e in ex["examples"].iloc[0].split(" | ")))
    with tab2:
        cpt = st.selectbox("Concept", labels, format_func=fmt, key="nb")
        cols = st.columns(max(1, len(conds) + 1))
        nb = edges[(edges["is_edge"]) & ((edges["concept_a"] == cpt) | (edges["concept_b"] == cpt))]
        nb = nb.assign(other=np.where(nb["concept_a"] == cpt, nb["concept_b"], nb["concept_a"]))
        nb = nb.sort_values("n_families", ascending=False).head(8)
        cols[0].write("**Colexified with (families)**")
        cols[0].write("\n".join(f"- {gloss.get(o, o)} ({n})" for o, n in zip(nb["other"], nb["n_families"])))
        for k, c in enumerate(conds, start=1):
            cols[k].write(f"**{CONDITION_LABELS[c]}**")
            cols[k].write("\n".join(f"- {g}" for g in neighbours(sims[c], order, gloss, cpt)))
    with tab3:
        for name in ("h1_alignment", "h2_blindspot", "h3_typology"):
            p = config.RESULTS_DIR / f"{name}.csv"
            if p.exists():
                st.subheader(name)
                st.dataframe(pd.read_csv(p), hide_index=True)
        for fig in ("fig11_synthesis", "fig3_h1_alignment", "fig8_h2_gap_by_model", "fig9_h3_families"):
            p = config.FIGURES_DIR / f"{fig}.png"
            if p.exists():
                st.image(str(p), caption=fig)


if __name__ == "__main__":
    main()
