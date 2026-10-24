"""Phase 3 · Community detection with seed-stability analysis (Proposal Step 3).

Methods (DECISIONS.md D-40, D-41):

* **Louvain** (networkx), weight = n_families, resolution 1.0, COMMUNITY_N_RUNS seeds.
* **Infomap** (``infomap`` package), two-level, undirected, same seeds — run when the
  package is installed, otherwise reported as PENDING USER RUN (never faked).

Stability: adjusted Rand index (ARI) and NMI between every pair of runs of a
method, and between methods. Reference partition (pre-specified rule): the
Louvain **medoid** run — the run with the highest mean ARI to all other runs —
relabelled by community size. It is frozen to
``data/processed/reference_partition.json`` and used unchanged in Phase 5.

Outputs: results/community_stability.json, results/communities.csv,
results/community_summary.csv, results/louvain_resolution_sensitivity.csv,
matrices/coassignment_louvain.npz (share of runs placing two concepts together).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import config
from src.utils.concept_order import load_concept_order, order_hash
from src.utils.io import read_parquet, write_result_json, write_result_table
from src.utils.logging_utils import get_logger, log_run_header
from src.utils.matrices import ConceptMatrix, save_matrix
from src.utils.seeds import set_all_seeds

log = get_logger(__name__, "phase3_communities")
REFERENCE_PATH = config.PROCESSED_DIR / "reference_partition.json"


# ------------------------------------------------------------------ methods --
def louvain_labels(g: nx.Graph, order: list[str], seed: int, resolution: float = config.LOUVAIN_RESOLUTION
                   ) -> np.ndarray:
    comms = nx.community.louvain_communities(g, weight="n_families", resolution=resolution, seed=seed)
    lab = {n: k for k, c in enumerate(comms) for n in c}
    return np.array([lab[c] for c in order])


def infomap_available() -> bool:
    try:
        import infomap  # noqa: F401
    except ImportError:
        return False
    return True


def infomap_labels(g: nx.Graph, order: list[str], seed: int) -> np.ndarray:
    import infomap

    im = infomap.Infomap(f"{config.INFOMAP_ARGS} --seed {seed} --num-trials 1")
    idx = {c: i for i, c in enumerate(order)}
    for c in order:
        im.add_node(idx[c])
    for a, b, d in g.edges(data=True):
        im.add_link(idx[a], idx[b], float(d["n_families"]))
    im.run()
    mods = im.get_modules()                      # {node_id: module_id}
    return np.array([mods[idx[c]] for c in order])


def run_many(fn, g: nx.Graph, order: list[str], n_runs: int) -> np.ndarray:
    """(n_runs, n_concepts) labels; seeds SEED, SEED+1, ... so every run is reproducible."""
    return np.vstack([fn(g, order, config.SEED + i) for i in range(n_runs)])


# ---------------------------------------------------------------- stability --
def pairwise_agreement(parts: np.ndarray) -> dict[str, np.ndarray]:
    n = len(parts)
    ari, nmi = [], []
    for i in range(n):
        for j in range(i + 1, n):
            ari.append(adjusted_rand_score(parts[i], parts[j]))
            nmi.append(normalized_mutual_info_score(parts[i], parts[j]))
    return {"ari": np.array(ari), "nmi": np.array(nmi)}


def cross_agreement(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([adjusted_rand_score(x, y) for x in a for y in b])


def describe(v: np.ndarray) -> dict[str, float]:
    return {"median": float(np.median(v)), "q25": float(np.quantile(v, .25)), "q75": float(np.quantile(v, .75)),
            "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())}


def medoid_run(parts: np.ndarray) -> int:
    """Index of the run with the highest mean ARI to all others (lowest index on ties)."""
    n = len(parts)
    m = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = adjusted_rand_score(parts[i], parts[j])
    return int(np.argmax((m.sum(axis=1) - 1) / (n - 1)))


def coassignment(parts: np.ndarray) -> np.ndarray:
    """Share of runs in which each pair of concepts shares a community."""
    n = parts.shape[1]
    co = np.zeros((n, n))
    for p in parts:
        co += p[:, None] == p[None, :]
    return co / len(parts)


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Community 0 = largest; ties broken by smallest member index (deterministic)."""
    uniq = np.unique(labels)
    key = sorted(uniq, key=lambda u: (-(labels == u).sum(), np.flatnonzero(labels == u)[0]))
    remap = {u: k for k, u in enumerate(key)}
    return np.array([remap[x] for x in labels])


# -------------------------------------------------------------- description --
def community_table(g: nx.Graph, order: list[str], labels: np.ndarray, co: np.ndarray,
                    concepts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = concepts.set_index("concept_id")
    wdeg = dict(g.degree(weight="n_families"))
    df = pd.DataFrame({"concept_id": order, "community": labels,
                       "gloss": [meta.at[c, "gloss"] for c in order],
                       "semantic_field": [meta.at[c, "semantic_field"] for c in order],
                       "weighted_degree": [wdeg.get(c, 0) for c in order],
                       "degree": [g.degree(c) for c in order]})
    core = []
    for i, lab in enumerate(labels):
        same = np.flatnonzero(labels == lab)
        same = same[same != i]
        core.append(float(co[i, same].mean()) if len(same) else np.nan)
    df["coassignment_with_own_community"] = core
    rows = []
    for lab, sub in df.groupby("community"):
        hubs = sub.sort_values(["weighted_degree", "concept_id"], ascending=[False, True])
        field = sub["semantic_field"].value_counts()
        rows.append({
            "community": lab, "size": len(sub),
            "label": " / ".join(hubs["gloss"].head(3)) if len(sub) > 1 else f"{sub['gloss'].iloc[0]} (isolated)",
            "hubs": ", ".join(hubs["gloss"].head(8)),
            "dominant_field": field.index[0] if len(field) else "",
            "dominant_field_share": float(field.iloc[0] / len(sub)) if len(field) else np.nan,
            "mean_coassignment": float(sub["coassignment_with_own_community"].mean()) if len(sub) > 1 else np.nan,
        })
    summary = pd.DataFrame(rows).sort_values("community")
    df = df.merge(summary[["community", "label"]], on="community")
    return df.sort_values(["community", "weighted_degree"], ascending=[True, False]), summary


def freeze_partition(order: list[str], labels: np.ndarray, provenance: dict, overwrite: bool = False) -> str:
    payload = {"concept_order_sha256": order_hash(order),
               "labels": {c: int(x) for c, x in zip(order, labels)}}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if REFERENCE_PATH.exists():
        old = json.loads(REFERENCE_PATH.read_text())
        if old["sha256"] == digest:
            return digest
        if not overwrite:
            raise RuntimeError(f"{REFERENCE_PATH} holds a different frozen partition; pass --refreeze "
                               "(this changes the H1 cluster-agreement reference).")
    REFERENCE_PATH.write_text(json.dumps({**payload, "sha256": digest, "provenance": provenance,
                                          "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                                         indent=1) + "\n")
    return digest


def load_reference_partition(order: list[str]) -> np.ndarray:
    ref = json.loads(REFERENCE_PATH.read_text())
    if ref["concept_order_sha256"] != order_hash(order):
        raise RuntimeError("Reference partition was built for a different concept order — rebuild it.")
    return np.array([ref["labels"][c] for c in order])


# --------------------------------------------------------------------- main --
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refreeze", action="store_true")
    args = ap.parse_args()
    set_all_seeds()
    log_run_header(log)
    order = load_concept_order()
    g = nx.read_graphml(config.PROCESSED_DIR / "network.graphml")
    for _, _, d in g.edges(data=True):          # graphml round-trip may type attributes as str
        d["n_families"] = float(d["n_families"])
    assert set(g.nodes) == set(order), "network nodes != canonical concepts"
    concepts = read_parquet(config.PROCESSED_DIR / "concepts.parquet")
    n = config.COMMUNITY_N_RUNS

    lv = run_many(louvain_labels, g, order, n)
    lv_stab = pairwise_agreement(lv)
    stats: dict = {
        "n_runs": n, "seeds": f"{config.SEED}..{config.SEED + n - 1}",
        "louvain": {"ari": describe(lv_stab["ari"]), "nmi": describe(lv_stab["nmi"]),
                    "n_communities": describe(np.array([len(np.unique(p)) for p in lv])),
                    "modularity": describe(np.array([
                        nx.community.modularity(g, [set(np.array(order)[p == k]) for k in np.unique(p)],
                                                weight="n_families") for p in lv]))},
    }
    if infomap_available():
        im = run_many(infomap_labels, g, order, n)
        im_stab = pairwise_agreement(im)
        stats["infomap"] = {"ari": describe(im_stab["ari"]), "nmi": describe(im_stab["nmi"]),
                            "n_communities": describe(np.array([len(np.unique(p)) for p in im]))}
        stats["louvain_vs_infomap_ari"] = describe(cross_agreement(lv, im))
        np.save(config.PROCESSED_DIR / "infomap_runs.npy", im)
    else:
        stats["infomap"] = "PENDING USER RUN: `pip install infomap==2.8.0`, then `python run.py network`"
        log.warning("infomap not installed: Infomap results are PENDING USER RUN")
    np.save(config.PROCESSED_DIR / "louvain_runs.npy", lv)

    # stability verdict, stated by a rule fixed in advance (D-41)
    med = stats["louvain"]["ari"]["median"]
    stats["louvain_stability_verdict"] = ("stable" if med >= 0.9 else "moderately stable" if med >= 0.75
                                          else "unstable")

    best = medoid_run(lv)
    ref = relabel_by_size(lv[best])
    co = coassignment(lv)
    save_matrix(config.MATRIX_DIR / "coassignment_louvain.npz",
                ConceptMatrix(co, order, "similarity", {"method": "louvain", "n_runs": str(n)}), canonical=order)
    stats["reference"] = {"rule": "Louvain medoid run (max mean ARI to the other runs)",
                          "run_index": best, "seed": config.SEED + best,
                          "n_communities": int(len(np.unique(ref))),
                          "n_singletons": int(sum((ref == k).sum() == 1 for k in np.unique(ref))),
                          "mean_ari_to_other_runs": float(np.mean([adjusted_rand_score(lv[best], p)
                                                                   for i, p in enumerate(lv) if i != best]))}
    digest = freeze_partition(order, ref, {"method": "louvain", "seed": config.SEED + best,
                                           "resolution": config.LOUVAIN_RESOLUTION, "producer": __name__},
                              overwrite=args.refreeze)
    stats["reference"]["sha256"] = digest

    # resolution sensitivity (exploratory)
    sens = []
    for res in (0.5, 0.75, 1.0, 1.5, 2.0):
        labs = [louvain_labels(g, order, config.SEED + i, res) for i in range(20)]
        sens.append({"resolution": res, "n_communities_median": float(np.median([len(np.unique(x)) for x in labs])),
                     "ari_to_reference_median": float(np.median([adjusted_rand_score(ref, x) for x in labs]))})
    write_result_table(pd.DataFrame(sens), "louvain_resolution_sensitivity", producer=__name__)

    table, summary = community_table(g, order, ref, co, concepts)
    write_result_table(table, "communities", producer=__name__)
    write_result_table(summary, "community_summary", producer=__name__)
    write_result_json(stats, "community_stability", producer=__name__)
    log.info("Louvain over %d seeds: ARI median %.3f (min %.3f) -> %s; reference = run %d with %d communities "
             "(%d singletons); Infomap: %s", n, med, stats["louvain"]["ari"]["min"],
             stats["louvain_stability_verdict"], best, stats["reference"]["n_communities"],
             stats["reference"]["n_singletons"],
             "done" if isinstance(stats["infomap"], dict) else "PENDING")


if __name__ == "__main__":
    main()
