"""All static figures (PDF + PNG, 300 dpi, Okabe-Ito / viridis).

Figures are registered with the result files they need. ``python -m
src.viz.figures`` builds every figure whose inputs exist and lists the rest as
PENDING, so the module is usable from Phase 1 onward and complete at Phase 7.
Every figure reads numbers from results/ or data/processed/ — never literals.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.utils.logging_utils import get_logger, log_run_header  # noqa: E402

log = get_logger(__name__, "figures")
C = config.OKABE_ITO


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": config.FIG_DPI,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.labelcolor": config.INK,
        "axes.edgecolor": config.INK_MUTED,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": config.GRID,
        "grid.linewidth": 0.6,
        "xtick.color": config.INK_MUTED,
        "ytick.color": config.INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "pdf.fonttype": 42,     # editable text in the PDF
    })


def save(fig: plt.Figure, stem: str) -> list[Path]:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in config.FIG_FORMATS:
        p = config.FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------- Phase 1 --
def fig_concept_coverage() -> list[Path]:
    """Fig. S1 — concept coverage curve and the Kneedle threshold."""
    curve = pd.read_csv(config.RESULTS_DIR / "concept_coverage_curve.csv")
    info = json.loads((config.RESULTS_DIR / "concept_threshold.json").read_text())
    t, n = info["threshold_used"], int(curve.loc[curve["min_families"] == info["threshold_used"],
                                                   "n_concepts"].iloc[0])
    tr = info["robustness_threshold"]
    nr = int(curve.loc[curve["min_families"] == tr, "n_concepts"].iloc[0])
    lo, hi = config.CONCEPT_SET_TARGET

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
    a.axhspan(lo, hi, color=C[5], alpha=0.12, lw=0)
    a.text(curve["min_families"].max(), lo + 8, f"target {lo}–{hi}", ha="right", va="bottom",
           fontsize=7.5, color=config.INK_MUTED)
    a.plot(curve["min_families"], curve["n_concepts"], color=C[0], marker="o", ms=3.5, lw=1.8)
    a.axvline(t, color=config.INK, lw=0.8, ls="--")
    a.scatter([t], [n], s=60, color=C[3], zorder=5, edgecolor="white", linewidth=1.5)
    a.annotate(f"knee: ≥ {t} families\n→ {n} concepts ({info['n_working']} working)", (t, n), xytext=(14, 40),
               textcoords="offset points", fontsize=8, color=config.INK,
               arrowprops={"arrowstyle": "-", "color": config.INK_MUTED, "lw": 0.6})
    a.scatter([tr], [nr], s=40, color="white", edgecolor=C[1], linewidth=1.8, zorder=5)
    a.annotate(f"robustness: ≥ {tr} → {nr} ({info['n_robustness_set']} working)", (tr, nr),
               xytext=(8, 14), textcoords="offset points", fontsize=7.5, color=config.INK_MUTED)
    a.set_xlabel("Minimum number of language families attesting the concept")
    a.set_ylabel("Concepts retained")
    a.set_title("a  Coverage curve")
    a.set_ylim(0, None)

    b.plot(curve["min_families"], curve["difference"], color=C[0], lw=1.8)
    b.axvline(t, color=config.INK, lw=0.8, ls="--")
    b.scatter([t], [curve["difference"].max()], s=60, color=C[3], zorder=5, edgecolor="white", linewidth=1.5)
    b.set_xlabel("Minimum number of families")
    b.set_ylabel("Distance below chord (normalised)")
    b.set_title("b  Kneedle difference curve")
    fig.suptitle(f"Concept selection on CLICS⁴: {info['n_working']} working concepts "
                 f"(after {info['n_excluded_by_override']} exclusions, "
                 f"+{len(info['anchors_added_below_threshold'])} anchors)",
                 x=0.01, ha="left", fontsize=9, color=config.INK_MUTED, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, "figS1_concept_coverage")


# ----------------------------------------------------------------- Phase 2 --
def fig_network_data_decisions() -> list[Path]:
    """Fig. S2 — why the network is built the way it is (D-07, D-37, D-39)."""
    import numpy as np

    from src.utils.io import read_parquet

    pairs = read_parquet(config.PROCESSED_DIR / "pairs.parquet")
    alt = read_parquet(config.PROCESSED_DIR / "pairs__all__segments.parquet")
    ties = pd.read_csv(config.RESULTS_DIR / "typological_matrix_ties.csv")
    gl = read_parquet(config.PROCESSED_DIR / "concepts.parquet").set_index("concept_id")["gloss"]

    fig, (pa, pb, pc) = plt.subplots(1, 3, figsize=(10.5, 3.3), gridspec_kw={"width_ratios": [1, 1.15, 1]})

    # a: support distribution of colexified pairs
    nf = pairs.loc[pairs["n_families"] >= 1, "n_families"]
    bins = np.unique(np.round(np.logspace(0, np.log10(nf.max() + 1), 22)).astype(int)) - 0.5
    pa.hist(nf, bins=bins, color=C[0], edgecolor="white", linewidth=0.8)
    pa.set_xscale("log")
    pa.set_yscale("log")
    pa.axvline(config.EDGE_MIN_FAMILIES - 0.5, color=config.INK, lw=0.8, ls="--")
    n1, n2 = int((nf == 1).sum()), int((nf >= config.EDGE_MIN_FAMILIES).sum())
    pa.text(0.97, 0.80, f"{n1:,} pairs in 1 family (dropped)\n{n2:,} edges in ≥ {config.EDGE_MIN_FAMILIES} families",
            transform=pa.transAxes, ha="right", va="top", fontsize=7.5, color=config.INK)
    pa.set_xlabel("Families colexifying the pair (log)")
    pa.set_ylabel("Concept pairs (log)")
    pa.set_title("a  Edge support")

    # b: split source entries
    m = pairs.merge(alt, on=["concept_a", "concept_b"], suffixes=("", "_all"))
    m = m[m["n_families_all"] >= 1]
    x, y = m["n_families_all"] + 1, m["n_families"] + 1
    pb.scatter(x, y, s=6, color=C[0], alpha=0.35, lw=0)
    lim = float(x.max()) * 1.6
    pb.plot([1, lim], [1, lim], color=config.INK_MUTED, lw=0.6, ls=":")
    pb.set_xscale("log")
    pb.set_yscale("log")
    pb.set_xlim(1, lim)
    pb.set_ylim(1, lim)
    top = m.assign(drop=m["n_families_all"] - m["n_families"]).nlargest(5, "drop").sort_values("n_families")
    for ly, r in zip(np.geomspace(1.5, 8, len(top)), top.itertuples()):
        px, py = r.n_families_all + 1, r.n_families + 1
        pb.scatter([px], [py], s=26, color=C[3], zorder=4, edgecolor="white", linewidth=0.8)
        pb.annotate(f"{gl[r.concept_a]}~{gl[r.concept_b]}: {r.n_families_all}→{r.n_families}",
                    (px, py), xytext=(14, ly), textcoords="data", ha="left", va="center", fontsize=7,
                    color=config.INK, arrowprops={"arrowstyle": "-", "color": config.INK_MUTED, "lw": 0.5})
    pb.set_xlabel("Families + 1, CLICS⁴ as published")
    pb.set_ylabel("Families + 1, separately recorded only")
    pb.set_title("b  Split source entries removed")

    # c: ties per typological metric
    names = ["shortest_path", "profile_cosine", "language_profile", "binary_profile", "rate"]
    labels = {"shortest_path": "shortest path (primary)", "profile_cosine": "profile cosine",
              "language_profile": "language profile", "binary_profile": "binary profile",
              "rate": "first-order rate"}
    t = ties.set_index("matrix").loc[names, "share_pairs_at_max"] * 100
    ypos = np.arange(len(names))[::-1]
    pc.barh(ypos, t, height=0.6,
            color=[C[0] if n == config.TYPOLOGICAL_DISTANCE_PRIMARY else C[5] for n in names])
    for yi, v in zip(ypos, t):
        pc.text(v + 1.5, yi, f"{v:.1f}%", va="center", fontsize=7.5, color=config.INK)
    pc.set_yticks(ypos, [labels[n] for n in names])
    pc.set_xlim(0, 115)
    pc.set_xlabel("Concept pairs tied at maximum distance (%)")
    pc.set_title("c  Ties by distance metric")
    pc.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, "figS2_network_decisions")


# ----------------------------------------------------------------- Phase 3 --
GREY = "#c8c8c8"


def community_colours(comm: pd.DataFrame, summary: pd.DataFrame) -> dict[int, str]:
    """Rule (D-44): colour the communities that contain the proposal's anchor
    concepts, then the largest remaining multi-concept ones, up to the 6 Okabe-Ito
    hues that passed the palette validator, in fixed order; every other community
    is grey."""
    anchor_comms = list(dict.fromkeys(comm.set_index("concept_id").loc[list(config.ANCHOR_CONCEPTS),
                                                                       "community"].tolist()))
    multi = summary[summary["size"] > 1].sort_values(["size", "community"], ascending=[False, True])
    ranked = sorted(anchor_comms, key=lambda c: int(summary.set_index("community").at[c, "size"]), reverse=True)
    ranked += [c for c in multi["community"] if c not in ranked]
    return {int(c): C[i] for i, c in enumerate(ranked[:6])}   # the 6 hues that passed the validator


def _load_communities():
    comm = pd.read_csv(config.RESULTS_DIR / "communities.csv")
    summary = pd.read_csv(config.RESULTS_DIR / "community_summary.csv")
    return comm, summary, community_colours(comm, summary)


def _short_label(label: str) -> str:
    return " / ".join(label.split(" / ")[:2])


def network_layout(core, lab: dict, seed: int = config.SEED) -> dict:
    """Community-aware force layout shared by Fig. 1 and the interactive network:
    within-community edges pull 6x harder; the outermost 1% are pulled in."""
    import networkx as nx
    import numpy as np

    for a_, b_, d in core.edges(data=True):
        d["w_layout"] = float(np.log1p(float(d["n_families"]))) * (6.0 if lab[a_] == lab[b_] else 1.0)
    pos = nx.spring_layout(core, weight="w_layout", seed=seed, iterations=500, k=1.6 / np.sqrt(len(core)))
    pts = np.array(list(pos.values()))
    lo, hi = np.quantile(pts, 0.01, axis=0), np.quantile(pts, 0.99, axis=0)
    return {n: np.clip(p, lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo)) for n, p in pos.items()}


def fig_network() -> list[Path]:
    """Fig. 1 — the family-weighted colexification network."""
    import networkx as nx
    import numpy as np

    g = nx.read_graphml(config.PROCESSED_DIR / "network.graphml")
    for _, _, d in g.edges(data=True):
        d["n_families"] = float(d["n_families"])
        d["w"] = float(np.log1p(d["n_families"]))
    comm, summary, colours = _load_communities()
    lab = comm.set_index("concept_id")["community"].to_dict()
    gloss = comm.set_index("concept_id")["gloss"].to_dict()
    core = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    pos = network_layout(core, lab)
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    ax.set_axis_off()
    ew = np.array([d["n_families"] for _, _, d in core.edges(data=True)])
    order_e = np.argsort(ew)                          # strongest edges drawn last
    edges = list(core.edges())
    for i in order_e:
        a, b = edges[i]
        same = lab[a] == lab[b] and lab[a] in colours
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=colours[lab[a]] if same else "#9a9a9a",
                lw=0.15 + 2.2 * np.sqrt(ew[i] / ew.max()), alpha=0.55 if same else 0.18,
                solid_capstyle="round", zorder=1)
    deg = dict(core.degree())
    nodes = sorted(core.nodes, key=lambda n: lab[n] in colours)  # grey first, coloured on top
    ax.scatter([pos[n][0] for n in nodes], [pos[n][1] for n in nodes],
               s=[8 + 2.2 * deg[n] for n in nodes], c=[colours.get(lab[n], GREY) for n in nodes],
               edgecolors="white", linewidths=0.6, zorder=2)
    # direct labels: top-2 hubs per coloured community + anchors
    hubs = (comm[comm["concept_id"].isin(core.nodes)].sort_values("weighted_degree", ascending=False)
            .groupby("community").head(2))
    hubs = hubs.groupby("community").head(1)
    show = set(hubs[hubs["community"].isin(colours)]["concept_id"]) | set(config.ANCHOR_CONCEPTS)
    for n in sorted(show & set(core.nodes)):
        ax.annotate(gloss[n], pos[n], xytext=(4, 3), textcoords="offset points", fontsize=7,
                    color=config.INK, zorder=3,
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.7})
    handles = [plt.Line2D([], [], marker="o", ls="", color=col, markersize=6,
                          label=_short_label(summary.set_index("community").at[k, "label"]))
               for k, col in colours.items()]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=GREY, markersize=6, label="other communities"))
    ax.legend(handles=handles, loc="upper left", fontsize=7, title="Community (top hubs)", title_fontsize=7.5,
              ncol=1, handletextpad=0.3, bbox_to_anchor=(1.0, 1.0), frameon=False)
    n_iso = g.number_of_nodes() - core.number_of_nodes()
    ax.set_title(f"Colexification network of {g.number_of_nodes()} concepts in CLICS⁴ "
                 f"({g.number_of_edges():,} edges, each attested in ≥ {config.EDGE_MIN_FAMILIES} families)",
                 fontsize=9, loc="left")
    ax.text(0, -0.03, f"Largest component shown ({core.number_of_nodes()} concepts; {n_iso} isolated concepts "
            "omitted). Node size = degree; edge width = number of families.\nLayout: force-directed (seed "
            f"{config.SEED}) with within-community edges weighted 6× to separate communities; outermost 1% pulled in.",
            transform=ax.transAxes, fontsize=7, color=config.INK_MUTED, va="top")
    fig.tight_layout()
    return save(fig, "fig1_network")


def _deoverlap(pos: dict, min_dy: float, min_dx: float, iters: int = 200) -> dict:
    """Greedy vertical separation of label anchor points (data units): any two
    labels closer than min_dx horizontally and min_dy vertically are pushed apart."""
    import numpy as np

    pos = {k: np.array(v, dtype=float) for k, v in pos.items()}
    keys = sorted(pos, key=lambda k: pos[k][1])
    for _ in range(iters):
        moved = False
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                dx, dy = pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]
                if abs(dx) < min_dx and abs(dy) < min_dy:
                    shift = (min_dy - abs(dy)) / 2 + 1e-9
                    sgn = 1.0 if dy >= 0 else -1.0
                    pos[b][1] += sgn * shift
                    pos[a][1] -= sgn * shift
                    moved = True
        if not moved:
            break
    return pos


NAMED_PAIRS = [("HAND", "ARM"), ("TREE", "WOOD"), ("YESTERDAY", "TOMORROW"), ("FINGER", "TOE"),
               ("SEE", "UNDERSTAND"), ("TONGUE", "LANGUAGE")]


def fig_typological_map() -> list[Path]:
    """Fig. 2 — 2-D map of the typological space (method chosen by trustworthiness),
    with the proposal's named concept pairs joined."""
    import json

    import numpy as np

    from src.utils.io import read_parquet

    choice = json.loads((config.RESULTS_DIR / "dr_choice.json").read_text())
    q = pd.read_csv(config.RESULTS_DIR / "dr_quality.csv").set_index("method").loc[choice["chosen"]]
    xy = read_parquet(config.PROCESSED_DIR / "typological_2d.parquet")
    xy = xy[xy["method"] == choice["chosen"]]
    comm, summary, colours = _load_communities()
    df = xy.merge(comm, on="concept_id")
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    grey = df[~df["community"].isin(colours)]
    ax.scatter(grey["x"], grey["y"], s=13, color=GREY, edgecolors="white", linewidths=0.4, zorder=1,
               label="other communities")
    for k, col in colours.items():
        sub = df[df["community"] == k]
        ax.scatter(sub["x"], sub["y"], s=18, color=col, edgecolors="white", linewidths=0.5, zorder=2,
                   label=_short_label(summary.set_index("community").at[k, "label"]))
    p = df.set_index("concept_id")
    for a, b in NAMED_PAIRS:
        ax.plot([p.at[a, "x"], p.at[b, "x"]], [p.at[a, "y"], p.at[b, "y"]], color=config.INK, lw=1.0,
                zorder=3, alpha=0.8)
    centre = df[["x", "y"]].median().to_numpy()
    span = np.ptp(df[["x", "y"]].to_numpy(), axis=0).max()
    pts = {cid: p.loc[cid, ["x", "y"]].to_numpy(dtype=float) for cid in config.ANCHOR_CONCEPTS}
    lab_pos = {}
    for cid, pt in pts.items():
        v = pt - centre
        v = v / (np.linalg.norm(v) + 1e-9)
        lab_pos[cid] = pt + v * 0.12 * span
    lab_pos = _deoverlap(lab_pos, min_dy=0.045 * span, min_dx=0.16 * span)
    for cid, pt in pts.items():
        ax.scatter([pt[0]], [pt[1]], s=34, facecolor="white", edgecolor=config.INK, linewidth=1.1, zorder=4)
        lp = lab_pos[cid]
        ax.annotate(p.at[cid, "gloss"], pt, xytext=lp, textcoords="data", fontsize=8,
                    fontweight="bold", color=config.INK, ha="left" if lp[0] >= pt[0] else "right", va="center",
                    zorder=5, arrowprops={"arrowstyle": "-", "color": config.INK_MUTED, "lw": 0.5},
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85})
    ax.set_xlabel("Dimension 1 (typological distance units)")
    ax.set_ylabel("Dimension 2 (typological distance units)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7, markerscale=1.2,
              title="Community (top hubs)", title_fontsize=7.5, frameon=False)
    names = {"mds_metric": "metric MDS (SMACOF)", "pca": "classical MDS", "umap": "UMAP",
             "mds_nonmetric": "non-metric MDS"}
    ax.set_title(f"Typological semantic space: {names.get(choice['chosen'], choice['chosen'])} of weighted "
                 "shortest-path distances", loc="left", fontsize=9)
    n_iso = len(choice.get("isolated_concepts_omitted", []))
    ax.text(0, -0.11, f"N = {choice['n_concepts_embedded']} concepts in the connected network ({n_iso} isolated concepts "
            f"have no position). Trustworthiness (k={config.TRUSTWORTHINESS_K}) {q['trustworthiness']:.2f}; "
            f"stress-1 {q['stress1']:.2f}; Shepard ρ {q['shepard_spearman']:.2f}.\n"
            "Black lines join the concept pairs named in the proposal; open circles mark them.",
            transform=ax.transAxes, fontsize=7, color=config.INK_MUTED, va="top")
    fig.tight_layout()
    return save(fig, "fig2_typological_map")


def fig_dr_comparison() -> list[Path]:
    """Fig. S3 — all 2-D maps of the same distances, with their quality scores."""
    import json

    from src.utils.io import read_parquet

    q = pd.read_csv(config.RESULTS_DIR / "dr_quality.csv").set_index("method")
    choice = json.loads((config.RESULTS_DIR / "dr_choice.json").read_text())
    xy = read_parquet(config.PROCESSED_DIR / "typological_2d.parquet")
    comm, summary, colours = _load_communities()
    panels = ["pca", "mds_metric", "mds_nonmetric", "umap", "pca_concept_lexeme"]
    titles = {"pca": "classical MDS (PCA)", "mds_metric": "metric MDS", "mds_nonmetric": "non-metric MDS",
              "umap": "UMAP", "pca_concept_lexeme": "PCA, concept × word form"}
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 2.9))
    for ax, m, letter in zip(axes, panels, "abcde"):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        sub = xy[xy["method"] == m].merge(comm, on="concept_id")
        ax.set_title(f"{letter}  {titles[m]}", fontsize=8.5)
        if sub.empty:
            ax.text(0.5, 0.5, "PENDING USER RUN\n(umap-learn not installed\nin build environment)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7.5, color=config.INK_MUTED)
            continue
        ax.scatter(sub["x"], sub["y"], s=5, c=[colours.get(k, GREY) for k in sub["community"]], lw=0)
        if m == "pca_concept_lexeme":
            cl = choice["concept_lexeme_pca"]
            txt = (f"variance {cl['explained_variance_ratio'][0]*100:.1f}% + "
                   f"{cl['explained_variance_ratio'][1]*100:.1f}%\nShepard ρ vs D "
                   f"{cl['agreement_with_typological_distance']['shepard_spearman']:.2f}")
        else:
            r = q.loc[m]
            txt = f"T = {r['trustworthiness']:.2f}, stress-1 = {r['stress1']:.2f}\nShepard ρ = {r['shepard_spearman']:.2f}"
            if "DEGENERATE" in str(r["note"]):
                txt += "  (degenerate)"
            if m == choice["chosen"]:
                txt += "  ← chosen"
        ax.text(0.02, -0.04, txt, transform=ax.transAxes, va="top", fontsize=7, color=config.INK)
    fig.suptitle(f"Two-dimensional maps of the same typological distances, {choice['n_concepts_embedded']} concepts "
                 "of the connected network (colours = communities as in Fig. 1)", x=0.01, ha="left", fontsize=9,
                 color=config.INK_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, "figS3_dr_comparison")


def fig_community_stability() -> list[Path]:
    """Fig. S4 — how stable the communities are across 100 Louvain seeds."""
    import json

    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    from src.utils.concept_order import load_concept_order
    from src.utils.matrices import load_matrix

    order = load_concept_order()
    runs = np.load(config.PROCESSED_DIR / "louvain_runs.npy")
    stab = json.loads((config.RESULTS_DIR / "community_stability.json").read_text())
    sens = pd.read_csv(config.RESULTS_DIR / "louvain_resolution_sensitivity.csv")
    co = load_matrix(config.MATRIX_DIR / "coassignment_louvain.npz", order).values
    comm, summary, colours = _load_communities()
    ari = [adjusted_rand_score(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))]

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(10.5, 3.3), gridspec_kw={"width_ratios": [1, 1.1, 0.9]})
    a.hist(ari, bins=30, color=C[0], edgecolor="white", linewidth=0.5)
    med = stab["louvain"]["ari"]["median"]
    a.axvline(med, color=config.INK, lw=0.8, ls="--")
    a.text(med, a.get_ylim()[1] * 0.95, f" median {med:.2f}", fontsize=7.5, va="top", color=config.INK)
    a.set_xlabel("ARI between two Louvain runs")
    a.set_ylabel(f"Run pairs (of {len(ari):,})")
    a.set_title("a  Seed stability")
    if isinstance(stab.get("infomap"), str):
        a.text(0.02, 0.97, "Infomap: PENDING USER RUN", transform=a.transAxes, fontsize=7, va="top",
               color=config.INK_MUTED)

    ref = comm.set_index("concept_id").loc[order, "community"].to_numpy()
    idx = np.lexsort((np.arange(len(order)), ref))
    im = b.imshow(co[np.ix_(idx, idx)], cmap=config.SEQUENTIAL_CMAP, vmin=0, vmax=1, interpolation="nearest")
    b.set_xticks([]); b.set_yticks([]); b.grid(False)
    cb = fig.colorbar(im, ax=b, fraction=0.046, pad=0.02)
    cb.set_label("Share of runs together", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    b.set_title("b  Co-assignment (ordered by reference)")
    b.set_xlabel("Concepts (427)")

    c.plot(sens["resolution"], sens["ari_to_reference_median"], color=C[0], marker="o", ms=4)
    for r in sens.itertuples():
        c.annotate(f"{int(r.n_communities_median)}", (r.resolution, r.ari_to_reference_median),
                   xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7, color=config.INK_MUTED)
    c.set_xlabel("Louvain resolution")
    c.set_ylabel("Median ARI to reference partition")
    c.set_ylim(0, 1.05)
    c.set_title("c  Resolution (labels = no. of communities)")
    fig.tight_layout()
    return save(fig, "figS4_community_stability")


# ----------------------------------------------------------------- Phase 5 --
def _condition_order() -> list[str]:
    return list(config.MODEL_CONDITIONS)


def _labels() -> dict[str, str]:
    from src.analysis.common import CONDITION_LABELS
    return CONDITION_LABELS


def fig_h1_alignment() -> list[Path]:
    """Fig. 3 — RSA between typological and model spaces, raw and partial, with CIs."""
    import json

    import numpy as np

    res = pd.read_csv(config.RESULTS_DIR / "h1_alignment.csv").set_index("condition")
    aset = json.loads((config.RESULTS_DIR / "analysis_concept_set.json").read_text())
    labels = _labels()
    conds = _condition_order()
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    y = np.arange(len(conds))[::-1]
    ax.axvline(0, color=config.INK_MUTED, lw=0.8)
    for yi, c in zip(y, conds):
        r = res.loc[c]
        if r["status"] != "done":
            ax.text(0.002, yi, "PENDING USER RUN", va="center", fontsize=7.5, color=config.INK_MUTED)
            continue
        ax.plot([r["rho_ci_low"], r["rho_ci_high"]], [yi, yi], color=C[0], lw=2, solid_capstyle="round")
        ax.scatter([r["rho_raw"]], [yi], s=46, color=C[0], zorder=3, edgecolor="white", linewidth=1)
        ax.scatter([r["rho_partial"]], [yi - 0.22], s=40, facecolor="white", edgecolor=C[3], linewidth=1.6, zorder=3)
        star = "" if not r.get("h1_supported", False) else "  (BH p < .05)"
        ax.text(r["rho_ci_high"] + 0.008, yi, f"ρ = {r['rho_raw']:.3f}{star}", va="center", fontsize=7.5,
                color=config.INK)
    ax.set_yticks(y, [labels[c] for c in conds])
    ax.grid(axis="y", visible=False)
    done = res[res["status"] == "done"]
    hi = max(0.05, float(done[["rho_ci_high", "rho_partial"]].max().max()) if len(done) else 0.1)
    lo = min(-0.02, float(done[["rho_ci_low", "rho_partial"]].min().min()) if len(done) else 0)
    ax.set_xlim(min(lo - 0.02, -0.05), max(0.4, hi + 0.15))
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.set_xlabel("Spearman ρ between typological and model distances")
    handles = [plt.Line2D([], [], marker="o", ls="-", color=C[0], label="raw ρ with 95% concept-bootstrap CI"),
               plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=C[3],
                          label="partial ρ (English frequency, semantic field)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
    ax.set_title("H1 · Alignment of model spaces with the typological space", loc="left", fontsize=9, pad=10)
    ax.text(0, -0.2, f"N = {aset['n_concepts']} concepts, {aset['n_pairs']:,} pairs (D-16 analysis set). "
            f"Mantel tests: {config.N_PERMUTATIONS:,} joint concept permutations, one-sided, "
            "Benjamini–Hochberg across conditions.", transform=ax.transAxes, fontsize=7, color=config.INK_MUTED)
    fig.tight_layout()
    return save(fig, "fig3_h1_alignment")


def fig_h1_heatmap() -> list[Path]:
    """Fig. 4 — correlations among the typological and all model distance matrices."""
    import numpy as np

    m = pd.read_csv(config.RESULTS_DIR / "h1_matrix_correlations.csv").set_index("matrix")
    labels = {"typology": "Typology (CLICS⁴)", **_labels()}
    names = list(m.index)
    vals = m.loc[names, names].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(0.9 * len(names) + 2.6, 0.75 * len(names) + 1.6))
    lim = max(0.1, np.nanmax(np.abs(vals[~np.eye(len(names), dtype=bool)])) if len(names) > 1 else 1)
    im = ax.imshow(vals, cmap=config.DIVERGING_CMAP, vmin=-lim, vmax=lim)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(vals[i, j]) > 0.6 * lim else config.INK)
    ax.set_xticks(range(len(names)), [labels.get(n, n) for n in names], rotation=35, ha="right")
    ax.set_yticks(range(len(names)), [labels.get(n, n) for n in names])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Spearman ρ (upper triangles)", fontsize=7.5)
    pending = [c for c in config.MODEL_CONDITIONS if c not in names]
    ax.set_title("Agreement between distance matrices", loc="left", fontsize=9)
    if pending:
        ax.text(0, -0.35 - 0.02 * len(names), "Not shown (PENDING USER RUN): "
                + ", ".join(_labels()[c] for c in pending), transform=ax.transAxes, fontsize=7,
                color=config.INK_MUTED)
    fig.tight_layout()
    return save(fig, "fig4_h1_matrix_correlations")


def fig_h1_procrustes() -> list[Path]:
    """Fig. 5 — typological map and Procrustes-aligned model maps."""
    from src.utils.io import read_parquet

    xy = read_parquet(config.PROCESSED_DIR / "h1_procrustes_coords.parquet")
    res = pd.read_csv(config.RESULTS_DIR / "h1_alignment.csv").set_index("condition")
    comm, summary, colours = _load_communities()
    spaces = ["typology"] + list(config.MODEL_CONDITIONS)
    labels = {"typology": "Typology (CLICS⁴)", **_labels()}
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    axes = axes.ravel()
    for ax, sp in zip(axes, spaces):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(labels[sp], fontsize=8.5)
        sub = xy[xy["space"] == sp].merge(comm, on="concept_id")
        if sub.empty:
            ax.text(0.5, 0.5, "PENDING USER RUN", ha="center", va="center", transform=ax.transAxes,
                    fontsize=7.5, color=config.INK_MUTED)
            continue
        ax.scatter(sub["x"], sub["y"], s=5, c=[colours.get(k, GREY) for k in sub["community"]], lw=0)
        if sp != "typology":
            r = res.loc[sp]
            ax.text(0.02, 0.02, f"m² = {r['procrustes_m2']:.2f}, PROTEST p = {r['p_protest']:.3g}",
                    transform=ax.transAxes, fontsize=7, color=config.INK)
    for ax in axes[len(spaces):]:
        ax.set_axis_off()
    n = int(xy[xy["space"] == "typology"].shape[0])
    fig.suptitle(f"Classical MDS of each space, aligned to the typological map by Procrustes rotation and scaling "
                 f"(N = {n} connected concepts; colours = communities as in Fig. 1). Lower m² = closer match.",
                 x=0.01, ha="left", fontsize=8.5, color=config.INK_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, "fig5_h1_procrustes")


def fig_h1_regression_clusters() -> list[Path]:
    """Fig. S5 — pair-level regression coefficients and cluster agreement across k."""
    import numpy as np

    reg = pd.read_csv(config.RESULTS_DIR / "h1_regression.csv")
    clu = pd.read_csv(config.RESULTS_DIR / "h1_cluster_agreement.csv")
    labels = _labels()
    conds = [c for c in config.MODEL_CONDITIONS if c in set(reg["condition"])]
    terms = {"log_families": "log(1 + families colexifying)", "zipf_min": "lower Zipf frequency",
             "zipf_max": "higher Zipf frequency", "same_field": "same semantic field"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
    off = np.linspace(-0.25, 0.25, len(conds)) if len(conds) > 1 else np.zeros(1)
    for k, c in enumerate(conds):
        r = reg[(reg["condition"] == c) & reg["term"].isin(terms)].set_index("term").loc[list(terms)]
        yy = np.arange(len(terms))[::-1] + off[k]
        a.errorbar(r["beta_std"], yy, xerr=[r["beta_std"] - r["ci_low"], r["ci_high"] - r["beta_std"]],
                   fmt="o", ms=4, color=C[k], lw=1.4, label=labels[c])
    a.axvline(0, color=config.INK_MUTED, lw=0.8)
    a.set_yticks(np.arange(len(terms))[::-1], list(terms.values()))
    a.grid(axis="y", visible=False)
    a.set_xlabel("Standardised coefficient (95% concept-bootstrap CI)")
    a.set_title("a  Predicting model cosine for each concept pair", fontsize=9)
    a.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=3, frameon=False)
    for k, c in enumerate(conds):
        for method, ls in (("agglomerative", "-"), ("kmeans", "--")):
            sub = clu[(clu["condition"] == c) & (clu["subset"] == "all") & (clu["method"] == method)].sort_values("k")
            if not sub.empty:
                b.plot(sub["k"], sub["ari"], ls=ls, marker="o", ms=3, color=C[k],
                       label=f"{labels[c]} · {method}")
    kref = clu.loc[clu["primary"], "k"]
    if len(kref):
        b.axvline(int(kref.iloc[0]), color=config.INK_MUTED, lw=0.8, ls=":")
        b.text(int(kref.iloc[0]), b.get_ylim()[1], " k of reference", fontsize=7, va="top", color=config.INK_MUTED)
    b.axhline(0, color=config.INK_MUTED, lw=0.6)
    b.set_xlabel("Number of model clusters k")
    b.set_ylabel("ARI with typological communities")
    b.set_title("b  Cluster agreement (exploratory)", fontsize=9)
    b.legend(fontsize=6.5)
    fig.tight_layout()
    return save(fig, "figS5_h1_regression_clusters")


# ----------------------------------------------------------------- Phase 6 --
def fig_h2_balance() -> list[Path]:
    """Fig. 6 — covariate balance before and after matching (Love plot)."""
    import numpy as np

    bal = pd.read_csv(config.RESULTS_DIR / "h2_balance.csv")
    import json
    des = json.loads((config.RESULTS_DIR / "h2_design.json").read_text())
    names = {"log_families": "log families colexifying", "zipf_min": "lower Zipf frequency",
             "same_field": "same semantic field", "zipf_max": "higher Zipf frequency (not matched on)"}
    pops = [("all_pairs", "all pairs"), ("no_input_colexified", "without input-colexified pairs")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.8), sharey=True)
    for ax, (pop, title) in zip(axes, pops):
        b = bal[bal["population"] == pop].set_index("covariate").loc[list(names)]
        y = np.arange(len(names))[::-1]
        ax.axvspan(-config.H2_SMD_THRESHOLD, config.H2_SMD_THRESHOLD, color=C[2], alpha=0.12, lw=0)
        ax.axvline(0, color=config.INK_MUTED, lw=0.8)
        for yi, (_, r) in zip(y, b.iterrows()):
            ax.plot([r["smd_before"], r["smd_after"]], [yi, yi], color=config.GRID, lw=1.5, zorder=1)
        ax.scatter(b["smd_before"], y, s=40, color=C[3], zorder=2, label="before matching")
        ax.scatter(b["smd_after"], y, s=40, color=C[0], zorder=3, label="after matching")
        ax.set_yticks(y, list(names.values()))
        ax.grid(axis="y", visible=False)
        p = des["populations"][pop]
        ax.set_title(f"{title}: {p['n_matched_present']} present vs {p['n_matched_absent']} absent "
                     f"(from {p['n_present']} vs {p['n_absent']})", fontsize=8.5)
        ax.set_xlabel("Standardised mean difference (present − absent)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"H2 · Covariate balance, 1:{des['match_k']} propensity matching, exact on semantic field "
                 f"(shaded band = |SMD| < {config.H2_SMD_THRESHOLD})", x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, "fig6_h2_balance")


def fig_h2_ranks() -> list[Path]:
    """Fig. 7 — neighbour ranks of English-present vs English-absent pairs, matched and unmatched."""
    import numpy as np

    ranks = pd.read_csv(config.RESULTS_DIR / "h2_ranks.csv")
    res = pd.read_csv(config.RESULTS_DIR / "h2_blindspot.csv")
    labels = _labels()
    conds = [c for c in config.MODEL_CONDITIONS if c in set(ranks["condition"])]
    fig, axes = plt.subplots(1, max(len(conds), 1), figsize=(4.6 * max(len(conds), 1) + 0.6, 3.8), squeeze=False)
    rng_ = np.random.default_rng(0)
    for ax, c in zip(axes[0], conds):
        prim_pop = "no_input_colexified" if c in ("xlmr_translated", "labse_translated") else "all_pairs"
        r = ranks[(ranks["condition"] == c) & (ranks["population"] == prim_pop)]
        groups = [("present", "English-present", C[0]), ("absent", "matched English-absent", C[1])]
        for k, (role, lab, col) in enumerate(groups):
            v = np.clip(r.loc[r["role"] == role, "rank"].to_numpy(), 1e-3, 1)
            x = k + rng_.uniform(-0.12, 0.12, len(v))
            ax.scatter(x, v, s=10, color=col, alpha=0.7, lw=0)
            ax.plot([k - 0.22, k + 0.22], [np.median(v)] * 2, color=config.INK, lw=1.6)
        ax.set_yscale("log")
        ax.set_xticks([0, 1], ["English-\npresent", "matched\nEnglish-absent"])
        ax.set_ylim(1e-3, 1.1)
        ax.set_ylabel("Mutual neighbour rank (0 = nearest; log scale)")
        row = res[(res["condition"] == c) & (res["design"] == "matched") & (res["population"] == prim_pop)].iloc[0]
        un = res[(res["condition"] == c) & (res["design"] == "unmatched")].iloc[0]
        ax.set_title(f"{labels[c]}\nmatched δ = {row['cliffs_delta']:.2f} [{row['delta_ci_low']:.2f}, "
                     f"{row['delta_ci_high']:.2f}], p = {row['p_one_sided']:.3f} · unmatched δ = {un['cliffs_delta']:.2f}",
                     fontsize=8)
        ax.set_yticks([1e-3, 1e-2, 1e-1, 1], ["≤ 0.001", "0.01", "0.1", "1"])
        ax.grid(axis="x", visible=False)
    fig.suptitle("H2 · Neighbour ranks of English-present vs matched\nEnglish-absent pairs (bars = medians; "
                 "δ > 0: present pairs closer)", x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return save(fig, "fig7_h2_ranks")


def fig_h2_gap_by_model() -> list[Path]:
    """Fig. 8 — the English blind-spot effect size per model (the group x model interaction)."""
    import json

    import numpy as np

    res = pd.read_csv(config.RESULTS_DIR / "h2_blindspot.csv")
    des = json.loads((config.RESULTS_DIR / "h2_design.json").read_text())
    labels = _labels()
    conds = list(config.MODEL_CONDITIONS)
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    y = np.arange(len(conds))[::-1]
    ax.axvline(0, color=config.INK_MUTED, lw=0.8)
    for yi, c in zip(y, conds):
        sub = res[(res["condition"] == c) & (res.get("design") == "matched") & (res["primary"] == True)]  # noqa: E712
        un = res[(res["condition"] == c) & (res.get("design") == "unmatched")]
        if sub.empty:
            ax.text(0.01, yi, "PENDING USER RUN", va="center", fontsize=7.5, color=config.INK_MUTED)
            continue
        r = sub.iloc[0]
        ax.plot([r["delta_ci_low"], r["delta_ci_high"]], [yi, yi], color=C[0], lw=2, solid_capstyle="round")
        ax.scatter([r["cliffs_delta"]], [yi], s=46, color=C[0], zorder=3, edgecolor="white", linewidth=1)
        if len(un):
            ax.scatter([un.iloc[0]["cliffs_delta"]], [yi - 0.22], s=36, marker="D", facecolor="white",
                       edgecolor=C[3], linewidth=1.4, zorder=3)
        ax.text(r["delta_ci_high"] + 0.03, yi + 0.12, f"δ = {r['cliffs_delta']:.2f}, p = {r['p_one_sided']:.3f}",
                va="center", fontsize=7.5)
    ax.set_yticks(y, [labels[c] for c in conds])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.set_xlabel("Cliff's δ: how much closer English-present pairs are than English-absent pairs")
    handles = [plt.Line2D([], [], marker="o", ls="-", color=C[0], label="matched (pre-specified), 95% BCa CI"),
               plt.Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor=C[3],
                          label="unmatched (confounded, for comparison)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
    ax.set_title("H2 · The English blind spot, model by model", loc="left", fontsize=9, pad=10)
    p = des["populations"]["all_pairs"]
    ax.text(0, -0.2, f"English gold standard: {des['audit_status']}. Matched sets: {p['n_matched_present']} present vs "
            f"{p['n_matched_absent']} absent pairs (multilingual-input rows exclude input-colexified pairs, D-49).",
            transform=ax.transAxes, fontsize=7, color=config.INK_MUTED)
    fig.tight_layout()
    return save(fig, "fig8_h2_gap_by_model")


# ----------------------------------------------------------------- Phase 7 --
def _h3_inputs():
    """H3 fit table, gradient table, proxies with status, best condition."""
    import json

    fit = pd.read_csv(config.RESULTS_DIR / "h3_family_fit.csv")
    grad = pd.read_csv(config.RESULTS_DIR / "h3_typology.csv")
    prox = pd.read_csv(config.RESULTS_DIR / "resource_proxies.csv")
    status = json.loads((config.RESULTS_DIR / "resource_proxies_status.json").read_text())
    h1 = pd.read_csv(config.RESULTS_DIR / "h1_alignment.csv")
    done = h1[h1["status"] == "done"].sort_values("rho_raw", ascending=False)
    best = str(done["condition"].iloc[0])
    return fit, grad, prox, status, best


def _proxy_choice(prox: pd.DataFrame, status: dict) -> tuple[str, str]:
    """Primary proxy when available (Wikipedia, D-57), else the Joshi lower bound."""
    if prox["wikipedia_log10_articles"].notna().sum() >= 4:
        return "wikipedia_log10_articles", "log10(1 + articles), largest Wikipedia edition"
    return "joshi_class", "Joshi et al. (2020) class, lower bound from the paper"


def _family_order(fit: pd.DataFrame, prox: pd.DataFrame, status: dict) -> pd.DataFrame:
    col, _ = _proxy_choice(prox, status)
    fam = fit.drop_duplicates("family_gc")[["family_gc", "family_name", "n_languages"]].merge(
        prox[["family_gc", col]], on="family_gc", how="left")
    return fam.sort_values([col, "n_languages"], ascending=[False, False], na_position="last").reset_index(drop=True)


def _proxy_tag(v, col: str) -> str:
    if pd.isna(v):
        return "–"
    return f"≥{int(v)}" if col == "joshi_class" else f"{v:.1f}"


def fig_h3_families() -> list[Path]:
    """Fig. 9 — H3 per-family fit: family x model heatmap (rarefied AUC) and full vs rarefied for the best model."""
    import numpy as np
    from matplotlib.colors import Normalize

    fit, grad, prox, status, best = _h3_inputs()
    labels = _labels()
    col, col_label = _proxy_choice(prox, status)
    fam = _family_order(fit, prox, status)
    conds = list(config.MODEL_CONDITIONS)
    n = len(fam)
    mat = np.full((n, len(conds)), np.nan)
    for i, f in enumerate(fam["family_gc"]):
        for j, c in enumerate(conds):
            v = fit.loc[(fit["family_gc"] == f) & (fit["condition"] == c), "auc_rarefied_mean"]
            if len(v):
                mat[i, j] = v.iloc[0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 0.27 * n + 2.0), gridspec_kw={"width_ratios": [1.15, 1]})
    norm = Normalize(0.5, max(0.85, np.nanmax(mat) if np.isfinite(mat).any() else 0.85))
    a.imshow(np.ma.masked_invalid(mat), cmap=config.SEQUENTIAL_CMAP, norm=norm, aspect="auto")
    for j, c in enumerate(conds):
        if np.isnan(mat[:, j]).all():
            a.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, n, facecolor="#f3f3f3", edgecolor="white",
                                      hatch="////", lw=0))
            a.text(j, (n - 1) / 2, "PENDING USER RUN", rotation=90, ha="center", va="center", fontsize=7,
                   color=config.INK_MUTED)
            continue
        for i in range(n):
            v = mat[i, j]
            if np.isfinite(v):
                a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                       color="white" if norm(v) < 0.6 else config.INK)
    a.set_xticks(range(len(conds)), [labels[c] for c in conds], rotation=35, ha="right", fontsize=7)
    a.set_yticks(range(n), [f"{r.family_name} ({r.n_languages}) · {_proxy_tag(getattr(r, col), col)}"
                            for r in fam.itertuples()], fontsize=7.5)
    a.grid(False)
    a.set_title("a  Rarefied AUC (20 languages × 200), family × model", fontsize=9)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=config.SEQUENTIAL_CMAP)
    cb = fig.colorbar(sm, ax=a, fraction=0.04, pad=0.02)
    cb.set_label("AUC (0.5 = chance)", fontsize=7)
    cb.outline.set_linewidth(0.4)

    sub = fit[fit["condition"] == best].set_index("family_gc").loc[fam["family_gc"]]
    y = np.arange(n)
    b.axvline(0.5, color=config.INK_MUTED, lw=0.8)
    b.hlines(y - 0.15, sub["auc_ci_low"], sub["auc_ci_high"], color=C[0], lw=2)
    b.scatter(sub["auc"], y - 0.15, s=30, color=C[0], edgecolor="white", lw=0.8, zorder=3,
              label="all languages, 95% concept-bootstrap CI")
    b.hlines(y + 0.15, sub["auc_rarefied_low"], sub["auc_rarefied_high"], color=C[1], lw=1.4)
    b.scatter(sub["auc_rarefied_mean"], y + 0.15, s=28, facecolor="white", edgecolor=C[1], lw=1.4, zorder=3,
              label="rarefied to 20 languages, 2.5–97.5% of 200 draws")
    b.set_ylim(n - 0.5, -0.5)
    b.set_yticks(y, [""] * n)
    b.grid(axis="y", visible=False)
    b.set_xlabel("AUC: P(colexified pair more similar than non-colexified pair)")
    b.set_title(f"b  Full vs rarefied, {labels[best]}", fontsize=9)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), fontsize=7, ncol=1)
    fig.text(0.01, 0.005, f"Families qualifying under D-09 (≥ {config.FAMILY_MIN_LANGUAGES} languages); "
             f"rows ordered by resource proxy ({col_label}; value after '·'). "
             f"Number of CLICS⁴ languages in brackets.", fontsize=6.8, color=config.INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return save(fig, "fig9_h3_families")


def fig_h3_density() -> list[Path]:
    """Fig. S6 — sampling density vs fit, before and after rarefaction (the D-25 confound check)."""
    import numpy as np

    fit, grad, prox, status, best = _h3_inputs()
    labels = _labels()
    sub = fit[fit["condition"] == best].sort_values("n_languages")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    x = sub["n_languages"].to_numpy()
    for xi, a0, a1 in zip(x, sub["auc"], sub["auc_rarefied_mean"]):
        ax.plot([xi, xi], [a0, a1], color=config.GRID, lw=1.2, zorder=1)
    ax.scatter(x, sub["auc"], s=32, color=C[0], edgecolor="white", lw=0.8, zorder=3, label="all languages")
    ax.scatter(x, sub["auc_rarefied_mean"], s=30, facecolor="white", edgecolor=C[1], lw=1.4, zorder=3,
               label="rarefied to 20 languages")
    for r in sub.itertuples():
        ax.annotate(r.family_name, (r.n_languages, r.auc), xytext=(4, 2), textcoords="offset points",
                    fontsize=6, color=config.INK_MUTED)
    ax.set_xscale("log")
    ax.set_xticks([20, 30, 50, 100, 200, 400], ["20", "30", "50", "100", "200", "400"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Languages in CLICS⁴ (log scale)")
    ax.set_ylabel("AUC")
    g = grad[(grad["condition"] == best) & (grad["proxy"] == "log10_n_languages_clics")].set_index("auc_kind")
    txt = "; ".join(f"{k}: ρ = {g.loc[k, 'rho']:.2f}, p = {g.loc[k, 'p_two_sided']:.3f}"
                    for k in ("full", "rarefied") if k in g.index)
    ax.set_title(f"Sampling density and family fit, {labels[best]}", fontsize=9)
    ax.text(0, -0.2, f"Spearman across {len(sub)} families (two-sided permutation p) — {txt}",
            transform=ax.transAxes, fontsize=6.8, color=config.INK_MUTED)
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 0.0), fontsize=7)
    fig.tight_layout()
    return save(fig, "figS6_h3_density")


def fig_synthesis() -> list[Path]:
    """Fig. 11 — synthesis: alignment (H1), English blind spot (H2) and the family gradient (H3)."""
    import numpy as np

    fit, grad, prox, status, best = _h3_inputs()
    labels = _labels()
    conds = list(config.MODEL_CONDITIONS)
    h1 = pd.read_csv(config.RESULTS_DIR / "h1_alignment.csv").set_index("condition")
    h2 = pd.read_csv(config.RESULTS_DIR / "h2_blindspot.csv")
    h2 = h2[(h2["design"] == "matched") & (h2["primary"] == True)].set_index("condition")  # noqa: E712
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(10.4, 3.7), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    y = np.arange(len(conds))[::-1]
    for ax, title in ((a, "a  H1 · alignment (RSA ρ)"), (b, "b  H2 · English blind spot (Cliff's δ)")):
        ax.axvline(0, color=config.INK_MUTED, lw=0.8)
        ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=9)
    for yi, cnd in zip(y, conds):
        if cnd in h1.index and h1.loc[cnd, "status"] == "done":
            r = h1.loc[cnd]
            a.plot([r["rho_ci_low"], r["rho_ci_high"]], [yi, yi], color=C[0], lw=2)
            a.scatter([r["rho_raw"]], [yi], s=36, color=C[0], edgecolor="white", zorder=3)
        else:
            a.text(0.01, yi, "pending", va="center", fontsize=7, color=config.INK_MUTED)
        if cnd in h2.index:
            r = h2.loc[cnd]
            b.plot([r["delta_ci_low"], r["delta_ci_high"]], [yi, yi], color=C[2], lw=2)
            b.scatter([r["cliffs_delta"]], [yi], s=36, color=C[2], edgecolor="white", zorder=3)
        else:
            b.text(0.02, yi, "pending", va="center", fontsize=7, color=config.INK_MUTED)
    a.set_yticks(y, [labels[c] for c in conds], fontsize=7.5)
    b.set_yticks(y, [""] * len(conds))
    a.set_xlim(-0.05, 0.5)
    b.set_xlim(-0.4, 0.8)
    a.set_xlabel("Spearman ρ, 95% CI")
    b.set_xlabel("δ > 0: English-present pairs closer")

    col, col_label = _proxy_choice(prox, status)
    d = fit.merge(prox[["family_gc", col]], on="family_gc", how="left")
    d = d[d["condition"] == best].dropna(subset=[col])
    d = d.sort_values([col, "auc_rarefied_mean"]).reset_index(drop=True)
    xs = d[col].to_numpy(float).copy()
    if col == "joshi_class":                     # spread families that share a class
        for v in np.unique(xs):
            k = np.flatnonzero(d[col].to_numpy(float) == v)
            xs[k] = v + (np.arange(k.size) - (k.size - 1) / 2) * 0.16
    c.vlines(xs, d["auc_rarefied_low"], d["auc_rarefied_high"], color=C[1], lw=1.2, alpha=0.7)
    c.scatter(xs, d["auc_rarefied_mean"], s=30, facecolor="white", edgecolor=C[1], lw=1.4, zorder=3)
    ly = d["auc_rarefied_mean"].to_numpy(float).copy()
    order = np.argsort(ly)
    side = np.where(xs < d[col].to_numpy(float) - 1e-9, -1, 1)   # left-shifted points label leftwards
    for _ in range(50):                          # push apart same-side labels closer than 0.016 in AUC
        for a0, a1 in zip(order[:-1], order[1:]):
            if side[a0] == side[a1] and abs(xs[a0] - xs[a1]) < 1.2 and ly[a1] - ly[a0] < 0.016:
                ly[a1] = ly[a0] + 0.016
    for xi, yi, yl, sd, r in zip(xs, d["auc_rarefied_mean"], ly, side, d.itertuples()):
        c.annotate(r.family_name, (xi, yi), xytext=(xi + 0.1 * sd, yl), textcoords="data",
                   fontsize=6, color=config.INK_MUTED, va="center", ha="left" if sd > 0 else "right",
                   arrowprops=dict(arrowstyle="-", color=config.GRID, lw=0.5) if abs(yl - yi) > 0.004 else None)
    if col == "joshi_class":
        c.set_xticks(range(6))
        c.set_xlim(-1.3, 6.3)
    c.axhline(0.5, color=config.INK_MUTED, lw=0.8)
    g = grad[(grad["condition"] == best) & (grad["proxy"] == col) & (grad["auc_kind"] == "rarefied")]
    note = ""
    if len(g) and pd.notna(g.iloc[0].get("rho")):
        note = f"ρ = {g.iloc[0]['rho']:.2f}, one-sided p = {g.iloc[0]['p_one_sided']:.3f} ({int(g.iloc[0]['n_families'])} families)"
    c.set_xlabel(col_label)
    c.set_ylabel("Rarefied AUC")
    c.set_title(f"c  H3 · family gradient, {labels[best]}", fontsize=9)
    c.text(0, -0.24, note + ("  ·  PROVISIONAL proxy" if col == "joshi_class" else ""),
           transform=c.transAxes, fontsize=6.8, color=config.INK_MUTED)
    fig.tight_layout()
    return save(fig, "fig11_synthesis")


# ------------------------------------------------------------------ registry --
@dataclass
class FigureSpec:
    name: str
    builder: Callable[[], list[Path]]
    requires: tuple[str, ...]   # paths relative to repo root


REGISTRY: list[FigureSpec] = [
    FigureSpec("figS1_concept_coverage", fig_concept_coverage,
               ("results/concept_coverage_curve.csv", "results/concept_threshold.json")),
    FigureSpec("figS2_network_decisions", fig_network_data_decisions,
               ("data/processed/pairs.parquet", "data/processed/pairs__all__segments.parquet",
                "results/typological_matrix_ties.csv")),
    FigureSpec("fig1_network", fig_network,
               ("data/processed/network.graphml", "results/communities.csv", "results/community_summary.csv")),
    FigureSpec("fig2_typological_map", fig_typological_map,
               ("data/processed/typological_2d.parquet", "results/dr_choice.json", "results/communities.csv")),
    FigureSpec("figS3_dr_comparison", fig_dr_comparison,
               ("data/processed/typological_2d.parquet", "results/dr_quality.csv", "results/communities.csv")),
    FigureSpec("figS4_community_stability", fig_community_stability,
               ("data/processed/louvain_runs.npy", "results/community_stability.json",
                "data/processed/matrices/coassignment_louvain.npz")),
    FigureSpec("fig3_h1_alignment", fig_h1_alignment,
               ("results/h1_alignment.csv", "results/analysis_concept_set.json")),
    FigureSpec("fig4_h1_matrix_correlations", fig_h1_heatmap, ("results/h1_matrix_correlations.csv",)),
    FigureSpec("fig5_h1_procrustes", fig_h1_procrustes,
               ("data/processed/h1_procrustes_coords.parquet", "results/h1_alignment.csv")),
    FigureSpec("figS5_h1_regression_clusters", fig_h1_regression_clusters,
               ("results/h1_regression.csv", "results/h1_cluster_agreement.csv")),
    FigureSpec("fig6_h2_balance", fig_h2_balance, ("results/h2_balance.csv", "results/h2_design.json")),
    FigureSpec("fig7_h2_ranks", fig_h2_ranks, ("results/h2_ranks.csv", "results/h2_blindspot.csv")),
    FigureSpec("fig8_h2_gap_by_model", fig_h2_gap_by_model, ("results/h2_blindspot.csv", "results/h2_design.json")),
    FigureSpec("fig9_h3_families", fig_h3_families,
               ("results/h3_family_fit.csv", "results/h3_typology.csv", "results/resource_proxies.csv")),
    FigureSpec("figS6_h3_density", fig_h3_density,
               ("results/h3_family_fit.csv", "results/h3_typology.csv", "results/resource_proxies.csv")),
    FigureSpec("fig11_synthesis", fig_synthesis,
               ("results/h1_alignment.csv", "results/h2_blindspot.csv", "results/h3_family_fit.csv",
                "results/h3_typology.csv", "results/resource_proxies.csv")),
]


def _exists(p: Path) -> bool:
    """A required input exists (or its sandbox-only Parquet fallback, D-33)."""
    return p.exists() or (p.suffix == ".parquet" and p.with_name(p.name + ".pkl").exists())


def main() -> None:
    log_run_header(log)
    apply_style()
    built, pending = [], []
    for spec in REGISTRY:
        missing = [r for r in spec.requires if not _exists(config.ROOT / r)]
        if missing:
            pending.append((spec.name, missing))
            continue
        built += spec.builder()
        log.info("built %s", spec.name)
    for name, missing in pending:
        log.warning("PENDING %s: missing %s", name, ", ".join(missing))
    log.info("%d files written, %d figures pending", len(built), len(pending))


if __name__ == "__main__":
    main()
