"""Phase 3 · Interactive colexification network (figures/interactive_network.html).

Hovering an edge lists its family and language counts and up to six example
forms ("Language (Family): form", one per family). Hovering a node shows its
gloss, semantic field, community and degree. Controls: find a concept
(highlights it and its neighbours) and a minimum-family-support slider.

Written directly against vis-network 9.1.9 (loaded from cdnjs) rather than via
pyvis, which only wraps the same library (DECISIONS.md D-45): no extra dependency,
and node positions are precomputed by the same seeded layout as Fig. 1, so the
file is identical on every run (physics is off).
"""
from __future__ import annotations

import json

import networkx as nx
import numpy as np

import config
from src.utils.io import read_parquet
from src.utils.logging_utils import get_logger, log_run_header
from src.viz.figures import GREY, _load_communities, _short_label, network_layout

log = get_logger(__name__, "phase3_interactive_network")
VIS_URL = "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js"
SCALE = 900.0


def build_payload() -> dict:
    g = nx.read_graphml(config.PROCESSED_DIR / "network.graphml")
    comm, summary, colours = _load_communities()
    lab = comm.set_index("concept_id")["community"].to_dict()
    meta = comm.set_index("concept_id")
    slab = summary.set_index("community")["label"].to_dict()
    core = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    pos = network_layout(core, lab)
    pts = np.array(list(pos.values()))
    centre, span = pts.mean(axis=0), np.ptp(pts, axis=0).max()

    iso = sorted(set(g.nodes) - set(core.nodes))
    for k, n in enumerate(iso):            # isolated concepts in a row beneath the network
        pos[n] = centre + np.array([(k - (len(iso) - 1) / 2) * span / 14, -0.62 * span])

    nodes = []
    for n in sorted(g.nodes):
        x, y = (pos[n] - centre) / span * SCALE
        c = int(lab[n])
        nodes.append({
            "id": n, "label": str(meta.at[n, "gloss"]), "x": round(float(x), 2), "y": round(float(-y), 2),
            "color": colours.get(c, GREY), "value": int(g.degree(n)) + 1,
            "title": (f"{meta.at[n, 'gloss']}  [{n}]\nField: {meta.at[n, 'semantic_field']}\n"
                      f"Community: {slab[c]}\nDegree: {g.degree(n)}"
                      + ("\n(no colexification in >= 2 families)" if n in iso else "")),
            "community": c,
        })
    edges_df = read_parquet(config.PROCESSED_DIR / "edges.parquet")
    edges_df = edges_df[edges_df["is_edge"]]
    edges = []
    for r in edges_df.itertuples():
        same = lab[r.concept_a] == lab[r.concept_b] and lab[r.concept_a] in colours
        ex = (r.examples or "").split(" | ")
        edges.append({
            "from": r.concept_a, "to": r.concept_b, "value": int(r.n_families), "nf": int(r.n_families),
            "color": {"color": colours[lab[r.concept_a]] if same else "#a8a8a8", "opacity": 0.55 if same else 0.25},
            "title": (f"{r.gloss_a} ~ {r.gloss_b}\n{r.n_families} families, {r.n_languages} languages "
                      f"(rate {r.rate_families:.2f} of {r.cov_families} families attesting both)\n"
                      "Examples:\n  " + "\n  ".join(ex)),
        })
    legend = [{"label": _short_label(slab[k]), "color": col} for k, col in colours.items()]
    legend.append({"label": "other communities", "color": GREY})
    return {"nodes": nodes, "edges": edges, "legend": legend,
            "n_nodes": len(nodes), "n_edges": len(edges), "min_families": config.EDGE_MIN_FAMILIES,
            "max_families": int(edges_df["n_families"].max())}


TEMPLATE = """<title>Colexification Network Explorer</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,600&display=swap">
<script src="__VIS__"></script>
<style>
  /* Layout: one working canvas under a compact header; controls wrap on phones. */
  :root {
    --bg: #f7f7f4; --panel: #ffffff; --ink: #1d1f22; --muted: #5f6368; --line: #dfe0dc; --accent: #0072B2;
    --halo: rgba(255,255,255,0.85);
    --display: "Source Serif 4", Georgia, "Times New Roman", serif;
    --body: "Source Sans 3", system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --bg: #15171a; --panel: #1d2024; --ink: #e9e9e6; --muted: #a2a5aa; --line: #32363b; --accent: #56B4E9;
    --halo: rgba(29,32,36,0.9); color-scheme: dark; } }
  :root[data-theme="dark"] {
    --bg: #15171a; --panel: #1d2024; --ink: #e9e9e6; --muted: #a2a5aa; --line: #32363b; --accent: #56B4E9;
    --halo: rgba(29,32,36,0.9); color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body { margin: 0; font: 15px/1.45 var(--body); background: var(--bg); color: var(--ink);
         display: flex; flex-direction: column;
         padding-left: max(16px, env(safe-area-inset-left, 0px));
         padding-right: max(16px, env(safe-area-inset-right, 0px)); }
  header { padding-block: 14px 6px; }
  h1 { font: 600 1.25rem/1.25 var(--display); margin: 0 0 4px; text-wrap: balance; }
  .sub { color: var(--muted); font-size: 0.85rem; margin: 0; max-width: 68ch; }
  .controls { display: flex; flex-wrap: wrap; gap: 10px 22px; align-items: center; padding-block: 8px; }
  label { font-size: 0.82rem; color: var(--muted); display: inline-flex; gap: 8px; align-items: center; }
  input[type=search] { font: inherit; font-size: 0.9rem; padding: 5px 10px; border: 1px solid var(--line);
    border-radius: 6px; background: var(--panel); color: var(--ink); width: min(220px, 60vw); }
  input[type=range] { width: min(180px, 45vw); accent-color: var(--accent); }
  input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  #minv { font-variant-numeric: tabular-nums; color: var(--ink); min-width: 2ch; }
  .legend { display: flex; flex-wrap: wrap; gap: 4px 14px; padding-block: 0 8px; font-size: 0.78rem; color: var(--muted); }
  .sw { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; }
  #net { flex: 1 1 auto; min-height: 380px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
  #status { padding-block: 6px 12px; font-size: 0.78rem; color: var(--muted); font-variant-numeric: tabular-nums; }
  div.vis-tooltip { white-space: pre-line; font: 12px/1.4 var(--body); max-width: min(420px, 85vw);
    background: var(--panel) !important; color: var(--ink) !important; border: 1px solid var(--line) !important; }
</style>
<header>
  <h1>Colexification network of __N__ concepts (CLICS⁴)</h1>
  <p class="sub">Each edge joins two concepts expressed by the same word, recorded separately, in at least
  __MIN__ language families. Hover an edge for counts and example forms; hover or click a concept to see its
  neighbours. Positions match Fig. 1 of the report.</p>
</header>
<div class="controls">
  <label for="q">Find concept <input id="q" type="search" list="glosses" placeholder="hand, tomorrow…"></label>
  <datalist id="glosses"></datalist>
  <label for="minf">Minimum families <input id="minf" type="range" min="__MIN__" max="60" value="__MIN__" step="1">
    <span id="minv">__MIN__</span></label>
</div>
<div class="legend" id="legend"></div>
<div id="net" role="img" aria-label="Interactive colexification network"></div>
<div id="status"></div>
<script>
const DATA = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const ink = () => css("--ink") || "#1d1f22";
const halo = () => css("--halo") || "rgba(255,255,255,0.85)";
const nodes = new vis.DataSet(DATA.nodes);
const allEdges = DATA.edges.map((e, i) => ({...e, id: i}));
const edges = new vis.DataSet(allEdges);
const network = new vis.Network(document.getElementById("net"), {nodes, edges}, {
  physics: false,
  interaction: { hover: true, tooltipDelay: 80, hideEdgesOnDrag: true },
  nodes: { shape: "dot", scaling: { min: 4, max: 22 }, borderWidth: 1,
           font: { size: 11, color: ink(), strokeWidth: 3, strokeColor: halo() } },
  edges: { scaling: { min: 0.4, max: 7 }, smooth: false, selectionWidth: 1.5 },
});
function restyleFonts() {
  network.setOptions({ nodes: { font: { color: ink(), strokeColor: halo() } } });
}
if (window.matchMedia) matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", restyleFonts);
new MutationObserver(restyleFonts).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

const legend = document.getElementById("legend");
for (const l of DATA.legend) {
  const s = document.createElement("span");
  const sw = document.createElement("span"); sw.className = "sw"; sw.style.background = l.color;
  s.appendChild(sw); s.appendChild(document.createTextNode(l.label)); legend.appendChild(s);
}
const dl = document.getElementById("glosses"), byGloss = new Map();
for (const n of DATA.nodes) {
  byGloss.set(n.label.toLowerCase(), n.id);
  const o = document.createElement("option"); o.value = n.label; dl.appendChild(o);
}
const status = document.getElementById("status");
let focusNote = "";
function setStatus() {
  status.textContent = `${edges.length} of ${DATA.n_edges} edges shown · ${DATA.n_nodes} concepts${focusNote}`;
}
const minf = document.getElementById("minf"), minv = document.getElementById("minv");
minf.addEventListener("input", () => {
  const t = +minf.value; minv.textContent = t;
  edges.clear(); edges.add(allEdges.filter(e => e.nf >= t)); setStatus();
});
function focusConcept(id) {
  const nb = new Set(network.getConnectedNodes(id)); nb.add(id);
  nodes.update(DATA.nodes.map(n => ({ id: n.id, color: nb.has(n.id) ? n.color : "rgba(150,150,150,0.22)",
    font: { color: nb.has(n.id) ? ink() : "rgba(150,150,150,0.45)" } })));
  network.selectNodes([id]);
  network.focus(id, { scale: 1.4, animation: { duration: 400 } });
  focusNote = ` · ${nb.size - 1} neighbours of “${nodes.get(id).label}”`; setStatus();
}
function resetFocus() {
  nodes.update(DATA.nodes.map(n => ({ id: n.id, color: n.color, font: { color: ink() } })));
  network.unselectAll(); focusNote = ""; setStatus();
}
document.getElementById("q").addEventListener("change", ev => {
  const id = byGloss.get(ev.target.value.trim().toLowerCase());
  id ? focusConcept(id) : resetFocus();
});
network.on("click", p => p.nodes.length ? focusConcept(p.nodes[0]) : resetFocus());
setStatus();
</script>
"""


def render(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return (TEMPLATE.replace("__VIS__", VIS_URL).replace("__DATA__", data)
            .replace("__N__", str(payload["n_nodes"])).replace("__MIN__", str(payload["min_families"])))


def main() -> None:
    log_run_header(log)
    payload = build_payload()
    out = config.FIGURES_DIR / "interactive_network.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(payload), encoding="utf-8")
    log.info("Wrote %s: %d nodes, %d edges (%.0f KB)", out.name, payload["n_nodes"], payload["n_edges"],
             out.stat().st_size / 1024)


if __name__ == "__main__":
    main()
