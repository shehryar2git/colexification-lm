"""Phase 7 · World map of language-level agreement with the best model (exploratory).

Each CLICS⁴ language is drawn at its Glottolog coordinates and coloured by its
per-language AUC (results/h3_language_agreement.csv): the probability that a pair
of concepts the language colexifies is more similar in the model than a pair it
keeps apart (both concepts attested in that language). Languages with fewer than
config.H3_MIN_LANGUAGE_COLEX colexified analysis pairs are shown as small grey
dots: their AUC is too noisy to colour (DECISIONS.md D-58).

Projection: Equal Earth (Šavrič, Patterson & Jenny 2019), Pacific-centred at
150°E so that Austronesian and the Pacific are not split at the map edge. Land
outlines: Natural Earth 1:110m (public domain), fetched at a pinned commit.

Writes figures/fig10_world_map.{pdf,png} and figures/world_map.html (self-contained
SVG, no external libraries; same token system as interactive_network.html).
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

import config  # noqa: E402
from src.analysis.common import CONDITION_LABELS  # noqa: E402
from src.utils.logging_utils import get_logger, log_run_header  # noqa: E402
from src.viz.figures import apply_style, save  # noqa: E402

log = get_logger(__name__, "world_map")
LON0 = 150.0
A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796
M = np.sqrt(3) / 2
AUC_RANGE = (0.4, 1.0)           # colour scale; 0.5 = chance is marked on the colour bar
MIN_LAT_LAND = -58.0             # Antarctica is dropped (no languages)


# --------------------------------------------------------------- projection --
def equal_earth(lon, lat, lon0: float = LON0, unwrap: bool = False):
    """Equal Earth projection; returns (x, y) in units of the earth radius.
    ``lon`` is first expressed relative to ``lon0`` in [-180, 180) unless
    ``unwrap`` is True (then it is taken as already relative and may exceed ±180)."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    d = lon if unwrap else ((lon - lon0 + 180.0) % 360.0) - 180.0
    lam, phi = np.radians(d), np.radians(lat)
    th = np.arcsin(M * np.sin(phi))
    th2, th6 = th ** 2, th ** 6
    x = 2 * np.sqrt(3) * lam * np.cos(th) / (3 * (9 * A4 * th6 * th2 + 7 * A3 * th6 + 3 * A2 * th2 + A1))
    y = th * (A4 * th6 * th2 + A3 * th6 + A2 * th2 + A1)
    return x, y


def outline(n: int = 181) -> np.ndarray:
    lat = np.linspace(-90, 90, n)
    xr, yr = equal_earth(np.full(n, 180.0), lat, unwrap=True)
    xl, yl = equal_earth(np.full(n, -180.0), lat[::-1], unwrap=True)
    return np.column_stack([np.r_[xr, xl], np.r_[yr, yl]])


def land_rings(path=None) -> list[np.ndarray]:
    """Projected exterior and interior rings of Natural Earth land polygons.
    Rings crossing the map seam (lon0 ± 180) are unwrapped and drawn shifted by
    ±360° as well; the caller clips everything to the outline."""
    gj = json.loads((path or config.NATURAL_EARTH_LAND).read_text(encoding="utf-8"))
    rings = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            for ring in poly:
                r = np.asarray(ring, dtype=float)
                if r[:, 1].max() < MIN_LAT_LAND:
                    continue
                d = ((r[:, 0] - LON0 + 180.0) % 360.0) - 180.0
                d = np.degrees(np.unwrap(np.radians(d)))
                for shift in (0.0, -360.0, 360.0):
                    dd = d + shift
                    if dd.max() < -180 or dd.min() > 180:
                        continue
                    x, y = equal_earth(dd, r[:, 1], unwrap=True)
                    rings.append(np.column_stack([x, y]))
    return rings


# ------------------------------------------------------------------- data ---
def best_condition() -> tuple[str, float]:
    h1 = pd.read_csv(config.RESULTS_DIR / "h1_alignment.csv")
    h1 = h1[h1["status"] == "done"].sort_values("rho_raw", ascending=False)
    return str(h1["condition"].iloc[0]), float(h1["rho_raw"].iloc[0])


def load_languages() -> pd.DataFrame:
    lang = pd.read_csv(config.RESULTS_DIR / "h3_language_agreement.csv")
    lang = lang.dropna(subset=["latitude", "longitude"])
    x, y = equal_earth(lang["longitude"], lang["latitude"])
    return lang.assign(x=x, y=y)


# ----------------------------------------------------------------- static ---
def fig_world_map() -> list:
    apply_style()
    cond, rho = best_condition()
    lang = load_languages()
    col = f"auc_{cond}"
    mapped = lang[lang["mapped"] & lang[col].notna()].sort_values(col)
    other = lang[~lang.index.isin(mapped.index)]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ol = outline()
    clip = PathPatch(MPath(ol), transform=ax.transData)
    ax.add_patch(PathPatch(MPath(ol), facecolor="#f2f4f6", edgecolor=config.INK_MUTED, lw=0.5, zorder=0))
    for r in land_rings():
        p = PathPatch(MPath(r), facecolor="#dcdfe3", edgecolor="#c4c8cd", lw=0.25, zorder=1)
        ax.add_patch(p)
        p.set_clip_path(clip)
    ax.scatter(other["x"], other["y"], s=2.5, c="#9a9fa6", lw=0, zorder=2, alpha=0.7)
    sc = ax.scatter(mapped["x"], mapped["y"], s=8, c=mapped[col], cmap=config.SEQUENTIAL_CMAP,
                    vmin=AUC_RANGE[0], vmax=AUC_RANGE[1], lw=0.25, edgecolor="white", zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(ol[:, 0].min() * 1.01, ol[:, 0].max() * 1.01)
    ax.set_ylim(ol[:, 1].min() * 1.02, ol[:, 1].max() * 1.02)
    ax.axis("off")
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, aspect=40,
                      extend="both")
    cb.set_label(f"Per-language AUC, {CONDITION_LABELS[cond]} "
                 "(P[colexified pair more similar than non-colexified pair])")
    cb.ax.axvline(0.5, color=config.INK, lw=1.0)
    cb.outline.set_linewidth(0.4)
    ax.set_title(f"Language-level agreement with the best available model ({CONDITION_LABELS[cond]}, "
                 f"H1 ρ = {rho:.2f})", loc="left")
    fig.text(0.02, -0.02, f"{len(mapped):,} languages with ≥ {config.H3_MIN_LANGUAGE_COLEX} colexified analysis "
            f"pairs coloured; {len(other):,} grey. Equal Earth projection, centred at {LON0:.0f}°E. "
            "Exploratory.", fontsize=7, color=config.INK_MUTED, va="top")
    return save(fig, "fig10_world_map")


# ------------------------------------------------------------------- html ---
def _svg_path(rings: list[np.ndarray], scale: float) -> str:
    parts = []
    for r in rings:
        pts = np.round(r * scale, 1)
        pts[:, 1] = -pts[:, 1]
        parts.append("M" + "L".join(f"{x:g},{y:g}" for x, y in pts) + "Z")
    return "".join(parts)


def build_payload(scale: float = 100.0) -> dict:
    lang = load_languages()
    conds = [c[len("auc_"):] for c in lang.columns if c.startswith("auc_")]
    best, rho = best_condition()
    cmap = plt.get_cmap(config.SEQUENTIAL_CMAP)
    lut = [matplotlib.colors.to_hex(cmap(v)) for v in np.linspace(0, 1, 11)]
    ol = outline() * scale
    ol[:, 1] = -ol[:, 1]
    langs = []
    for r in lang.itertuples():
        langs.append({"n": r.language_name, "g": r.language_gc, "f": r.family_name if isinstance(r.family_name, str)
                      else "", "x": round(float(r.x) * scale, 2), "y": round(float(-r.y) * scale, 2),
                      "k": int(r.n_colex_pairs), "c": int(r.n_concepts),
                      "a": {c: (None if pd.isna(getattr(r, f"auc_{c}")) else round(float(getattr(r, f"auc_{c}")), 3))
                            for c in conds}})
    fams = lang.groupby("family_name")["language_gc"].size().sort_values(ascending=False)
    return {"land": _svg_path(land_rings(), scale),
            "outline": "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in ol) + "Z",
            "viewBox": [round(float(ol[:, 0].min()) - 4, 1), round(float(ol[:, 1].min()) - 4, 1),
                        round(float(np.ptp(ol[:, 0])) + 8, 1), round(float(np.ptp(ol[:, 1])) + 8, 1)],
            "langs": langs, "conditions": [{"id": c, "label": CONDITION_LABELS.get(c, c)} for c in conds],
            "best": best, "best_rho": round(rho, 3), "lut": lut, "range": list(AUC_RANGE),
            "min_colex": config.H3_MIN_LANGUAGE_COLEX,
            "families": [f for f in fams.index if fams[f] >= 10]}


TEMPLATE = """<title>Colexification Agreement Atlas</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,600&display=swap">
<style>
  /* Layout: compact header and controls over one full-width map; legend and status beneath. */
  :root {
    --bg: #f7f7f4; --panel: #ffffff; --ink: #1d1f22; --muted: #5f6368; --line: #dfe0dc; --accent: #0072B2;
    --ocean: #eef1f3; --land: #d9dcd8; --coast: #c3c7c2; --nodata: #a3a7ab;
    --display: "Source Serif 4", Georgia, "Times New Roman", serif;
    --body: "Source Sans 3", system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --bg: #15171a; --panel: #1d2024; --ink: #e9e9e6; --muted: #a2a5aa; --line: #32363b; --accent: #56B4E9;
    --ocean: #1a1d21; --land: #33373c; --coast: #444950; --nodata: #6c7177; color-scheme: dark; } }
  :root[data-theme="dark"] {
    --bg: #15171a; --panel: #1d2024; --ink: #e9e9e6; --muted: #a2a5aa; --line: #32363b; --accent: #56B4E9;
    --ocean: #1a1d21; --land: #33373c; --coast: #444950; --nodata: #6c7177; color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.45 var(--body); background: var(--bg); color: var(--ink);
         padding-left: max(16px, env(safe-area-inset-left, 0px));
         padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 0 16px; }
  header { padding-block: 14px 4px; }
  h1 { font: 600 1.25rem/1.25 var(--display); margin: 0 0 4px; text-wrap: balance; }
  .sub { color: var(--muted); font-size: 0.85rem; margin: 0; max-width: 72ch; }
  .controls { display: flex; flex-wrap: wrap; gap: 10px 22px; align-items: center; padding-block: 10px; }
  label { font-size: 0.82rem; color: var(--muted); display: inline-flex; gap: 8px; align-items: center; }
  select { font: inherit; font-size: 0.88rem; padding: 4px 8px; border: 1px solid var(--line); border-radius: 6px;
           background: var(--panel); color: var(--ink); max-width: 60vw; }
  input[type=checkbox] { accent-color: var(--accent); }
  select:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .mapwrap { position: relative; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
  svg { display: block; width: 100%; height: auto; }
  .ocean { fill: var(--ocean); stroke: var(--line); stroke-width: 0.6; }
  .land { fill: var(--land); stroke: var(--coast); stroke-width: 0.3; }
  circle.nd { fill: var(--nodata); opacity: 0.65; }
  circle.on { stroke: var(--panel); stroke-width: 0.35; }
  circle.hl { stroke: var(--ink); stroke-width: 1.1; }
  #tip { position: absolute; pointer-events: none; background: var(--panel); color: var(--ink);
         border: 1px solid var(--line); border-radius: 6px; padding: 6px 9px; font-size: 0.8rem;
         line-height: 1.35; max-width: min(300px, 80vw); box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
  .tipname { font-weight: 600; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 22px; align-items: center; padding-block: 10px 2px;
            font-size: 0.78rem; color: var(--muted); }
  .ramp { display: inline-grid; grid-template-columns: auto auto auto; gap: 2px 6px; align-items: center; }
  .bar { width: min(240px, 55vw); height: 10px; border-radius: 2px; position: relative; }
  .tick { position: absolute; top: -3px; bottom: -3px; width: 1.5px; background: var(--ink); }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--nodata);
         margin-right: 5px; vertical-align: 0; }
  #status { font-size: 0.78rem; color: var(--muted); font-variant-numeric: tabular-nums; padding-block: 4px 0; }
  .note { font-size: 0.76rem; color: var(--muted); max-width: 80ch; margin: 8px 0 0; }
</style>
<header>
  <h1>Where does a language model agree with each language's colexifications?</h1>
  <p class="sub">Each dot is a language in CLICS⁴ at its Glottolog location. Colour shows how well the model's
  similarities pick out the concept pairs that language expresses with one word: 0.5 is chance, 1.0 means every
  colexified pair is more similar than every pair the language keeps apart. Exploratory analysis.</p>
</header>
<div class="controls">
  <label for="cond">Model <select id="cond"></select></label>
  <label for="fam">Family <select id="fam"><option value="">All families</option></select></label>
  <label for="few"><input id="few" type="checkbox" checked> Show languages with too few colexifications</label>
</div>
<div class="mapwrap" id="wrap">
  <svg id="map" role="img" aria-label="World map of per-language agreement scores">
    <path class="ocean" id="ocean"></path><path class="land" id="land"></path><g id="pts"></g>
  </svg>
  <div id="tip" hidden></div>
</div>
<div class="legend">
  <span class="ramp"><span id="lo"></span><span class="bar" id="bar"><span class="tick" id="chance"></span></span>
    <span id="hi"></span></span>
  <span>Tick = chance (0.5)</span>
  <span><span class="dot"></span>fewer than __MIN__ colexified pairs (not scored)</span>
</div>
<div id="status"></div>
<p class="note">AUC per language over the analysis concept set (pairs whose two concepts the language attests).
Equal Earth projection centred at 150°E; land outlines from Natural Earth. Source: results/h3_language_agreement.csv.</p>
<script>
const D = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("map");
svg.setAttribute("viewBox", D.viewBox.join(" "));
document.getElementById("ocean").setAttribute("d", D.outline);
document.getElementById("land").setAttribute("d", D.land);
const clip = document.createElementNS(NS, "clipPath"); clip.id = "clip";
const cp = document.createElementNS(NS, "path"); cp.setAttribute("d", D.outline); clip.appendChild(cp);
const defs = document.createElementNS(NS, "defs"); defs.appendChild(clip); svg.prepend(defs);
document.getElementById("land").setAttribute("clip-path", "url(#clip)");
const [lo, hi] = D.range;
document.getElementById("lo").textContent = lo.toFixed(1);
document.getElementById("hi").textContent = hi.toFixed(1);
document.getElementById("bar").style.background = `linear-gradient(90deg, ${D.lut.join(",")})`;
document.getElementById("chance").style.left = `${(0.5 - lo) / (hi - lo) * 100}%`;
function colour(v) {
  const t = Math.min(1, Math.max(0, (v - lo) / (hi - lo))) * (D.lut.length - 1);
  const i = Math.min(D.lut.length - 2, Math.floor(t)), f = t - i;
  const p = h => [1, 3, 5].map(k => parseInt(h.slice(k, k + 2), 16));
  const a = p(D.lut[i]), b = p(D.lut[i + 1]);
  return `rgb(${a.map((x, k) => Math.round(x + (b[k] - x) * f)).join(",")})`;
}
const condSel = document.getElementById("cond"), famSel = document.getElementById("fam");
for (const c of D.conditions) {
  const o = document.createElement("option"); o.value = c.id;
  o.textContent = c.label + (c.id === D.best ? " (best H1 fit)" : ""); condSel.appendChild(o);
}
condSel.value = D.best;
for (const f of D.families) { const o = document.createElement("option"); o.value = f; o.textContent = f; famSel.appendChild(o); }
const few = document.getElementById("few"), pts = document.getElementById("pts");
const tip = document.getElementById("tip"), wrap = document.getElementById("wrap"), status = document.getElementById("status");
let current = [];
function draw() {
  const cond = condSel.value, fam = famSel.value;
  pts.textContent = "";
  const sel = D.langs.filter(l => !fam || l.f === fam);
  const scored = sel.filter(l => l.k >= D.min_colex && l.a[cond] != null).sort((a, b) => a.a[cond] - b.a[cond]);
  const rest = sel.filter(l => !(l.k >= D.min_colex && l.a[cond] != null));
  current = [];
  const add = (l, on) => {
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", l.x); c.setAttribute("cy", l.y); c.setAttribute("r", on ? 1.6 : 0.9);
    c.setAttribute("class", on ? "on" : "nd");
    if (on) c.setAttribute("fill", colour(l.a[cond]));
    c.dataset.i = current.length; current.push(l); pts.appendChild(c);
  };
  if (few.checked) rest.forEach(l => add(l, false));
  scored.forEach(l => add(l, true));
  const vals = scored.map(l => l.a[cond]).sort((a, b) => a - b);
  const med = vals.length ? vals[Math.floor(vals.length / 2)] : NaN;
  const above = vals.filter(v => v > 0.5).length;
  status.textContent = `${scored.length} languages scored` + (vals.length ?
    ` · median AUC ${med.toFixed(3)} · ${above} of ${vals.length} above chance` : "") +
    ` · ${rest.length} not scored` + (fam ? ` · ${fam}` : "");
}
function showTip(ev) {
  const t = ev.target; if (t.tagName !== "circle") { tip.hidden = true; return; }
  pts.querySelectorAll(".hl").forEach(e => e.classList.remove("hl")); t.classList.add("hl");
  const l = current[+t.dataset.i], v = l.a[condSel.value];
  tip.textContent = "";
  const n = document.createElement("div"); n.className = "tipname"; n.textContent = l.n; tip.appendChild(n);
  const add = s => { const d = document.createElement("div"); d.textContent = s; tip.appendChild(d); };
  add(`${l.f || "family unknown"} · ${l.g}`);
  add(`${l.c} concepts attested, ${l.k} colexified pairs`);
  add(l.k >= D.min_colex && v != null ? `AUC ${v.toFixed(3)}` : "too few colexified pairs to score");
  tip.hidden = false;
  const r = wrap.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
  tip.style.left = Math.min(x + 12, r.width - tip.offsetWidth - 6) + "px";
  tip.style.top = Math.max(6, y - tip.offsetHeight - 10) + "px";
}
svg.addEventListener("pointermove", showTip);
svg.addEventListener("pointerleave", () => { tip.hidden = true; pts.querySelectorAll(".hl").forEach(e => e.classList.remove("hl")); });
[condSel, famSel, few].forEach(e => e.addEventListener("change", draw));
draw();
</script>
"""


def render(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", data).replace("__MIN__", str(payload["min_colex"]))


def main() -> None:
    log_run_header(log)
    needed = [config.RESULTS_DIR / "h3_language_agreement.csv", config.RESULTS_DIR / "h1_alignment.csv",
              config.NATURAL_EARTH_LAND]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        log.warning("PENDING world map: missing %s", ", ".join(missing))
        return
    fig_world_map()
    out = config.FIGURES_DIR / "world_map.html"
    out.write_text(render(build_payload()), encoding="utf-8")
    log.info("Wrote fig10_world_map and %s (%.0f KB)", out.name, out.stat().st_size / 1024)


if __name__ == "__main__":
    main()
