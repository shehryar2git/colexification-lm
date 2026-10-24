# Do Language Models Think in English?

**Testing multilingual embedding spaces against the world's colexification patterns**

Many languages use one word for meanings English keeps apart: Hindi–Urdu *kal* is both "yesterday" and
"tomorrow"; Russian *рука* is both "hand" and "arm". Concepts colexified across many unrelated
families are thought to be conceptually close for humans. This project asks whether the semantic
spaces of language models reproduce that cross-linguistic map of meaning — or mainly reproduce English.

We build a typological semantic space from CLICS⁴ (3,447 language varieties, 247 families) and compare
it with six model conditions, from English word2vec to multilingual transformers fed non-English input.

| Hypothesis | Question | Confirmatory test |
|---|---|---|
| **H1 Alignment** | Are concepts colexified in more families closer in model space? | One-sided Mantel test on RSA ρ, 10,000 permutations, per model |
| **H2 English blind spot** | Do models capture colexifications English has (*see/understand*) better than matched ones it lacks (*hand/arm*)? | Matched design; Cliff's δ per model; group × model interaction |
| **H3 Uneven typology** | Is the fit higher for well-resourced families? | Per-family RSA under rarefaction vs resource proxy |

> **Status: all 8 phases complete.** The pipeline runs end to end; **of the six model conditions, word2vec is computed here**.
> Headline numbers and the three figures to read first are in [Results summary](#results-summary) below, the full
> paper is `report/report.md` (and `report/report.pdf`), and every methodological choice is in `DECISIONS.md`.
> Anything the build environment could not compute is marked **PENDING USER RUN** and listed with its command in
> `report/appendix/E_pending.md`: five of the six model conditions, the human audit of the English gold standard,
> the Wikipedia resource proxy, and Infomap/UMAP.

**To finalise the English gold standard:** open `audit/english_colexification_manual_check.csv`, read each row's
`suggested_decision` and `notes`, and write `present`, `absent` or `unsure` in `decision` (leave a row empty to accept the
suggestion only provisionally; fill `annotator` with your initials). Then run `python run.py analyse report`.

---

## Reproduce

```bash
git clone <this repo> colexification-lm && cd colexification-lm
make setup                      # conda env from environment.yml (Python 3.11, pinned deps)
conda activate colexification-lm
make lock                       # record the exact environment
make test                       # unit tests on synthetic fixtures (seconds)
make all                        # full pipeline: data → network → embeddings → analyses → figures → report
```

Without conda: `python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`.

**Windows / no `make`:** every target is a thin wrapper around `run.py`, so `python run.py fetch data`,
`python run.py test` and `python run.py all` do the same thing. `git` must be on the PATH for `fetch`.

`make help` lists every target. Approximate cost (CPU): data + network ~10 min; embeddings ~1–2 h
(word2vec download is 1.6 GB; cached after first run); analyses ~30–60 min (10,000-permutation tests).

| Target | Phase | Produces |
|---|---|---|
| `make fetch` | 1 | Pinned CLICS⁴ v1.0, Glottolog v5.3, Concepticon v3.4.0, WordNet 3.0 → `data/raw/` (sparse, ~15 s) |
| `make data` | 1 | `forms/concepts/languages.parquet`, frozen `concept_order.json`, English-colex candidates (~15 s) |
| `make network` | 2–3 | Family-weighted network, 11 typological matrices, 16 family matrices, audit sheet, communities, 2-D maps (~2 min) |
| `make embeddings` | 4 | Six model similarity matrices in `data/processed/matrices/` |
| `make analyse` | 5–8 | `results/h1_alignment.csv` (+ regression, clusters, matrix correlations; ~5 min per model), `h2_blindspot.csv`, `h3_typology.csv`, `robustness.csv` (~20 min) |
| `make figures` | 1–7 | `figures/*.pdf|png` (300 dpi); `figures/interactive_network.html`; `figures/world_map.html` |
| `make report` | 8 | `report/report.md` regenerated from `results/`, then `report/report.pdf` |

## Running the language models (Phase 4)

The build environment could not download Hugging Face models, so five of the six model conditions must be run on your
machine. Everything else (data, network, word2vec) is reproducible without them.

```bash
pip install -r requirements.txt          # torch, transformers, sentence-transformers, gensim
python run.py embeddings                 # BERT, XLM-R, LaBSE, multilingual input; cached after the first run
```

* Downloads: word2vec 1.7 GB (from GitHub, checksum-verified), BERT-base ≈ 0.4 GB, XLM-R base ≈ 1.1 GB, LaBSE ≈ 1.8 GB.
* Time on a laptop CPU (estimate, not measured here): roughly 20–40 minutes for all models (427 concepts × 10 templates); a GPU is used automatically.
* Memory: < 4 GB. word2vec is streamed, so it never loads all 3 million vectors.
* `results/model_manifest.json` records the exact model revisions that were used.
* `results/embedding_coverage.csv` lists any condition that is still pending, with the command that produces it.

## H3 resource proxies (Phase 7)

`python run.py analyse` queries Wikidata and the Wikimedia API once (≈ 350 small requests, a few minutes) and caches the
answers in `data/raw/wikipedia/`. For the full Joshi et al. (2020) taxonomy, place the authors' `lang2tax.txt` at
`data/raw/joshi/lang2tax.txt`; without it the pipeline uses only the 37 languages whose class the paper states
(lower bounds, labelled PROVISIONAL). `results/resource_proxies_status.json` says which proxy was used.

Optional dashboard: `pip install streamlit==1.39.0 && streamlit run src/viz/dashboard.py`.

## Repository layout

```
config.py          every path, seed, model name, threshold — the only place values are set
DECISIONS.md       every methodological choice, its justification and citation (pre-registered 2026-10-23)
src/data/          CLICS loading, Glottolog metadata, concept selection, English colexification flags
src/network/       colexification network, typological matrices, communities
src/embeddings/    word2vec, BERT, XLM-R, LaBSE, multilingual input, similarity matrices
src/analysis/      H1, H2, H3, robustness
src/viz/           figures, interactive network, optional Streamlit dashboard
src/utils/         seeds, logging, canonical concept order, matrix + results I/O
notebooks/         thin narrative notebooks 01–08 (logic lives in src/)
results/           every statistic as CSV/JSON, each with a .provenance.json sidecar
figures/           figures used in the report
audit/             human-verifiable audit of English colexifications (H2 gold standard)
report/            report.md, references.bib, compiled report.pdf
tests/             pytest suite on synthetic fixtures
```

### Two guarantees enforced in code

1. **One concept order.** `data/processed/concept_order.json` is frozen in Phase 1 with a SHA-256. Every
   matrix stores that hash and is refused on load if it differs (`src/utils/matrices.py`).
2. **Every number is traceable.** Results are written only through `src/utils/io.py`, which adds a
   provenance sidecar (producing script, seed, git commit, package versions). The report reads from `results/`.

## Data and licences

| Source | Version | Licence | Citation |
|---|---|---|---|
| CLICS⁴ | v1.0 (`1ddece6`) | CC-BY-4.0 | Tjuka, Forkel, Rzymski & List (2026), doi:10.5281/zenodo.16900179 |
| Glottolog CLDF | v5.3 (`072ca0d`) | CC-BY-4.0 | Hammarström, Forkel, Haspelmath & Bank |
| Concepticon | v3.4.0 (`918bc44`) | CC-BY-4.0 | List et al. |
| WordNet 3.0 | nltk_data repo (`550b662`), read directly | WordNet licence | Miller (1995) |
| word2vec Google News | gensim-data | Apache-2.0 | Mikolov et al. (2013) |
| Natural Earth land 1:110m | `ca96624` | public domain | naturalearthdata.com |
| BERT, XLM-R, LaBSE | Hugging Face (commit hashes in `results/model_manifest.json`) | Apache-2.0 | see report |

Code: MIT licence (`LICENSE`).

## Results summary

<!-- RESULTS:START -->
*Generated by `python -m src.report.build_report` on 2026-10-24 from `results/`; do not edit by hand.*

**Conditions computed: 1 of 6** (word2vec (EN, static)).
Pending: BERT-base (EN), XLM-R base (multi), LaBSE (multi), XLM-R (multi input), LaBSE (multi input). English gold standard: **PROVISIONAL (Claude suggestions)**.

| | Headline | Where |
|---|---|---|
| **H1 · alignment** | ρ = 0.129 [0.099, 0.160], p < .001 for word2vec (EN, static); partial ρ = 0.084 after controlling English frequency and semantic field | `results/h1_alignment.csv` |
| **H2 · English blind spot** | matched δ = 0.17 [-0.05, 0.37], p = .065 (35 vs 103 pairs); unmatched δ = 0.59 — most of the apparent blind spot is confounding | `results/h2_blindspot.csv` |
| **H3 · typological unevenness** | all 16 qualifying families above chance (AUC 0.66–0.92); resource gradient ρ = -0.05, *p* = .559, 10 families under the provisional proxy | `results/h3_family_fit.csv` |
| **Robustness** | 23 of 38 checks reproduce the primary conclusion, 5 pending; of the 10 that do not, 6 are H2 variants that would have called the blind spot *significant* — the H2 null is boundary-sensitive (report §6) | `results/robustness_summary.csv` |

**The three figures to look at first**

1. `figures/fig11_synthesis.pdf` — the three hypotheses on one panel.
2. `figures/fig1_network.pdf` — the colexification network the whole study is built on
   (interactive: `figures/interactive_network.html`).
3. `figures/fig9_h3_families.pdf` — per-family fit, full and rarefied
   (world map: `figures/fig10_world_map.pdf`, interactive `figures/world_map.html`).

Read `report/report.md` for the full paper and `DECISIONS.md` for every methodological choice.
<!-- RESULTS:END -->

## Citation

If you use this pipeline, please cite the report (`report/report.pdf`) and the data sources above.
