# DECISIONS

Every methodological choice in this project, why it was made, and what supports it.
Status codes: **ADOPTED** (fixed), **PROVISIONAL** (to be confirmed or set empirically in the named phase),
**DEVIATION** (differs from the proposal; reason given). Each alternative listed under a decision is run
as a robustness check (Phase 8, `src/analysis/robustness.py`) unless marked otherwise.

Pre-specification: all decisions below were recorded on **2026-10-23, before any analysis was run**.
Later edits are appended with a date; nothing in this file is rewritten after results are seen.

---

## 0. Environment facts observed during Phase 0 (2026-10-23)

These were checked directly, not assumed:

| Fact | Evidence |
|---|---|
| CLICS⁴ v1.0 is the current CLICS release, published as one CLDF dataset | `git ls-remote github.com/clics/clics4` → tag `v1.0` = `1ddece6`; README cites DOI 10.5281/zenodo.16900179 |
| CLICS⁴ v1.0 covers 3,447 varieties, 247 Glottolog families, 1,730 concepts | `cldf/languages.csv`, `cldf/concepts.csv.zip` row counts |
| Its CLDF tables are `languages.csv` (ID, Name, Macroarea, Latitude, Longitude, Glottocode, ISO639P3code, Glottolog_Name, Family, Concept_Count, Form_Count, Contribution_ID, Family_Name), `forms.csv.zip` (ID, Language_ID, Parameter_ID, Form, Segments, Comment, Source, Value, …), `concepts.csv.zip`, `colexifications.csv.zip` (Source_Concept, Target_Concept, Language_Count, Family_Count, Family_Weight, Languages, Families, …), `contributions.csv` (95 datasets) | header inspection |
| The precomputed colexification table has 51,562 pairs; Family_Count median = 1, mean = 1.72, max = 226 | `colexifications.csv.zip` |
| *kal* is attested: Hindi (NorthEuraLex) gives कल /kəl/ for both YESTERDAY and TOMORROW. Urdu (DunnLex, 228 concepts) does not have the pair; Punjabi is absent from CLICS⁴ | `forms.csv.zip` filtered on ISO `hin`/`urd`/`pan`/`pnb` |
| YESTERDAY~TOMORROW: 59 languages / 15 families. ARM~HAND: 404 / 53. TREE~WOOD: 256 / 58. FINGER~TOE: 131 / 39. LANGUAGE~TONGUE: 105 / 13. **SEE~UNDERSTAND: 4 / 3** | `colexifications.csv.zip` |
| GitHub is reachable from the build sandbox; PyPI, Hugging Face, Zenodo and clics.clld.org are not | proxy log |

The last line means Phases 1–3 can be executed and verified in the sandbox (with pandas; see D-03),
while **every model-embedding step (Phase 4) and every result that depends on it is PENDING USER RUN**.

---

## Data

### D-01 · CLICS⁴ v1.0 as the primary dataset — DEVIATION (version), ADOPTED
**Choice.** Use CLICS⁴ v1.0 (Tjuka, Forkel, Rzymski & List 2026; commit `1ddece6`) rather than the CLICS³
release described in Rzymski et al. (2020), which the proposal cites.
**Why.** CLICS⁴ is the maintained successor by the same team: 95 source datasets vs 30, forms uniformly
transcribed to IPA (so colexification is judged on comparable phonological forms rather than mixed
orthographies), and stricter language-selection criteria (247 families). The proposal's
"more than 3,000 varieties" holds (3,447). Pinning an exact commit keeps it reproducible.
**Citation in report.** Both: Rzymski et al. (2020) for the method, Tjuka et al. (2026; preprint
arXiv:2503.11377) and the dataset DOI for the data actually used.
**Alternative (robustness v-b).** Re-run H1/H2 headline tests on CLICS³ to show conclusions do not hinge on the release.

### D-02 · Glottolog v5.3 for genealogy and geography — ADOPTED
CLICS⁴ ships Family / Macroarea / coordinates already, taken from an earlier Glottolog. We re-attach
from glottolog-cldf v5.3 (commit `072ca0d`) by Glottocode so all metadata come from one pinned
classification; disagreements with the shipped values are counted and written to
`results/glottolog_reconciliation.csv`. Missing coordinates fall back to the CLICS value, then to the
parent language's, and the source of each coordinate is recorded.

### D-03 · Recompute colexifications from forms; validate against the shipped table — ADOPTED
**Choice.** Build the network from `forms.csv` ourselves rather than reading `colexifications.csv`.
**Why.** We need per-family counts, coverage (which families attest *both* concepts, D-08), per-family
subnetworks (H3) and example forms per edge (interactive network) — none is available precomputed.
**Identity criterion.** Two concepts are colexified in a language iff they share an identical `Form`
(the CLICS normalised IPA form), following the CLICS procedure. `Value` (source orthography) is kept for display only.
**Validation.** Our language and family counts per pair must match CLICS⁴'s `Language_Count` and
`Family_Count`; the agreement rate is reported in `results/clics_validation.json`. Any systematic
disagreement is investigated before proceeding.
**Loader.** `pycldf` is used when installed (reads the metadata JSON and resolves column names by CLDF
property, not by literal header). A pandas reader keyed on the same CLDF property URLs is the fallback,
and a test asserts the two give identical tables.

### D-04 · Unit of counting = language (Glottocode), not variety — ADOPTED
Several languages appear in more than one source dataset (e.g. Russian in NorthEuraLex and DunnLex;
Turkish in four datasets). Language counts collapse varieties to their Glottolog *language*-level
Glottocode, so a language counts once per colexification even when several datasets attest it.
Family counts are unaffected by construction.

### D-05 · Concept-set threshold chosen at the knee of the coverage curve — PROVISIONAL (Phase 1)
**Choice.** Keep concepts attested in ≥ *t* families. Plot number of retained concepts against *t*
(5 to 200) and choose *t* at the knee located with the Kneedle algorithm (Satopää et al. 2011),
**constrained** so the set has 300–500 concepts as the proposal requires. If the unconstrained knee
falls outside that range, we report it, use the admissible *t* nearest to it, and say so.
**Why families, not languages.** Language counts are dominated by well-sampled families (Indo-European,
Austronesian); family counts are the genealogically independent unit (Dryer 1989; Youn et al. 2016).
A second threshold, one step away, is fixed in Phase 1 for robustness check (iv).

### D-06 · What counts as a family — ADOPTED
Top-level Glottolog family. **Isolates count as their own one-language family**, as in CLICS. Glottolog
pseudo-families that are not genealogical units (Bookkeeping, Unclassifiable, Sign Language, Pidgin,
Mixed Language, Artificial Language, Speech Register, Unattested) are **excluded**, because similarity
within them is not evidence of independent innovation.

### D-07 · Minimum edge support: ≥ 2 families — ADOPTED
At least three quarters of all CLICS⁴ colexifications occur in a single family (the 75th percentile of Family_Count is 1). Single-family
colexifications mix inherited polysemy, chance homophony and transcription accidents, and cannot be
distinguished from shared inheritance. The network keeps edges attested in ≥ 2 families.
**Alternatives (robustness i).** ≥ 1 family (all edges) and ≥ 3 families.

### D-08 · Primary typological distance: cosine distance between coverage-normalised colexification profiles — SUPERSEDED by D-39 (2026-10-23, Phase 2)
**Definition.** For concepts *i ≠ j*, let *F_ij* = number of families colexifying them and *C_ij* =
number of families in which both are attested. Rate *R_ij = F_ij / C_ij* (0 if *F_ij* < 2); *R_ii = 1*.
Each concept's profile is its row of *R*; typological distance *D_ij = 1 − cos(R_i, R_j)*.
**Why this metric** (the fork; both options were considered):
* *First-order rate (1 − R_ij)* is the most literal reading of "colexified in more families", but the
  great majority of pairs have *R_ij = 0* (exact share reported in Phase 2), so the matrix is mostly ties and Spearman-based RSA has little to work
  with. It is kept for H1's **regression**, where the first-order weight is the predictor of interest,
  and as robustness (ii).
* *Profile cosine* is dense, captures the proposal's wording ("from shared colexifications"), and treats
  two concepts as close when they colexify with the same neighbours — the semantic-map logic of
  François (2008) and the network-similarity logic of Youn et al. (2016). Normalising by *C_ij* removes
  the effect of how often two concepts happen to be sampled together, which in CLICS varies by an order
  of magnitude. Setting *R_ii = 1* makes a direct colexification contribute to profile overlap.
**Alternatives (robustness ii–iii).** First-order rate; binary profiles; language-weighted profiles;
weighted shortest-path distance (edge length 1 / F_ij), with disconnected pairs set to max + 1 and the share of such pairs reported.

### D-09 · Per-family subnetworks — PROVISIONAL (Phase 2)
A family qualifies for H3 if it has ≥ 20 languages in CLICS⁴ and attests ≥ 50% of the working concepts.
Its distance matrix is computed with D-08 *within the family* (counting languages, since there is one
family) on the concepts it attests; correlations with models use that concept subset. The list of
qualifying families and their concept coverage is reported in Phase 2. Families named in the proposal
that fail the rule are reported as such, not silently dropped.

### D-10 · "Niger-Congo" is represented by Glottolog's Atlantic-Congo — ADOPTED
Glottolog does not recognise Niger-Congo as a demonstrated family; its largest corresponding unit is
Atlantic-Congo (which contains Bantu, hence Swahili). We use Atlantic-Congo and write
"Atlantic-Congo (≈ Niger-Congo)" in figures and text.

### D-11 · English glosses — ADOPTED, curated in Phase 1
Each concept's gloss is its Concepticon gloss, lower-cased, with disambiguators removed
(e.g. `BARK (OF TREE)` → `bark`). Where that yields a multiword, ambiguous or archaic word, a
manual override goes into `src/data/gloss_overrides.csv` (columns: concept_id, concepticon_gloss,
gloss, reason). Every override is listed in the report appendix. Concepts whose English gloss collapses
two working concepts into one string (e.g. two concepts both glossed `hair`) are flagged, because
identical glosses would give identical model vectors and a spurious perfect "alignment".

### D-12 · Semantic field — ADOPTED
The `same_field` control (H1) and the H2 matching variable come from Concepticon's `SEMANTICFIELD`
attribute (v3.4.0), not from a hand-made scheme, so they are independent of our expectations.

---

## Models

### D-13 · Contextual pooling — ADOPTED
For BERT and XLM-R: embed each gloss in the ten templates in `config.TEMPLATES`, take the
**mean over the target word's sub-tokens** in the **final layer**, and average across templates.
**Why.** [CLS] / whole-sentence pooling mixes in the template words; target-token pooling with
aggregation across contexts gives better static word representations from contextual models
(Bommasani, Davis & Cardie 2020; Vulić et al. 2020).
**Anisotropy.** Contextual embeddings occupy a narrow cone, inflating all cosines (Ethayarajh 2019).
Primary: subtract the mean vector over the working concepts before computing cosines. Raw cosines are robustness.
**Alternatives.** Layer sweep (every layer); templates-only vs bare word; random halves of the template set
(template sensitivity is reported as the spread of RSA over 50 random 5-template subsets).

### D-14 · LaBSE via sentence-transformers; XLM-R via transformers — DEVIATION (minor)
The proposal lists both under sentence-transformers. LaBSE is a sentence encoder with a trained pooled
output, so it is used as intended (sentence-transformers, normalised output, same templates).
`xlm-roberta-base` has no trained sentence head; loading it through sentence-transformers would apply
untrained mean pooling over all tokens. It is therefore treated like BERT (target-token pooling,
D-13), which also makes the BERT ↔ XLM-R contrast (English-only vs multilingual pre-training) a
like-for-like comparison.

### D-15 · Static word2vec OOV rule — ADOPTED
Lookup order: exact gloss → lower-case → phrase token with underscores (Google News has phrases)
→ mean of in-vocabulary content tokens → missing. Missing concepts are NaN rows (masked), never
imputed. Coverage per model is a reported result (`results/embedding_coverage.csv`).

### D-16 · Concept set for model comparisons — ADOPTED
Primary analyses use the **intersection** of concepts embedded by all six conditions, so differences
between models are not differences in concept sets. Per-model maximal sets are a robustness check.

### D-17 · Multilingual-input condition — ADOPTED
* **Forms.** Taken from CLICS⁴ `Value` for Urdu, Hindi, Russian, Swahili, Indonesian, Turkish when the
  value is in the language's standard script (checked by Unicode block: Arabic for Urdu, Devanagari
  for Hindi, Cyrillic for Russian, Latin for the others). Values in transcription or with
  annotations are replaced from a curated table `src/embeddings/translations_curated.csv` with a source
  for every entry (e.g. a named dictionary). Every form has a provenance flag.
* **Hindustani counted once.** Hindi and Urdu vectors are averaged into one Hindustani vector first, so
  one language is not double-weighted. Hindi is included because it carries *kal* in CLICS⁴ and Urdu does not.
* **Input.** Bare word only (no English template), and an English bare-word baseline is computed
  from the same model so the input-language effect is isolated from the template effect.
* **Pooling across languages.** Mean of the available language vectors; a concept needs ≥ 3 of the 5
  language units, otherwise NaN. A concept × language coverage table is reported.
* **Known weakness.** Indonesian in CLICS⁴ comes from a 224-concept list, so most Indonesian forms
  will come from the curated table.

### D-18 · Model revisions — ADOPTED
Hugging Face is unreachable from the build sandbox, so models are loaded at `revision="main"` and the
resolved commit hash, file checksums and library versions are written to `results/model_manifest.json`
at embedding time. That manifest, not `config.py`, is the record of what was run.

---

## Statistics

### D-19 · RSA and Mantel test — ADOPTED
Spearman ρ between upper triangles of typological distance and model distance (1 − cosine).
Significance by a Mantel test permuting rows and columns of one matrix **jointly**, 10,000
permutations, *p* = (k + 1) / (N + 1). H1 is directional, so the pre-specified test is one-sided
(ρ > 0); two-sided *p* is also reported. 95% CIs by a **concept-level bootstrap** (resample concepts,
rebuild both matrices, 2,000 resamples), since resampling pairs would ignore their dependence.
Validated in tests against a planted monotone relation and against independent matrices.

### D-20 · Partial Mantel, with a caveat — ADOPTED
Partial Mantel (Smouse, Long & Sokal 1986) controlling for (a) English word-frequency similarity
(|Δ Zipf|) and (b) same semantic field. Partial Mantel tests are known to be anticonservative under
spatial or structured autocorrelation (Guillot & Rousset 2013), so the inference that matters is carried
by the regression in D-21; the raw-to-partial drop is reported as an effect-size statement.

### D-21 · Pair-level regression — DEVIATION (minor), ADOPTED
`cos_sim ~ z(log1p family_weight) + z(zipf_min) + z(zipf_max) + same_field`.
The proposal's `log_freq_a + log_freq_b` depends on arbitrary pair order; min/max of the two
frequencies is the symmetric equivalent. Pairs share concepts, so OLS standard errors are wrong.
Inference: **MRQAP with double semi-partialling** (Dekker, Krackhardt & Snijders 2007; 10,000
permutations) as primary; **dyadic cluster-robust** SEs (Aronow, Samii & Assenova 2015) as secondary.
Standardised coefficients with bootstrap CIs.

### D-22 · Cluster agreement — ADOPTED
Reference partition: the Phase-3 consensus community partition (frozen). Model partitions:
agglomerative (average linkage, cosine distance) and k-means on L2-normalised vectors, for
k ∈ {5, 10, …, 40} and, **pre-specified as primary**, k = number of reference communities.
ARI and NMI against a null from permuting concept labels (10,000 permutations).

### D-23 · Procrustes — ADOPTED
Classical MDS (2-D) of each space; orthogonal Procrustes with scaling; *m²* statistic with PROTEST
permutation test (Jackson 1995; Peres-Neto & Jackson 2001). Maps are for display; the statistic is
reported but the RSA in D-19 carries the hypothesis test, since 2-D projections discard most variance.

### D-24 · H2 design (pre-specified) — ADOPTED
* **Outcome.** For pair (a, b) in a model: rank of b among a's neighbours and of a among b's, by cosine,
  averaged and divided by (N − 1). 0 = nearest neighbour, ~0.5 = chance.
* **Population.** Pairs with an edge (≥ 2 families, D-07) between two working concepts.
* **Group (a) English-present.** Candidate if some English lemma has a WordNet sense for each concept
  (shared lemma across the synsets of the two glosses). **Every candidate is audited by hand** in
  `audit/english_colexification_manual_check.csv` under written criteria (the word must be ordinary
  present-day usage for both senses; metaphorical extensions noted in a dictionary count, technical or
  archaic senses do not). Claude pre-fills a suggested decision; **the human annotator's column is
  authoritative** and the audit yield is reported.
* **Group (b) English-absent.** All other qualifying pairs, including hand/arm, tree/wood,
  yesterday/tomorrow, finger/toe.
* **Matching.** Propensity score on log family count, min Zipf frequency of the two glosses, and
  exact matching on same-field status; 1:k nearest neighbour without replacement, caliper 0.2 SD of
  the logit propensity (Austin 2011). Balance: standardised mean differences before/after, shown in a
  Love plot; |SMD| < 0.1 = balanced. If balance fails, this is stated and the unmatched analysis is
  reported alongside, labelled as such.
* **Anticipated difficulty, recorded before analysis.** The proposal's flagship English-present example,
  SEE~UNDERSTAND, occurs in only 3 families in CLICS⁴, against 39–58 for the English-absent flagships.
  English-present colexifications may be systematically rarer, which is exactly why matching on family
  count is non-negotiable; the number of English-present pairs surviving the ≥ 2-family rule may also be small,
  and statistical power will be reported.
* **Tests.** Per model: Mann–Whitney U and Cliff's δ with BCa bootstrap CI (2,000). Across models:
  linear mixed model `rank ~ group * model` with a random intercept per pair (the unit repeated
  across the six conditions) and crossed variance components for concept a and concept b (statsmodels
  variance-component formulation). The **group × model interaction** is the key term (does the gap
  shrink in multilingual models?). This replaces the proposal's `(1|concept)`, which does not fit a
  design where each observation involves two concepts. A pair-cluster bootstrap CI for the interaction is
  reported alongside the model-based one.

### D-25 · H3 confound control: rarefaction — ADOPTED
Larger families yield denser, less noisy networks, which by itself would raise RSA for Indo-European.
Each qualifying family is therefore also subsampled to 20 languages, 200 times; the mean and spread
of RSA under rarefaction is reported next to the full-family value. H3 is supported only if the
resource gradient survives rarefaction.

### D-26 · H3 resource proxy and power — PROVISIONAL (Phase 7)
Two proxies: (1) Wikipedia article counts of each family's largest languages from the Wikimedia API
(network access needed — PENDING USER RUN if unavailable); (2) Joshi et al. (2020) resource classes
(0–5), transcribed into `src/analysis/joshi_classes.csv` from the paper with page citation. Family
proxy = log of the maximum over its languages. With 5–8 qualifying families, any family-level
correlation has very low power: it is reported as descriptive, with an exact permutation *p*, and is
not used to claim H3 on its own.

### D-27 · Multiple comparisons — ADOPTED
Benjamini–Hochberg FDR at q = 0.05 within each family of tests: H1 (6 conditions), H2 (6 per-model
tests), H3 (conditions × families). Both raw and adjusted *p* are reported.

### D-28 · Confirmatory vs exploratory — ADOPTED
Confirmatory: D-19 one-sided Mantel per condition (H1); D-24 per-model Cliff's δ and the
group × model interaction (H2); D-25 rarefied family-RSA gradient (H3). Everything else (layer
sweeps, cluster agreement over k, case studies, world map) is labelled exploratory in the report.

---

## Engineering

### D-29 · Environment — ADOPTED
Python 3.11. `gensim` 4.3.x requires `numpy < 2` and `scipy < 1.14`, which fixes the numeric stack.
The pins in `requirements.txt` were written without PyPI access from the build sandbox; `make setup`
on the user's machine is the check, and `make lock` writes the authoritative full freeze.

### D-30 · Determinism — ADOPTED
One seed (`config.SEED = 20261016`) feeds every generator through `src/utils/seeds.py`; each analysis
draws an independent stream `rng(offset)` so adding an analysis never changes earlier results.
`N_JOBS = 1` for UMAP and scikit-learn. Torch deterministic algorithms are on; GPU and CPU runs may
still differ in the last decimal places, so "identical results" means identical within 1e-6.

### D-31 · Canonical concept order — ADOPTED
`data/processed/concept_order.json` is frozen once in Phase 1 with its SHA-256. Every saved matrix
stores that hash and is refused on load if it differs (`src/utils/matrices.py`). Changing the concept
set requires `make refreeze`, which deletes all matrices.

### D-32 · No fabricated output — ADOPTED
Any result that could not be computed in the build environment is marked **PENDING USER RUN** with
the command that produces it. The report reads every number from `results/`; nothing is typed in.

---

## Phase 1 additions (2026-10-23, after data loading, before any network or model analysis)

### Real CLDF schema of CLICS⁴ v1.0 (as resolved by `src/data/load_clics.py`)
`Wordlist-metadata.json` declares a CLDF Wordlist with four tables. Columns used, resolved by property URL
(literal names only for the non-standard columns); the machine-readable map is `results/clics_schema.json`.

| Component | File | Columns used (CLDF property → header) |
|---|---|---|
| FormTable | `forms.csv.zip` | id→ID, languageReference→Language_ID, parameterReference→Parameter_ID, form→Form, value→Value |
| LanguageTable | `languages.csv` | id→ID, name→Name, glottocode→Glottocode, iso639P3code→ISO639P3code, macroarea→Macroarea, latitude→Latitude, longitude→Longitude, contributionReference→Contribution_ID; plus literal `Family`, `Family_Name` |
| ParameterTable | `concepts.csv.zip` | id→ID, concepticonReference→Concepticon_ID; plus literal `Concepticon_Gloss`, `Language_Count`, `Family_Count` |
| ContributionTable | `contributions.csv` | not used (95 rows) |

`StructureDataset-metadata.json` additionally describes `colexifications.csv.zip` (used only to validate our
recount in Phase 2) and `values.csv.zip` (not used). Tables are zipped; both readers handle `<url>.zip`.
Concept IDs in this project are **Concepticon glosses** (unique across all 1,730 CLICS⁴ concepts; checked in code).

### D-04 result
3,447 varieties → 3,438 included (9 in the Glottolog *Bookkeeping* pseudo-family) → **2,002 languages in 246
families, 71 of them isolates.** 324 varieties are Glottolog dialects (mapped to their language) and 41 are coded
with a subgroup node (kept as their own unit). Our Glottolog 5.3 family assignment agrees with CLICS⁴'s shipped
`Family` for all 3,447 varieties (`results/glottolog_reconciliation.csv` is empty). All coordinates come from the
language-level Glottolog record; 6 languages list several macroareas, and the first is used (flagged).

**Observation recorded for H3, before any H3 analysis:** sampling density in CLICS⁴ runs opposite to
training-data density. Indo-European contributes 63 languages; Austronesian 401, Atlantic-Congo 234,
Sino-Tibetan 180, Pama-Nyungan 123. A family-level comparison without equal-size subsampling would confound
"resource-rich" with "sparsely sampled" (strengthens D-25).

### D-05 result — ADOPTED
Kneedle knee at **t = 85 families → 436 concepts**, inside the 300–500 target, so no constraint was needed. The knee
is the same (85) for all six candidate ranges tried (5–150 … 30–200; `results/concept_threshold.json`).
Robustness threshold (iv): **t = 100** (369 concepts before exclusions, 358 in the working set).

### D-05b · Anchor concepts — ADOPTED (decided after seeing coverage counts; stated as such)
Three of the proposal's twelve named concepts sit below the knee: WOOD (85, which the knee keeps), UNDERSTAND (75)
and LANGUAGE (76). Without them, the proposal's flagship English-present examples (SEE~UNDERSTAND,
TONGUE~LANGUAGE) cannot be tested. Rule: a named concept is kept if attested in ≥ 50 families. Both qualify, so
the working set gains 2 concepts. This floor was chosen *after* seeing that the two concepts have 75–76 families;
the robustness set (t = 100) contains no anchor additions, so every conclusion is also checked without them.

### D-11 result and D-35 · English glosses and gloss collisions — ADOPTED
After removing Concepticon disambiguators, 34 candidate concepts collapsed onto 16 shared English strings
(e.g. FLY (INSECT) / FLY (MOVE THROUGH AIR) → "fly"). Identical glosses would give identical model vectors and a
spurious perfect alignment, so `src/data/gloss_overrides.csv` (47 rows: 36 glosses, 11 exclusions, each with a reason) resolves them:
* **Distinct ordinary word available** → override (e.g. *housefly*, *conceal*, *stink*, *perspire*, *slim*,
  *believe*, *worn*, *offspring*, *pale*, *lightweight*, *cleave*, *head hair*, *body hair*): 24 concepts in 11
  collision groups. Each carries a `collision_group`. **Pairs within a collision group are excluded from H2**, because their model
  similarity reflects our choice of substitute word, not the model.
* **No English word distinguishes them** → excluded from the working set (11 concepts): WE (EXCLUSIVE) and
  WE (INCLUSIVE); eight siblings distinguished by the sex of ego; THOU (its only glosses are archaic *thou* or
  *you*, already used for YOU). These exclusions are themselves a limitation in the direction of H2:
  distinctions English cannot express cannot be posed to an English-gloss model at all.
* **Other overrides** (12) replace multiword or misleading glosses: *alive*, *under*, *search*, *forearm*, *shin*,
  *man*, *thigh*, *sugarcane*, *soil* (not *earth*, which is dominated by the planet sense), *defecate*, *chief*, and
  *I* (capitalised, because word2vec is case-sensitive). Override glosses keep their case; default glosses are
  lower-cased.
Final working set: **427 concepts**, 8 with multiword glosses. A code check guarantees all 427 glosses are unique.

### D-12 note
Concepticon's semantic fields are used as given. Some are counter-intuitive (e.g. the pronoun I is filed under
*Kinship*). We do not edit them, so the `same_field` control is external to our hypotheses; the consequence is a
noisier control, discussed under Limitations. Field sizes in the working set: The body 86, Spatial relations 38,
Sense perception 36, … Emotions and values 16 (thin, as CLICS coverage of emotion concepts is low).

### D-24 update · Two WordNet evidence routes — ADOPTED
A shared-lemma rule alone misses TONGUE~LANGUAGE: WordNet files the 'language' sense of *tongue* under the synset
*natural_language*, not *language*. Candidates are therefore generated by two recorded routes:
(1) **shared lemma** across the synsets of both glosses; (2) **definition** — a sense of one gloss word is defined
with the other gloss (e.g. *tongue*: "a human written or spoken language used by a community").
Route (2) over-generates (e.g. *wood* is defined as a substance "under the bark of trees", a meronymy, not a
colexification). The audit, not the rule, decides. Proper-noun synsets (e.g. *Natalie_Wood*) are ignored.
Result: 3,579 candidate pairs among 90,951 (845 shared-lemma, 2,590 definition, 144 both); 408 of 427 glosses
are in WordNet (the 19 not found are function words, multiword glosses and *housefly*). The audit sheet is
restricted to candidates that are also CLICS edges and is written once the network exists (Phase 2).

### D-33 · Parquet fallback in the build sandbox only — ADOPTED
The build sandbox cannot install pyarrow. `src/utils/io.py` writes Parquet whenever an engine is installed
(always, via `requirements.txt`) and falls back to `<name>.parquet.pkl` **only** when the environment variable
`COLEX_ALLOW_PICKLE_FALLBACK=1` is set. The fallback never triggers in a normal run.

### D-34 · WordNet read directly from the WNdb files — ADOPTED (replaces the NLTK dependency)
`src/data/wordnet_db.py` reads Princeton WordNet 3.0 `index.*`, `data.*` and `*.exc` from the zip in the
nltk_data repository (pinned commit `550b662`), with morphy's documented base-form rules. NLTK is no longer
required. Synset IDs are `<offset>-<pos>`, checkable in any WordNet browser.

### D-36 · Cross-platform runner — ADOPTED
`run.py` is the single implementation of the pipeline; the Makefile only calls it. On Windows without `make`,
`python run.py all` is equivalent to `make all`. Raw data are fetched as shallow, sparse checkouts of the pinned
commits (Phase 1 fetch: about 15 s instead of downloading the 1.1 GB CLICS⁴ repository).

---

## Phase 2 additions (2026-10-23, after building the network, before any model embedding)

No model similarity, embedding or hypothesis test existed when any decision below was taken.
Each was decided on properties of the typological data alone.

### D-03 amendment · Colexification identity is `Segments`, as in CLICS — ADOPTED
My first recount used the raw `Form` string and found 45,353 colexified pairs against CLICS⁴'s 51,562, with family
counts agreeing on only 96% of shared pairs. Inspection showed that CLICS identifies colexification on the
**`Segments`** column (tokenised IPA in which tone and stress marks and stray characters are normalised away:
*sámà* → `s a m a`, *da'u* → `d a u`, *yepoi_* → `y e p oi`). With `Segments` as the identity key and CLICS's
counting units (variety Glottocode, CLICS family, all varieties), **the recount matches the shipped
`colexifications.csv` exactly: 51,562 pairs, with variety, language and family counts equal for 100% of pairs**
(`results/clics_validation.json`). Exact-`Form` identity is kept as robustness variant `identity_form`.
Caveat: `Segments` also merges forms that differ only in tone, so for tonal languages some "colexifications" are
tonal minimal pairs. This is CLICS's convention; `identity_form` bounds its effect.

### D-37 · Colexifications must come from separately recorded entries — DEVIATION from CLICS, ADOPTED
**Finding.** 6.33% of all CLICS⁴ form rows share their source entry (CLDF `Local_ID`) with another concept in the
same variety. One source item is mapped to two Concepticon concepts, e.g. Lalo `248_handarm` (source concept
"ARM OR HAND") becomes one HAND row and one ARM row with the same form, and CLICS then counts a colexification.
For the most extreme pairs, nearly all the evidence is of this kind: in ~95% of the varieties colexifying
THINK (BELIEVE)~THINK (REFLECT), HIDE (CONCEAL)~HIDE (ONESELF), HE~SHE and BURN (SOMETHING)~BURNING, both forms
come from a single source entry; for ARM~HAND it is 59%; for TREE~WOOD and YESTERDAY~TOMORROW, 0%.
**The fork.**
* (A) *CLICS as published*: reproduces the database exactly and keeps comparability with the literature.
* (B) *Separately recorded only*: drop form rows whose source entry maps to more than one concept, so a
  colexification counts only when the two concepts were recorded as separate items and received the same form.
**Recommendation and choice: (B) as primary.** A split entry shows that the *source* did not distinguish the two
meanings, not that the *language* does not. Sometimes the author recorded genuine polysemy; often the
questionnaire slot was disjunctive. (B) is the conservative evidence standard. It matters at the top of the network
(HE~SHE 225 → 27 families; THINK~THINK 215 → 23; BEAT~STRIKE 92 → 1; THIN~THIN 206 → 5) but little across the whole
of it: the Spearman correlation of colexification rates over all 90,951 pairs is 0.92, every (B)-edge is an
(A)-edge, and the shortest-path matrices correlate at 0.93. (A) is robustness variant `evidence_all`.
Concept *selection* (Phase 1) counted all forms as attestations; it is not repeated under (B), because a split
entry still shows the concept was elicited. Only colexification and coverage use the filter.

### D-38 · One maximally diverse dataset carries some of the strongest edges — RECORDED, robustness added
FOREARM~UPPER ARM (199 families) and OLD (AGED)~OLD (USED) (187) are among the strongest edges, but almost all
of their support comes from one dataset: Johansson et al. (2020, `johanssonsoundsymbolic`), a sound-symbolism sample
of ~225 languages chosen to be roughly one per family. It elicits the two concepts as separate items (different
`Local_ID`s), and many languages give the same word (Alabama *sakba* for both), so these are genuine data under D-37.
But family weighting lets a single family-diverse dataset dominate the counts for its ~290 concepts. Robustness
variant `no_johansson` recomputes the network without it (Spearman 0.91 with the primary matrix).
OLD (AGED)~OLD (USED) is a same-collision-group pair (D-35) and is already excluded from H2.

### D-39 · Primary typological distance changed to weighted shortest path — DEVIATION from D-08, ADOPTED
**Why D-08 failed.** D-08 chose cosine distance between colexification profiles, stating it would be "dense". On the
real network it is not: **73.5% of concept pairs are tied at the maximum distance** (they share no neighbours). As a
consequence the matrix is unstable under the edge threshold: Spearman 0.53 with itself at ≥ 1 family and 0.49 at ≥ 3
families. Whole blocks of pairs flip into or out of the tie, which would make any RSA result an artefact of the threshold.
**The alternative named in the proposal.** Weighted shortest path (edge length 1 / n_families; disconnected pairs =
max + 1) has 5.5% ties (the 12 isolated concepts) and is stable: Spearman with its own variants is **0.998 (≥ 1 family),
0.924 (≥ 3), 0.928 (CLICS as published), 0.971 (exact Form), 0.911 (without Johansson)**. It also implements the
semantic-map idea directly: concepts are close when connected through a chain of well-attested colexifications
(François 2008). It is the primary metric from now on.
**Consequence to be reported:** the two metrics correlate only 0.50, so the choice is material. Profile cosine
(and the binary, language-weighted and first-order variants) are computed and reported as robustness metrics
(`results/typological_matrix_correlations.csv`, `results/typological_matrix_ties.csv`, Fig. S2c).
D-08's first-order rate remains the predictor of H1's regression (pair level), where ties are not a problem.

### D-09 result and D-25 amendment · Family subnetworks and the H3 statistic — ADOPTED (pre-specified)
16 families qualify (≥ 20 languages, ≥ 50% concept coverage): Austronesian (401 languages), Atlantic-Congo (234),
Sino-Tibetan (180), Nuclear Trans New Guinea (133), Pama-Nyungan (123), Afro-Asiatic (89), Indo-European (63),
Tupian (53), Austroasiatic (38), Nakh-Daghestanian (33), Turkic (32), Uralic (26), Tai-Kadai (24), Hmong-Mien (23),
Tucanoan (21), Arawakan (20). All five families named in the proposal qualify.
Within a single family the network is very sparse (48–535 within-family edges among ~420 concepts), so any
within-family distance matrix is **99.4–99.9% ties** (`results/family_matrices.csv`). A Spearman "RSA" on such a
matrix is in effect a rank-biserial correlation between model distance and a colexified/not indicator.
**H3's pre-specified statistic is therefore the pair-level AUC**: for each family and model, the probability that a
random within-family colexified pair (≥ 2 languages) has higher model similarity than a random non-colexified pair,
among pairs the family attests. The Spearman correlation with the stored `family_<slug>.npz` (1 − within-family
rate) is reported alongside, as the proposal asks. Rarefaction (D-25) applies unchanged.

### D-07 result
Of 11,822 colexified working-set pairs, 9,363 (79%) occur in a single family and are dropped; **2,459 edges**
remain (median support 2 families, 90th percentile 10, maximum 199). The network has one component of 415
concepts and 12 isolated concepts (e.g. HUNDRED, ISLAND, MAIZE, OWL).

### H2 population (for Phase 6)
2,459 edges minus 11 same-collision-group pairs = **2,448 pairs**. 432 of them are WordNet English-colexification
candidates and form the audit sheet (`audit/english_colexification_manual_check.csv`): 120 by shared lemma,
257 by definition, 55 by both. The audit criteria are fixed in `src/data/english_colex.py` (`AUDIT_CRITERIA`).

---

## Phase 3 additions (2026-10-24, after building the typological space, before any model embedding)

### D-40 · Community detection — ADOPTED
Louvain (networkx; weight = n_families; resolution 1.0) over 100 seeds (SEED … SEED + 99). Infomap (two-level,
undirected, same seeds, `--num-trials 1`) is implemented in `src/network/communities.py` and runs whenever the
`infomap` package is installed; it is not installable in the build sandbox, so **all Infomap results are PENDING USER
RUN** (`results/community_stability.json` says so explicitly rather than omitting the method).

### D-41 · Stability verdict and the reference partition — ADOPTED (rule fixed before the run)
Verdict thresholds on the median pairwise ARI across the 100 runs: ≥ 0.90 "stable", ≥ 0.75 "moderately stable",
otherwise "unstable". **Result: median ARI 0.886 (IQR in results file; minimum 0.646) → moderately stable.** This is
reported as such in the paper. The reference partition is the Louvain **medoid** run (highest mean ARI to the other 99
runs): run 84 (seed SEED + 84), 29 communities (17 with ≥ 2 concepts, 12 isolated concepts), relabelled by size and
frozen in `data/processed/reference_partition.json` (SHA-256 recorded, tied to the concept-order hash; changing it
requires `--refreeze`). A resolution sweep (exploratory) shows the default resolution is also the most self-consistent
setting (median ARI to the reference 0.91 at 1.0, against 0.68 at 0.5 and 0.46 at 2.0).
**Caveat for interpretation:** the largest community (104 concepts: *eat, say, see, hear, die, …*) is a heterogeneous
block of everyday verbs with no dominant semantic field (largest field share 15%). It is stable (mean
co-assignment 0.98) but is not a semantic domain in the usual sense; H1's cluster agreement will be reported with and
without it (pre-specified here).

### D-42 · Two-dimensional maps — ADOPTED
All maps embed the primary distance matrix, so quality scores are comparable: classical (Torgerson) MDS = PCA of the
double-centred squared distances; metric and non-metric SMACOF implemented in `src/network/typological_space.py`
(scikit-learn's MDS API differs between the pinned 1.5 and later versions; the implementation is unit-tested against
known configurations); UMAP when `umap-learn` is installed (**PENDING USER RUN** in the sandbox).
**Connected component only.** A first run on all 427 concepts was dominated by the 12 isolated concepts: their
conventional distance (max + 1 to everything) placed them on an outer ring and compressed the other 415 into a blob, and
it made non-metric MDS degenerate (near-zero stress against its own disparities, Shepard ρ 0.33). The max + 1 value is a
convention that keeps a rank correlation computable; it is not geometry. Maps therefore embed the 415 connected
concepts; the isolated ones are listed in `results/dr_choice.json` and in the figure caption.
**Result (415 concepts):** metric MDS trustworthiness 0.79, stress-1 0.30, Shepard ρ 0.74; non-metric MDS 0.78 / 0.32 /
0.79 (no longer degenerate); classical MDS 0.74 / 0.41 / 0.46. By the pre-specified rule (highest trustworthiness, ties →
lower stress; degenerate or pending maps never chosen) **metric MDS** is the display map. If UMAP runs on the user's
machine the rule is re-applied automatically; `results/dr_quality.csv` is authoritative.
**Consequence for Phase 5 (pre-specified now):** the isolated concepts enter the primary RSA through the max + 1
convention; a robustness RSA on the 415 connected concepts is added to the robustness table.

### D-43 · Replicating the Oct 10 practice structure — ADOPTED (exploratory)
The proposal's "concept × language matrix, the same structure as in the Oct 10 practice" is implemented as two matrices.
`concept_language.npz` records attestation (which language has a form for which concept), and is a coverage record.
`concept_lexeme.npz` is the practice session's structure at CLICS scale: concepts × language-specific word forms (one column
per `Segments` form in one variety that covers ≥ 2 working concepts), 427 × 31,153. PCA of it explains 1.7% and 1.5% of
variance on its first two axes and its map anti-correlates with the typological distances (Shepard ρ −0.27): with tens of
thousands of sparse binary columns, Euclidean PCA mainly separates concepts by how many word forms they share, not which.
This is reported as an exploratory result that motivates the family-weighted network analysis.

### D-44 · Community colours in figures — ADOPTED
Okabe–Ito hues in the order validated by the palette checker (adjacent-pair CVD ΔE ≥ 9.6). Six communities are coloured:
first those containing the proposal's twelve named concepts (5 communities), then the largest remaining one; all others
are grey and are named in the interactive network. Every concept named in the proposal falls inside a single community
with its partner (HAND/ARM/FINGER/TOE; TREE/WOOD; YESTERDAY/TOMORROW; TONGUE/LANGUAGE; SEE/UNDERSTAND).

### D-45 · Interactive network without pyvis — ADOPTED
pyvis only wraps vis-network, so `src/viz/interactive_network.py` writes the vis-network page directly (vis-network 9.1.9
from cdnjs, pinned). Node positions come from the same seeded, community-aware layout as Fig. 1 and physics is off, so the
HTML is byte-identical across runs (verified). pyvis is removed from `requirements.txt`. Edge tooltips list family and
language counts, the coverage-normalised rate and up to six example forms, one per family.

### Fig. 1 layout note
The force-directed layout weights within-community edges 6× so that communities separate visually, and pulls the
outermost 1% of nodes inward. Both are display choices only (stated in the caption); no statistic uses layout coordinates.

---

## Phase 4 additions (2026-10-24, embeddings; no hypothesis test run)

### Environment result
torch, transformers and sentence-transformers cannot be installed in the build sandbox and Hugging Face is unreachable,
so **BERT-base, XLM-R base, LaBSE and both multilingual-input conditions are PENDING USER RUN**
(`results/embedding_coverage.csv` lists each with its command). The word2vec file is hosted on GitHub releases, which the
sandbox can reach, so **the word2vec condition is computed** (426 / 427 concepts).

### D-46 · word2vec read directly from the binary file — ADOPTED
`src/embeddings/static_w2v.py` streams the original binary word2vec file (gensim-data release asset, SHA-256
`cdd97c3b…d89d9` pinned in `config.W2V_SHA256` and checked on download) and keeps only the candidate keys, so memory is a
few MB instead of ~3.6 GB for loading all 3 M vectors with gensim. The format is the original word2vec binary format, so
the vectors are identical to gensim's; a test compares them against gensim whenever gensim is installed (marked slow).
The vocabulary is sorted by corpus frequency, so each concept's vocabulary rank is stored as a frequency proxy.

### D-15 amendment · word2vec lookup — ADOPTED
Rule order: exact gloss → lower case → **American spelling** (`axe` → `ax`, `grey` → `gray`; Google News is US English;
the gloss itself is unchanged, this is a lookup step) → phrase token (`upper_arm`) → mean of in-vocabulary content tokens.
Result: 416 exact, 2 spelling variants, 8 token means (*body hair, head hair, in front of, lie down, old man, old woman,
turn around, upper arm*), **1 missing: *and*** (Google News dropped common function words). Per-concept record:
`results/embedding_lookup_w2v_gnews.csv`.

### D-47 · Centring applies to all six conditions — ADOPTED (extends D-13)
D-13 specified mean-centring for the contextual models. It is applied to every condition, so all six are compared
on the same footing; uncentred matrices (`model_<cond>__raw.npz`) are a robustness check. For word2vec, centring moves
the mean pairwise cosine from 0.118 to −0.001 (`results/model_similarity_summary.csv`).

### D-48 · Multilingual-input forms: selection and coverage — ADOPTED
Forms are CLICS⁴ `Value` strings. A value is usable only if its first letter is in the language's standard script and it
contains no phonetic notation (IPA letters, modifier letters, combining diacritics after NFC normalisation, digits,
underscores). Per ISO code the dataset with the most usable values is chosen automatically:
Hindi, Russian and Turkish from NorthEuraLex (331 concepts each), Swahili from WOLD (403), Indonesian from
TransNewGuinea.org (170). The transcription datasets are rejected by the rule itself (e.g. DunnLex Russian 0 / 200
usable, DunnLex Urdu 0 / 199, Johansson Swahili 96 / 275).
**Urdu:** CLICS⁴'s only Urdu list is a phonetic transcription, so it yields no usable forms; the Hindustani unit is
represented by Hindi (Devanagari), the same spoken lexicon in another script. If Urdu-script forms are added through the
curated file, the code averages Hindi and Urdu vectors into one Hindustani vector.
**Coverage:** 331 of 427 concepts have forms in ≥ 3 of the 5 units (the NorthEuraLex concept list); 96 do not.
None of the twelve named concepts is affected.
**Curated fallback:** `src/embeddings/translations_curated.csv` ships empty; every form used is CLICS data. A separate
file, `translations_curated_DRAFT.csv`, holds 288 Hindi / Russian / Turkish translations for the 96 uncovered concepts,
written by Claude and **unverified** (14 flagged entries). It is never read by the pipeline. Entries enter the analysis
only if a person verifies them against a dictionary, adds a citable source and copies them into the real curated file.

### D-16 consequence · Common concept set — RECORDED
D-16 pre-specified that cross-model analyses use the concepts embedded by all six conditions. The multilingual-input
conditions cap that set at **≤ 331 concepts** (the NorthEuraLex list, which is Eurasia-centred basic vocabulary). This is
kept as pre-specified; each model on its own maximal concept set is reported as a robustness check.

### D-49 · Pairs colexified in the input languages — ADOPTED (pre-registered before any model output)
In the multilingual-input condition, a concept pair that one of the five input languages itself colexifies receives the
*same input string* in that language (Hindi कल for YESTERDAY and TOMORROW), so its similarity rises by construction,
not because the model learned anything. 49 of the 2,459 network edges are affected
(`results/translated_input_colexified_pairs.csv`), including four of the proposal's English-absent flagships:
TREE~WOOD (Swahili, Turkish), ARM~HAND (Swahili), FINGER~TOE (Hindi, Russian, Swahili), YESTERDAY~TOMORROW (Hindi).
**Rule for H2:** for the two multilingual-input conditions the primary comparison excludes these pairs; the analysis
including them is reported alongside. The same exclusion is applied to all six conditions in a sensitivity analysis, so
that every model is compared on one pair set. The *kal* case study reports the translated condition with this caveat
stated next to the number.

### Engineering note
`load_concept_order()` and `freeze_concept_order()` bound their default path when the module was imported, so redirecting
`config.CONCEPT_ORDER_PATH` (as tests do) had no effect. Found by a Phase 4 test; the path is now resolved at call time.
No result was affected: the pipeline never changes that path.

---

## Phase 5 additions (2026-10-24, H1)

### D-50 · English frequency from wordfreq's pinned data file — ADOPTED (re-pins requirements)
The Phase 0 pin `wordfreq==3.1.1` could not be verified: the wordfreq repository has no `v3.1.1` tag, and the pin was
written without PyPI access. wordfreq's word lists were frozen at version 3.0, so the project now pins the
`v3.0.2` release (commit `372f6db`). `src/data/frequency.py` reads its `large_en.msgpack.gz` directly (minimal msgpack
decoder) and reproduces `wordfreq.zipf_frequency(word, "en")` from the package source: bucket i has frequency
10^(−i/100); phrase frequency 1/f = Σ 1/fᵢ; 3 significant digits; Zipf = log₁₀ f + 9 rounded to 2 decimals. The file is
fetched by `run.py fetch` like every other source. A test compares against the wordfreq package whenever it is installed;
`requirements.txt` now pins `wordfreq==3.0.2` for that cross-check only. Check values: *the* 7.73 (the published value),
*hand* 5.41, *upper arm* 4.42, *housefly* 1.80.

### D-51 · Statistical implementation and its validation — ADOPTED
`src/utils/stats.py`, validated in `tests/test_phase5_stats.py` before any real-data result was computed:
* **Mantel**: equal to Spearman ρ of the upper triangles; with 6 concepts the Monte Carlo p matches the **exact** p over
  all 720 concept permutations (±0.01); false-positive rate under the null 5% (300 simulated datasets).
* **MRQAP-DSP**: the permuted predictor's t is computed through the Frisch–Waugh–Lovell theorem against a fixed
  projection (≈ 5× faster than refitting); a test proves it equals the full-refit t to 10⁻⁸. Calibration under the null:
  5.3% false positives at α = .05 over 150 simulated datasets.
* **Dyadic cluster-robust SE**: equals a brute-force O(N²) sum over all pairs of dyads sharing a concept.
* **PROTEST**: m² = 0 for a rotated, scaled, shifted copy; large for unrelated configurations.
* **Cliff's δ**: equals the brute-force mean of sign differences; BCa CI via scipy.
* **Benjamini–Hochberg**: checked against a hand-computed example.
Initially one unit test asserted p > .05 for the null predictor on a single simulated dataset and failed (p = .006);
the calibration check showed the test procedure is correct and that one draw was simply a 1-in-100 event, so the unit test
now checks calibration over many draws instead of one p-value.

### D-52 · H1 analysis details fixed before the run — ADOPTED
* **Concept set** (`results/analysis_concept_set.json`): D-16 implemented as "≥ 3 multilingual units AND finite in every
  available model matrix". The set is determined by data, so it is the same whether or not the contextual models
  have been run.
* **BH family**: the available conditions. In the build sandbox that is word2vec alone, so its adjusted p equals its raw p;
  on a full run the family is all six conditions (the results file records the family size in `bh_family`).
* **Regression coding**: `same_field` = 1 for pairs in the same Concepticon field; all predictors and the outcome
  z-scored, so coefficients are standardised. `log_families` = log(1 + n_families), 0 for never-colexified pairs.
* **Cluster agreement**: primary = agglomerative average-linkage clustering of the model distance at k = the number of
  reference communities present in the analysis set; 10,000-permutation null for the primary ARI, 2,000 for the
  exploratory grid (the p-value resolution there is ample for an exploratory display).
  As pre-registered in D-41, it is also computed without the largest (verb) community.
* **Procrustes**: classical MDS of both spaces over the connected concepts of the analysis set (D-42); PROTEST with
  10,000 permutations.

### H1 result for word2vec (recorded 2026-10-24; the other five conditions are PENDING USER RUN)
Analysis set: 330 concepts, 54,285 pairs (331 multilingual-covered concepts minus *and*, which word2vec lacks).
All numbers from `results/h1_alignment.csv`, `results/h1_regression.csv`, `results/h1_cluster_agreement.csv`.
* **Confirmatory:** Spearman ρ = 0.129, 95% concept-bootstrap CI [0.099, 0.160]; one-sided Mantel p = 1/10,001 (no
  permutation reached the observed value; the null distribution has SD 0.009). H1 is supported for word2vec.
* **Against the proposal's expectation:** the proposal anticipated ρ ≈ 0.2–0.4. word2vec falls below that range. Whether
  the contextual and multilingual models reach it is the open question for the user's run.
* **Controls:** partial ρ = 0.084 controlling for English frequency difference and semantic field jointly (drop 0.045,
  about one third of the raw value); 0.114 controlling for frequency only, 0.097 for field only.
* **Pair regression** (standardised, R² = 0.21): colexification support β = 0.18 [0.15, 0.22], MRQAP p = 1/10,001,
  dyadic-robust p < .001; same semantic field β = 0.33 [0.28, 0.38]; lower-frequency member β = 0.23 [0.17, 0.29];
  higher-frequency member β = −0.06 [−0.10, −0.03]. Colexification predicts word2vec similarity over and above
  frequency and semantic field.
* **Exploratory:** cluster agreement ARI = 0.17 (NMI 0.47) at the reference k = 22, permutation p = 1/10,001; 0.21 without
  the large verb community; ARI is roughly flat (0.12–0.22) across k = 5–40. Procrustes m² = 0.83 on 325 connected
  concepts (PROTEST p = 1/10,001): the two-dimensional shapes match only weakly, which is expected when both spaces
  are high-dimensional and their two leading axes carry little of the structure.

---

## Phase 6 additions (2026-10-24, H2)

### D-55 · English gold standard: Claude's suggestions, human decision authoritative — ADOPTED
All 432 audit rows carry a `suggested_decision` written by Claude under the fixed criteria in
`src/data/english_colex.py` (`AUDIT_CRITERIA`): **present** if one ordinary English word is a standard everyday rendering of
each concept in at least one common use (*right* = correct and the right-hand side; *see* = understand; *tongue* =
language; *stone* = seed of a fruit); **absent** if the link is only a definition (part-of, kind-of, related concept), slang,
archaic, or a sense unrelated to the concept; **unsure** for real borderline cases (excluded from H2).
Result: **49 present, 53 unsure, 330 absent**; a reason for every row is in the sheet's `notes` and in
`audit/claude_suggestions.csv`. The `decision` column is for the human annotator and always overrides the suggestion.
`results/h2_blindspot.csv` labels every result with the audit status; it reads **PROVISIONAL (Claude suggestions)** until
every row has a human decision. One known consequence of D-35 is recorded: in HIDE (CONCEAL)~SKIN the English
colexifying word *hide* is not shown to the models, because the gloss was overridden to *conceal*.

### D-24 update · Anticipated difficulty was wrong in direction — RECORDED
D-24 anticipated that English-present colexifications might be cross-linguistically rarer. Within the analysis set the
opposite holds: English-present pairs are attested in a median of about 16 families against 2 for English-absent pairs
(SMD of log family count 1.87), and they are also more often same-field and higher-frequency. Matching remains essential;
only the direction of the confound was mispredicted.

### D-53 · H2 populations and matching — ADOPTED (fixed before any rank was computed)
* Population: network edges between analysis-set concepts, minus same-collision-group pairs and audited *unsure* pairs:
  38 present, 1,736 absent. Without input-colexified pairs (D-49): 31 present, 1,714 absent.
* **1:3** nearest-neighbour matching on the logit propensity (log family count, lower Zipf frequency), exact on
  same semantic field, without replacement, caliper 0.2 SD; treated pairs in a seeded random order.
* Primary population per condition: all pairs for English-input conditions; without input-colexified pairs for the two
  multilingual-input conditions (D-49). The group × model interaction uses the population without input-colexified pairs
  for all conditions, so every model is compared on the same pairs.
* Statistics: one-sided Mann–Whitney U (English-present ranks lower), Cliff's δ (absent vs present; > 0 = present
  closer) with BCa CI (2,000), BH across conditions; interaction = per-model gaps and contrasts with word2vec, pair-cluster
  bootstrap over matched sets, plus a mixed model (pair, concept a, concept b variance components) when statsmodels is
  available. The unmatched comparison is always reported alongside and labelled as confounded.

### D-54 · Propensity model and matching-order sensitivity — ADOPTED
The propensity model is an unpenalised logistic regression fitted by Newton–Raphson in
`src/analysis/h2_blindspot.py` (tested against known coefficients) instead of scikit-learn, whose `penalty` argument
changed between the pinned 1.5 and 1.9. Switching from scikit-learn's iterative solver to the exact maximum-likelihood
fit moved the propensity scores slightly and changed which controls greedy matching chose: δ moved from 0.176 to 0.172,
and the log-family-count SMD in the secondary population moved from 0.097 to 0.110. That fragility is itself a
result, so a **matching-order sensitivity analysis** was added: the full matching is repeated under 200 seeded orders
of the treated pairs and the spread of δ and p is reported (`results/h2_matching_sensitivity.csv`). The seeded run
remains the pre-specified primary result.

### H2 result for word2vec (recorded 2026-10-24; English gold standard PROVISIONAL; five conditions PENDING USER RUN)
All numbers from `results/h2_blindspot.csv`, `h2_balance.csv`, `h2_matching_sensitivity.csv`, `h2_case_study.csv`.
* **Balance:** primary population balanced on all matched covariates (|SMD| ≤ 0.06 after, from 1.87 / 0.46 / 0.48 before);
  35 of 38 present pairs matched to 103 absent pairs; *carry~take*, *back~behind* and *small~young* had no absent pair within
  the caliper. Secondary population (without input-colexified pairs): log family count SMD 0.110 after matching, just
  above the 0.1 threshold, so it is reported as not fully balanced.
* **Confirmatory (pre-specified seeded run):** Cliff's δ = 0.17, 95% BCa CI [−0.05, 0.37], one-sided p = .065.
  **H2 is not supported for word2vec at α = .05.**
* **Unmatched (confounded):** δ = 0.59 [0.46, 0.70], p = 2 × 10⁻¹⁰. Most of the apparent English blind spot is explained
  by English-present colexifications being the widespread, same-field, frequent ones.
* **Matching-order sensitivity:** over 200 orders δ = 0.19 (2.5–97.5%: 0.16–0.23) and p < .05 in 68% of runs (median
  p = .044). The effect is consistently positive and small; its significance depends on matching details.
  Without input-colexified pairs: δ ≈ 0.13, p < .05 in 0% of runs.
* **Case study:** word2vec places all four English-absent flagships close together: yesterday~tomorrow are each
  other's 3rd nearest neighbour (mutual rank 0.006), arm~hand 0.012, tree~wood 0.014, finger~toe 0.030. These pairs are
  not blind spots for word2vec; they are distributionally related in English text.

---

## Phase 7 additions (2026-10-24, H3; fixed before any family-level number was inspected unless marked RESULT)

### D-56 · Resource proxies: what could be obtained, and how — ADOPTED
* **Wikipedia (primary, D-57).** `src/analysis/resource_proxies.py` lists every Wikipedia edition (Wikimedia
  `sitematrix`), reads each open edition's article count (`siteinfo` statistics), and maps each edition to a language
  through Wikidata (P424 Wikimedia language code, P1394 Glottocode, P220 ISO 639-3). The edition goes to the Glottolog
  family of that Glottocode at **any** level (so `zh`, `ar`, `ms` reach Sino-Tibetan, Afro-Asiatic and Austronesian
  without an ISO macrolanguage table), else via ISO 639-3. Family proxy = log10(1 + articles of its largest edition).
  Simple English and closed editions are excluded; ambiguous and unmatched editions are written to
  `results/wikipedia_editions_unmatched.csv` with their share of all articles. Raw responses are cached with a UTC
  timestamp in `data/raw/wikipedia/` (article counts change daily; the cached snapshot is what the results use).
  **The build environment cannot reach Wikimedia or Wikidata: PENDING USER RUN** (`python run.py analyse`).
* **Joshi et al. (2020) class (secondary).** The authors' `lang2tax.txt` is supplementary material that could not be
  located at a reachable URL (not on arXiv, not in the ACL Anthology record). If the user places it at
  `data/raw/joshi/lang2tax.txt` it is used in full (family = max class over its languages; unlisted languages = 0).
  Otherwise the pipeline uses **only languages whose class the paper itself states**
  (`src/analysis/joshi_classes_verified.csv`, 37 languages with page locations: Table 1, Fig. 3 caption, Table 3,
  Sec. 5.3). Mandarin, Turkish and Hindi are stated only as "class 5 and class 4" and enter at 4. Konkani (an ISO
  macrolanguage) and Popoloca (several languages) are left out as ambiguous. Family values are therefore **lower
  bounds**, and 6 of 16 families have no listed language. Results using it are labelled PROVISIONAL.
* No numbers were transcribed from memory.

### D-57 · H3 confirmatory test — ADOPTED (pre-specified)
Per condition: one-sided Spearman ρ across qualifying families between the **rarefied** mean AUC (D-25) and the
**Wikipedia** proxy; permutation p (exact for ≤ 9 families, else 100,000 Monte-Carlo permutations); BH across the
conditions. H3 is supported for a condition only if ρ > 0 and BH-adjusted p < .05. The same correlation with full-family
AUC, with the Joshi proxy, and with sampling density (log10 number of CLICS⁴ languages) is reported as descriptive.
Per-family AUCs: one-sided concept-label permutation test (10,000; ranks of the similarity matrix permuted, which is
exact because the eligible pair set is closed under concept permutation), concept-bootstrap 95% CI (2,000), BH across
conditions × families (D-27).

### D-09 amendment, made precise — "pairs the family attests"
Eligible pairs = pairs of analysis-set concepts that are **both attested somewhere in the family** (the `both_attested`
definition already stored in `family_pairs.parquet`), not pairs co-attested in one language. This keeps the eligible set
closed under concept permutation (needed for the exact permutation identity above). In rarefied draws the same rule
is applied to the 20 sampled languages, and an edge needs ≥ 2 of them.

### D-58 · World map (exploratory) — ADOPTED
Per-language AUC over the analysis set: the language's own colexified pairs vs its non-colexified pairs, both concepts
attested in the language. A language is coloured only with ≥ 5 colexified analysis pairs (below that the AUC rests on a
handful of pairs); the rest are drawn grey. The model shown is the condition with the highest H1 ρ. Equal Earth
projection (Šavrič, Patterson & Jenny 2019) centred at 150°E so the Pacific is not split; land from Natural Earth 1:110m
(public domain, pinned commit), Antarctica omitted. The interactive version is a self-contained SVG page with no
external libraries (plotly/geopandas were not needed and are not in the pinned environment).

### D-59 · What rarefaction does and does not control — RECORDED (after seeing the results)
Rarefaction equalises the number of languages, not phylogenetic depth. With 20 languages drawn from a deep, diverse
family (Austronesian, Atlantic-Congo) few colexifications reach the 2-language threshold (mean 14–20 positive pairs),
and those that do are the most widespread ones, which models capture best; shallow families (Nakh-Daghestanian, Uralic,
Tai-Kadai) keep 60–70 positive pairs. Rarefied AUC is therefore systematically higher than full-family AUC for deep
families. Arawakan has exactly 20 languages, so its rarefaction interval has zero width. This is reported as a limitation
of the D-25 control; it was not foreseen when D-25 was written.

### H3 result for word2vec (RESULT, recorded 2026-10-24; Wikipedia proxy and five conditions PENDING USER RUN)
All numbers from `results/h3_family_fit.csv`, `results/h3_typology.csv`, `results/h3_language_agreement.csv`.
* word2vec separates within-family colexified from non-colexified pairs above chance in **all 16 families**
  (full-family AUC 0.66–0.92; every BH-adjusted permutation p ≤ .001). Highest: Indo-European 0.92 [0.86, 0.96];
  lowest: Sino-Tibetan 0.66 [0.60, 0.71] and Hmong-Mien 0.66 [0.48, 0.84].
* Rarefied AUC ranges 0.69 (Hmong-Mien) to 0.95 (Indo-European).
* **Sampling density** does not drive the full-family fit (ρ = −0.18 with log languages, two-sided p = .50, 16 families).
* **Resource gradient:** with the paper-verified Joshi lower bounds (10 families) ρ = −0.05, one-sided p = .56 for
  rarefied AUC (0.35, p = .16 for full AUC). No evidence for H3 under this provisional proxy; the confirmatory test
  awaits the Wikipedia proxy.
* Indo-European, the family of the model's training language, has the highest fit both full and rarefied. With one
  English model this cannot be separated from "resource" in general: the five pending conditions are what can.
* **Languages (exploratory):** 1,409 of 2,002 languages have ≥ 5 colexified analysis pairs; median AUC 0.80, and 1,378
  (97.8%) are above 0.5. Macroarea medians 0.77 (Eurasia, Papunesia) to 0.84 (South America).

---

## Phase 8 additions (2026-10-24, robustness and report)

### D-60 · CLICS³ as a robustness dataset — ADOPTED (implements D-01's alternative v-b)
CLICS³ does not ship a CLDF form table in its repository; it ships the aggregated database and the **published
colexification networks**. The robustness check therefore reads `clics3-network.gml.zip`
(`graphs/network-3-families.gml`, pinned commit `19b4051`): 4,228 edges over 2,919 Concepticon concepts with
`FamilyWeight` and `LanguageWeight`. Node `Gloss` values are Concepticon glosses, which are our concept IDs, so the
mapping needs no translation table. The parser skips the `words`/`wofam` payloads (most of the file) by a line-prefix
test. Two caveats are stated with the result: CLICS³'s published network is already thresholded at ≥ 3 families, so the
comparable CLICS⁴ variant is `typ_minfam3`, not the primary; and 400 of our 427 working concepts appear in it.

### D-61 · What "survives" means — ADOPTED (pre-specified before the checks were run)
A robustness check **survives** when it reproduces the *conclusion* of the primary analysis — the sign of the effect
and its significance at α = 0.05 — not when it reproduces the number. `results/robustness.csv` also records, for every
check, the verdict that check would support on its own (`variant_conclusion`), so a reader can see whether a
"does not survive" row disagrees in direction or only in significance. Two H3 rows are **controls**, where finding no
relation is the pass: family fit must not track how densely CLICS⁴ samples a family.
Robustness checks use 2,000 permutations and 1,000 bootstrap draws rather than the primary's 10,000 / 2,000; the
p-value resolution (1/2,001) is ample for a survives / does-not-survive decision.

### D-62 · The report is generated, not written by hand — ADOPTED
`report/report_template.md` holds the prose with `{{placeholder}}` slots; `src/report/build_report.py` fills every slot
from a file in `results/` and writes `results/report_values.json`, which maps each value to its source file. A missing
placeholder is a hard error, so the report cannot be built with a number that no results file produced. Interpretive
sentences that depend on the data (whether H1 / H2 / H3 are supported, how many conditions have been run, how many
families are above chance) are chosen in the builder from pre-written alternatives, never written into the template, so
that re-running after `python run.py analyse` yields a report whose claims match the new numbers. The APA reference list
is generated from `references.bib` by a small BibTeX reader in the same module, and `report/csl/apa.csl`
(CC-BY-SA 3.0, Citation Style Language project, commit `bd78cc4`) renders the in-text citations when pandoc compiles
the PDF.

### H2 robustness result — RECORDED (the most consequential finding of Phase 8)
Of 7 H2 variants, **4 would on their own have declared the English blind spot significant** at α = .05: a 0.10 SD
caliper (δ = 0.21, p = .032), raw cosine instead of the neighbour rank (δ = 0.18, p = .0498), and either treatment of
the audited *unsure* pairs (as absent: δ = 0.22, p = .025; as present: δ = 0.14, p = .039). The pre-specified analysis
(1:3, caliper 0.20 SD, unsure excluded, neighbour rank) gives δ = 0.17, p = .065. The effect sits on the significance
boundary at this sample size, so the pre-specified null must be reported as "too small to resolve with 35 matched
pairs", never as "no English blind spot". This is recorded here, in `results/robustness.csv`, in §6 of the report and
in the Limitations section.

### H1 and H3 robustness results — RECORDED
H1 survives every alternative: edge rules ≥ 1 / ≥ 3 families (ρ = 0.130 / 0.118), four alternative metrics
(0.136–0.140), all-entries evidence (0.139), exact-form identity (0.135), dropping the most family-diverse dataset
(0.117), the stricter concept threshold (0.147), dropping anchor concepts (0.128), connected concepts only (0.141),
each model's maximal concept set (0.118), uncentred cosines (0.142), the CLICS³ network (0.103), and
leave-one-family-out over all 16 qualifying families (0.119–0.130, every p ≤ .0005). H3's per-family result survives
the proposal's Spearman statistic as well as the pre-specified AUC, and the two sampling-density controls pass.
