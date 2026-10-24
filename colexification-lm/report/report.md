---
title: "Do language models think in English? Colexification as a test of cross-linguistic semantic structure"
date: "2026-10-24"
suppress-bibliography: true
---

# Do language models think in English? Colexification as a test of cross-linguistic semantic structure

**Generated** 2026-10-24 from `results/` at commit `fdcd96d` by `python -m src.report.build_report`.
Every number below is substituted from a file in `results/`; the mapping from each number to its source file is
`results/report_values.json`. Conditions computed: **1 of 6**
(word2vec (EN, static)). Pending: BERT-base (EN), XLM-R base (multi), LaBSE (multi), XLM-R (multi input), LaBSE (multi input).

---

## Abstract

Languages differ in which meanings they give a single word. Hindi–Urdu *kal* means both 'yesterday' and 'tomorrow';
Russian *рука* covers both 'hand' and 'arm'; English keeps all four apart. When many unrelated languages colexify two
meanings — express them with one word — that is evidence the two meanings are conceptually close for human beings
[@francois2008semantic; @youn2016universal]. This study asks whether the semantic spaces of language models reproduce
that cross-linguistic structure, or mainly reproduce English. We build a family-weighted colexification network from
CLICS⁴ (3,438 language varieties, 2,002 languages, 246 families) over
427 concepts, and compare its geometry with up to six model conditions, from static English word2vec through
BERT, XLM-R and LaBSE to multilingual transformers fed non-English word forms. Three pre-registered hypotheses are
tested: that model similarity tracks colexification (H1), that models capture colexifications English happens to have
better than matched ones it lacks (H2, the *English blind spot*), and that the fit is uneven across language families
(H3). For word2vec (EN, static), the alignment is positive but modest (ρ = 0.129, 95% CI [0.099, 0.160],
*p* < .001), and about 35% of it is explained by English word frequency and semantic field together.
The English blind spot does not survive a matched design: unmatched, English-present colexifications look far closer
(δ = 0.59) than English-absent ones, but after matching on cross-linguistic breadth, frequency and
semantic field the difference falls to δ = 0.17 [-0.05, 0.37], *p* = .065. Within every one of the 16 qualifying families the model separates colexified from non-colexified pairs above chance
(AUC 0.66–0.92), with Indo-European highest. We report 38 robustness checks, of which
23 reproduce the primary conclusion. The remaining 5 model conditions, the human audit of the
English gold standard and the Wikipedia resource proxy are marked PENDING USER RUN and are not estimated anywhere in this
report.

**Keywords:** colexification, lexical typology, distributional semantics, multilingual language models, CLICS,
representational similarity analysis.

---

## 1. Introduction

A Hindi or Urdu speaker who says *kal* may mean yesterday or tomorrow; which one is fixed by the verb, not by the noun.
To an English speaker the merger looks like an accident of vocabulary, even a defect. It is neither. The same pattern
recurs across South Asia and beyond, and it has a transparent conceptual motivation: both days are *the day adjacent to
today*, one step away from the present in time. English does not lexicalise that concept, so an English speaker has no
single word for it — but the concept is available, and many languages name it.

Patterns of this kind are the subject matter of lexical typology. When one word in one language covers two concepts, the
two are said to be **colexified** [@francois2008semantic]. Aggregated over many unrelated languages, colexification is a
measure of conceptual proximity: pairs colexified in many independent families (ARM~HAND, TREE~WOOD, SKIN~BARK) are
pairs that human communities repeatedly judge to be close enough to share a name [@youn2016universal;
@jackson2019emotion]. The Database of Cross-Linguistic Colexifications [CLICS; @rzymski2020clics; @tjuka2025clics4paper]
turns this into a network over concepts, built from standardised wordlists in thousands of varieties.

Language models also carry a semantic space. Distributional models place words in a vector space whose geometry
reflects co-occurrence in text; multilingual models are trained on text in a hundred languages and are routinely
described as having a *shared* or *language-agnostic* semantic space [@conneau2020xlmr; @feng2022labse]. If that
description is right in a strong sense, a multilingual model's geometry should look like the world's colexification
geometry rather than like English. If it is right only in a weak sense — the model aligns other languages onto an
English-shaped space — then the model's similarities should track English lexicalisation, and should be systematically
worse wherever English draws a distinction the rest of the world does not.

That is the question this study asks, in three parts, pre-registered in `DECISIONS.md` before any analysis was run:

* **H1 (alignment).** Concept pairs colexified in more language families are closer in model space.
* **H2 (the English blind spot).** Models capture colexifications that English itself has (SEE~UNDERSTAND) better than
  matched colexifications it lacks (ARM~HAND), with the gap shrinking for multilingual models and for multilingual input.
* **H3 (typological unevenness).** The fit between model space and a family's colexification pattern increases with that
  family's digital resources.

The contribution is less the direction of H1, which is unsurprising, than the discipline applied to H2 and H3.
English-present colexifications are not a random sample of colexifications: as this study documents, they are
dramatically more widespread cross-linguistically than English-absent ones, more often within one semantic field, and
built from more frequent words. An unmatched comparison therefore measures those confounds rather than an English bias.
The matched design, the audited English gold standard, and the explicit separation of confirmatory from exploratory
analyses are what make the answer interpretable.

## 2. Related work

**Colexification as conceptual structure.** Semantic maps [@haspelmath2003geometry] and colexification networks
[@francois2008semantic] formalise the observation that polysemy is not arbitrary. @youn2016universal show that the
structure of a 22-concept subdomain is near-universal across 81 languages; @jackson2019emotion extend the method to
emotion concepts and find both universal structure and cultural variation. @xu2020conceptual show that conceptual
relations such as causal and taxonomic links predict which pairs colexify; @brochhagen2022goldilocks argue that
colexification balances informativeness against effort, and @regier2015efficient make the same efficiency argument for
category systems in general. CLICS³ [@rzymski2020clics] and CLICS⁴ [@tjuka2025clics4paper; @tjuka2026clics4data] provide
the aggregated database this study uses.

**Model semantic spaces and their typological reach.** Static embeddings [@mikolov2013distributed] and contextual
encoders [@devlin2019bert] both yield similarity geometries that can be compared with human data.
@bommasani2020interpreting show how to reduce contextual encoders to static vectors by pooling over contexts, which is
the method used here. @ethayarajh2019contextual document how anisotropic those spaces are, motivating the mean-centring
applied in this study. For multilingual models, @vulic2020probing probe how much lexical-semantic knowledge survives
in XLM-R, and @joshi2020state quantify the resource inequality across the world's languages that H3 is about.
@thompson2020cultural (cited in the proposal) is the closest antecedent for the present design: they compare
distributional semantic alignment across languages and find that alignment decays with cultural distance. The step
taken here is to replace cross-language alignment with the typological colexification network itself, and to test the
English-specific blind spot directly.

**Representational similarity.** Comparing two similarity structures over the same items is representational
similarity analysis in cognitive neuroscience [RSA; @kriegeskorte2008rsa] and the Mantel test in ecology
[@mantel1967detection], with partial variants [@smouse1986partial] and dyadic regression extensions
[@dekker2007mrqap]; this study uses all three.

## 3. Data

### 3.1 Colexification

The colexification data are CLICS⁴ v1.0 [@tjuka2026clics4data], pinned to one commit. The release covers
3,447 varieties; 3,438 remain after dropping Glottolog's *Bookkeeping* pseudo-family,
which reduce to 2,002 distinct Glottolog languages in 246 families (of which 71 are
isolates, counted as one-language families as in CLICS). Genealogy, macroareas and coordinates are re-attached from
Glottolog 5.3 [@glottolog53]; the re-attachment disagrees with the family shipped in CLICS⁴ for
0 varieties.

Rather than reading CLICS⁴'s precomputed colexification table, we recompute colexifications from the form table, so
that per-family counts, coverage and example forms are available. Two concepts are colexified in a variety when they
share an identical IPA segment string. Validated against the shipped table over all concepts and varieties, the recount
reproduces it exactly: 51,562 pairs, with variety, language and family counts equal for
100% of them (`results/clics_validation.json`).

One data-quality decision matters enough to state here. Some source datasets map a single lexicon entry to several
concepts; such an entry produces a colexification that is an artefact of the mapping rather than a fact about the
language. Counting those entries would, for instance, make HE~SHE one of the most widely colexified pairs in the world.
The primary analysis therefore uses only forms whose source entry maps to one concept, and the alternative is kept as a
robustness check (it yields 2,984 edges instead of 2,459, with the pair-level rates correlating
at ρ = 0.92).

### 3.2 Concepts

CLICS⁴ covers 1,730 Concepticon concept sets, most of them thinly attested. We retain concepts attested
in at least 85 language families, the knee of the coverage curve located by the Kneedle algorithm
[@satopaa2011kneedle] (Fig. S1; `figures/figS1_concept_coverage.pdf`), which yields 436 concepts — inside the
proposal's 300–500 target without further constraint. Two concepts named in the proposal fall just below the knee
(UNDERSTAND, LANGUAGE) and are re-admitted by a pre-specified anchor rule; 11 concepts are excluded and
36 English glosses are overridden where the Concepticon gloss would collide with another concept's
(for example FLY (INSECT) and FLY (MOVE THROUGH AIR) would otherwise both be *fly* and receive identical vectors).
The working set is **427 concepts** across 22 semantic fields, frozen with its order and a
SHA-256 (`10cab02b133a…`) that every matrix in the project carries and is checked against on load. A second
set of 358 concepts at a stricter threshold is kept for robustness check (iv).

### 3.3 English lexical data

English glosses come from Concepticon [@concepticon34]. English colexification is read from WordNet 3.0
[@miller1995wordnet]: 408 of the 427 glosses are in WordNet, and 3,579 concept
pairs share a lemma or appear in one another's definitions. Candidates are not used as a gold standard directly; they
are audited (§4.4). English word frequencies for the controls are Zipf values from wordfreq's frozen 3.0 word list
[@speer2022wordfreq], read from the pinned data file.

## 4. Methods

All seeds, paths and thresholds are set in `config.py`, and every result is written with a provenance sidecar
recording the producing script, the git commit and package versions.

### 4.1 The typological space

An edge joins two concepts colexified in at least two independent families, the unit of genealogical independence being
the top-level Glottolog family [@dryer1989large]. Of 11,822 colexified pairs in the working set,
9,363 (79%) occur in a single family and are dropped as potentially inherited or
borrowed; **2,459 edges** remain, with a median support of 2 families, a 90th percentile of 10 and a
maximum of 199. The network has one component of 415 concepts plus 12 isolated
concepts; median degree is 9 and the most connected concept is SAY
(Fig. 1, `figures/fig1_network.pdf`; interactive version `figures/interactive_network.html`).

![](../figures/fig1_network.png)

**Figure 1.** The family-weighted colexification network of the 427 working concepts. An edge joins two
concepts expressed by one word, recorded as separate lexicon entries, in at least two independent language families;
node size is degree, edge width is the number of families, colour is the Louvain community. 12 concepts
with no qualifying colexification are omitted. Interactive version: `figures/interactive_network.html`.

The concept × concept typological distance is the **weighted shortest path** with edge length 1 / (number of families),
disconnected pairs taking the maximum finite distance plus one. The obvious alternative, one minus the colexification
rate, is 73% ties in the relevant variant and unstable under the edge threshold; the shortest path
has only 5.5% ties and correlates 0.998 with its own variants. Four alternative metrics are computed
and used in the robustness table.

Communities are detected with Louvain [@blondel2008louvain] over 100 seeds. Agreement between runs is ARI
0.89 (IQR 0.85–0.92) at modularity 0.66 and a median of
29 communities: *moderately stable*, which is reported as a result rather than hidden behind a single
run (Fig. S4). The medoid run is frozen as the reference partition. Infomap [@rosvall2008infomap] is implemented and
PENDING USER RUN: `pip install infomap==2.8.0`, then `python. For the two-dimensional maps we compare PCA, classical, metric and non-metric MDS
[@kruskal1964mds] by trustworthiness [@venna2006trustworthiness]; **mds metric** wins (trustworthiness
0.79, stress-1 0.30) and is used in Fig. 2, over the 415 connected concepts.

![](../figures/fig2_typological_map.png)

**Figure 2.** The typological semantic space in two dimensions (mds metric, trustworthiness 0.79,
stress-1 0.30), over the 415 concepts of the network's connected component. Colour is the
frozen reference community; labelled points are the concepts named in the proposal and the pairs discussed in the
text. Distances in the plane understate the true distances: see the stress value.

### 4.2 Model spaces

Six conditions span the proposal's progression: static English word2vec; BERT-base; XLM-R and LaBSE on English input;
and XLM-R and LaBSE on **multilingual input**. For word2vec, multiword glosses and out-of-vocabulary items follow a
documented lookup chain, which leaves 426 of 427 concepts embedded and *and* missing.
Contextual vectors are the mean of the target word's sub-token states in the final layer, averaged over ten templates
[@bommasani2020interpreting]. Multilingual input uses the actual word forms attested in CLICS⁴ for five typologically
diverse units (Hindustani, Russian, Swahili, Indonesian, Turkish), accepted only when the form is written in the
language's standard script; 331 concepts have forms in at least 3 of the
five units, and 0 forms come from outside CLICS. Every condition's vectors are mean-centred before
cosines are computed, which moves word2vec's mean pairwise cosine from 0.12 to -0.001
[@ethayarajh2019contextual]; uncentred matrices are kept for robustness.

Cross-model analyses use the concepts every condition can embed — **330 concepts, 54,285
pairs** — so that differences between models are not differences in concept sets.

### 4.3 H1: alignment

The confirmatory test is RSA [@kriegeskorte2008rsa]: the Spearman correlation between the upper triangles of the
typological and model distance matrices, with a one-sided Mantel test [@mantel1967detection] of 10,000 joint
row-and-column permutations, Benjamini–Hochberg corrected across conditions [@benjamini1995fdr], and a 95% confidence
interval from a concept-level bootstrap. Partial Mantel tests [@smouse1986partial] control for English word-frequency
similarity and for same-semantic-field membership. A pair-level regression of model similarity on colexification
support, the two concepts' frequencies and a same-field indicator is fitted with MRQAP double semi-partialling
[@dekker2007mrqap] and dyadic cluster-robust standard errors [@aronow2015dyadic]. Cluster agreement (ARI, NMI against
the frozen reference partition) and Procrustes/PROTEST comparison of the 2-D maps [@jackson1995protest;
@peresneto2001procrustes] are exploratory.

### 4.4 H2: the English blind spot

**Gold standard.** Every WordNet candidate pair is written to an audit sheet with its evidence, a suggested decision
and a reason, for a human annotator to confirm or overturn. The audit status of the present run is
**PROVISIONAL (Claude suggestions)**; until a person has decided every row, all H2 numbers in this report inherit that status.

**Design.** The population is the 2,459 network edges between analysis-set concepts, minus pairs whose glosses
collide and minus pairs audited *unsure*: 38 English-present and 1,736 English-absent pairs.
The two groups are not comparable as they stand — English-present pairs are attested in far more families (standardised
mean difference 1.87), are more often in one semantic field, and are built from more frequent words — so the
comparison is **matched**: 1:3 nearest-neighbour matching on the logit propensity of being English-present given
log family count and the lower of the two Zipf frequencies, exact on semantic field, without replacement, caliper
0.22 on the logit scale [@austin2011propensity]. Covariate balance before and after is reported as a Love
plot (Fig. 6) and a table; after matching the largest absolute standardised mean difference is 0.06.

**Outcome and test.** For each pair, the mutual nearest-neighbour rank in the model space (0 = each concept is the
other's nearest neighbour). One-sided Mann–Whitney U [@mannwhitney1947] and Cliff's δ [@cliff1993dominance] with a BCa
bootstrap interval [@efron1987bca]; a group × model interaction with a pair-cluster bootstrap when more than one
condition is available. Because greedy matching depends on the order in which treated units are processed, the whole
matching is repeated under 200 seeded orders and the spread of δ and p is reported.

### 4.5 H3: typological unevenness

Within a single family the colexification matrix is more than 99% ties, so a Spearman correlation is not informative.
The pre-specified statistic is therefore the pair-level **AUC**: the probability that a within-family colexified pair
(attested in ≥ 2 languages of that family) is more similar in model space than a non-colexified pair, among pairs whose
two concepts the family attests. Inference is a concept-label permutation test (10,000 permutations) with BH correction
across conditions × families; the Spearman alternative is reported alongside. A family qualifies with ≥ 20 languages
and ≥ 50% concept coverage, which admits 16 families including all five named in the proposal.

Because larger families yield denser, less noisy networks, every family is additionally **rarefied** to 20 languages,
200 times. The confirmatory H3 test is the one-sided Spearman correlation across families between the rarefied AUC and
a digital-resource proxy: Wikipedia article counts of the family's largest edition (primary) and the resource class of
@joshi2020state (secondary). Exploratory analyses add a per-language AUC mapped at Glottolog coordinates (Fig. 10) and
macroarea summaries.

## 5. Results

### 5.1 H1 — alignment between the model space and the typological space

| Model condition | ρ (RSA) | 95% CI | Mantel *p* | Partial ρ | ARI | Procrustes *m*² |
| --- | --- | --- | --- | --- | --- | --- |
| word2vec (EN, static) | 0.129 | [0.099, 0.160] | < .001 | 0.084 | 0.17 | 0.83 |
| BERT-base (EN) | PENDING USER RUN |  |  |  |  |  |
| XLM-R base (multi) | PENDING USER RUN |  |  |  |  |  |
| LaBSE (multi) | PENDING USER RUN |  |  |  |  |  |
| XLM-R (multi input) | PENDING USER RUN |  |  |  |  |  |
| LaBSE (multi input) | PENDING USER RUN |  |  |  |  |  |

> **Table 1.** H1. RSA between the typological and model distance matrices over N = 330 concepts
> (54,285 pairs). ρ = Spearman correlation of the upper triangles; CI = 95% concept-bootstrap interval;
> Mantel *p* = one-sided, 10,000 joint permutations, Benjamini–Hochberg corrected across conditions; partial ρ controls
> English frequency and semantic field jointly; ARI against the reference Louvain partition at k = 22;
> *m*² = Procrustes residual. Source: `results/h1_alignment.csv`.

![](../figures/fig3_h1_alignment.png)

**Figure 3.** H1. Spearman ρ between the typological and model distance matrices, per condition, over
N = 330 concepts (54,285 pairs). Filled points are the raw RSA with 95% concept-bootstrap
intervals; open points are the partial correlation controlling English word frequency and semantic field.
Conditions not yet embedded are marked PENDING USER RUN.

For word2vec (EN, static), concepts colexified in more families are closer in model space: ρ = 0.129 [0.099, 0.160],
*p* < .001 against a permutation null with SD 0.009 (Fig. 3, `figures/fig3_h1_alignment.pdf`).
H1 is supported for every condition computed so far. The effect is real but smaller than the proposal anticipated (ρ ≈ 0.2–0.4).

Two controls matter. Partialling out English word frequency and semantic-field membership together lowers the
correlation to ρ = 0.084 — a drop of 0.045, about 35% of the raw value (frequency alone:
0.114; field alone: 0.097). The pair-level regression tells the same story with the
dependency structure respected: standardised β = 0.18 [0.15, 0.22] for colexification
support (*p* < .001 by MRQAP), against β = 0.33 [0.28, 0.38] for same semantic field
and β = 0.23 [0.17, 0.29] for the lower-frequency member. Colexification carries information about
model similarity beyond frequency and field, but it is not the dominant term.

Exploratory comparisons agree that the match is loose rather than structural. Cluster agreement with the typological
communities is ARI = 0.17 (NMI 0.47) at the reference k, roughly flat at 0.11–0.22 across
k = 5–40 and 0.23 when the large verb community is removed (Fig. S5). Procrustes alignment of the two 2-D maps leaves *m*² = 0.83 on 325 connected concepts: the
leading two dimensions of the two spaces are not the same two, as one should expect when both are high-dimensional.

### 5.2 H2 — the English blind spot under a matched design

| Model condition | *n* present / absent | δ matched | 95% CI | *p* (BH) | δ unmatched |
| --- | --- | --- | --- | --- | --- |
| word2vec (EN, static) | 35 / 103 | 0.17 | [-0.05, 0.37] | .065 | 0.59 |
| BERT-base (EN) | PENDING USER RUN |  |  |  |  |
| XLM-R base (multi) | PENDING USER RUN |  |  |  |  |
| LaBSE (multi) | PENDING USER RUN |  |  |  |  |
| XLM-R (multi input) | PENDING USER RUN |  |  |  |  |
| LaBSE (multi input) | PENDING USER RUN |  |  |  |  |

> **Table 2.** H2. Cliff's δ > 0 means English-present colexifications sit at a *lower* (closer) neighbour rank than
> matched English-absent ones. Matched = the pre-specified 1:3 propensity design; unmatched = the same
> populations with no matching, reported because the contrast between the two columns is the result. English gold
> standard: **PROVISIONAL (Claude suggestions)**. Source: `results/h2_blindspot.csv`.

![](../figures/fig6_h2_balance.png)

**Figure 6.** H2 covariate balance (Love plot). Standardised mean differences between English-present and
English-absent colexification pairs before and after 1:3 propensity matching, for each matched covariate.
The shaded band is |SMD| < 0.1. Before matching, English-present pairs are attested in far more families
(SMD 1.87); after matching the largest imbalance is 0.06.

The unmatched comparison looks like a decisive English blind spot: δ = 0.59 [0.45, 0.70],
*p* < .001. It is not one. English-present colexifications in this population are attested in far more families than
English-absent ones (geometric means 14 against 3; standardised mean difference
1.87, Fig. 6), they
are more often within one semantic field, and they use more frequent words. All three predict model similarity on their
own (§5.1). Matching removes the imbalance — every matched covariate has |SMD| ≤ 0.06 — at the cost of
sample size: 35 of 38 English-present pairs find controls within the caliper
(3 do not), giving 35 present and 103 absent pairs.

In the matched comparison the gap falls to **δ = 0.17 [-0.05, 0.37], *p* = .065** (median rank 0.014
for present pairs against 0.027 for absent ones; Fig. 7). H2 is not supported at α = .05: after matching, English-present and English-absent colexifications are not reliably different in model space. The direction is the one H2
predicts and the effect is consistently positive, but at this sample size it is not distinguishable from zero at
α = .05, and its significance depends on matching details: over 200 seeded matching orders,
δ = 0.19 (2.5–97.5%: 0.16–0.23) and *p* < .05 in 68% of runs (median
*p* = .044). Excluding pairs that an input language itself colexifies gives δ = 0.13 with *p* < .05 in
0% of runs. The group × model interaction, which is the term that would show multilingual models
closing the gap, is PENDING USER RUN: needs at least two model conditions.

![](../figures/fig7_h2_ranks.png)

**Figure 7.** H2. Distribution of the mutual nearest-neighbour rank (0 = each concept is the other's nearest
neighbour) for the 35 matched English-present pairs and their 103 matched
English-absent controls, for word2vec (EN, static). Cliff's δ = 0.17 [-0.05, 0.37], *p* = .065.

The named cases are instructive:

| Pair | English | Families | Model | cos | Mutual rank | Input-colexified |
| --- | --- | --- | --- | --- | --- | --- |
| tomorrow ~ yesterday | absent | 15 | w2v_gnews | 0.44 | 0.006 | yes |
| arm ~ hand | absent | 47 | w2v_gnews | 0.35 | 0.012 | yes |
| tree ~ wood | absent | 57 | w2v_gnews | 0.28 | 0.014 | yes |
| finger ~ toe | absent | 39 | w2v_gnews | 0.34 | 0.030 | yes |
| see ~ understand | present | 3 | w2v_gnews | 0.39 | 0.014 | no |
| language ~ tongue | present | 13 | w2v_gnews | 0.34 | 0.006 | yes |

> **Table 3.** The proposal's flagship pairs traced through each computed model. Mutual rank 0 means each concept is
> the other's nearest neighbour among the 330 analysis concepts. *Input-colexified* marks pairs that one of
> the five multilingual-input languages itself colexifies, which inflates similarity by construction in those two
> conditions. Source: `results/h2_case_study.csv`.

The English-absent flagships are not blind spots for word2vec (EN, static) at all: among the 330 analysis
concepts, on average only 2 lie between *yesterday* and *tomorrow*. English text makes them
distributionally close —
they occur in the same frames, with the same verbs — even though English lexicalises them separately. This is the
clearest single lesson of H2: distributional similarity in English is not the same thing as English lexicalisation, and
a test that confuses the two will find a blind spot that is not there.

![](../figures/fig8_h2_gap_by_model.png)

**Figure 8.** H2 by model: Cliff's δ with 95% BCa intervals under the pre-specified matched design (filled) and
without matching (open diamonds), for each condition. δ > 0 means English-present colexifications are closer in that
model's space than matched English-absent ones. English gold standard: PROVISIONAL (Claude suggestions).

### 5.3 H3 — per-family fit and the resource gradient

| Family | Languages | Colex. pairs | AUC | 95% CI | *p* (BH) | AUC rarefied | Joshi class | log₁₀ Wikipedia |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Indo-European | 63 | 106 | 0.92 | [0.86, 0.96] | < .001 | 0.95 | ≥5 | pending |
| Arawakan | 20 | 50 | 0.90 | [0.81, 0.96] | < .001 | 0.90 | – | pending |
| Uralic | 26 | 86 | 0.88 | [0.80, 0.95] | < .001 | 0.90 | ≥4 | pending |
| Nakh-Daghestanian | 33 | 101 | 0.88 | [0.78, 0.95] | < .001 | 0.90 | – | pending |
| Pama-Nyungan | 123 | 79 | 0.87 | [0.78, 0.93] | < .001 | 0.93 | ≥0 | pending |
| Austroasiatic | 38 | 53 | 0.86 | [0.73, 0.95] | < .001 | 0.90 | ≥4 | pending |
| Afro-Asiatic | 89 | 101 | 0.85 | [0.75, 0.93] | < .001 | 0.90 | ≥3 | pending |
| Turkic | 32 | 74 | 0.81 | [0.69, 0.90] | < .001 | 0.79 | ≥4 | pending |
| Austronesian | 401 | 270 | 0.79 | [0.73, 0.85] | < .001 | 0.88 | ≥3 | pending |
| Tai-Kadai | 24 | 69 | 0.78 | [0.66, 0.89] | < .001 | 0.80 | ≥2 | pending |
| Tupian | 53 | 94 | 0.77 | [0.66, 0.88] | < .001 | 0.82 | – | pending |
| Atlantic-Congo | 234 | 166 | 0.76 | [0.68, 0.84] | < .001 | 0.93 | ≥2 | pending |
| Nuclear Trans New Guinea | 133 | 114 | 0.75 | [0.62, 0.85] | < .001 | 0.90 | – | pending |
| Tucanoan | 21 | 66 | 0.72 | [0.58, 0.85] | < .001 | 0.73 | – | pending |
| Hmong-Mien | 23 | 35 | 0.66 | [0.48, 0.84] | < .001 | 0.69 | – | pending |
| Sino-Tibetan | 180 | 430 | 0.66 | [0.60, 0.71] | < .001 | 0.80 | ≥4 | pending |

> **Table 4.** H3, word2vec (EN, static). AUC = probability that a within-family colexified pair is more similar in model
> space than a non-colexified pair, among pairs whose concepts the family attests; CI = 95% concept bootstrap;
> *p* = BH-adjusted concept-label permutation test; *AUC rarefied* = mean over 200 subsamples of 20 languages.
> Joshi classes are lower bounds taken from the published paper (§4.5); Wikipedia counts are pending.
> Source: `results/h3_family_fit.csv`, `results/resource_proxies.csv`.

![](../figures/fig9_h3_families.png)

**Figure 9.** H3. (a) Rarefied AUC (20 languages × 200 draws) for every qualifying family and model condition,
rows ordered by the available resource proxy. (b) Full-family AUC with 95% concept-bootstrap intervals against the
rarefied distribution, for word2vec (EN, static). AUC 0.5 is chance.

Within each of the 16 qualifying families, word2vec (EN, static) separates the family's own colexifications
from its non-colexifications well above chance: AUC 0.66–0.92, every BH-adjusted
*p* < .001 (Fig. 9). The highest is Indo-European (0.92 [0.86, 0.96]), the lowest Sino-Tibetan.
After rarefaction the range is 0.69–0.95.

Whether that ordering reflects digital resources is what H3 asks, and this run cannot answer it.
**H3 cannot be decided yet: the confirmatory proxy (Wikipedia article counts) is PENDING USER RUN.** The two proxies stand as follows:

* *Wikipedia article counts* (confirmatory) — PENDING USER RUN: `python run.py analyse` with internet access (query.wikidata.org, meta.wikimedia.org, *.wikipedia.org)
* *Resource classes of @joshi2020state* (secondary) — PROVISIONAL: 37 languages whose class the paper states (lower bounds per family); for the full taxonomy place lang2tax.txt at data/raw/joshi/lang2tax.txt

Under the secondary proxy, which is available for 10 of the 16 families, the rarefied
gradient is ρ = -0.05, *p* = .559, 10 families — no evidence for H3, and none against it either, given the power available with so
few families.

Two things can be said now. First, the ordering is not an artefact of how densely CLICS⁴ samples each family: full-family
AUC correlates ρ = -0.18 with the log number of languages (*p* = .505, Fig. S6). Second,
Indo-European — the family of the model's training language — has the highest fit of all, both full and rarefied. With a
single English model, "the model's own family" and "the best-resourced family" are the same thing; separating them is
exactly what the 5 pending conditions are for.

Exploratorily, 1,409 of 2,002 languages have enough colexified pairs to score
individually; their median AUC is 0.80 and 97.9% are above chance (Fig. 10). Macroarea
medians run only from 0.77 (Papunesia) to 0.84 (South America): whatever the model
captures is not confined to one region of the world.

![](../figures/fig10_world_map.png)

**Figure 10.** Per-language agreement with word2vec (EN, static): for each CLICS⁴ language at its Glottolog coordinates,
the probability that a pair of concepts it colexifies is more similar in model space than a pair it keeps apart
(both concepts attested in that language). 1,409 languages with at least five colexified analysis
pairs are coloured; the rest are grey. Equal Earth projection centred at 150°E. Exploratory.

![](../figures/fig11_synthesis.png)

**Figure 11.** Synthesis. (a) H1 alignment per condition, (b) the H2 English-present/absent gap per condition,
(c) the H3 family gradient: rarefied AUC against the resource proxy, one point per family. Together these answer the
title question: H1 is supported for every condition computed so far; H2 is not supported at α = .05: after matching, English-present and English-absent colexifications are not reliably different in model space; H3 cannot be decided yet: the confirmatory proxy (Wikipedia article counts) is PENDING USER RUN.

Figure 11 (`figures/fig11_synthesis.pdf`) puts the three hypotheses on one panel.

## 6. Robustness

Every alternative recorded in `DECISIONS.md` was re-run. A check **survives** when it reproduces the primary analysis's
conclusion — the sign of the effect and its significance at α = 0.05 — not when it reproduces the number.

| H | Check | Statistic | Variant says | Conclusion |
| --- | --- | --- | --- | --- |
| H1 | Only concepts connected in the colexification network | 0.141 | supported | survives |
| H1 | This model's maximal concept set instead of the… | 0.118 | supported | survives |
| H1 | Without the concepts kept only by the anchor rule | 0.128 | supported | survives |
| H1 | Concept set at the robustness threshold t = 100… | 0.147 | supported | survives |
| H1 | CLICS3's published family-weighted network (>= 3… | 0.103 | supported | survives |
| H1 | Layer sweep: hidden layers -1, -2, -4 and the mean… | pending |  | PENDING USER RUN |
| H1 | Template sensitivity: each of the 10 templates… | pending |  | PENDING USER RUN |
| H1 | Leave-one-family-out over 16 families: range of rho | 0.119 | supported | survives |
| H1 | Uncentred cosine similarities instead of… | 0.142 | supported | survives |
| H1 | Binary (unweighted) colexification profiles | 0.139 | supported | survives |
| H1 | All CLICS4 entries, including forms whose source… | 0.139 | supported | survives |
| H1 | Colexification identity on the exact form string… | 0.135 | supported | survives |
| H1 | Language-weighted profiles instead of family-weighted | 0.140 | supported | survives |
| H1 | Edge rule >= 1 family instead of >= 2 | 0.130 | supported | survives |
| H1 | Edge rule >= 3 families instead of >= 2 | 0.118 | supported | survives |
| H1 | Without the single most family-diverse source dataset | 0.117 | supported | survives |
| H1 | Profile cosine on family rates | 0.138 | supported | survives |
| H1 | First-order distance 1 - family rate instead of… | 0.136 | supported | survives |
| H2 | Caliper 0.10 SD instead of 0.20 SD | 0.212 | supported | does not survive |
| H2 | English gold standard restricted to rows a person… | pending |  | PENDING USER RUN |
| H2 | Leave-one-family-out over 16 families: range of… | 0.095 | not supported, supported | does not survive |
| H2 | 1:1 matching instead of 1:3 | 0.166 | not supported | survives |
| H2 | Edge rule >= 1 family instead of >= 2, population… | 0.239 | supported | does not survive |
| H2 | Edge rule >= 3 family instead of >= 2, population… | 0.197 | supported | does not survive |
| H2 | Without pairs that an input language itself… | 0.147 | not supported | survives |
| H2 | Raw cosine as the outcome instead of the mutual… | 0.185 | supported | does not survive |
| H2 | Concept set at the robustness threshold t = 100… | 0.047 | not supported | survives |
| H2 | No matching at all (confounded | 0.590 | supported | does not survive |
| H2 | Audited *unsure* pairs counted as English-absent… | 0.221 | supported | does not survive |
| H2 | Audited *unsure* pairs counted as English-present… | 0.142 | supported | does not survive |
| H3 | Every qualifying family's fit, full-family AUC… | 0.658 | every family above chance | survives |
| H3 | Family gradient: full AUC vs Joshi et al. (2020)… | 0.352 | not supported | does not survive |
| H3 | Family gradient: rarefied AUC vs Joshi et al.… | -0.050 | not supported | does not survive |
| H3 | Control: family fit must *not* track how densely… | -0.179 | no relation | survives (control) |
| H3 | Control: family fit must *not* track how densely… | 0.321 | no relation | survives (control) |
| H3 | Family gradient: full AUC vs Wikipedia | pending | not computable | PENDING USER RUN |
| H3 | Family gradient: rarefied AUC vs Wikipedia | pending | not computable | PENDING USER RUN |
| H3 | Spearman correlation with the within-family rate… | 0.015 | every family above chance | survives |

> **Table 5.** Robustness. 38 checks: 23 survive, 10 do not, 5 are
> pending a user run. *Variant says* is the verdict that check would support on its own; *Conclusion* compares it with
> the primary analysis. Two rows are controls, where finding **no** relation is the pass. Statistic column: the value
> or range across conditions. Full detail, including every leave-one-family-out run, is Appendix C and
> `results/robustness.csv`.

H1 is robust: every alternative edge rule, distance metric, evidence rule, concept set and centring choice leaves
the conclusion in place, with ρ in a narrow band. Leave-one-family-out over all 16 qualifying families
changes nothing (min rho 0.119, max rho 0.130), so no single family — Indo-European included — is carrying the alignment, and
rebuilding the typological space from the **CLICS³** published network gives ρ = 0.103, so the conclusion
does not hinge on the release (CLICS³'s network has 4,228 edges over 2,919 concepts, of which 400 are in our working set and 886 edges fall inside it).

**H2's null result is the fragile one, and this is the most important thing in the table.** Of
10 H2 design variants, 6 would on their own have declared the English blind
spot supported at α = .05: Audited *unsure* pairs counted as English-absent instead of excluded; Audited *unsure* pairs counted as English-present instead of excluded; Caliper 0.10 SD instead of 0.20 SD; Edge rule >= 1 family instead of >= 2, population rebuilt; Edge rule >= 3 family instead of >= 2, population rebuilt; Raw cosine as the outcome instead of the mutual neighbour rank. Leave-one-family-out tells the same story from the other side:
2 of the 16 runs cross into significance (dropping Afro-Asiatic, Atlantic-Congo),
while the rest reproduce the null. Each variant is a defensible analysis choice, and the pre-specified one is not
privileged by anything except having been fixed in advance. The honest reading is that the matched effect sits close
to the significance boundary with 35 matched present pairs, so that small design choices move it
across — which is exactly why the design was pre-registered, and exactly why the result should not be reported as
"no English blind spot". It should be reported as an effect too small to resolve at this sample size.

The table is grouped by the proposal's five robustness axes in `results/robustness_by_axis.csv`:
(i) binary versus family-weighted colexification, (ii) alternative distance metrics, (iii) the first-order distance in
place of the shortest path, (iv) a different concept-set threshold, and (v) leave-one-family-out. For H1 every axis is
covered by the typological matrices; for H2, axes (i), (iv) and (v) are implemented by rebuilding the pair population
and re-running the whole matched design, and axes (ii)–(iii) do not apply because H2 never uses the typological
distance matrix.

For H3, the two sampling-density rows are controls: they pass by showing no relation between family fit and how
densely CLICS⁴ samples a family. The resource-gradient rows do not support H3 under the provisional proxy, and the
confirmatory proxy is pending.

## 7. Discussion

**The title question, answered in one sentence.** On the evidence available here, the model's semantic space partly
reproduces the world's colexification structure rather than only English — the alignment holds
within every one of the 16 families tested, not just Indo-European — but the alignment is weak, 35% of it is attributable to English word
frequency and semantic field, and the strongest version of the English-blind-spot claim does not survive a matched
comparison.

That answer needs three qualifications.

First, *weak* is the operative word. ρ ≈ 0.129 over 54,285 pairs is a reliable but small effect, and
the cluster and Procrustes comparisons agree that the two spaces are not the same shape. A reader who wants to use a
language model as a stand-in for a typological semantic map should not: the model will get the broad strokes
(body parts cluster, verbs of motion cluster) and miss the structure that makes colexification interesting.

Second, the English blind spot as originally framed is partly an artefact of what English-present colexifications
*are*. They are the widespread, same-field, high-frequency pairs — exactly the pairs any distributional model handles
well. Once that is matched away, the residual effect is small and, in this run, not significant. The honest statement
is not "there is no English blind spot" but "the blind spot is not large enough to detect with
35 matched pairs, and most of what looked like one was confounding". Increasing the number of
English-present pairs — a larger concept set, or a gold standard that is not restricted to WordNet — is the obvious way
to get power. The fragility documented in §6 points the same way: 6 of
10 reasonable variants of this analysis would have called the effect significant, which is what an
underpowered test close to the boundary looks like.

Third, the *kal* case cuts against the naive expectation in an interesting way. word2vec (EN, static) places *yesterday*
and *tomorrow* among each other's nearest neighbours despite English keeping them apart, because distributional similarity
tracks contextual substitutability, not lexicalisation. If a model's space were a direct image of its training
language's lexicon, that could not happen. What a model inherits from English is subtler than its word boundaries.

For H3, the single informative result is the one that was not the hypothesis: the fit is above chance
within every one of the 16 families tested, from Indo-European to Pama-Nyungan and Tucanoan, and the ordering across families is not explained by how
densely CLICS⁴ samples them. Whether it tracks digital resources is a question this report leaves open, with the code
written and the command stated.

## 8. Limitations

**CLICS coverage is uneven and not a random sample of languages.** Sampling density runs opposite to digital resources
— Austronesian has 401 languages in CLICS⁴ against Indo-European's 63 — so family-level comparisons confound resource level with sampling. Rarefaction addresses the number of
languages but not phylogenetic depth: in a deep family, 20 languages retain only the most widespread colexifications,
which are the easiest for any model, so rarefied AUCs are inflated for deep families relative to shallow ones.

**The English gloss is a mediation layer.** Every model sees an English string, even in the multilingual-input
conditions where the *input* is a non-English form but the *concept inventory* is Concepticon's English-labelled one.
A colexification that English cannot gloss cleanly cannot enter this study at all.

**WordNet is an English-specific, coverage-limited gold standard.** It is a lexicographic resource, not a corpus of
usage; its sense inventory is famously fine-grained; and 408 of 427 glosses are in it at all.
The audit (Appendix B) is the mitigation, and until it carries human decisions the H2 results are labelled
PROVISIONAL (Claude suggestions).

**Template and pooling artefacts.** Contextual vectors depend on the templates used to elicit them and on the layer
pooled. Both sensitivities are implemented and both are PENDING USER RUN.

**The H2 null is boundary-sensitive.** As §6 shows, several defensible variants of the matched analysis would have
reported a significant blind spot. Nothing here should be read as evidence that the blind spot is absent; the data
rule out only a large one.

**The design is correlational.** Nothing here identifies a causal path from training data to semantic geometry. A model
that fits Indo-European best may do so because of training data, because of the typological properties of those
languages, or because Concepticon's concept inventory is itself Eurocentric.

**Multilingual input is partial.** The 331-concept multilingual-input set comes from a
Eurasia-centred wordlist, Urdu's only CLICS⁴ list is phonetic transcription so the Hindustani unit is carried by Hindi,
and pairs that an input language itself colexifies receive identical input strings — handled by exclusion, but a
reminder that the condition is not a clean "multilingual" counterfactual.

**Most of the comparison is still pending.** 1 of 6 conditions are computed. The
central claim of the proposal — that the English blind spot narrows as one moves from English word2vec to multilingual
models with multilingual input — is a claim about an interaction that cannot be estimated from one condition. Appendix
E lists every pending item with its command.

## 9. Conclusion

A family-weighted colexification network over 427 concepts and 2,002 languages provides a
demanding, non-circular test of whether a language model's semantic space reflects how the world's languages carve up
meaning. For the one condition computed here, the answer is: partly, weakly, and more evenly across language families
than the English-blind-spot framing predicts. The apparent blind spot largely dissolves under a matched design, and the
pairs the proposal expected models to miss — *kal*'s yesterday and tomorrow above all — turn out to be close in
word2vec (EN, static)'s space even though English lexicalises them apart. The pipeline, the audit sheet and the pre-registered analysis plan are
in place for the remaining 5 conditions; running them is a single command, and every number in this report will be
regenerated from the files they write.

---

## References

44 works, APA style, generated from `report/references.bib` (the proposal's references plus those added
in this study). In-text keys such as `@francois2008semantic` are rendered as author–year citations when the report is
compiled with `make report`.

Aronow, P. M., Samii, C., & Assenova, V. A. (2015). Cluster-robust variance estimation for dyadic data. *Political Analysis*, *23*(4), 564–577.

Austin, P. C. (2011). Optimal caliper widths for propensity-score matching when estimating differences in means and differences in proportions in observational studies. *Pharmaceutical Statistics*, *10*(2), 150–161.

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, *57*(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

Blondel, V. D., Guillaume, J., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics: Theory and Experiment*, *2008*(10), P10008. https://doi.org/10.1088/1742-5468/2008/10/P10008

Bommasani, R., Davis, K., & Cardie, C. (2020). Interpreting Pretrained Contextualized Representations via Reductions to Static Embeddings. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 4758–4781).

Brochhagen, T., & Boleda, G. (2022). When do languages use the same word for different meanings? The Goldilocks Principle in colexification. *Cognition*.

Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions. *Psychological Bulletin*, *114*(3), 494–509. https://doi.org/10.1037/0033-2909.114.3.494

Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 8440–8451). https://doi.org/10.18653/v1/2020.acl-main.747

Dekker, D., Krackhardt, D., & Snijders, T. A. B. (2007). Sensitivity of MRQAP tests to collinearity and autocorrelation conditions. *Psychometrika*, *72*(4), 563–581.

Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019* (pp. 4171–4186). https://doi.org/10.18653/v1/N19-1423

Dryer, M. S. (1989). Large linguistic areas and language sampling. *Studies in Language*, *13*(2), 257–292.

Earth, N. (n.d.). Natural Earth vector data, 1:110m land. https://github.com/nvkelso/natural-earth-vector, commit ca96624. Public domain.

Efron, B. (1987). Better bootstrap confidence intervals. *Journal of the American Statistical Association*, *82*(397), 171–185. https://doi.org/10.1080/01621459.1987.10478410

Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. In *Proceedings of EMNLP-IJCNLP 2019* (pp. 55–65).

Feng, F., Yang, Y., Cer, D., Arivazhagan, N., & Wang, W. (2022). Language-agnostic BERT sentence embedding. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics* (pp. 878–891). https://doi.org/10.18653/v1/2022.acl-long.62

François, A. (2008). Semantic maps and the typology of colexification: Intertwining polysemous networks across languages. In *From Polysemy to Semantic Change* (pp. 163–215). John Benjamins.

Guillot, G., & Rousset, F. (2013). Dismantling the Mantel tests. *Methods in Ecology and Evolution*, *4*(4), 336–344.

Hammarström, H., Forkel, R., Haspelmath, M., & Bank, S. (2025). Glottolog 5.3. Max Planck Institute for Evolutionary Anthropology, https://glottolog.org. CLDF release, commit 072ca0d.

Haspelmath, M. (2003). The geometry of grammatical meaning: Semantic maps and cross-linguistic comparison. In *The New Psychology of Language* (pp. 211–242). Lawrence Erlbaum.

Jackson, D. A. (1995). PROTEST: A PROcrustean Randomization TEST of community environment concordance. *'Ecoscience*, *2*(3), 297–303.

Jackson, J. C., Watts, J., Henry, T. R., List, J., Forkel, R., Mucha, P. J., Greenhill, S. J., Gray, R. D., & Lindquist, K. A. (2019). Emotion semantics show both cultural variation and universal structure. *Science*, *366*(6472), 1517–1522. https://doi.org/10.1126/science.aaw8160

Joshi, P., Santy, S., Budhiraja, A., Bali, K., & Choudhury, M. (2020). The State and Fate of Linguistic Diversity and Inclusion in the NLP World. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 6282–6293).

Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational similarity analysis – connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, *2*, 4.

Kruskal, J. B. (1964). Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. *Psychometrika*, *29*(1), 1–27. https://doi.org/10.1007/BF02289565

List, J., Tjuka, A., van Dam, M., Weinhold, J., & Forkel, R. (2025). CLLD Concepticon 3.4.0. Max Planck Institute for Evolutionary Anthropology, https://concepticon.clld.org.

Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *The Annals of Mathematical Statistics*, *18*(1), 50–60. https://doi.org/10.1214/aoms/1177730491

Mantel, N. (1967). The detection of disease clustering and a generalized regression approach. *Cancer Research*, *27*(2), 209–220.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In *Advances in Neural Information Processing Systems 26* (pp. 3111–3119).

Miller, G. A. (1995). WordNet: A lexical database for English. *Communications of the ACM*, *38*(11), 39–41. https://doi.org/10.1145/219717.219748

Peres-Neto, P. R., & Jackson, D. A. (2001). How well do multivariate data sets match? The advantages of a Procrustean superimposition approach over the Mantel test. *Oecologia*, *129*(2), 169–178.

Regier, T., Kemp, C., & Kay, P. (2015). Word meanings across languages support efficient communication. In *The Handbook of Language Emergence* (pp. 237–263). Wiley-Blackwell.

Rosvall, M., & Bergstrom, C. T. (2008). Maps of random walks on complex networks reveal community structure. *Proceedings of the National Academy of Sciences*, *105*(4), 1118–1123. https://doi.org/10.1073/pnas.0706851105

Rzymski, C., Tresoldi, T., Greenhill, S. J., Wu, M., Schweikhard, N. E., Koptjevskaja-Tamm, M., Gast, V., Bodt, T. A., Hantgan, A., Kaiping, G. A., Chang, S., Lai, Y., Morozova, N., Arjava, H., Hübler, N., Koile, E., Pepper, S., Proos, M., Van Epps, B., Blanco, I., Hündorf, C., Ojo, L., Forkel, R., & List, J. (2020). The Database of Cross-Linguistic Colexifications, reproducible analysis of cross-linguistic polysemies. *Scientific Data*, *7*, 13. https://doi.org/10.1038/s41597-019-0341-x

Satopää, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011). Finding a ``Kneedle'' in a Haystack: Detecting Knee Points in System Behavior. In *31st International Conference on Distributed Computing Systems Workshops* (pp. 166–171).

Smouse, P. E., Long, J. C., & Sokal, R. R. (1986). Multiple regression and correlation extensions of the Mantel test of matrix correspondence. *Systematic Zoology*, *35*(4), 627–632.

Speer, R. (2022). wordfreq: v3.0. https://doi.org/10.5281/zenodo.7199437

Thompson, B., Roberts, S. G., & Lupyan, G. (2020). Cultural influences on word meanings revealed through large-scale semantic alignment. *Nature Human Behaviour*, *4*(10), 1029–1038. https://doi.org/10.1038/s41562-020-0924-8

Tjuka, A., Forkel, R., Rzymski, C., & List, J. (2025). Advancing the Database of Cross-Linguistic Colexifications with New Workflows and Data. Proceedings of the 16th International Conference on Computational Semantics (IWCS). Preprint arXiv:2503.11377.

Tjuka, A., Forkel, R., Rzymski, C., & List, J. (2026). CLICS$^4$: An Improved Database of Cross-Linguistic Colexifications [Dataset, Version 1.0]. MCL Chair at the University of Passau. https://doi.org/10.5281/zenodo.16900179

Venna, J., & Kaski, S. (2006). Local multidimensional scaling. *Neural Networks*, *19*(6–7), 889–899. https://doi.org/10.1016/j.neunet.2006.05.014

Vuli'c, I., Ponti, E. M., Litschko, R., Glavaš, G., & Korhonen, A. (2020). Probing Pretrained Language Models for Lexical Semantics. In *Proceedings of EMNLP 2020* (pp. 7222–7240).

Xu, Y., Duong, K., Malt, B. C., Jiang, S., & Srinivasan, M. (2020). Conceptual relations predict colexification across languages. *Cognition*, *201*, 104280. https://doi.org/10.1016/j.cognition.2020.104280

Youn, H., Sutton, L., Smith, E., Moore, C., Wilkins, J. F., Maddieson, I., Croft, W., & Bhattacharya, T. (2016). On the universal structure of human lexical semantics. *Proceedings of the National Academy of Sciences*, *113*(7), 1766–1771. https://doi.org/10.1073/pnas.1520752113

Šavrič, B., Patterson, T., & Jenny, B. (2019). The Equal Earth map projection. *International Journal of Geographical Information Science*, *33*(3), 454–465. https://doi.org/10.1080/13658816.2018.1504949


## Appendices

* **Appendix A** — the 427 working concepts with their coverage: `report/appendix/A_concepts.md`
* **Appendix B** — the English colexification audit: `report/appendix/B_english_audit.md`
* **Appendix C** — every robustness check: `report/appendix/C_robustness.md`
* **Appendix D** — figure files: `report/appendix/D_figures.md`
* **Appendix E** — everything marked PENDING USER RUN, with commands: `report/appendix/E_pending.md`

## Data, code and reproducibility

Code: `https://github.com/<user>/colexification-lm` (MIT). Data: CLICS⁴ v1.0 (CC-BY-4.0, doi:10.5281/zenodo.16900179),
Glottolog 5.3, Concepticon 3.4.0, WordNet 3.0, wordfreq 3.0 data, Natural Earth 1:110m. A fresh clone reproduces
everything in this report with `make all`; every statistic is written to `results/` with a provenance sidecar recording
the script, seed, git commit and package versions that produced it.
