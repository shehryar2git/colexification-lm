"""Central configuration for the colexification-lm project.

Every path, seed, model name, threshold and hyperparameter used anywhere in
``src/`` is defined here and nowhere else. Scripts import from this module;
they never hardcode values. Changing a value here and re-running ``make all``
is the only supported way to alter an analysis.

Values marked ``# DECIDED IN PHASE n`` are set empirically during that phase
(e.g. the concept-set threshold is chosen from the coverage curve) and the
chosen value, with its justification, is recorded in DECISIONS.md.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 20261016  # proposal date; used for numpy, random, torch, sklearn, UMAP, Louvain
N_JOBS: int = 1       # >1 can break bit-for-bit determinism in UMAP / sklearn; keep 1 for `make all`

# ---------------------------------------------------------------------------
# Paths (all relative to the repository root)
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MATRIX_DIR: Path = PROCESSED_DIR / "matrices"
EMBEDDING_DIR: Path = PROCESSED_DIR / "embeddings"   # cached raw vectors per model
RESULTS_DIR: Path = ROOT / "results"
FIGURES_DIR: Path = ROOT / "figures"
REPORT_DIR: Path = ROOT / "report"
AUDIT_DIR: Path = ROOT / "audit"
LOG_DIR: Path = ROOT / "logs"

CONCEPT_ORDER_PATH: Path = PROCESSED_DIR / "concept_order.json"   # immutable once frozen (Phase 1)
GLOSS_OVERRIDES_PATH: Path = ROOT / "src" / "data" / "gloss_overrides.csv"
ENGLISH_AUDIT_PATH: Path = AUDIT_DIR / "english_colexification_manual_check.csv"

# ---------------------------------------------------------------------------
# Data sources (pinned to exact releases; see DECISIONS.md D-01, D-02)
# ---------------------------------------------------------------------------
CLICS_REPO: str = "https://github.com/clics/clics4"
CLICS_TAG: str = "v1.0"
CLICS_COMMIT: str = "1ddece619d026d7fc11a4dcab3fa71ecd059efe7"   # verified 2026-10-23
CLICS_DOI: str = "10.5281/zenodo.16900179"
CLICS_DIR: Path = RAW_DIR / "clics4"

# Robustness only (D-01, D-60): the CLICS3 aggregate described in Rzymski et al. (2020).
# Its published family-weighted network (>= 3 families), as shipped in the repository.
CLICS3_REPO: str = "https://github.com/clics/clics3"
CLICS3_COMMIT: str = "19b4051fd3ac5dbd5d9cb97b6f581e92ad744edf"   # verified 2026-10-24
CLICS3_REPO_PATH: str = "clics3-network.gml.zip"
CLICS3_DIR: Path = RAW_DIR / "clics3"
CLICS3_GML_ZIP: Path = CLICS3_DIR / CLICS3_REPO_PATH
CLICS3_GML_MEMBER: str = "graphs/network-3-families.gml"

GLOTTOLOG_REPO: str = "https://github.com/glottolog/glottolog-cldf"
GLOTTOLOG_TAG: str = "v5.3"
GLOTTOLOG_COMMIT: str = "072ca0d0410039fb8b779be8fc165bac575d2cda"  # verified 2026-10-23
GLOTTOLOG_DIR: Path = RAW_DIR / "glottolog-cldf"

# Concepticon: English glosses and semantic fields of concept sets (D-11, D-18)
CONCEPTICON_REPO: str = "https://github.com/concepticon/concepticon-data"
CONCEPTICON_TAG: str = "v3.4.0"
CONCEPTICON_COMMIT: str = "918bc44e123952a6ab5733be36c2d463799c23b4"  # verified 2026-10-23
CONCEPTICON_DIR: Path = RAW_DIR / "concepticon-data"

# WordNet 3.0 database files (Princeton WNdb format), read directly by
# src/data/wordnet_db.py — no NLTK dependency (D-34). Fetched from the
# nltk_data repository at a pinned commit, sparse checkout of one file.
WORDNET_REPO: str = "https://github.com/nltk/nltk_data"
WORDNET_COMMIT: str = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"  # gh-pages, verified 2026-10-23
WORDNET_REPO_PATH: str = "packages/corpora/wordnet.zip"
WORDNET_DIR: Path = RAW_DIR / "wordnet"
WORDNET_ZIP: Path = WORDNET_DIR / WORDNET_REPO_PATH

# English word frequencies: wordfreq's frozen 3.0 word list (D-50), read directly.
WORDFREQ_REPO: str = "https://github.com/rspeer/wordfreq"
WORDFREQ_COMMIT: str = "372f6dbb3bd2ad4b675a3b18b94d44d0dd5fea8b"   # tag v3.0.2, verified 2026-10-24
WORDFREQ_REPO_PATH: str = "wordfreq/data/large_en.msgpack.gz"
WORDFREQ_DIR: Path = RAW_DIR / "wordfreq"
WORDFREQ_FILE: Path = WORDFREQ_DIR / WORDFREQ_REPO_PATH

# World land outline for the static world map (Natural Earth 1:110m, public domain)
NATURAL_EARTH_REPO: str = "https://github.com/nvkelso/natural-earth-vector"
NATURAL_EARTH_COMMIT: str = "ca96624a56bd078437bca8184e78163e5039ad19"   # v5.1.2 era, verified 2026-10-24
NATURAL_EARTH_REPO_PATH: str = "geojson/ne_110m_land.geojson"
NATURAL_EARTH_DIR: Path = RAW_DIR / "natural-earth"
NATURAL_EARTH_LAND: Path = NATURAL_EARTH_DIR / NATURAL_EARTH_REPO_PATH

# Glottolog pseudo-families that are not genealogical units (D-06)
NON_GENEALOGICAL_FAMILIES: tuple[str, ...] = (
    "Bookkeeping", "Unclassifiable", "Sign Language", "Pidgin",
    "Mixed Language", "Artificial Language", "Speech Register", "Unattested",
)

# ---------------------------------------------------------------------------
# Concept selection (Proposal Step 1; DECISIONS.md D-05)
# ---------------------------------------------------------------------------
CONCEPT_SET_TARGET: tuple[int, int] = (300, 500)
# Candidate thresholds on the number of *families* attesting a concept. The
# knee of the coverage curve is chosen in Phase 1 and written below.
CONCEPT_MIN_FAMILIES_CANDIDATES: tuple[int, ...] = tuple(range(10, 201, 5))
# None = choose automatically at the Kneedle knee (Phase 1 result: t = 85 ->
# 436 concepts, results/concept_threshold.json). Set an integer to override.
CONCEPT_MIN_FAMILIES: int | None = None
# Robustness check (iv): the next stricter threshold, knee + 15 families (D-05).
CONCEPT_MIN_FAMILIES_ROBUSTNESS_OFFSET: int = 15
# Concepts named in the proposal as test cases (H2 and the case study). They are
# kept if attested in >= ANCHOR_MIN_FAMILIES families even when below the knee (D-05b).
ANCHOR_CONCEPTS: tuple[str, ...] = (
    "HAND", "ARM", "TREE", "WOOD", "YESTERDAY", "TOMORROW",
    "FINGER", "TOE", "SEE", "UNDERSTAND", "TONGUE", "LANGUAGE",
)
ANCHOR_MIN_FAMILIES: int = 50

# ---------------------------------------------------------------------------
# Network construction (Proposal Step 2; D-06 to D-09)
# ---------------------------------------------------------------------------
EDGE_MIN_FAMILIES: int = 2      # an edge must be attested in >= 2 independent families (D-07)
# D-37: which colexification evidence counts. "separate_entries" (primary) drops
# form rows whose source entry was mapped to more than one concept (e.g. a source
# item "ARM OR HAND" split into ARM and HAND), so a colexification needs the two
# concepts to have been recorded separately. "all" reproduces CLICS4 exactly.
COLEX_EVIDENCE: str = "separate_entries"
COLEX_EVIDENCE_ALTERNATIVES: tuple[str, ...] = ("all",)
COLEX_IDENTITY: str = "segments"     # D-03: CLICS criterion; "form" = exact string (robustness)
ISOLATE_AS_OWN_FAMILY: bool = True  # Glottolog isolates count as one-language families (D-06)
# D-08 amended (D-39): shortest path is primary — the profile metric left 73.5% of
# pairs tied at the maximum distance and was unstable across edge thresholds.
TYPOLOGICAL_DISTANCE_PRIMARY: str = "shortest_path"   # weighted, edge length 1 / n_families
TYPOLOGICAL_DISTANCE_ALTERNATIVES: tuple[str, ...] = (
    "profile_cosine",    # 1 - cosine of coverage-normalised family-rate profiles (original D-08)
    "rate",              # 1 - first-order family rate
    "binary_profile",    # profile cosine on unweighted edges
    "language_profile",  # profile cosine on language rates
)
# Operationalisation variants, each computed with the primary metric (robustness i-v)
TYPOLOGICAL_VARIANTS: tuple[str, ...] = (
    "minfam1", "minfam3", "evidence_all", "identity_form", "no_johansson",
)
# Per-family subnetworks (Proposal Step 7, H3; D-09)
FAMILY_MIN_LANGUAGES: int = 20
FAMILY_MIN_CONCEPT_COVERAGE: float = 0.5   # share of working concepts attested in the family
FAMILIES_OF_INTEREST: tuple[str, ...] = (
    "Indo-European",
    "Austronesian",
    "Atlantic-Congo",   # Glottolog's name for the core of "Niger-Congo" (D-10)
    "Sino-Tibetan",
    "Pama-Nyungan",
)
FAMILY_RAREFACTION_N_LANGUAGES: int = 20   # equal-size subsampling for H3 (D-23)
FAMILY_RAREFACTION_REPEATS: int = 200
# H3 resource proxies (D-26, D-56). Wikipedia counts need network access to the
# Wikimedia/Wikidata APIs; Joshi et al. (2020) classes are read from their
# lang2tax.txt when the user supplies it, else from the paper-verified subset.
WIKIDATA_SPARQL_URL: str = "https://query.wikidata.org/sparql"
WIKIMEDIA_API_TEMPLATE: str = "https://{code}.wikipedia.org/w/api.php"
WIKIPEDIA_DIR: Path = RAW_DIR / "wikipedia"
JOSHI_LANG2TAX_PATH: Path = RAW_DIR / "joshi" / "lang2tax.txt"
JOSHI_VERIFIED_PATH: Path = ROOT / "src" / "analysis" / "joshi_classes_verified.csv"
H3_MIN_LANGUAGE_COLEX: int = 5         # world map: a language is coloured only with >= 5 colexified analysis pairs
H3_EXACT_PERMUTATION_MAX_N: int = 9    # family-level correlation: exact permutation p up to 9 families
H3_MC_PERMUTATIONS: int = 100_000      # Monte-Carlo permutations above that

# ---------------------------------------------------------------------------
# Community detection and dimensionality reduction (Proposal Step 3)
# ---------------------------------------------------------------------------
COMMUNITY_METHODS: tuple[str, ...] = ("louvain", "infomap")
COMMUNITY_N_RUNS: int = 100
LOUVAIN_RESOLUTION: float = 1.0
INFOMAP_ARGS: str = "--two-level --undirected --silent"

DR_METHODS: tuple[str, ...] = ("pca", "mds_metric", "mds_nonmetric", "umap")
UMAP_N_NEIGHBORS: int = 15
UMAP_MIN_DIST: float = 0.1
TRUSTWORTHINESS_K: int = 10

# ---------------------------------------------------------------------------
# Models (Proposal Step 4; D-12 to D-16)
# Revisions are "main" because Hugging Face is unreachable from the build
# sandbox; the resolved commit hash of every model is written to
# results/model_manifest.json at embedding time so the run is auditable.
# ---------------------------------------------------------------------------
MODELS: dict[str, dict[str, str]] = {
    "w2v_gnews": {
        "kind": "static",
        "name": "word2vec-google-news-300",   # gensim-data identifier
        "label": "word2vec (EN, static)",
    },
    "bert_base": {
        "kind": "contextual",
        "name": "bert-base-uncased",
        "revision": "main",
        "label": "BERT-base (EN)",
    },
    "xlmr_base": {
        "kind": "contextual",
        "name": "xlm-roberta-base",
        "revision": "main",
        "label": "XLM-R base (multi)",
    },
    "labse": {
        "kind": "sentence",
        "name": "sentence-transformers/LaBSE",
        "revision": "main",
        "label": "LaBSE (multi)",
    },
}
# Multilingual-input conditions (two extra matrices: XLM-R and LaBSE fed
# non-English forms). Together with the four above these are the six conditions.
TRANSLATED_INPUT_MODELS: tuple[str, ...] = ("xlmr_base", "labse")
MODEL_CONDITIONS: tuple[str, ...] = (
    "w2v_gnews", "bert_base", "xlmr_base", "labse", "xlmr_translated", "labse_translated",
)

# Input languages for the multilingual condition, selected by ISO 639-3 and
# resolved to CLICS language IDs in Phase 4 (D-15). Urdu is thin in CLICS4
# (228 concepts), so Hindi is included as the Hindustani fallback that carries
# the kal colexification.
TRANSLATED_INPUT_LANGUAGES: dict[str, str] = {
    "urd": "Urdu",
    "hin": "Hindi",
    "rus": "Russian",
    "swh": "Swahili",
    "ind": "Indonesian",
    "tur": "Turkish",
}

# word2vec Google News, read directly from the binary file (D-46). The gensim-data
# release asset; SHA-256 recorded from the build-sandbox download on 2026-10-24.
W2V_URL: str = ("https://github.com/piskvorky/gensim-data/releases/download/"
                "word2vec-google-news-300/word2vec-google-news-300.gz")
W2V_SHA256: str = "cdd97c3b73abcd8e1437ad3837b8faf98fc21258f61f031721fe3e2096ed89d9"
W2V_PATH: Path = RAW_DIR / "word2vec" / "word2vec-google-news-300.gz"
# Function words ignored when averaging the tokens of a multiword gloss (D-15)
GLOSS_STOPWORDS: frozenset[str] = frozenset({"a", "an", "the", "of", "in", "on", "to", "for", "at", "by", "be"})
# British -> American spellings for vocabularies built from US text (Google News);
# a lookup step only, the gloss itself is unchanged (D-15 amendment)
GLOSS_SPELLING_VARIANTS: dict[str, str] = {"axe": "ax", "grey": "gray"}

# Embedding post-processing (D-47): subtract each condition's mean vector over the
# working concepts before cosine (all six conditions); raw cosine is a robustness check.
CENTER_EMBEDDINGS: bool = True

# Multilingual input (D-17 amended, D-48): language units and the scripts their
# forms must be written in. Hindi and Urdu form one unit (Hindustani).
TRANSLATED_UNITS: dict[str, tuple[str, ...]] = {
    "hindustani": ("hin", "urd"),
    "russian": ("rus",),
    "swahili": ("swh",),
    "indonesian": ("ind",),
    "turkish": ("tur",),
}
TRANSLATED_SCRIPTS: dict[str, str] = {   # Unicode script name of the first letter
    "hin": "DEVANAGARI", "urd": "ARABIC", "rus": "CYRILLIC", "swh": "LATIN", "ind": "LATIN", "tur": "LATIN",
}
TRANSLATED_MIN_UNITS: int = 3          # a concept needs forms in >= 3 of the 5 units
TRANSLATED_CURATED_PATH: Path = ROOT / "src" / "embeddings" / "translations_curated.csv"

TEMPLATES: tuple[str, ...] = (   # D-13; "{w}" is replaced by the gloss
    "{w}",
    "the word '{w}'",
    "The word \"{w}\" is a common word.",
    "This is about {w}.",
    "Here is a sentence about {w}.",
    "{w} is a word in English.",
    "I thought about {w}.",
    "The meaning of {w}.",
    "People often talk about {w}.",
    "A text about {w}.",
)
CONTEXTUAL_LAYER: int = -1          # final hidden layer (D-13); layer sweep is a robustness check
CONTEXTUAL_POOLING: str = "target_mean"   # mean over the target word's sub-tokens
EMBED_BATCH_SIZE: int = 32
MAX_SEQ_LEN: int = 32

# ---------------------------------------------------------------------------
# Statistics (Proposal Steps 5-7; D-17 to D-22)
# ---------------------------------------------------------------------------
N_PERMUTATIONS: int = 10_000
N_BOOTSTRAP: int = 2_000
# Robustness checks (Phase 8): many variants, each re-running a permutation test.
# Fewer permutations (p resolution 1/2,001) is ample to decide whether a conclusion holds.
N_PERMUTATIONS_ROBUSTNESS: int = 2_000
N_BOOTSTRAP_ROBUSTNESS: int = 1_000
ALPHA: float = 0.05
FDR_METHOD: str = "fdr_bh"           # Benjamini-Hochberg across models x families
CLUSTER_K_RANGE: tuple[int, ...] = tuple(range(5, 41, 5))
H2_RANK_NORMALISE: bool = True       # nearest-neighbour rank / (N - 1)
H2_MATCH_CALIPER_SD: float = 0.2     # caliper in SD of the logit propensity (Austin 2011)
H2_MATCH_K: int = 3                  # 1:k matching without replacement (D-53, fixed before outcomes were computed)
H2_SMD_THRESHOLD: float = 0.1        # |SMD| < 0.1 counts as balanced
WORDFREQ_LANG: str = "en"
WORDFREQ_WORDLIST: str = "best"

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
FIG_DPI: int = 300
FIG_FORMATS: tuple[str, ...] = ("pdf", "png")
# Okabe-Ito colour-blind-safe palette (Okabe & Ito 2008), in the fixed order
# that passed the dataviz palette validator (adjacent CVD dE >= 9.6, normal-vision
# dE >= 16.4 on a white surface). Orange, purple and sky fall below 3:1 contrast,
# so every figure using them carries direct labels. Black is reserved for annotations.
OKABE_ITO: tuple[str, ...] = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#000000",
)
INK: str = "#1f1f1f"          # primary text
INK_MUTED: str = "#6b6b6b"    # axis labels, annotations
GRID: str = "#e6e6e6"
SEQUENTIAL_CMAP: str = "viridis"
DIVERGING_CMAP: str = "RdBu_r"
