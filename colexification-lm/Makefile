# colexification-lm — one-command reproduction: `make all`
# Every target delegates to run.py, which is the single implementation and
# also works without make (Windows: `python run.py all`).

PY ?= python
PANDOC_OPTS = --citeproc --bibliography=references.bib --csl=csl/apa.csl --resource-path=..:. --toc \
  -V geometry:margin=2.4cm -V fontsize=10pt -V colorlinks=true -V linkcolor=blue \
  -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono"

.PHONY: all setup lock fetch data network embeddings analyse figures report dashboard test test-all refreeze clean clean-all help

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n",$$1,$$2}'

all:  ## Full pipeline, end to end
	$(PY) run.py all
	$(MAKE) report

setup:  ## Create the pinned conda environment
	conda env create -f environment.yml || conda env update -f environment.yml

lock:  ## Freeze the exact installed environment
	$(PY) -m pip freeze --exclude-editable > requirements.lock.txt

fetch:  ## Download pinned raw data (CLICS4, Glottolog, Concepticon, WordNet)
	$(PY) run.py fetch

data: fetch  ## Phase 1: tidy tables, Glottolog metadata, concept set, English-colex candidates
	$(PY) run.py data

network: data  ## Phases 2-3: network, matrices, communities, typological maps
	$(PY) run.py network

embeddings: data  ## Phase 4: six model conditions (slow on CPU; cached)
	$(PY) run.py embeddings

analyse: network embeddings  ## Phases 5-7: H1, H2, H3, robustness
	$(PY) run.py analyse

figures:  ## All figures whose inputs exist (others are listed as PENDING)
	$(PY) run.py figures

report: figures  ## Rebuild report/report.md from results/, then compile report.pdf
	$(PY) -m src.report.build_report
	cd report && (pandoc report.md $(PANDOC_OPTS) --pdf-engine=tectonic -o report.pdf \
	  || pandoc report.md $(PANDOC_OPTS) --pdf-engine=xelatex -o report.pdf \
	  || pandoc report.md $(PANDOC_OPTS) --pdf-engine=xelatex -V fontfamily=mathptmx -o report.pdf \
	  || pandoc report.md --citeproc --bibliography=references.bib --csl=csl/apa.csl -o report.html \
	  || echo "report.md is built; install pandoc and a LaTeX engine for the PDF")

dashboard:  ## Optional Streamlit app
	streamlit run src/viz/dashboard.py

test:  ## Fast unit tests on synthetic fixtures
	$(PY) run.py test

test-all:  ## Unit tests + invariants on the real processed data
	$(PY) -m pytest -q

refreeze:  ## DANGER: re-freeze a changed concept set; invalidates all matrices
	$(PY) -m src.data.select_concepts --refreeze
	$(PY) -c "import config; [p.unlink() for p in config.MATRIX_DIR.glob('*.npz')]"

clean:  ## Remove processed data, results and figures (keeps raw downloads, embeddings, concept order)
	$(PY) -c "import config, shutil; \
	[p.unlink() for p in config.PROCESSED_DIR.rglob('*.parquet')]; \
	[p.unlink() for p in config.MATRIX_DIR.glob('*.npz')]; \
	[p.unlink() for d in (config.RESULTS_DIR, config.FIGURES_DIR) for p in d.iterdir() if p.name != '.gitkeep']"

clean-all: clean  ## Also remove raw downloads and the embedding cache
	$(PY) -c "import config, shutil; \
	[shutil.rmtree(d, ignore_errors=True) for d in (config.CLICS_DIR, config.GLOTTOLOG_DIR, config.CONCEPTICON_DIR, config.WORDNET_DIR)]; \
	[p.unlink() for p in config.EMBEDDING_DIR.iterdir() if p.name != '.gitkeep']"
