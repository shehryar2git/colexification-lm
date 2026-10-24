"""Cross-platform pipeline runner (Windows, macOS, Linux).

``make <target>`` simply calls ``python run.py <target>``; on Windows without
``make``, call run.py directly:

    python run.py fetch        # download pinned raw data (git required)
    python run.py data         # Phase 1
    python run.py all          # everything

Each step runs as ``python -m <module>`` in a fresh interpreter with
PYTHONHASHSEED fixed, so steps are isolated and reproducible.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import config

# ---------------------------------------------------------------------------
# Raw sources: (target dir, repo, commit, sparse paths). Shallow fetch of one
# exact commit with sparse checkout, so only the files we use are downloaded
# (CLICS4 is ~1.1 GB in full; its cldf/ folder is a small fraction of that).
# ---------------------------------------------------------------------------
SOURCES: list[tuple[Path, str, str, list[str]]] = [
    (config.CLICS_DIR, config.CLICS_REPO, config.CLICS_COMMIT, ["cldf"]),
    (config.GLOTTOLOG_DIR, config.GLOTTOLOG_REPO, config.GLOTTOLOG_COMMIT, ["cldf"]),
    (config.CONCEPTICON_DIR, config.CONCEPTICON_REPO, config.CONCEPTICON_COMMIT,
     ["concepticondata/concepticon.tsv"]),
    (config.WORDNET_DIR, config.WORDNET_REPO, config.WORDNET_COMMIT, [config.WORDNET_REPO_PATH]),
    (config.WORDFREQ_DIR, config.WORDFREQ_REPO, config.WORDFREQ_COMMIT, [config.WORDFREQ_REPO_PATH]),
    (config.NATURAL_EARTH_DIR, config.NATURAL_EARTH_REPO, config.NATURAL_EARTH_COMMIT,
     [config.NATURAL_EARTH_REPO_PATH]),
    (config.CLICS3_DIR, config.CLICS3_REPO, config.CLICS3_COMMIT, [config.CLICS3_REPO_PATH]),
]

STEPS: dict[str, list[str]] = {
    "data": [
        "src.data.load_clics",
        "src.data.attach_glottolog",
        "src.data.select_concepts",
        "src.data.english_colex",
    ],
    "network": [
        "src.network.build_network",
        "src.network.build_matrices",
        "src.data.english_colex",       # re-run: writes the H2 audit sheet now that edges exist
        "src.network.communities",
        "src.network.typological_space",
    ],
    "embeddings": [
        "src.embeddings.static_w2v",
        "src.embeddings.contextual",
        "src.embeddings.translated_input",
        "src.embeddings.similarity",
    ],
    "analyse": [
        "src.analysis.h1_alignment",
        "src.analysis.h2_blindspot",
        "src.analysis.resource_proxies",
        "src.analysis.h3_typology",
        "src.analysis.robustness",
    ],
    "figures": ["src.viz.figures", "src.viz.interactive_network", "src.viz.world_map"],
    "report": ["src.report.build_report"],
}
ORDER = ["fetch", "data", "network", "embeddings", "analyse", "figures", "report"]


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def fetch_source(target: Path, repo: str, commit: str, sparse: list[str]) -> None:
    ok = target / ".ok"
    if ok.exists() and ok.read_text().strip() == commit:
        print(f"[fetch] {target.name}: cached at {commit[:7]}")
        return
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    print(f"[fetch] {target.name}: {repo} @ {commit[:7]} ({', '.join(sparse)})")
    _git(["init", "--quiet"], target)
    _git(["remote", "add", "origin", repo], target)
    _git(["sparse-checkout", "init", "--no-cone"], target)
    _git(["sparse-checkout", "set", "--no-cone", *sparse], target)
    _git(["fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", commit], target)
    _git(["checkout", "--quiet", "FETCH_HEAD"], target)
    got = subprocess.run(["git", "rev-parse", "HEAD"], cwd=target, capture_output=True,
                         text=True, check=True).stdout.strip()
    if got != commit:
        raise RuntimeError(f"{target.name}: expected {commit}, got {got}")
    ok.write_text(commit + "\n")


def fetch() -> None:
    if shutil.which("git") is None:
        sys.exit("git is required for `fetch` (https://git-scm.com).")
    for src in SOURCES:
        fetch_source(*src)


def run_module(module: str) -> None:
    if importlib.util.find_spec(module) is None:
        # Modules arrive phase by phase; a missing one is reported, never faked.
        print(f"\n[run] PENDING: {module} is not implemented yet (later phase) - skipped")
        return
    env = dict(os.environ, PYTHONHASHSEED=str(config.SEED), PYTHONPATH=str(config.ROOT),
               TOKENIZERS_PARALLELISM="false")
    print(f"\n[run] python -m {module}")
    subprocess.run([sys.executable, "-m", module], cwd=config.ROOT, env=env, check=True)


def run_target(target: str) -> None:
    if target == "fetch":
        fetch()
    elif target == "test":
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "not slow and not network"],
                       cwd=config.ROOT, check=True)
    elif target == "all":
        for t in ORDER:
            run_target(t)
    else:
        for module in STEPS[target]:
            run_module(module)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("targets", nargs="+", choices=[*ORDER, "test", "all"])
    for t in p.parse_args().targets:
        run_target(t)


if __name__ == "__main__":
    main()
