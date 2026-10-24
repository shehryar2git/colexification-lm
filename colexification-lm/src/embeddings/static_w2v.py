"""Phase 4 · Static English embeddings: word2vec Google News (300 d).

The binary file is read directly and streamed (DECISIONS.md D-46): only the
vectors for candidate keys are kept, so memory stays at a few MB instead of
the ~3.6 GB that loading all 3 million vectors (e.g. with gensim) requires.
The vectors are identical to gensim's (the file format is the original
word2vec binary format); ``tests/test_phase4_embeddings.py`` checks the reader
on a synthetic file, and a cross-check against gensim runs when it is installed.

Lookup rule per gloss (D-15), first match wins:
  1. exact gloss            ("I", "hand")
  2. lower-cased gloss
  2b. American spelling of a British gloss (config.GLOSS_SPELLING_VARIANTS: axe -> ax, grey -> gray)
  3. phrase token           ("upper arm" -> "upper_arm", exact then lower-cased)
  4. mean of the in-vocabulary content tokens (function words in GLOSS_STOPWORDS ignored)
  5. missing -> NaN row (never imputed); coverage is a reported result.
The vocabulary rank (the file is sorted by corpus frequency) is stored as a
frequency proxy.

Outputs: data/processed/embeddings/w2v_gnews.npz, results/embedding_lookup_w2v_gnews.csv
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.embeddings.store import coverage, exists, save_embeddings
from src.utils.concept_order import load_concept_order
from src.utils.io import read_parquet, write_result_table
from src.utils.logging_utils import get_logger, log_run_header
from src.utils.seeds import set_all_seeds

log = get_logger(__name__, "phase4_static_w2v")
CONDITION = "w2v_gnews"


# ----------------------------------------------------------------- download --
def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def ensure_file(path: Path = config.W2V_PATH, url: str = config.W2V_URL, expected: str = config.W2V_SHA256) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    log.info("Downloading %s (1.7 GB) to %s", url, path)
    with urllib.request.urlopen(url) as r, open(part, "wb") as out:   # honours HTTPS_PROXY
        shutil.copyfileobj(r, out, length=1 << 22)
    got = sha256(part)
    if got != expected:
        part.unlink()
        raise RuntimeError(f"word2vec checksum mismatch: {got} != {expected}")
    part.rename(path)
    return path


# ------------------------------------------------------------------- reader --
def stream_vectors(path: Path, wanted: set[str], chunk: int = 1 << 24) -> tuple[dict[str, np.ndarray], dict[str, int], int]:
    """Read a (gzipped) binary word2vec file, keeping only ``wanted`` keys.
    Returns ({key: vector}, {key: 0-based vocabulary rank}, vocabulary size)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    found: dict[str, np.ndarray] = {}
    rank: dict[str, int] = {}
    with opener(path, "rb") as f:
        header = f.readline().split()
        vocab, dim = int(header[0]), int(header[1])
        nbytes = dim * 4
        buf = b""
        pos = 0
        for i in range(vocab):
            # make sure the buffer holds a full record (word <= 1000 bytes + vector)
            if len(buf) - pos < nbytes + 1024:
                buf = buf[pos:] + f.read(chunk)
                pos = 0
            while buf[pos:pos + 1] in (b"\n", b" "):
                pos += 1
            sp = buf.index(b" ", pos)
            word = buf[pos:sp].decode("utf-8", errors="replace")
            start = sp + 1
            if word in wanted:
                found[word] = np.frombuffer(buf, dtype="<f4", count=dim, offset=start).astype(np.float32)
                rank[word] = i
                if len(found) == len(wanted):
                    break
            pos = start + nbytes
    return found, rank, vocab


# --------------------------------------------------------------------- rule --
def candidate_keys(gloss: str) -> dict[str, list[str]]:
    """Ordered lookup candidates for each rule step."""
    g = gloss.strip()
    tokens = [t for t in g.replace("-", " ").split() if t.lower() not in config.GLOSS_STOPWORDS] or g.split()
    return {
        "exact": [g],
        "lower": [g.lower()],
        "spelling_variant": [config.GLOSS_SPELLING_VARIANTS[g.lower()]] if g.lower() in config.GLOSS_SPELLING_VARIANTS else [],
        "phrase": [g.replace(" ", "_"), g.lower().replace(" ", "_")] if " " in g else [],
        "token_mean": tokens if " " in g else [],
    }


def lookup(gloss: str, found: dict[str, np.ndarray], rank: dict[str, int]) -> tuple[np.ndarray | None, str, str, float]:
    c = candidate_keys(gloss)
    for rule in ("exact", "lower", "spelling_variant", "phrase"):
        for k in c[rule]:
            if k in found:
                return found[k], rule, k, float(rank[k])
    toks = [t for t in c["token_mean"] if t in found or t.lower() in found]
    if toks:
        keys = [t if t in found else t.lower() for t in toks]
        vec = np.mean([found[k] for k in keys], axis=0)
        return vec, "token_mean", "+".join(keys), float(np.mean([rank[k] for k in keys]))
    return None, "missing", "", float("nan")


def embed(glosses: pd.Series, path: Path) -> tuple[np.ndarray, pd.DataFrame, int]:
    wanted: set[str] = set()
    for g in glosses:
        for keys in candidate_keys(g).values():
            wanted.update(keys)
            wanted.update(k.lower() for k in keys)
    found, rank, vocab = stream_vectors(path, wanted)
    rows, vecs = [], []
    dim = len(next(iter(found.values()))) if found else 300
    for cid, g in glosses.items():
        v, rule, key, r = lookup(g, found, rank)
        vecs.append(v if v is not None else np.full(dim, np.nan, dtype=np.float32))
        rows.append({"concept_id": cid, "gloss": g, "rule": rule, "key": key, "vocab_rank": r})
    return np.vstack(vecs), pd.DataFrame(rows), vocab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute even if the cache exists")
    args = ap.parse_args()
    set_all_seeds()
    log_run_header(log)
    if exists(CONDITION) and not args.force:
        log.info("%s cached; skipping (use --force to recompute)", CONDITION)
        return
    order = load_concept_order()
    concepts = read_parquet(config.PROCESSED_DIR / "concepts.parquet").set_index("concept_id")
    glosses = concepts.loc[order, "gloss"]
    path = ensure_file()
    vecs, table, vocab = embed(glosses, path)
    cov = coverage(vecs)
    meta = {"model": config.MODELS[CONDITION]["name"], "source": config.W2V_URL, "sha256": config.W2V_SHA256,
            "vocab_size": vocab, "rule_counts": table["rule"].value_counts().to_dict(), **cov}
    save_embeddings(CONDITION, vecs, order, meta,
                    extras={"vocab_rank": table["vocab_rank"].to_numpy(dtype=np.float64)})
    write_result_table(table, f"embedding_lookup_{CONDITION}", producer=__name__)
    log.info("word2vec: %d/%d concepts embedded; rules %s; missing: %s", cov["n_embedded"], cov["n_concepts"],
             meta["rule_counts"], table.loc[table["rule"] == "missing", "gloss"].tolist())


if __name__ == "__main__":
    main()
