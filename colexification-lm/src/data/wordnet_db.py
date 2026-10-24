"""Minimal, dependency-free reader for the Princeton WordNet 3.0 database files.

Reads ``index.<pos>`` / ``data.<pos>`` / ``<pos>.exc`` straight from the
WordNet zip (DECISIONS.md D-34), implementing the documented WNdb format
(wndb(5WN)) and the morphy base-form rules (morphy(7WN)). It covers exactly
what this project needs: lemma -> synsets, synset -> lemmas + definition.

Synset IDs are written ``<offset>-<pos>`` (e.g. ``05564590-n``), the format used
by the Open Multilingual Wordnet, so every audit entry can be checked by hand
in any WordNet browser.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import config

POS_FILES = {"n": "noun", "v": "verb", "a": "adj", "r": "adv"}

# morphy(7WN) detachment rules
MORPHY_RULES = {
    "n": [("s", ""), ("ses", "s"), ("xes", "x"), ("zes", "z"), ("ches", "ch"),
          ("shes", "sh"), ("men", "man"), ("ies", "y")],
    "v": [("s", ""), ("ies", "y"), ("es", "e"), ("es", ""), ("ed", "e"), ("ed", ""),
          ("ing", "e"), ("ing", "")],
    "a": [("er", ""), ("est", ""), ("er", "e"), ("est", "e")],
    "r": [],
}


@dataclass(frozen=True)
class Synset:
    id: str
    pos: str
    lemmas: tuple[str, ...]
    definition: str
    proper: bool = False   # instance / proper-noun synset (capitalised words, e.g. 'Natalie_Wood')


class WordNet:
    def __init__(self, zip_path: Path | None = None) -> None:
        zip_path = Path(zip_path or config.WORDNET_ZIP)
        if not zip_path.exists():
            raise FileNotFoundError(f"{zip_path} missing — run `python run.py fetch`.")
        self.index: dict[str, dict[str, tuple[str, ...]]] = {p: {} for p in POS_FILES}
        self.synsets: dict[str, Synset] = {}
        self.exceptions: dict[str, dict[str, tuple[str, ...]]] = {p: {} for p in POS_FILES}
        with zipfile.ZipFile(zip_path) as zf:
            names = {Path(n).name: n for n in zf.namelist()}
            for pos, stem in POS_FILES.items():
                self._read_index(zf.read(names[f"index.{stem}"]).decode("utf-8"), pos)
                self._read_data(zf.read(names[f"data.{stem}"]).decode("utf-8"), pos)
                self._read_exc(zf.read(names[f"{stem}.exc"]).decode("utf-8"), pos)

    # ---------------------------------------------------------------- parsing
    def _read_index(self, text: str, pos: str) -> None:
        for line in text.splitlines():
            if not line or line.startswith(" "):        # licence header lines start with spaces
                continue
            f = line.split()
            lemma, synset_cnt, p_cnt = f[0], int(f[2]), int(f[3])
            offsets = f[4 + p_cnt + 2: 4 + p_cnt + 2 + synset_cnt]
            # adjective satellites live in data.adj with ss_type 's'; id keeps pos 'a'
            self.index[pos][lemma] = tuple(f"{o}-{pos}" for o in offsets)

    def _read_data(self, text: str, pos: str) -> None:
        for line in text.splitlines():
            if not line or line.startswith(" "):
                continue
            head, _, gloss = line.partition(" | ")
            f = head.split()
            offset, w_cnt = f[0], int(f[3], 16)
            words = []
            proper = False
            for i in range(w_cnt):
                w = f[4 + 2 * i]
                proper = proper or w[:1].isupper()
                if pos == "a" and w.endswith(")") and "(" in w:   # strip adjective markers e.g. (p)
                    w = w[: w.index("(")]
                words.append(w.lower())
            definition = gloss.split(";")[0].strip().strip('"')
            sid = f"{offset}-{pos}"
            self.synsets[sid] = Synset(sid, pos, tuple(words), definition, proper)

    def _read_exc(self, text: str, pos: str) -> None:
        for line in text.splitlines():
            f = line.split()
            if len(f) >= 2:
                self.exceptions[pos][f[0]] = tuple(f[1:])

    # ----------------------------------------------------------------- lookup
    def morphy(self, word: str, pos: str) -> list[str]:
        """Base forms of ``word`` that exist in the index for ``pos``."""
        w = word.lower().replace(" ", "_")
        idx = self.index[pos]
        found: list[str] = []
        for cand in (w, *self.exceptions[pos].get(w, ())):
            if cand in idx and cand not in found:
                found.append(cand)
        if found:
            return found
        for suffix, repl in MORPHY_RULES[pos]:
            if w.endswith(suffix):
                cand = w[: len(w) - len(suffix)] + repl
                if cand in idx and cand not in found:
                    found.append(cand)
        return found

    def synsets_for(self, word: str, include_proper: bool = False) -> list[Synset]:
        out: list[Synset] = []
        seen: set[str] = set()
        for pos in POS_FILES:
            for base in self.morphy(word, pos):
                for sid in self.index[pos][base]:
                    if sid not in seen:
                        seen.add(sid)
                        if include_proper or not self.synsets[sid].proper:
                            out.append(self.synsets[sid])
        return out


@lru_cache(maxsize=1)
def load_wordnet() -> WordNet:
    return WordNet()
