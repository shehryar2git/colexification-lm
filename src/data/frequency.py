"""English word frequencies (Zipf scale) from wordfreq's frozen word list.

Reads ``large_en.msgpack.gz`` (wordfreq v3.0.2, pinned commit) directly and
reproduces ``wordfreq.zipf_frequency(word, "en")`` for the glosses used here
(letters and spaces only), following the package source (DECISIONS.md D-50):

* the file is "cBpack": a gzipped msgpack list ``[header, bucket_0, bucket_1, ...]``;
  a word in bucket i has frequency 10 ** (-i / 100);
* a multiword gloss has frequency 1 / f = sum(1 / f_token); a missing token
  gives the minimum (Zipf 0);
* the frequency is rounded to 3 significant digits, then
  Zipf = log10(f) + 9, rounded to 2 decimals.

A test compares against the ``wordfreq`` package whenever it is installed.
"""
from __future__ import annotations

import gzip
import math
import re
import struct
from functools import lru_cache
from pathlib import Path

import config


# ------------------------------------------------------------ msgpack -------
class _Reader:
    """Minimal msgpack decoder: the types that occur in cBpack files."""

    def __init__(self, data: bytes) -> None:
        self.b = data
        self.i = 0

    def take(self, n: int) -> bytes:
        out = self.b[self.i:self.i + n]
        self.i += n
        return out

    def unpack(self):
        t = self.b[self.i]
        self.i += 1
        if t <= 0x7F:
            return t
        if 0x80 <= t <= 0x8F:
            return {self.unpack(): self.unpack() for _ in range(t & 0x0F)}
        if 0x90 <= t <= 0x9F:
            return [self.unpack() for _ in range(t & 0x0F)]
        if 0xA0 <= t <= 0xBF:
            return self.take(t & 0x1F).decode("utf-8")
        if t >= 0xE0:
            return t - 0x100
        simple = {0xC0: None, 0xC2: False, 0xC3: True}
        if t in simple:
            return simple[t]
        fmt = {0xCA: (">f", 4), 0xCB: (">d", 8), 0xCC: (">B", 1), 0xCD: (">H", 2), 0xCE: (">I", 4),
               0xCF: (">Q", 8), 0xD0: (">b", 1), 0xD1: (">h", 2), 0xD2: (">i", 4), 0xD3: (">q", 8)}
        if t in fmt:
            f, n = fmt[t]
            return struct.unpack(f, self.take(n))[0]
        lens = {0xD9: (">B", 1), 0xDA: (">H", 2), 0xDB: (">I", 4)}
        if t in lens:
            f, n = lens[t]
            return self.take(struct.unpack(f, self.take(n))[0]).decode("utf-8")
        arrs = {0xDC: (">H", 2), 0xDD: (">I", 4)}
        if t in arrs:
            f, n = arrs[t]
            return [self.unpack() for _ in range(struct.unpack(f, self.take(n))[0])]
        maps = {0xDE: (">H", 2), 0xDF: (">I", 4)}
        if t in maps:
            f, n = maps[t]
            return {self.unpack(): self.unpack() for _ in range(struct.unpack(f, self.take(n))[0])}
        raise ValueError(f"unsupported msgpack type byte 0x{t:02x} at offset {self.i - 1}")


def read_cbpack(path: Path) -> list:
    with gzip.open(path, "rb") as f:
        data = _Reader(f.read()).unpack()
    header = data[0]
    if not (isinstance(header, dict) and header.get("format") == "cB" and header.get("version") == 1):
        raise ValueError(f"{path} is not a cBpack v1 file: header {header!r}")
    return data[1:]


@lru_cache(maxsize=2)
def frequency_dict(path: str | None = None) -> dict[str, float]:
    buckets = read_cbpack(Path(path) if path else config.WORDFREQ_FILE)
    freqs: dict[str, float] = {}
    for i, words in enumerate(buckets):
        f = 10 ** (-i / 100)
        for w in words:
            freqs.setdefault(w, f)
    return freqs


# ------------------------------------------------------------- frequency ---
def _round_sig3(x: float) -> float:
    lead = math.floor(-math.log(x, 10))
    return round(x, lead + 3)


def zipf(word: str, freqs: dict[str, float] | None = None, minimum: float = 0.0) -> float:
    if not re.fullmatch(r"[A-Za-z][A-Za-z' -]*", word):
        raise ValueError(f"zipf(): only plain English words are supported, got {word!r}")
    freqs = freqs if freqs is not None else frequency_dict()
    floor = 10 ** (minimum - 9)
    tokens = word.lower().replace("-", " ").split()
    one_over = 0.0
    for t in tokens:
        if t not in freqs:
            return round(math.log(floor, 10) + 9, 2)
        one_over += 1.0 / freqs[t]
    f = max(1.0 / one_over, floor)
    return round(math.log(_round_sig3(f), 10) + 9, 2)
