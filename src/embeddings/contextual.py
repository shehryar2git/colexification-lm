"""Phase 4 · Contextual embeddings: BERT-base (English), XLM-R base, LaBSE (multilingual).

Pooling (DECISIONS.md D-13, D-14):

* BERT and XLM-R (``kind = contextual``): each gloss is placed in the ten
  ``config.TEMPLATES``; the vector is the **mean of the hidden states of the
  gloss's own sub-tokens** (located through the fast tokenizer's character
  offsets), taken from the final layer and averaged over templates. The mean
  over templates of *every* layer is also stored for the layer sweep.
* LaBSE (``kind = sentence``): the model's trained, normalised sentence
  embedding of each templated sentence (sentence-transformers), averaged over
  templates.

Per-template vectors are stored for the template-sensitivity analysis
(Phase 5) and template 0 (the bare word) gives the English bare-word baseline
for the multilingual-input condition (D-17).

The numerical core (``render``, ``pool_target``, ``embed_contextual``,
``embed_sentences``) is plain numpy and is tested with a fake encoder; only
``HFTokenEncoder`` / ``STSentenceEncoder`` need torch. Without torch the step
logs PENDING USER RUN and exits cleanly.

Outputs: data/processed/embeddings/{bert_base,xlmr_base,labse}.npz,
results/model_manifest.json (resolved model revisions and library versions).
"""
from __future__ import annotations

import argparse
import json
from typing import Callable, Protocol, Sequence

import numpy as np

import config
from src.embeddings.store import coverage, exists, save_embeddings
from src.utils.concept_order import load_concept_order
from src.utils.io import read_parquet
from src.utils.logging_utils import get_logger, log_run_header, package_versions
from src.utils.seeds import set_all_seeds

log = get_logger(__name__, "phase4_contextual")
CONTEXTUAL = ("bert_base", "xlmr_base", "labse")
MANIFEST = config.RESULTS_DIR / "model_manifest.json"


# --------------------------------------------------------------- numerics ---
def render(template: str, word: str) -> tuple[str, tuple[int, int]]:
    """Fill a template; return the text and the character span of the word."""
    if template.count("{w}") != 1:
        raise ValueError(f"template must contain exactly one {{w}}: {template!r}")
    start = template.index("{w}")
    text = template.replace("{w}", word)
    return text, (start, start + len(word))


def pool_target(hidden: np.ndarray, offsets: Sequence[tuple[int, int]], span: tuple[int, int]) -> np.ndarray:
    """Mean over the tokens whose character offsets overlap ``span``.

    ``hidden`` is (n_layers, seq_len, dim) or (seq_len, dim). Special tokens
    have empty offsets (s == e) and are never selected."""
    s0, e0 = span
    idx = [i for i, (s, e) in enumerate(offsets) if e > s and s < e0 and e > s0]
    if not idx:
        raise ValueError(f"no token overlaps span {span}; offsets={list(offsets)[:12]}")
    return hidden[..., idx, :].mean(axis=-2)


class TokenEncoder(Protocol):
    def __call__(self, texts: list[str]) -> list[tuple[np.ndarray, list[tuple[int, int]]]]:
        """For each text: (hidden states (n_layers, seq_len, dim), token character offsets)."""


def embed_contextual(words: Sequence[str], templates: Sequence[str], encode: TokenEncoder,
                     layer: int = config.CONTEXTUAL_LAYER, batch_size: int = config.EMBED_BATCH_SIZE
                     ) -> dict[str, np.ndarray]:
    """Target-token pooling over templates.

    Returns ``vectors`` (n_words, dim) = mean over templates at ``layer``;
    ``per_template`` (n_templates, n_words, dim) at ``layer``;
    ``per_layer`` (n_layers, n_words, dim) = mean over templates at every layer."""
    jobs = [(t, w) for t in range(len(templates)) for w in range(len(words))]
    pooled: dict[tuple[int, int], np.ndarray] = {}
    for b in range(0, len(jobs), batch_size):
        batch = jobs[b:b + batch_size]
        rendered = [render(templates[t], words[w]) for t, w in batch]
        outs = encode([text for text, _ in rendered])
        for (t, w), (_, span), (hidden, offsets) in zip(batch, rendered, outs):
            pooled[(t, w)] = pool_target(np.asarray(hidden, dtype=np.float64), offsets, span)
    n_layers, dim = pooled[(0, 0)].shape
    per = np.stack([[pooled[(t, w)] for w in range(len(words))] for t in range(len(templates))])  # T,N,L,D
    return {"vectors": per[:, :, layer, :].mean(axis=0),
            "per_template": per[:, :, layer, :],
            "per_layer": per.mean(axis=0).transpose(1, 0, 2)}


def embed_sentences(words: Sequence[str], templates: Sequence[str],
                    encode: Callable[[list[str]], np.ndarray]) -> dict[str, np.ndarray]:
    """Sentence-encoder pooling: mean over templates of the sentence embeddings."""
    per = np.stack([np.asarray(encode([render(t, w)[0] for w in words]), dtype=np.float64) for t in templates])
    return {"vectors": per.mean(axis=0), "per_template": per}


# ------------------------------------------------------------- real models --
def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class HFTokenEncoder:
    """Hugging Face encoder returning all hidden states and fast-tokenizer offsets."""

    def __init__(self, name: str, revision: str = "main") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True)
        if not self.tok.is_fast:
            raise RuntimeError(f"{name}: a fast tokenizer is required for character offsets")
        self.model = AutoModel.from_pretrained(name, revision=revision, output_hidden_states=True).eval()
        self.model.to(device())
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision

    def __call__(self, texts: list[str]) -> list[tuple[np.ndarray, list[tuple[int, int]]]]:
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=config.MAX_SEQ_LEN,
                       return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping").tolist()
        enc = {k: v.to(device()) for k, v in enc.items()}
        with self.torch.no_grad():
            hs = self.model(**enc).hidden_states            # tuple of (B, S, D), embeddings layer first
        hs = self.torch.stack(hs, dim=1).float().cpu().numpy()   # B, L, S, D
        mask = enc["attention_mask"].cpu().numpy().astype(bool)
        out = []
        for i in range(len(texts)):
            keep = mask[i]
            offs = [tuple(o) if keep[j] else (0, 0) for j, o in enumerate(offsets[i])]
            out.append((hs[i], offs))
        return out


class STSentenceEncoder:
    def __init__(self, name: str, revision: str = "main") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, revision=revision, device=device())
        cfg = getattr(self.model[0], "auto_model", None)
        self.revision = getattr(getattr(cfg, "config", None), "_commit_hash", None) or revision

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=config.EMBED_BATCH_SIZE, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)


def load_encoder(condition: str):
    spec = config.MODELS[condition]
    cls = STSentenceEncoder if spec["kind"] == "sentence" else HFTokenEncoder
    return cls(spec["name"], spec.get("revision", "main"))


def update_manifest(condition: str, info: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    data[condition] = info
    MANIFEST.write_text(json.dumps(data, indent=2, default=str) + "\n")


def run_condition(condition: str, words: list[str], order: list[str]) -> None:
    spec = config.MODELS[condition]
    enc = load_encoder(condition)
    if spec["kind"] == "sentence":
        res = embed_sentences(words, config.TEMPLATES, enc)
    else:
        res = embed_contextual(words, config.TEMPLATES, enc)
    cov = coverage(res["vectors"])
    meta = {"model": spec["name"], "requested_revision": spec.get("revision", "main"),
            "resolved_revision": enc.revision, "kind": spec["kind"], "templates": list(config.TEMPLATES),
            "layer": config.CONTEXTUAL_LAYER if spec["kind"] != "sentence" else "sentence-pooled", **cov}
    extras = {k: v.astype(np.float32) for k, v in res.items() if k != "vectors"}
    save_embeddings(condition, res["vectors"], order, meta, extras)
    import torch
    update_manifest(condition, {**meta, "device": device(), "torch": torch.__version__,
                                "packages": package_versions()})
    log.info("%s: %d/%d embedded (revision %s)", condition, cov["n_embedded"], cov["n_concepts"], enc.revision)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(CONTEXTUAL), choices=list(CONTEXTUAL))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    set_all_seeds()
    log_run_header(log)
    todo = [m for m in args.models if args.force or not exists(m)]
    if not todo:
        log.info("All requested contextual conditions are cached.")
        return
    if not torch_available():
        log.warning("PENDING USER RUN: torch/transformers not installed; %s not embedded. "
                    "Install requirements.txt and run `python run.py embeddings`.", ", ".join(todo))
        return
    order = load_concept_order()
    concepts = read_parquet(config.PROCESSED_DIR / "concepts.parquet").set_index("concept_id")
    words = concepts.loc[order, "gloss"].tolist()
    for m in todo:
        run_condition(m, words, order)


if __name__ == "__main__":
    main()
