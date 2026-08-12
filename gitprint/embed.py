"""Embedding backend with a lazy fallback chain.

1. jinaai/jina-embeddings-v2-base-code — code-aware sentence transformer
2. microsoft/codebert-base       — BERT for code (mean pooling)
3. sentence-transformers/all-MiniLM-L6-v2 — tiny, fast, generic

Models are downloaded on first use and cached. Set GITPRINT_MODEL to pin one.
The "model: ..." notice is printed once per model (first actual load), then
quiet — a marker file under ~/.gitprint records what has been announced.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Keep stdout/stderr clean: suppress tqdm/progress bars from the model libs.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

MODELS = [
    "jinaai/jina-embeddings-v2-base-code",
    "microsoft/codebert-base",
    "sentence-transformers/all-MiniLM-L6-v2",
]

_MARKER_DIR = Path.home() / ".gitprint"
_SINGLETON = {"model": None, "name": None}


def _announce_once(name: str) -> None:
    """Print the model notice only the first time this model is loaded."""
    marker = _MARKER_DIR / f"embed-announced-{name.replace('/', '__')}"
    if marker.exists():
        return
    _MARKER_DIR.mkdir(parents=True, exist_ok=True)
    marker.touch()
    print(f"[embed] model: {name}", file=sys.stderr)


def _load_sbert(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def _embed_codebert(name: str, texts: list[str]) -> "list[list[float]]":
    from transformers import AutoModel, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    out = []
    for t in texts:
        enc = tok(t, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1)
        out.append(pooled[0].tolist())
    return out


def embed_batch(texts: list[str]) -> tuple[list[list[float]], str]:
    """Return (vectors, model_name). Lazily loads the first usable model."""
    if _SINGLETON["model"] is not None:
        return _SINGLETON["model"](texts), _SINGLETON["name"]

    pinned = os.environ.get("GITPRINT_MODEL")
    candidates = [pinned] if pinned else MODELS
    last_err = None
    for name in candidates:
        if not name:
            continue
        try:
            if "codebert" in name:
                vecs = _embed_codebert(name, texts)
                _SINGLETON.update(model=lambda t: _embed_codebert(name, t), name=name)
            else:
                m = _load_sbert(name)
                _SINGLETON.update(
                    model=lambda t: m.encode(t, normalize_embeddings=True).tolist(),
                    name=name)
            _announce_once(name)
            return vecs, name
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"failed to load any embedding model (all {candidates}): {last_err}")
