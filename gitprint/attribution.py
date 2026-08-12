"""Attribution — nearest-author scoring by embedding cosine + lexical distance."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .embed import embed_batch
from .features import chunk_text, extract_features, lang_of
from .profiles import build_blackbox


def _norm(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype="float64")
    n = np.linalg.norm(a)
    return a / n if n else a


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return float(np.dot(_norm(a), _norm(b)))


def _lex_sim(feat, baseline: dict) -> float:
    """Similarity in [0,1] between a sample's features and an author baseline."""
    keys = ["avg_ident_len", "short_ident_ratio", "hex_ident_ratio",
            "comment_ratio", "unique_ratio", "string_escape_ratio",
            "snake", "camel", "pascal"]
    values = feat.vector()
    dists = []
    for k, x in zip(keys, values):
        b = baseline.get(k)
        if not b or b.get("std", 0) == 0:
            continue
        z = abs(x - b["mean"]) / b["std"]
        dists.append(1.0 / (1.0 + z))
    if not dists:
        return 0.0
    return sum(dists) / len(dists)


def attribute_file(path: Path, profile: dict,
                   embed_weight: float = 0.6) -> list[dict]:
    """Score a source file against every author in the profile."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lang = lang_of(path.name)
    feat = extract_features(text, lang)

    results = []
    centroids = profile.get("centroids", {})
    if centroids:
        vecs, _model = embed_batch(chunk_text(text))
        file_vec = np.mean(vecs, axis=0).tolist()
    else:
        file_vec = None

    for author in profile.get("n_samples", {}):
        score = 0.0
        parts = {}
        if file_vec is not None and author in centroids:
            c = _cosine(file_vec, centroids[author])
            parts["embedding"] = c
            score += embed_weight * c
        if author in profile.get("lexical", {}):
            l = _lex_sim(feat, profile["lexical"][author])
            parts["lexical"] = l
            score += (1 - embed_weight) * l
        results.append({
            "author": author,
            "score": round(score, 4),
            "parts": {k: round(v, 4) for k, v in parts.items()},
            "n_samples": profile["n_samples"].get(author, 0),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def attribute_snippet(text: str, profile: dict,
                      embed_weight: float = 0.6) -> list[dict]:
    """Score a raw code snippet against every author."""
    feat = extract_features(text, "python" if not any(
        c in text for c in ("{", "}", ";" )) else "js")
    results = []
    centroids = profile.get("centroids", {})
    if centroids:
        vecs, _model = embed_batch(chunk_text(text))
        file_vec = np.mean(vecs, axis=0).tolist()
    else:
        file_vec = None
    for author in profile.get("n_samples", {}):
        score, parts = 0.0, {}
        if file_vec is not None and author in centroids:
            c = _cosine(file_vec, centroids[author])
            parts["embedding"] = c
            score += embed_weight * c
        if author in profile.get("lexical", {}):
            l = _lex_sim(feat, profile["lexical"][author])
            parts["lexical"] = l
            score += (1 - embed_weight) * l
        results.append({
            "author": author, "score": round(score, 4),
            "parts": {k: round(v, 4) for k, v in parts.items()},
            "n_samples": profile["n_samples"].get(author, 0),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def best_match(path: Path, profile: dict) -> str:
    res = attribute_file(path, profile)
    return res[0]["author"] if res else "unknown"
