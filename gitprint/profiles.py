"""Profile building — per-author lexical baselines + embedding centroids."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .embed import embed_batch
from .features import chunk_text, extract_features, lang_of
from .gitmine import author_files


def _slug(repo: Path) -> str:
    name = repo.resolve().name
    try:
        import subprocess
        remote = subprocess.run(
            ["git", "-C", str(repo), "config", "--get", "remote.origin.url"],
            capture_output=True, text=True).stdout.strip()
        if remote:
            name = remote.rstrip(".git").split(":")[-1].split("/")[-1] or name
    except Exception:
        pass
    return name


def build_profile(repo: Path, after: str | None = None,
                  min_samples: int = 3, max_samples_per_author: int = 60,
                  skip_obfuscated: bool = True) -> dict:
    """Mine repo and build {author: baseline + centroid} profile."""
    repo = Path(repo)
    files = author_files(repo, after=after)

    lexical: dict[str, dict] = {}
    centroids: dict[str, list[float]] = {}
    n_samples: dict[str, int] = {}
    sample_texts: list[str] = []
    sample_author: list[str] = []

    for author, paths in sorted(files.items()):
        lex = {k: [] for k in [
            "avg_ident_len", "short_ident_ratio", "hex_ident_ratio",
            "comment_ratio", "unique_ratio", "string_escape_ratio",
            "snake", "camel", "pascal",
        ]}
        vecs = []
        count = 0
        for fp in paths:
            if count >= max_samples_per_author:
                break
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lang = lang_of(fp.name)
            f = extract_features(text, lang)
            if skip_obfuscated and f.obfuscation_score > 60:
                continue
            for k in lex:
                val = f.vector()[list(lex).index(k)]
                lex[k].append(val)
            for chunk in chunk_text(text):
                sample_texts.append(chunk)
                sample_author.append(author)
                count += 1
                if count >= max_samples_per_author:
                    break
        if count < min_samples:
            continue
        for k in lex:
            arr = np.array(lex[k], dtype="float64")
            lexical.setdefault(author, {})[k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()) if arr.std() > 0 else 0.0,
            }
        n_samples[author] = count

    if sample_texts:
        vecs, model = embed_batch(sample_texts)
        agg: dict[str, list[list[float]]] = {}
        for author, v in zip(sample_author, vecs):
            agg.setdefault(author, []).append(v)
        for author, vs in agg.items():
            centroids[author] = (np.mean(vs, axis=0)).tolist()

    return {
        "slug": _slug(repo),
        "repo": str(repo),
        "meta": {
            "authors": sorted(set(sample_author)),
            "samples": sum(n_samples.values()),
            "model": None,
        },
        "lexical": lexical,
        "centroids": centroids,
        "n_samples": n_samples,
    }


def build_blackbox(sample_files: list[Path], min_samples: int = 3) -> dict:
    """Build a pseudo-profile from raw files without git history (one 'unknown' author)."""
    texts: list[str] = []
    lexical: dict[str, dict] = {}
    count = 0
    for fp in sample_files:
        text = fp.read_text(encoding="utf-8", errors="replace")
        lang = lang_of(fp.name)
        f = extract_features(text, lang)
        for chunk in chunk_text(text):
            texts.append(chunk)
            count += 1
    vecs, model = embed_batch(texts)
    return {
        "slug": "blackbox",
        "repo": "blackbox (files)",
        "meta": {"authors": ["unknown"], "samples": count, "model": model},
        "lexical": {},
        "centroids": {"unknown": (np.mean(vecs, axis=0)).tolist()} if vecs else {},
        "n_samples": {"unknown": count},
    }
