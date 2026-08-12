"""Smell / drift checks and obfuscation triage (deterministic)."""
from __future__ import annotations

from pathlib import Path

from .attribution import _lex_sim
from .features import extract_features, lang_of

WARN_FEATURES = {
    "avg_ident_len": "average identifier length too short (obfuscation/minify signal)",
    "short_ident_ratio": "short identifier ratio too high",
    "hex_ident_ratio": "hex identifier ratio too high",
    "comment_ratio": "comment density abnormal",
    "string_escape_ratio": "string escape density too high",
    "unique_ratio": "token repetition rate abnormal",
}

KEYS = ["avg_ident_len", "short_ident_ratio", "hex_ident_ratio",
        "comment_ratio", "unique_ratio", "string_escape_ratio",
        "snake", "camel", "pascal"]


def _baseline_row(k: str, x: float, baseline: dict | None) -> dict:
    if not baseline or k not in baseline:
        return {"feature": k, "value": round(x, 4), "baseline": None,
                "z": None, "anomaly": False, "note": "no baseline"}
    b = baseline[k]
    std = b["std"] if b["std"] and b["std"] > 0 else (abs(b["mean"]) * 0.1 or 0.001)
    z = abs(x - b["mean"]) / std
    return {
        "feature": k, "value": round(x, 4), "baseline": round(b["mean"], 4),
        "z": round(z, 2), "anomaly": z > 2,
        "note": WARN_FEATURES.get(k) if z > 2 else None,
    }


def check_file(path: Path, profile: dict | None) -> tuple[list[dict], str | None, float]:
    """Z-score the file against the best-matching author baseline."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    feat = extract_features(text, lang_of(path.name))
    vec = feat.vector()

    best_author, best_sim = None, 0.0
    if profile and profile.get("lexical"):
        for author, lex in profile["lexical"].items():
            sim = _lex_sim(feat, lex)
            if sim > best_sim:
                best_author, best_sim = author, sim

    baseline = (profile or {}).get("lexical", {}).get(best_author or "", {}) or None
    rows = [_baseline_row(k, v, baseline) for k, v in zip(KEYS, vec)]
    return rows, best_author, best_sim


def obfuscate_file(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    f = extract_features(text, lang_of(path.name))
    level = "normal"
    if f.obfuscation_score >= 70:
        level = "high"
    elif f.obfuscation_score >= 45:
        level = "suspicious"
    return {
        "file": str(path),
        "score": round(f.obfuscation_score, 1),
        "level": level,
        "signatures": f.obfuscation_hits,
        "indicators": {
            "hex_ident_ratio": round(f.hex_ident_ratio, 4),
            "short_ident_ratio": round(f.short_ident_ratio, 4),
            "avg_ident_len": round(f.avg_ident_len, 2),
            "string_escape_ratio": round(f.string_escape_ratio, 4),
        },
        "lines": len(text.splitlines()),
    }


def scan_directory(root: Path, recursive: bool = True) -> list[dict]:
    it = root.rglob("*") if recursive else root.glob("*")
    out = []
    for p in it:
        if p.is_file() and lang_of(p.name):
            try:
                out.append(obfuscate_file(p))
            except OSError:
                continue
    out.sort(key=lambda d: d["score"], reverse=True)
    return out
