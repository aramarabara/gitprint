"""Lexical style features and obfuscation indicators (deterministic, model-free)."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
HEXISH = re.compile(r"^_?0[xX][0-9a-fA-F]+$|^[0-9a-fA-F]{8,}$")
OCTAL_ESC = re.compile(r"\\x[0-9a-fA-F]{2,4}")
SINGLE_LETTER = re.compile(r"^[a-z]$")
SHORT_IDENT = re.compile(r"^[a-z]{1,3}$")

OBFUS_SIGNATURES = {
    "javascript-obfuscator": re.compile(r"_0x[a-f0-9]{4,8}"),
    "obfuscator-io": re.compile(r"(?:_0x[a-f0-9]{4,8}|String\.fromCharCode)"),
    "hex-escape-chains": re.compile(r"(?:\\x[0-9a-f]{2}){4,}"),
    "eval-chain": re.compile(r"\beval\(|\bFunction\s*\("),
    "fromCharCode": re.compile(r"fromCharCode"),
    "atob-b64": re.compile(r"\batob\("),
    "base64-blob": re.compile(r"[A-Za-z0-9+/]{60,}={0,2}"),
}

LANG_EXT = {
    ".py": "python", ".js": "js", ".mjs": "js", ".cjs": "js", ".ts": "ts",
    ".tsx": "ts", ".jsx": "js", ".go": "go", ".rs": "rust", ".java": "java",
    ".rb": "ruby", ".php": "php", ".c": "c", ".h": "c", ".cpp": "cpp",
    ".cs": "csharp", ".sh": "shell", ".kt": "kotlin", ".swift": "swift",
    ".html": "html", ".css": "css", ".vue": "js", ".svelte": "js",
}


def entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    c = Counter(s)
    return -sum((cnt / n) * math.log2(cnt / n) for cnt in c.values())


def _convention(name: str) -> str | None:
    if "_" in name and name.islower():
        return "snake"
    if "_" in name and name.isupper():
        return "screaming"
    if re.fullmatch(r"[A-Z][a-zA-Z0-9]*", name):
        return "pascal"
    if re.fullmatch(r"[a-z][a-zA-Z0-9]*", name) and any(ch.isupper() for ch in name[1:]):
        return "camel"
    if name.islower():
        return "lower"
    return "other"


@dataclass
class LexicalFeatures:
    """Normalized lexical fingerprint of a code sample."""

    n_idents: int = 0
    avg_ident_len: float = 0.0
    convention_ratio: dict = field(default_factory=lambda: {
        "snake": 0.0, "camel": 0.0, "pascal": 0.0, "screaming": 0.0,
        "lower": 0.0, "other": 0.0,
    })
    short_ident_ratio: float = 0.0
    hex_ident_ratio: float = 0.0
    comment_ratio: float = 0.0
    line_len_mean: float = 0.0
    line_len_std: float = 0.0
    entropy_mean: float = 0.0
    unique_ratio: float = 0.0
    num_strings: int = 0
    string_escape_ratio: float = 0.0
    obfuscation_score: float = 0.0
    obfuscation_hits: dict = field(default_factory=dict)

    def vector(self) -> list[float]:
        """Fixed-order vector for profile baselines."""
        return [
            self.avg_ident_len,
            self.short_ident_ratio,
            self.hex_ident_ratio,
            self.comment_ratio,
            self.unique_ratio,
            self.string_escape_ratio,
            self.convention_ratio.get("snake", 0.0),
            self.convention_ratio.get("camel", 0.0),
            self.convention_ratio.get("pascal", 0.0),
        ]


def _identifiers(text: str) -> list[str]:
    return IDENT.findall(text)


def extract_features(text: str, lang: str | None = None) -> LexicalFeatures:
    idents = _identifiers(text)
    f = LexicalFeatures()
    f.n_idents = len(idents)
    if idents:
        f.avg_ident_len = sum(len(i) for i in idents) / len(idents)
        conv = Counter(_convention(i) for i in idents)
        n = len(idents)
        for k in f.convention_ratio:
            f.convention_ratio[k] = conv.get(k, 0) / n
        f.short_ident_ratio = sum(1 for i in idents if SHORT_IDENT.match(i)) / n
        f.hex_ident_ratio = sum(1 for i in idents if HEXISH.match(i)) / n
        f.unique_ratio = len(set(idents)) / n
        f.entropy_mean = sum(entropy(i) for i in idents) / n

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines:
        lengths = [len(ln) for ln in lines]
        f.line_len_mean = sum(lengths) / len(lengths)
        f.line_len_std = (sum((x - f.line_len_mean) ** 2 for x in lengths) / len(lengths)) ** 0.5
        commentish = 0
        if lang == "python":
            commentish = sum(1 for ln in lines if ln.lstrip().startswith("#"))
        else:
            commentish = sum(1 for ln in lines if ln.lstrip().startswith("//") or ln.lstrip().startswith("*"))
        f.comment_ratio = commentish / len(lines)

    f.num_strings = len(re.findall(r'"([^"\\]|\\.)*"|\'([^\'\\]|\\.)*\'', text))
    escapes = OCTAL_ESC.findall(text)
    f.string_escape_ratio = min(1.0, len(escapes) / max(1, len(escapes) + len(idents)))

    for name, rx in OBFUS_SIGNATURES.items():
        n_hits = len(rx.findall(text))
        if n_hits:
            f.obfuscation_hits[name] = n_hits

    score = 0.0
    if f.hex_ident_ratio > 0.02:
        score += 25
    if f.string_escape_ratio > 0.15:
        score += 20
    if f.avg_ident_len < 4.5:
        score += 15
    if f.short_ident_ratio > 0.4:
        score += 15
    for name, rx in OBFUS_SIGNATURES.items():
        if name in ("javascript-obfuscator", "obfuscator-io", "hex-escape-chains", "eval-chain"):
            if f.obfuscation_hits.get(name, 0) > 0:
                score += 25
    if len(f.obfuscation_hits) >= 2:
        score += 10
    f.obfuscation_score = min(100.0, score)
    return f


def chunk_text(text: str, chunk_chars: int = 3000) -> list[str]:
    """Split code into embeddable chunks at line boundaries."""
    lines = text.splitlines(keepends=True)
    chunks, cur = [], []
    size = 0
    for ln in lines:
        cur.append(ln)
        size += len(ln)
        if size >= chunk_chars:
            chunks.append("".join(cur))
            cur, size = [], 0
    if cur:
        chunks.append("".join(cur))
    return chunks or [""]


def lang_of(path: str) -> str | None:
    ext = path.rsplit(".", 1)[-1].lower()
    return LANG_EXT.get("." + ext) if ext else None
