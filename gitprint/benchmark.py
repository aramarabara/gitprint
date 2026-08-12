"""Benchmark runner — GCJ/Abuhamad-style authorship attribution evaluation.

Behavior:
  1. load dataset/authors/<author>/*.py (>= min_files per author)
  2. per-file train/test split (fixed seed)
  3. evaluate both methods on the same data:
     - run_embedding  : code-embedding centroid cosine (diff-hunk = file split into chunks)
     - run_lexical_lr : lexical 9-d vector + softmax logistic regression
  4. metrics: Top-1 Accuracy / Pair-AUC / FPR@TPR=0.9

Metric meaning (Abuhamad et al. 2018 / Caliskan-Islam et al. 2015 conventions):
  - Top-1 Acc     : fraction where the file's argmax author == true author
  - Pair-AUC      : rank fraction of "same-author pair score > different-author pair score" (Mann-Whitney)
  - FPR@TPR=0.9   : false-positive rate (different author flagged as same) while catching 90% of same-author
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from .embed import embed_batch
from .features import chunk_text, extract_features

CHUNK_CHARS = 3000
MIN_FILES = 8


def load_dataset(dataset_dir: str | Path) -> dict[str, list[Path]]:
    authors_dir = Path(dataset_dir) / "authors"
    out: dict[str, list[Path]] = {}
    for d in sorted(authors_dir.iterdir()):
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.py"))
        if len(files) >= MIN_FILES:
            out[d.name] = files
    return out


def split(author_files: dict[str, list[Path]], seed: int = 42,
          ratio: float = 0.7) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    rng = random.Random(seed)
    train, test = {}, {}
    for a, files in author_files.items():
        fs = list(files)
        rng.shuffle(fs)
        k = max(1, int(len(fs) * ratio))
        train[a], test[a] = fs[:k], fs[k:]
    return train, test


def _read(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="replace")


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / (n + 1e-9)


def run_embedding(train_map: dict[str, list[Path]],
                  test_map: dict[str, list[Path]],
                  skip_obfuscated: bool = True):
    """Code-embedding centroid. Each test file is embedded chunk(hunk)-wise → file-vector mean.

    Returns: (S, y, authors, model, V) — S=per-file author cosine (for closed-set Top-1),
    V=file embedding vectors (for pair/verification).
    """
    authors = sorted(train_map)
    aindex = {a: i for i, a in enumerate(authors)}

    texts, labels = [], []
    for a in authors:
        for fp in train_map[a]:
            body = _read(fp)
            if skip_obfuscated and extract_features(body, "python").obfuscation_score > 60:
                continue
            for ch in chunk_text(body, CHUNK_CHARS):
                texts.append(ch)
                labels.append(a)
    vecs, model = embed_batch(texts)
    agg: dict[str, list[np.ndarray]] = {}
    for a, v in zip(labels, vecs):
        agg.setdefault(a, []).append(np.asarray(v))
    C = _normalize_rows(np.stack([np.mean(agg[a], axis=0) for a in authors]))

    S_rows, V_rows, y_true = [], [], []
    for a in authors:
        for fp in test_map[a]:
            ch_vecs, _ = embed_batch(chunk_text(_read(fp), CHUNK_CHARS))
            V_file = np.mean(np.asarray(ch_vecs), axis=0)
            score = (_normalize_rows(V_file[None, :]) @ C.T)[0]
            S_rows.append(score)
            V_rows.append(V_file)
            y_true.append(aindex[a])
    return np.asarray(S_rows), np.asarray(y_true), authors, model, np.asarray(V_rows)


def _onehot(y: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


def run_lexical_lr(train_map: dict[str, list[Path]],
                   test_map: dict[str, list[Path]],
                   iters: int = 800, lr: float = 0.3):
    """lexical 9-d vector + softmax logistic regression (numpy only)."""
    authors = sorted(train_map)
    aindex = {a: i for i, a in enumerate(authors)}

    X, y = [], []
    for a in authors:
        for fp in train_map[a]:
            X.append(extract_features(_read(fp), "python").vector())
            y.append(aindex[a])
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd

    W = np.zeros((X.shape[1], len(authors)))
    for _ in range(iters):
        logits = Xs @ W
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        grad = Xs.T @ (p - _onehot(y, len(authors))) / len(Xs)
        W -= lr * grad

    S_rows, V_rows, y_true = [], [], []
    for a in authors:
        for fp in test_map[a]:
            x = extract_features(_read(fp), "python").vector()
            xs = (np.asarray(x) - mu) / sd
            logits = xs @ W
            z = logits - logits.max()
            p = np.exp(z)
            S_rows.append(p / p.sum())
            V_rows.append(xs)
            y_true.append(aindex[a])
    return np.asarray(S_rows), np.asarray(y_true), authors, None, np.asarray(V_rows)


def _cos_pairs(V: np.ndarray, y: np.ndarray, seed: int = 1,
               max_pos: int = 800, max_neg: int = 2000):
    """Per-file pair cosine similarity — positive (same author) / negative (different author)."""
    Vn = _normalize_rows(V)
    rng = np.random.RandomState(seed)
    pos: list[float] = []
    for a in np.unique(y):
        idx = np.where(y == a)[0]
        if len(idx) < 2:
            continue
        pairs = [(idx[i], idx[j]) for i in range(len(idx))
                 for j in range(i + 1, len(idx))]
        rng.shuffle(pairs)
        for i, j in pairs[:max_pos]:
            pos.append(float(Vn[i] @ Vn[j]))
    neg: list[float] = []
    tries = 0
    while len(neg) < max_neg and tries < max_neg * 20:
        tries += 1
        i, j = rng.randint(0, len(V)), rng.randint(0, len(V))
        if i == j or y[i] == y[j]:
            continue
        neg.append(float(Vn[i] @ Vn[j]))
    scores = np.concatenate([np.asarray(pos), np.asarray(neg)])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return scores, labels


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _fpr_at_tpr(scores: np.ndarray, labels: np.ndarray,
                tpr_target: float = 0.9) -> tuple[float, float]:
    order = np.argsort(-scores)
    s, l = scores[order], labels[order]
    n_pos, n_neg = l.sum(), len(l) - l.sum()
    if n_pos == 0 or n_neg == 0:
        return 1.0, 0.0
    cum = np.cumsum(l)
    for i in range(len(s)):
        if cum[i] / n_pos >= tpr_target:
            fp = (i + 1) - cum[i]
            return float(fp / n_neg), float(s[i])
    return 1.0, float(s[-1])


def pair_metrics(V: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """same-author pairs > different-author pairs — cosine rank AUC + FPR@TPR=0.9."""
    scores, labels = _cos_pairs(V, y)
    return _roc_auc(scores, labels), _fpr_at_tpr(scores, labels)[0]


def top1_accuracy(S: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(S.argmax(1) == y))


def report(name: str, S: np.ndarray, y: np.ndarray, V: np.ndarray | None = None) -> dict:
    acc = top1_accuracy(S, y)
    print(f"\n== {name}")
    print(f"  Top-1 Accuracy : {acc:.3f}  ({int((S.argmax(1) == y).sum())}/{len(y)} files)")
    if V is not None:
        auc, fpr = pair_metrics(V, y)
        print(f"  Pair-AUC       : {auc:.3f}  (same-author pairs > different-author pairs)")
        print(f"  FPR@TPR=0.9    : {fpr:.3f}")
    return {"method": name, "top1_acc": acc,
            "pair_auc": (pair_metrics(V, y)[0] if V is not None else None),
            "fpr_at_tpr_90": (pair_metrics(V, y)[1] if V is not None else None),
            "n_test": len(y)}
