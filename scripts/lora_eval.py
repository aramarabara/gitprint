#!/usr/bin/env python3
"""Δloss evaluation — per-token CE of base vs a single adapter, grouped (internal/external).

Usage: python lora_eval.py --model <hf> --adapter <dir> --eval-json <file>
  eval.json: {"holdout": {author: [paths]}, "external": {author: [paths]}}
Output: per-group mean/tail (entropy amplification) statistics + per-file Δloss.
"""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

MAX_LEN = 512


def per_token_ce(model, tok, ids, max_len: int = MAX_LEN) -> np.ndarray:
    ids = ids[: max_len + 1]
    toks = mx.array(ids)[None, :]
    logits = model(toks)
    if isinstance(logits, dict):
        logits = logits.get("logits") or logits["lm_logits"]
    logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    labels = mx.array(ids[1:])[:, None]
    m = logits.max(axis=-1, keepdims=True)
    lse = m + mx.log(mx.exp(logits - m).sum(axis=-1, keepdims=True))
    logit_l = mx.take_along_axis(logits, labels, axis=-1)
    ce = (lse - logit_l)[:, 0]
    out = np.asarray(ce, dtype="float64")
    del logits, m, lse, logit_l, ce
    return out


def _stats(arr: np.ndarray) -> dict:
    return {
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "frac2": float((arr > 2.0).mean()),
        "frac4": float((arr > 4.0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-Coder-1.5B-4bit")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--max-files-per-author", type=int, default=100)
    args = ap.parse_args()

    base, tok = load(args.model)
    adpt, _ = load(args.model, adapter_path=args.adapter)
    evalm = json.loads(Path(args.eval_json).read_text())

    rows = []
    for group in ("holdout", "external"):
        for author, files in evalm[group].items():
            for fp in files[: args.max_files_per_author]:
                ids = tok.encode(Path(fp).read_text(encoding="utf-8", errors="replace"))
                if len(ids) < 20:
                    continue
                b = per_token_ce(base, tok, ids)
                a = per_token_ce(adpt, tok, ids)
                rows.append({
                    "group": "HOLDOUT(internal,unseen)" if group == "holdout" else "EXTERNAL",
                    "author": author, "name": Path(fp).name,
                    "base": _stats(b), "adapt": _stats(a),
                    "d_mean": float(b.mean() - a.mean()),
                    "d_frac2": float((b > 2.0).mean() - (a > 2.0).mean()),
                    "d_p90": float(np.percentile(b, 90) - np.percentile(a, 90)),
                })
                print(f"  {rows[-1]['group'][:20]:20s} {author[:14]:14s} "
                      f"{rows[-1]['name'][:32]:32s} Δmean={rows[-1]['d_mean']:+.3f} "
                      f"Δfrac2={rows[-1]['d_frac2']:+.3f}", file=sys.stderr)

    print("\n== Δloss group summary")
    for group in ("HOLDOUT(internal,unseen)", "EXTERNAL"):
        rr = [r for r in rows if r["group"] == group]
        if not rr:
            continue
        print(f"\n  [{group}] {len(rr)} files")
        print(f"    Δmean          : {np.mean([r['d_mean'] for r in rr]):+.4f}")
        print(f"    Δfrac(ent>2.0) : {np.mean([r['d_frac2'] for r in rr]):+.4f}")
        print(f"    base  mean/frac2 : {np.mean([r['base']['mean'] for r in rr]):.3f} / "
              f"{np.mean([r['base']['frac2'] for r in rr]):.3f}")
        print(f"    adapt mean/frac2 : {np.mean([r['adapt']['mean'] for r in rr]):.3f} / "
              f"{np.mean([r['adapt']['frac2'] for r in rr]):.3f}")

    # Guard verdict: scan Δmean threshold -> FPR@TPR=0.9
    pos = [r["d_mean"] for r in rows if r["group"] == "HOLDOUT(internal,unseen)"]
    neg = [r["d_mean"] for r in rows if r["group"] == "EXTERNAL"]
    if pos and neg:
        scores = np.array(pos + neg)
        labels = np.array([1] * len(pos) + [0] * len(neg))
        order = np.argsort(-scores)
        s, l = scores[order], labels[order]
        cum = np.cumsum(l)
        n_pos, n_neg = l.sum(), len(l) - l.sum()
        for i in range(len(s)):
            if cum[i] / n_pos >= 0.9:
                fp = (i + 1) - cum[i]
                print(f"\n  [guard] FPR@TPR=0.9 : {fp / n_neg:.3f} (Δmean threshold {s[i]:+.4f})")
                break
        print(f"  [guard] positive={len(pos)} negative={len(neg)}")


if __name__ == "__main__":
    main()
