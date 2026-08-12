#!/usr/bin/env python3
"""Prepare LoRA fine-tuning data for the GCJ benchmark — per-author {train,valid}.jsonl."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gitprint.benchmark import load_dataset, split
from gitprint.features import chunk_text

AUTHORS = ["theXYZT", "kamyu104", "KirarinSnow", "blerou"]
CHUNK_CHARS = 1500
SEED = 42


def main(dataset: str, out_root: str):
    data = load_dataset(dataset)
    train, test = split(data, seed=SEED, ratio=0.7)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    (out / "split.json").write_text(json.dumps({
        a: [p.name for p in v] for a, v in {**{f"train-{a}": v for a, v in train.items()},
                                            **{f"test-{a}": v for a, v in test.items()}}.items()},
        indent=2))

    for a in AUTHORS:
        if a not in train:
            continue
        files = train[a]
        rng = random.Random(SEED)
        samples = []
        for fp in files:
            body = fp.read_text(encoding="utf-8", errors="replace")
            for ch in chunk_text(body, CHUNK_CHARS):
                if ch.strip():
                    samples.append({"text": ch})
        rng.shuffle(samples)
        n_valid = max(2, int(len(samples) * 0.1))
        valid, t = samples[:n_valid], samples[n_valid:]
        if len(t) < 10:
            valid, t = t[: max(1, len(t) // 5)], t[len(t) // 5:]
        adir = out / a
        adir.mkdir(exist_ok=True)
        (adir / "train.jsonl").write_text("\n".join(
            json.dumps(s, ensure_ascii=False) for s in t))
        (adir / "valid.jsonl").write_text("\n".join(
            json.dumps(s, ensure_ascii=False) for s in valid))
        print(f"{a}: train={len(t)} valid={len(valid)} samples")
    print(f"[lora-data] done -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/gcj-dataset",
         sys.argv[2] if len(sys.argv) > 2 else "/tmp/lora-data")
