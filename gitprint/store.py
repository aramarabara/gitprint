"""Profile persistence — JSON (lexical baselines) + .npy (embedding centroids)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Store:
    def __init__(self, base: Path):
        self.base = Path(base)
        self.base.mkdir(parents=True, exist_ok=True)

    def save(self, repo_slug: str, lexical: dict, centroids: dict[str, list[float]],
             n_samples: dict[str, int], meta: dict):
        name = repo_slug.replace("/", "__")
        (self.base / f"{name}.json").write_text(json.dumps({
            "meta": meta,
            "lexical": lexical,
            "n_samples": n_samples,
        }, indent=2, ensure_ascii=False))
        if centroids:
            np.save(self.base / f"{name}.npy", {
                "authors": list(centroids),
                "vecs": np.array([centroids[a] for a in centroids], dtype="float32"),
            })
        return self.base / f"{name}.json"

    def load(self, repo_slug: str) -> dict:
        name = repo_slug.replace("/", "__")
        json_p = self.base / f"{name}.json"
        npy_p = self.base / f"{name}.npy"
        if not json_p.exists():
            raise FileNotFoundError(f"No profile: {json_p}")
        data = json.loads(json_p.read_text())
        data["centroids"] = {}
        if npy_p.exists():
            arr = np.load(npy_p, allow_pickle=True).item()
            data["centroids"] = {
                a: v.tolist() for a, v in zip(arr["authors"], arr["vecs"])
            }
        return data

    def list(self) -> list[Path]:
        return sorted(self.base.glob("*.json"))
