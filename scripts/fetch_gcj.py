#!/usr/bin/env python3
"""GCJ-solutions dataset collector — one search + per-repo tarball (codeload) download.

api.github.com is used once for the search (repo list). Sources come from
codeload tarballs so the core rate limit is not hit. Saved to a persistent
path (default ~/.gitprint/bench/gcj-dataset).
"""
import io
import json
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.github.com"


def gh(url: str, retries: int = 3) -> dict:
    for i in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": "gitprint-bench"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 403 and i < retries - 1:
                time.sleep(5)
                continue
            print(f"  ! HTTP {e.code}: {url}")
            return {}
        except Exception as e:
            print(f"  ! {e}: {url}")
            return {}
    return {}


def _skip_path(p: str) -> bool:
    low = p.lower()
    if not p.endswith(".py"):
        return True
    if any(x in low for x in ("test", "setup.py", "sample", "/build/",
                              "node_modules", "readme", "__pycache__", ".ipynb")):
        return True
    if low.count("/") > 4:
        return True
    return False


def fetch(search_q: str, out: Path, max_authors: int = 30,
          min_files: int = 8, max_files: int = 25):
    q = urllib.parse.quote(search_q)
    data = gh(f"{API}/search/repositories?q={q}&sort=stars&per_page={max_authors}")
    repos = data.get("items", [])
    print(f"[fetch] {len(repos)} candidate repos")

    authors_dir = out / "authors"
    authors_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    n_authors = 0
    for repo in repos:
        if n_authors >= max_authors:
            break
        full = repo["full_name"]
        owner = repo["owner"]["login"]
        branch = repo.get("default_branch", "main")
        tar_url = f"https://codeload.github.com/{full}/tar.gz/{branch}"
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    tar_url, headers={"User-Agent": "gitprint-bench"}), timeout=60) as r:
                blob = r.read()
        except Exception as e:
            print(f"  ! tarball {full}: {e}")
            time.sleep(0.5)
            continue
        if len(blob) < 1000:
            print(f"  ! tarball {full}: empty")
            continue

        dest = authors_dir / owner
        dest.mkdir(exist_ok=True)
        n = 0
        try:
            tar = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
            for m in sorted(tar.getmembers(), key=lambda m: m.name):
                if not m.isfile():
                    continue
                rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
                if _skip_path(rel):
                    continue
                try:
                    body = tar.extractfile(m).read().decode("utf-8", errors="replace")
                except Exception:
                    continue
                if not body.strip():
                    continue
                (dest / rel.replace("/", "_")).write_text(body)
                n += 1
                if n >= max_files:
                    break
        except Exception as e:
            print(f"  ! extract {full}: {e}")
        if n >= min_files:
            saved[owner] = n
            n_authors += 1
            print(f"  ok {owner}: {n} .py  ({full.split('/')[-1]}@{branch})")
        else:
            print(f"  - {owner}: only {n} (below min {min_files})")
        time.sleep(0.3)

    (out / "manifest.json").write_text(json.dumps(
        {"query": search_q, "authors": saved}, indent=2))
    print(f"[fetch] done — {n_authors} authors -> {out}")
    return n_authors


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1
               else Path.home() / ".gitprint/bench/gcj-dataset")
    max_authors = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    n = fetch("google code jam solutions language:python", out, max_authors=max_authors)
    sys.exit(0 if n >= 8 else 1)
