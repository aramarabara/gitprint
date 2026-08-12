"""Git history mining — per-author code ownership via numstat aggregation."""
from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path

from .features import LANG_EXT


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {r.stderr.strip()[:300]}")
    return r.stdout


def _normalize_author(name: str) -> str:
    return name.strip().lower()


def mine_ownership(repo: Path, after: str | None = None) -> dict[str, dict[str, int]]:
    """Return {author: {filepath: added_lines}} based on full numstat history."""
    fmt = "--pretty=format:--%h--%ad--%aN"
    cmd = ["log", "--all", "--numstat", "--date=short", fmt, "--no-renames"]
    if after:
        cmd.append(f"--after={after}")
    raw = _git(repo, *cmd)

    owner: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    cur_author = None
    for line in raw.splitlines():
        if line.startswith("--"):
            parts = line.strip().split("--")
            if len(parts) >= 4:
                cur_author = _normalize_author(parts[3])
            continue
        if cur_author is None:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added, _del, fpath = parts[0], parts[1], parts[2]
        if added == "-" or added.isdigit() is False:
            continue
        ext = fpath.rsplit(".", 1)[-1].lower()
        if f".{ext}" not in LANG_EXT:
            continue
        owner[cur_author][fpath] += int(added)
    return {a: dict(fs) for a, fs in owner.items()}


def author_files(repo: Path, after: str | None = None,
                 min_lines: int = 10) -> dict[str, list[Path]]:
    """Assign each source file to its dominant author (by added lines)."""
    ownership = mine_ownership(repo, after=after)
    by_author: dict[str, list[Path]] = defaultdict(list)
    assigned = set()
    for author, files in ownership.items():
        for fpath, lines in files.items():
            if lines < min_lines:
                continue
            if fpath in assigned:
                continue
            assigned.add(fpath)
            by_author[author].append(Path(repo) / fpath)
    return {a: paths for a, paths in by_author.items() if paths}


def contributors(repo: Path) -> list[str]:
    raw = _git(repo, "shortlog", "-sn", "--all")
    authors = []
    for line in raw.splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2:
            authors.append(parts[1])
    return authors
