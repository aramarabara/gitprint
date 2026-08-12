"""gitprint guard — repository team-style guard (diff -> LoRA fine-tune -> Δloss report).

Supply-chain security use case: does a commit in my repo match the "team style"?
  - fine-tune a team adapter on added lines only (no full repo needed)
  - per commit/file Δloss = base_loss − adapt_loss (team-style fit)
  - Δloss > 0 = team style / near 0·negative = outside (external contributor, malicious)

Flow: mine -> build -> fit(mlx_lm) -> eval -> report(HTML)
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from .features import chunk_text

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-1.5B-4bit"
MIN_ADDED_CHARS = 120


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {r.stderr.strip()[:300]}")
    return r.stdout


def author_commits(repo: Path, team: list[str]) -> dict[str, list[str]]:
    """{author name: [commit shas]} — from the full history."""
    raw = _git(repo, "log", "--format=%H\t%aN", "--all")
    out: dict[str, list[str]] = {}
    for line in raw.splitlines():
        if "\t" not in line:
            continue
        sha, name = line.split("\t", 1)
        out.setdefault(name.strip(), []).append(sha)
    return out


def commit_files(repo: Path, sha: str) -> list[str]:
    """.py file paths modified by this commit."""
    raw = _git(repo, "show", "--name-only", "--format=", sha, "--", "*.py")
    return [p for p in raw.splitlines() if p.strip() and p.endswith(".py")]


def file_at_commit(repo: Path, sha: str, path: str) -> str:
    """Full file content as of that commit."""
    try:
        return _git(repo, "show", f"{sha}:{path}")
    except RuntimeError:
        return ""


def added_lines(repo: Path, sha: str) -> str:
    raw = _git(repo, "show", "--format=", "--unified=0", sha, "--", "*.py")
    lines = []
    for ln in raw.splitlines():
        if ln.startswith("+++") or not ln.startswith("+"):
            continue
        lines.append(ln[1:])
    return "\n".join(lines)


def build_lora_data(repo: Path, team_commits: list[str], out_dir: Path,
                    n_train: int = 120, n_valid: int = 12,
                    chunk_chars: int = 1500, seed: int = 42,
                    min_file_chars: int = 200) -> int:
    """Full content at commit of each file a team commit modified -> {train,valid}.jsonl."""
    rng = random.Random(seed)
    samples: list[dict] = []
    valid_samples: list[dict] = []
    used_files: list[str] = []
    for sha in team_commits:
        for path in commit_files(repo, sha):
            text = file_at_commit(repo, sha, path)
            if len(text) < min_file_chars:
                continue
            chunks = [ch for ch in chunk_text(text, chunk_chars) if ch.strip()]
            for ch in chunks:
                samples.append({"text": ch})
            used_files.append(f"{sha[:8]}:{path}")
            if len(used_files) >= n_train + n_valid:
                break
        if len(used_files) >= n_train + n_valid:
            break
    rng.shuffle(samples)
    valid_samples = samples[:n_valid]
    train = samples[n_valid:]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.jsonl").write_text("\n".join(
        json.dumps(s, ensure_ascii=False) for s in train))
    (out_dir / "valid.jsonl").write_text("\n".join(
        json.dumps(s, ensure_ascii=False) for s in valid_samples))
    (out_dir / "used_files.json").write_text(json.dumps(used_files, indent=2))
    return len(train)


def fit_lora(model: str, data_dir: Path, adapter_path: Path, config: Path,
             iters: int = 300, layers: int = 8, rank: int = 32,
             lr: float = 1e-4, batch: int = 2, quiet: bool = True) -> None:
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model, "--train", "--data", str(data_dir),
        "--fine-tune-type", "lora", "--num-layers", str(layers),
        "--iters", str(iters), "--batch-size", str(batch),
        "--learning-rate", str(lr),
        "--adapter-path", str(adapter_path), "-c", str(config),
        "--seed", "42",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[guard] fine-tuning failed: {r.stderr[-800:]}")
    if not quiet:
        print(r.stdout[-1200:])


# ---------- evaluation ----------

def per_token_ce(model, tok, ids, max_len: int = 512) -> np.ndarray:
    import mlx.core as mx
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


def evaluate(model: str, adapter: Path,
             samples: list[dict], max_len: int = 512,
             backend: str = "auto") -> list[dict]:
    """samples: [{group, name, text}] -> [{group, name, base, adapt, d_mean, d_frac2}]

    backend: 'mlx' | 'llama' | 'auto' (auto = macOS mlx / otherwise llama).
    """
    from .backend import make_backend
    base = make_backend(backend, model)
    adpt = make_backend(backend, model, adapter_path=str(adapter))
    rows = []
    for s in samples:
        ids = base.encode(s["text"])
        if len(ids) < 20:
            continue
        b = base.token_ce(ids, max_len)
        a = adpt.token_ce(ids, max_len)
        rows.append({
            "group": s["group"], "name": s["name"],
            "base_mean": float(b.mean()), "adapt_mean": float(a.mean()),
            "d_mean": float(b.mean() - a.mean()),
            "d_frac2": float((b > 2.0).mean() - (a > 2.0).mean()),
        })
    return rows


def collect_samples(repo: Path, all_commits: dict[str, list[str]],
                    team: list[str], n_eval_team: int = 25,
                    n_eval_other: int = 30,
                    inspect: list[Path] | None = None,
                    min_file_chars: int = 200) -> list[dict]:
    """Eval samples: team/other-contributor commits' files at that point + given external files."""
    rng = random.Random(7)
    team_lower = {t.lower() for t in team}
    samples: list[dict] = []

    def _file_samples(commit_list: list[str], group: str, limit: int) -> None:
        rng.shuffle(commit_list)
        added = 0
        for sha in commit_list:
            for path in commit_files(repo, sha):
                text = file_at_commit(repo, sha, path)
                if len(text) < min_file_chars:
                    continue
                samples.append({"group": group,
                                "name": f"{sha[:8]}:{Path(path).name}", "text": text})
                added += 1
                if added >= limit:
                    return
            if added >= limit:
                return

    team_shas = [s for name, shas in all_commits.items()
                 if any(t in name.lower() for t in team_lower) for s in shas]
    _file_samples(team_shas, "team-holdout", n_eval_team)

    other_shas = [s for name, shas in all_commits.items()
                  if not any(t in name.lower() for t in team_lower) for s in shas]
    _file_samples(other_shas, "other-contributor", n_eval_other)

    for p in inspect or []:
        if p.is_dir():
            for fp in sorted(p.glob("*.py"))[:60]:
                try:
                    text = fp.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if text.strip():
                    samples.append({"group": "external-file", "name": fp.name, "text": text})
        elif p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace")
            samples.append({"group": "external-file", "name": p.name, "text": text})
    return samples


def summarize(rows: list[dict]) -> dict:
    out = {}
    for g in ("team-holdout", "other-contributor", "external-file"):
        rr = [r for r in rows if r["group"] == g]
        if not rr:
            continue
        out[g] = {
            "n": len(rr),
            "d_mean": float(np.mean([r["d_mean"] for r in rr])),
            "d_frac2": float(np.mean([r["d_frac2"] for r in rr])),
        }
    return out


def guard_report(rows: list[dict], repo: Path, team: list[str],
                 out_html: Path, top_n: int = 10) -> str:
    """HTML report — sorted by suspicion + 3-level labels + evidence.

    Labels (Δmean base−adapt):
      🟢 safe          d_mean >  +0.05
      🟡 different style -0.05 <= d_mean <= +0.05  (safe but style slightly differs)
      🔴 suspicious    d_mean <  -0.05
    """
    def verdict(d):
        if d > 0.05:
            return ("🟢", "safe")
        if d >= -0.05:
            return ("🟡", "different style")
        return ("🔴", "suspicious")

    summ = summarize(rows)
    # most suspicious first = lowest Δmean
    ordered = sorted(rows, key=lambda x: x["d_mean"])
    table = ""
    for i, r in enumerate(ordered):
        mark, label = verdict(r["d_mean"])
        strong = ' class="strong"' if i < top_n else ""
        table += (f"<tr{strong}><td>{mark}</td><td>{label}</td>"
                  f"<td>{r['group']}</td><td>{r['name']}</td>"
                  f"<td>{r['base_mean']:.3f}</td><td>{r['adapt_mean']:.3f}</td>"
                  f"<td><b>{r['d_mean']:+.3f}</b></td><td>{r['d_frac2']:+.3f}</td></tr>")
    summ_html = ""
    for g, v in summ.items():
        summ_html += (f"<div class='card'><h3>{g}</h3>"
                      f"<div class='big'>Δmean {v['d_mean']:+.3f}</div>"
                      f"<div>n={v['n']} · Δfrac2 {v['d_frac2']:+.3f}</div></div>")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>gitprint guard — {repo.name}</title>
<style>
body{{font-family:ui-monospace,Menlo,monospace;margin:2rem;background:#0f1117;color:#d6dbe4}}
h1{{font-size:1.3rem}} h2{{font-size:1.05rem;margin-top:2rem}}
.cards{{display:flex;gap:1rem;flex-wrap:wrap}}
.card{{background:#171b24;border:1px solid #2a3040;border-radius:8px;padding:1rem;min-width:180px}}
.big{{font-size:1.5rem;margin:.3rem 0}}
table{{border-collapse:collapse;width:100%;margin-top:1rem;font-size:.82rem}}
th,td{{border:1px solid #2a3040;padding:.35rem .6rem;text-align:left}}
th{{background:#1c2230}} tr:nth-child(even){{background:#131720}}
tr.strong{{background:#2a2030}}
.g{{color:#9aa7bd}}
</style></head><body>
<h1>🧬 gitprint guard — <span class="g">{repo.name}</span></h1>
<p>team: {", ".join(team)} · samples {len(rows)} · Δloss = base − adapt · top {top_n} highlighted</p>
<div class="cards">{summ_html}</div>
<h2>Verdicts (most suspicious first)</h2>
<table><tr><th>verdict</th><th>label</th><th>group</th><th>name</th><th>base</th><th>adapt</th><th>Δmean</th><th>Δfrac2</th></tr>{table}</table>
</body></html>"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html)
    return str(out_html)
