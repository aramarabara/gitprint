"""gitprint CLI — Code Provenance Intelligence."""
from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
from pathlib import Path

from .attribution import attribute_file, attribute_snippet
from .profiles import build_blackbox, build_profile
from .smell import check_file, obfuscate_file, scan_directory
from .store import Store

DEFAULT_PROFILE_DIR = Path.home() / ".gitprint" / "profiles"


def _store(base: str | None) -> Store:
    return Store(Path(base) if base else DEFAULT_PROFILE_DIR)


def cmd_profile(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gitprint profile")
    ap.add_argument("repo")
    ap.add_argument("--after", default=None, help="only commits after YYYY-MM-DD")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--no-skip-obfuscated", action="store_true")
    args = ap.parse_args(argv)
    profile = build_profile(
        Path(args.repo), after=args.after,
        min_samples=args.min_samples,
        skip_obfuscated=not args.no_skip_obfuscated)
    p = _store(args.out).save(
        profile["slug"], profile["lexical"], profile["centroids"],
        profile["n_samples"], profile["meta"])
    print(f"[profile] {profile['slug']} — saved: {p}")
    print(f"  authors: {profile['meta']['authors']}")
    print(f"  samples: {profile['meta']['samples']}")
    for a in profile["meta"]["authors"]:
        print(f"    {a}: {profile['n_samples'].get(a, 0)} chunks")
    return 0


def cmd_blackbox(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gitprint blackbox")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--name", default="blackbox")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    profile = build_blackbox([Path(f) for f in args.files])
    profile["slug"] = args.name
    p = _store(args.out).save(
        args.name, profile["lexical"], profile["centroids"],
        profile["n_samples"], profile["meta"])
    print(f"[blackbox] {args.name} — saved: {p} (samples={profile['meta']['samples']})")
    return 0


def _load(store: Store, slug: str | None) -> dict:
    profiles = store.list()
    if not profiles:
        sys.exit(f"No profiles — run `gitprint profile <repo>` first (dir: {store.base})")
    if slug:
        slug = slug.replace("/", "__")
        p = store.base / f"{slug}.json"
        if not p.exists():
            sys.exit(f"No profile: {p}\navailable: {[q.name for q in profiles]}")
        return store.load(slug.replace("__", "/"))
    if len(profiles) > 1:
        sys.exit(f"--profile required. available: {[q.stem for q in profiles]}")
    return store.load(profiles[0].stem)


def cmd_who(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gitprint who")
    ap.add_argument("target", help="source file or `snippet:` prefixed code")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args(argv)
    store = _store(args.out)
    profile = _load(store, args.profile)

    if args.target.startswith("snippet:"):
        results = attribute_snippet(args.target[len("snippet:"):], profile)
    else:
        fp = Path(args.target)
        if not fp.exists():
            sys.exit(f"No such file: {fp}")
        results = attribute_file(fp, profile)
    print(f"== who {args.target}")
    for r in results[:args.top]:
        parts = "  ".join(f"{k}={v}" for k, v in r["parts"].items())
        print(f"  {r['score']:.3f}  {r['author']}  ({parts})")
    return 0


def cmd_check(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gitprint check")
    ap.add_argument("target", help="file or `snippet:`")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    store = _store(args.out)
    profile = _load(store, args.profile)

    if args.target.startswith("snippet:"):
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
            tf.write(args.target[len("snippet:"):])
            tmp = Path(tf.name)
        rows, best, sim = check_file(tmp, profile)
        tmp.unlink()
    else:
        rows, best, sim = check_file(Path(args.target), profile)
    print(f"== check {args.target}")
    if best:
        print(f"  best match: {best} (lexical sim={sim:.3f})")
    anomalies = [r for r in rows if r.get("anomaly")]
    for r in rows:
        mark = "⚠️ " if r.get("anomaly") else "   "
        z = f" z={r['z']}" if r.get("z") is not None else ""
        note = f"  <- {r['note']}" if r.get("note") else ""
        print(f"{mark}{r['feature']}={r['value']} (baseline={r.get('baseline')}){z}{note}")
    if not anomalies:
        print("style OK — within the best-matching author's baseline")
    return 0


def _fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=20) as r:
        return r.read().decode("utf-8", errors="replace")


def cmd_obfu(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gitprint obfu")
    ap.add_argument("target", help="file, directory, or http(s) URL")
    args = ap.parse_args(argv)
    t = args.target
    if t.startswith(("http://", "https://")):
        try:
            body = _fetch(t)
        except Exception as e:
            sys.exit(f"failed to fetch {t}: {e}")
        tmp = Path(tempfile.mkdtemp()) / "remote.js"
        tmp.write_text(body, encoding="utf-8")
        t = str(tmp)
    p = Path(t)
    if not p.exists():
        sys.exit(f"no such file or directory: {p}")
    if p.is_dir():
        rows = scan_directory(p)
        print(f"== obfu scan {p} ({len(rows)} files)")
        for r in rows:
            sig = ",".join(r["signatures"]) or "-"
            print(f"  {r['score']:5.1f} {r['level']:10s} {r['file']}  [{sig}]")
        return 0
    if not p.is_file():
        sys.exit(f"not a regular file: {p}")
    r = obfuscate_file(p)
    print(f"== obfu {r['file']}")
    print(f"  score: {r['score']}  level: {r['level']}  lines: {r['lines']}")
    for k, v in r["indicators"].items():
        print(f"    {k}: {v}")
    if r["signatures"]:
        print("  signatures:")
        for k, v in r["signatures"].items():
            print(f"    {k}: {v}")
    return 0


def cmd_benchmark(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gitprint benchmark")
    ap.add_argument("--dataset", required=True, help="authors/<author>/*.py layout")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=float, default=0.7)
    ap.add_argument("--embed-only", action="store_true")
    ap.add_argument("--lr-only", action="store_true")
    args = ap.parse_args(argv)

    from .benchmark import load_dataset, report, run_embedding, run_lexical_lr, split
    data = load_dataset(args.dataset)
    if len(data) < 2:
        sys.exit(f"Not enough authors: {len(data)} (min 2)")
    print(f"[bench] dataset {args.dataset} — {len(data)} authors "
          f"({sum(len(v) for v in data.values())} files)")
    train, test = split(data, seed=args.seed, ratio=args.split)
    n_train = sum(len(v) for v in train.values())
    n_test = sum(len(v) for v in test.values())
    print(f"[bench] split seed={args.seed} ratio={args.split} — train {n_train} / test {n_test}")

    if not args.lr_only:
        S, y, authors, _model, V = run_embedding(train, test)
        report("embedding-cosine (CodeBERT)", S, y, V)
    if not args.embed_only:
        S, y, authors, _, V = run_lexical_lr(train, test)
        report("lexical-lr (9d + softmax)", S, y, V)
    return 0


def cmd_guard(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gitprint guard")
    ap.add_argument("--repo", required=True, help="git repository path")
    ap.add_argument("--team", required=True, help="comma-separated developer names (fine-tune target)")
    ap.add_argument("--inspect", default=None,
                    help="extra files/dirs to check (comma separated, e.g. suspicious code)")
    ap.add_argument("--out", default="guard-report.html")
    ap.add_argument("--workdir", default=None, help="work dir (default ~/.gitprint/guard/<repo>)")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-Coder-1.5B-4bit")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--n-train-commits", type=int, default=120,
                    help="files to use for team training (per modified file in commits)")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "mlx", "llama"],
                    help="auto: macOS=mlx, otherwise llama.cpp(GGUF)")
    ap.add_argument("--n-eval-team", type=int, default=25)
    ap.add_argument("--n-eval-other", type=int, default=30)
    args = ap.parse_args(argv)

    from .guard import (added_lines, author_commits, build_lora_data,
                        collect_samples, evaluate, fit_lora, guard_report)
    import random as _r
    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists():
        sys.exit(f"[guard] not a git repository: {repo}")
    team = [t.strip() for t in args.team.split(",") if t.strip()]
    work = Path(args.workdir) if args.workdir else \
        Path.home() / ".gitprint/guard" / repo.name
    data_dir = work / "lora-data"
    adapter = work / "adapter.npz"
    config = work / "config.yaml"

    print(f"[guard] repo={repo.name} team={team} workdir={work}")
    print("[guard] 1/5 git mining (commits -> authors)...")
    all_commits = author_commits(repo, team)
    team_commits = [s for name, shas in all_commits.items()
                    if any(t.lower() in name.lower() for t in team) for s in shas]
    print(f"[guard]   team commits {len(team_commits)} / total authors {len(all_commits)}")

    print("[guard] 2/5 training data generation (team commit added lines)...")
    n_train = build_lora_data(repo, team_commits, data_dir,
                              n_train=args.n_train_commits)
    if n_train < 5:
        sys.exit(f"[guard] too few team training samples: {n_train} (check team names)")
    print(f"[guard]   train samples {n_train}")

    print(f"[guard] 3/5 LoRA fine-tuning (rank={args.rank}, layers={args.layers}, "
          f"iters={args.iters})...")
    config.write_text(f"lora_parameters:\n  rank: {args.rank}\n  alpha: {args.rank*2}\n"
                      f"  dropout: 0.0\n  scale: 10.0\n")
    fit_lora(args.model, data_dir, adapter, config,
             iters=args.iters, layers=args.layers, rank=args.rank)
    print("[guard]   adapter saved")

    inspect_paths = [Path(p) for p in (args.inspect.split(",") if args.inspect else [])]
    print("[guard] 4/5 collecting eval samples...")
    samples = collect_samples(repo, all_commits, team,
                              n_eval_team=args.n_eval_team,
                              n_eval_other=args.n_eval_other,
                              inspect=inspect_paths)
    print(f"[guard]   eval samples {len(samples)} ("
          + ", ".join(f"{g}={sum(1 for s in samples if s['group']==g)}"
                      for g in ("team-holdout", "other-contributor", "external-file")) + ")")

    print("[guard] 5/5 Δloss evaluation + report...")
    rows = evaluate(args.model, adapter, samples, backend=args.backend)
    html_path = guard_report(rows, repo, team, Path(args.out))
    summ = {}
    for g in ("team-holdout", "other-contributor", "external-file"):
        rr = [r for r in rows if r["group"] == g]
        if rr:
            summ[g] = f"Δmean {sum(r['d_mean'] for r in rr)/len(rr):+.3f} (n={len(rr)})"
    print("[guard] == results (top 10, most suspicious first) ==")
    print(f"{'verdict':6s} {'Δmean':>7s}  {'name':42s}  group")
    for r in sorted(rows, key=lambda x: x["d_mean"])[:10]:
        d = r["d_mean"]
        if d > 0.05: mark = "🟢safe"
        elif d >= -0.05: mark = "🟡diff"
        else: mark = "🔴suspicious"
        print(f"{mark:6s} {d:+.3f}  {r['name'][:42]:42s}  {r['group']}")
    for g, v in summ.items():
        print(f"  [{g}] {v}")
    print(f"[guard] report: {html_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        print("Usage: gitprint {profile|blackbox|who|check|obfu|benchmark|guard} ...")
        return 0 if args else 1
    cmd, rest = args[0], args[1:]
    cmds = {
        "profile": cmd_profile,
        "blackbox": cmd_blackbox,
        "who": cmd_who,
        "check": cmd_check,
        "obfu": cmd_obfu,
        "benchmark": cmd_benchmark,
        "guard": cmd_guard,
    }
    if cmd not in cmds:
        print(f"Unknown command: {cmd}\navailable: {', '.join(cmds)}")
        return 1
    return cmds[cmd](rest)


if __name__ == "__main__":
    sys.exit(main())
