# gitprint — Code Provenance Intelligence

> **verify who really wrote your code.**

`gitprint` combines git history mining with pretrained code embeddings to build a
**per-developer style fingerprint**, then answers whether a piece of code was
written by someone on your team, whether it drifts from the owner's usual
habits, and whether it is obfuscated.

Black-box principle: **only your own code and assets** are analyzed. This is not
a tool for identifying who wrote someone else's code — it verifies the
**integrity of your own codebase**.

---

## Why

In software supply-chain security, "who wrote this code" is the first question.
Names in commit logs can be forged, but **coding habits are hard to fake**:

- identifier naming conventions (snake_case vs camelCase vs short abbreviations)
- identifier length and token diversity
- comment density, error/log wording
- contextual habits captured by code embeddings (CodeBERT etc.)

We combine the two (deterministic lexical features + embedding centroids) to
build a **per-author style profile** and answer the questions below.

### Primary use cases

| Question | Command |
|---|---|
| **Does this repo's PR/commit match the team style? (supply-chain guard)** | `gitprint guard --repo <path> --team "Name,Name"` |
| Does this commit/file match its usual owner? (account takeover, external injection) | `gitprint check` |
| Which team member's style is this code fragment? | `gitprint who` |
| Is a deployed JS bundle obfuscated / contaminated? (Magecart etc.) | `gitprint obfu` |
| Need a reference profile from a few files with no git history? | `gitprint blackbox` |

---

## Install

```sh
# One-liner, isolated, PATH managed for you (recommended)
pipx install gitprint

# Or
pip install gitprint          # requires gitprint's bin dir to be on PATH
#   (venv)  .venv/bin/gitprint
#   (user)  ~/.local/bin/gitprint

# Or run without installing
uvx gitprint

# Or from source
git clone https://github.com/you/gitprint && cd gitprint
python3 -m venv .venv && .venv/bin/pip install -e . && .venv/bin/gitprint --help
```

> **PATH**: pip installs the `gitprint` executable into the environment's bin
> directory. If the command isn't found, add that directory to your `PATH`
> (or use `pipx`, which handles it for you).

> **Environment**: macOS (Apple Silicon, MLX). The first run downloads the code
> model (`mlx-community/Qwen2.5-Coder-1.5B-4bit`, ~1GB) from HuggingFace
> (one-time cold start, then cached). The model is open-weight and free to swap.
> **Cross-platform**: Linux/Windows use `--backend llama` (GGUF via llama.cpp).
> A peft converter for the mlx format is on the roadmap.

## Usage

### 0) Supply-chain guard — check PRs/commits against a team style baseline (flagship) 🛡️

```sh
gitprint guard --repo ./my-repo --team "Alice,Bob" [--inspect path,...]
```

End-to-end flow:
```
git history → auto-extract diffs/files of the named developers → LoRA fine-tune (~10 min, once)
→ then check every commit/file as base vs adapter Δloss → suspicious-first report (HTML)
```

Verdict labels (Δloss = base − adapt):

| Label | Meaning |
|---|---|
| 🟢 safe | matches the team style fingerprint |
| 🟡 different style | safe but style differs slightly |
| 🔴 suspicious | strongly deviates from the team style (external injection / malicious candidate) |

> **Real-world validation (2026-08)**: an adapter trained on the `click` repo's
> team (David Lord, Armin Ronacher) flagged **all 15 real malicious PyPI
> packages** (colorama typosquatting family, etc.), **never seen during
> training**, as "outside the team style". **AUC 0.936**, FPR@TPR=0.9 **7.3%**.
> Malicious code lives outside the statistical distribution of normal code, so
> it doesn't match any team's style.
>
> **Note on the benchmark artifacts**: the evaluation dataset contains live
> malware samples, so we deliberately do **not** redistribute them with this
> repo. The adapter, training data, and evaluation pipeline are reproducible
> from the public `click` repository; only the malicious evaluation set is
> intentionally omitted.

### 1) Build a profile — your team's style fingerprint

```sh
gitprint profile ./my-repo --out ~/.gitprint/profiles
```

From the full `git --all --numstat` history, assigns each file to its dominant
author (most contributed lines) and stores per-author lexical baselines +
embedding centroids. (`~/.gitprint/profiles/<repo>.json/.npy`)

### 2) Whose style is it — attribution

```sh
gitprint who ./my-repo/src/module.py --profile my-repo
gitprint who 'snippet:def q(a,b):\n    n=0\n    for i in a:\n        n+=len(i)' --profile my-repo
```

Combines embedding cosine similarity + lexical similarity (0.6/0.4) into an
author ranking.

### 3) Style-drift check — inspection guard

```sh
gitprint check ./my-repo/src/module.py --profile my-repo
```

Finds the best-matching author, computes a z-score against **that author's
baseline**, and flags features with `z > 2`. (As a CI/PR guard: "this change
does not match the owner's usual habits".)

### 4) Obfuscation detection — injected skimmers / supply-chain contamination

```sh
gitprint obfu ./dist/bundle.js
gitprint obfu ./dist/            # scan a whole directory
gitprint obfu https://example.com/assets/app.js
```

- hex identifier ratio, short identifier ratio, `\x` escape density, avg identifier length
- signature matching: `javascript-obfuscator`, `obfuscator-io`, `eval-chain`, `base64-blob`
- 0–100 score with a `normal / suspicious / high` level

### 5) Black-box profile — files only, no git

```sh
gitprint blackbox ./extracted/src/*.js --name vendor-x
gitprint who ./extracted/src/mystery.js --profile vendor-x
```

Builds an "unknown" profile from files received without a source repo, to set a
comparison baseline.

---

## How it works

```
git log --all --numstat ──► per-file owner determination (most contributed lines)
        │
        ▼
per-author code chunk collection ──► ① lexical features (naming/identifier/comment/entropy)
        │                             ② embedding centroids (CodeBERT contextual habits)
        ▼
profile saved (JSON + .npy)
        │
        ▼
who ──► cosine similarity + lexical combined ranking
check ──► z-score against best-matching author baseline (drift/smell)
obfu ──► obfuscation indicators + signature score
guard ──► git diff → LoRA fine-tune (team adapter) → Δloss team in/out decision
```

- **A profile is a behavioral baseline.** An attacker can forge a name but not habits.
- Combined with black-box tracking you can compare the style fingerprint of a
  deployed asset against your internal profile (e.g. `gitprint who ./captured/bundle.js --profile my-team`).

## Roadmap

- [ ] automated blackbox↔profile cross-matching (ui-snap-style pipeline)
- [ ] line-level `git blame` attribution (file-level approximation → line-level)
- [ ] commit time-series style tracking (`track`): drift/legacy discovery over time
- [ ] GitHub Action / pre-commit guard
- [ ] deobfuscation heuristics v2

## Supported environments & testing matrix

CI (`.github/workflows/ci.yml`) runs unit tests, a full-install smoke test, and
a per-platform backend import check on every push/PR:

| OS | Python | inference backend | CI jobs |
|---|---|---|---|
| macOS (arm64) | 3.10 / 3.12 | MLX (default) | unit + install + backend-smoke |
| macOS (x86_64) | 3.10 / 3.12 | llama.cpp | unit + install |
| Linux | 3.10 / 3.12 | llama.cpp | unit + install + backend-smoke |
| Windows | 3.10 / 3.12 | llama.cpp | unit + install + backend-smoke |

- **Embedding models** (used by `who`/`profile`/`check`): loaded on first use,
  cached locally; `CODEDNA_MODEL` pins one. Fallback chain:
  `jina-embeddings-v2-base-code` → `microsoft/codebert-base` → `all-MiniLM-L6-v2`.
- **Inference backends** (used by `guard`): `--backend auto` picks MLX on macOS
  arm64, llama.cpp elsewhere. Override with `--backend mlx|llama`.
- **Python**: `>=3.10`, tested on 3.10 and 3.12.
- First run downloads the code model (~1GB, one-time, cached) from HuggingFace.

## References

- A. Caliskan-Islam, R. Harang, A. Liu, A. Narayanan, C. Voss, F. Yamaguchi,
  R. Greenstadt. *De-anonymizing Programmers via Code Stylometry.* USENIX
  Security, 2015. — the stylistic-identifier framework our lexical features
  follow (identifier naming, comment density, language-independent vectors).
- M. Abuhamad, T. Rhouma, D. Mohaisen. *Code Authorship Identification:
  Methods and Challenges.* ACM Computing Surveys, 2018. — the survey our
  Top-1 Accuracy / Pair-AUC / FPR@TPR evaluation conventions follow.
- Microsoft. *CodeBERT: A Pre-Trained Model for Programming and Natural
  Languages.* arXiv:2002.08155, 2020. — the code embedding model used for
  centroid-based attribution (with a fallback chain).

## License

MIT
