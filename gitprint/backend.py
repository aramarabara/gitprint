"""Model backend abstraction — per-token CE interface.

Per-platform backends:
  - mlx          : macOS (Apple Silicon) — training/inference
  - llama_cpp    : Linux/Windows — GGUF inference (llama-cpp-python)
  - transformers : all platforms (peft adapters) — GPU/CPU

Common interface: TokenCEBackend.load(..., adapter=...) → (encode, token_ce)
  - encode(text) -> list[int]
  - token_ce(ids, max_len) -> np.ndarray  (next-token cross-entropy per token)
"""
from __future__ import annotations

import os

import numpy as np

# Keep stdout/stderr clean: suppress tqdm/progress bars from the model libs.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


class TokenCEBackend:
    """per-token cross-entropy computing backend."""

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def token_ce(self, ids: list[int], max_len: int = 512) -> np.ndarray:
        raise NotImplementedError


class MLXBackend(TokenCEBackend):
    """Apple Silicon MLX backend (4-bit base + mlx-lm adapter)."""

    def __init__(self, model_path: str, adapter_path: str | None = None):
        from mlx_lm import load
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        self.name = f"mlx:{model_path}" + (f"+adapter:{adapter_path}" if adapter_path else "")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def token_ce(self, ids: list[int], max_len: int = 512) -> np.ndarray:
        import mlx.core as mx
        ids = list(ids)[: max_len + 1]
        if len(ids) < 2:
            return np.zeros(1)
        logits = self.model(mx.array(ids)[None, :])
        if isinstance(logits, dict):
            logits = logits.get("logits") or logits["lm_logits"]
        logits = logits[0, :-1, :]
        labels = mx.array(ids[1:])[:, None]
        m = logits.max(-1, keepdims=True)
        lse = m + mx.log(mx.exp(logits - m).sum(-1, keepdims=True))
        ce = (lse - mx.take_along_axis(logits, labels, -1))[:, 0]
        out = np.asarray(ce, dtype="float64")
        del logits, m, lse, ce
        return out


class LlamaCppBackend(TokenCEBackend):
    """llama.cpp backend (GGUF, Linux/Windows). Uses llama-cpp-python.

    Per-token CE: tokenize → batch decode(logits) → cross-entropy.
    """

    def __init__(self, gguf_path: str, n_gpu_layers: int = 0, n_ctx: int = 4096,
                 n_threads: int = 4):
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_threads=n_threads,
            verbose=False, logits_all=True,
        )
        self.name = f"llama:{gguf_path}"

    def encode(self, text: str) -> list[int]:
        return self.llm.tokenize(text.encode("utf-8"), add_bos=True)

    def decode(self, ids: list[int]) -> str:
        return self.llm.detokenize(ids).decode("utf-8", errors="replace")

    def token_ce(self, ids: list[int], max_len: int = 512) -> np.ndarray:
        ids = list(ids)[: max_len + 1]
        if len(ids) < 2:
            return np.zeros(1)
        self.llm.reset()
        self.llm.eval(ids)
        scores = self.llm._scores  # (n_tokens × vocab) flattened or deque
        n_vocab = self.llm.n_vocab()
        if isinstance(scores, np.ndarray):
            logits = scores.reshape(-1, n_vocab)
        else:
            logits = np.concatenate(
                [np.asarray(s, dtype="float64").reshape(-1, n_vocab) for s in scores])
        logits = np.asarray(logits, dtype="float64")
        logits = logits[:-1]  # token i predicts the next token
        labels = np.asarray(ids[1:])
        m = logits.max(-1, keepdims=True)
        exp = np.exp(logits - m)
        lse = m + np.log(exp.sum(-1, keepdims=True))
        return (lse[:, 0] - logits[np.arange(len(labels)), labels])


def resolve_backend_kind(kind: str, system: str, machine: str) -> str:
    """Resolve 'auto' using platform info ('mlx' | 'llama'). Pure function, testable."""
    if kind not in ("auto", "mlx", "llama"):
        raise ValueError(f"Unknown backend: {kind}")
    if kind != "auto":
        return kind
    if system == "Darwin" and machine == "arm64":
        return "mlx"
    return "llama"


def make_backend(kind: str, model_path: str,
                 adapter_path: str | None = None) -> TokenCEBackend:
    """'mlx' | 'llama' | 'auto'. auto = mlx on macOS, otherwise llama."""
    import platform
    resolved = resolve_backend_kind(kind, platform.system(), platform.machine())
    if resolved == "mlx":
        return MLXBackend(model_path, adapter_path)
    if resolved == "llama":
        return LlamaCppBackend(model_path)
    raise ValueError(f"Unknown backend: {kind}")
