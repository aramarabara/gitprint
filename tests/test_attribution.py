import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gitprint.attribution import _lex_sim
from gitprint.features import extract_features, lang_of

FIXTURE = Path(__file__).parent / "fixtures" / "duo"
UNKNOWN = Path(__file__).parent / "fixtures" / "unknown_alice_style.py"
FEAT_KEYS = ["avg_ident_len", "short_ident_ratio", "hex_ident_ratio",
             "comment_ratio", "unique_ratio", "string_escape_ratio",
             "snake", "camel", "pascal"]


def _baseline(paths: list[Path]) -> dict:
    agg = {k: [] for k in FEAT_KEYS}
    for p in paths:
        f = extract_features(p.read_text(encoding="utf-8"), lang_of(p.name))
        for k, v in zip(FEAT_KEYS, f.vector()):
            agg[k].append(v)
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in agg.items()}


ALICE_BASELINE = _baseline([
    FIXTURE / "payment_processor.py",
    FIXTURE / "data_warehouse_client.py",
    FIXTURE / "report_generator.py",
])
BOB_BASELINE = _baseline([
    FIXTURE / "pay.py",
    FIXTURE / "dw.py",
    FIXTURE / "rep.py",
])


def test_unknown_file_matches_alice_style_lexically():
    text = UNKNOWN.read_text(encoding="utf-8")
    f = extract_features(text, "python")
    sim_alice = _lex_sim(f, ALICE_BASELINE)
    sim_bob = _lex_sim(f, BOB_BASELINE)
    assert sim_alice > sim_bob
    assert sim_alice > 0.3
    assert sim_bob < sim_alice * 0.8
