import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gitprint.features import chunk_text, extract_features


def test_normal_python_not_obfuscated():
    f = extract_features(
        "def process_records(total_processed_records):\n"
        "    # normalize the incoming data\n"
        "    return abs(total_processed_records)\n", "python")
    assert f.obfuscation_score < 40
    assert f.avg_ident_len > 5
    assert f.hex_ident_ratio == 0.0


def test_obfuscated_js_detected():
    text = ("var _0x4b3c=['\\x6a\\x73\\x6f\\x6e'];\n"
            "var a=function(c){return c['toString']()};")
    f = extract_features(text, "js")
    assert f.obfuscation_score >= 70
    assert "javascript-obfuscator" in f.obfuscation_hits or \
        "hex-escape-chains" in f.obfuscation_hits


def test_chunking_preserves_content():
    text = "\n".join(f"line {i}" for i in range(100))
    chunks = chunk_text(text, chunk_chars=300)
    assert sum(len(c) for c in chunks) == len(text)


def test_identifiers_collected():
    f = extract_features("def get_user_data(user_id):\n    return user_id\n", "python")
    assert f.n_idents >= 2
