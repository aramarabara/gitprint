import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gitprint.backend import resolve_backend_kind


@pytest.mark.parametrize("system,machine,expected", [
    ("Darwin", "arm64", "mlx"),
    ("Darwin", "x86_64", "llama"),
    ("Linux", "x86_64", "llama"),
    ("Linux", "aarch64", "llama"),
    ("Windows", "AMD64", "llama"),
])
def test_auto_resolves_by_platform(system, machine, expected):
    assert resolve_backend_kind("auto", system, machine) == expected


@pytest.mark.parametrize("kind", ["mlx", "llama"])
def test_explicit_kind_passthrough(kind):
    assert resolve_backend_kind(kind, "Linux", "x86_64") == kind


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        resolve_backend_kind("bogus", "Linux", "x86_64")
