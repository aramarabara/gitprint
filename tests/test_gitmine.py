import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gitprint.gitmine import author_files, mine_ownership

FIXTURE = Path(__file__).parent / "fixtures" / "duo"


def test_mine_ownership_two_authors():
    own = mine_ownership(FIXTURE)
    authors = {a.lower() for a in own}
    assert "alice dev" in own or "alice dev" in authors
    assert len(own) >= 2


def test_author_files_assignment():
    files = author_files(FIXTURE)
    assert len(files) >= 2
    all_paths = [p for paths in files.values() for p in paths]
    assert len(all_paths) >= 5
    assert any(p.name == "payment_processor.py" for p in all_paths)
