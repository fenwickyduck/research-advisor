from __future__ import annotations

from pathlib import Path

import pytest

from advisor import db

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def conn(tmp_path: Path):
    connection = db.connect(tmp_path / "test.db")
    yield connection
    connection.close()
