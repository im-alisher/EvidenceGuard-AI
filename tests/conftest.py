from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def mini_beir_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "mini_beir"
