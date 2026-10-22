from __future__ import annotations

from pathlib import Path

import pytest

from evidenceguard.data import Document


@pytest.fixture
def mini_beir_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "mini_beir"


@pytest.fixture
def retrieval_documents() -> tuple[Document, ...]:
    return (
        Document(doc_id="doc-a", title="", text="alpha"),
        Document(doc_id="doc-b", title="", text="beta"),
        Document(doc_id="doc-c", title="", text="gamma"),
        Document(doc_id="doc-d", title="", text="delta"),
    )
