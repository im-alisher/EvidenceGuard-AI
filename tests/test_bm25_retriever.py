from __future__ import annotations

from pathlib import Path

import pytest

from evidenceguard.data import Document, Query, load_beir_dataset
from evidenceguard.errors import DataValidationError
from evidenceguard.retrieval.bm25 import BM25Retriever


def test_bm25_ranks_matching_document_first(
    retrieval_documents: tuple[Document, ...],
) -> None:
    retriever = BM25Retriever(retrieval_documents)
    query = Query(query_id="query-1", text="alpha")

    results = retriever.retrieve(query, top_k=2)

    assert retriever.name == "bm25"
    assert [result.doc_id for result in results] == ["doc-a", "doc-b"]
    assert results[0].score > 0.0
    assert [result.rank for result in results] == [1, 2]


def test_bm25_accepts_string_query_and_top_k(
    retrieval_documents: tuple[Document, ...],
) -> None:
    retriever = BM25Retriever(retrieval_documents)

    results = retriever.retrieve("gamma", top_k=1)

    assert [result.doc_id for result in results] == ["doc-c"]


def test_bm25_retrieves_from_loaded_beir_fixture(mini_beir_dir: Path) -> None:
    dataset = load_beir_dataset(mini_beir_dir)
    retriever = BM25Retriever(dataset.documents)
    query = next(query for query in dataset.queries if query.query_id == "q-properties")

    results = retriever.retrieve(query, top_k=1)

    assert [result.doc_id for result in results] == ["doc-e"]


def test_bm25_indexes_document_titles() -> None:
    documents = (
        Document(doc_id="doc-a", title="alpha", text="omega"),
        Document(doc_id="doc-b", title="", text="beta"),
        Document(doc_id="doc-c", title="", text="gamma"),
        Document(doc_id="doc-d", title="", text="delta"),
    )
    retriever = BM25Retriever(documents)

    results = retriever.retrieve("alpha", top_k=1)

    assert [result.doc_id for result in results] == ["doc-a"]


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("k1", -0.1),
        ("b", 1.1),
        ("epsilon", -0.1),
        ("k1", float("nan")),
    ],
)
def test_bm25_rejects_invalid_parameters(
    retrieval_documents: tuple[Document, ...],
    parameter: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=parameter):
        BM25Retriever(retrieval_documents, **{parameter: value})


def test_bm25_rejects_duplicate_documents(
    retrieval_documents: tuple[Document, ...],
) -> None:
    with pytest.raises(DataValidationError, match="Duplicate"):
        BM25Retriever((retrieval_documents[0], retrieval_documents[0]))
