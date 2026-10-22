from __future__ import annotations

import pytest

from evidenceguard.data import Document, Query
from evidenceguard.errors import DataValidationError
from evidenceguard.retrieval.base import (
    RetrievalResult,
    build_results,
    query_text,
    validate_documents,
    validate_top_k,
)
from evidenceguard.retrieval.text import document_text, tokenize


def test_retrieval_result_exposes_document_and_validates_rank() -> None:
    document = Document(doc_id="doc-1", title="", text="Text.")
    result = RetrievalResult(document=document, score=1.5, rank=1)

    assert result.doc_id == "doc-1"
    with pytest.raises(DataValidationError, match="rank"):
        RetrievalResult(document=document, score=1.5, rank=0)


def test_retrieval_result_rejects_non_finite_score() -> None:
    document = Document(doc_id="doc-1", title="", text="Text.")

    with pytest.raises(DataValidationError, match="finite"):
        RetrievalResult(document=document, score=float("nan"), rank=1)


def test_build_results_uses_doc_id_for_deterministic_ties() -> None:
    documents = (
        Document(doc_id="doc-b", title="", text="Beta"),
        Document(doc_id="doc-a", title="", text="Alpha"),
    )

    results = build_results(documents, [1.0, 1.0], top_k=2)

    assert [result.doc_id for result in results] == ["doc-a", "doc-b"]
    assert [result.rank for result in results] == [1, 2]


def test_build_results_sorts_scores_and_caps_at_top_k() -> None:
    documents = (
        Document(doc_id="doc-a", title="", text="Alpha"),
        Document(doc_id="doc-b", title="", text="Beta"),
        Document(doc_id="doc-c", title="", text="Gamma"),
    )

    results = build_results(documents, [0.0, 2.0, 1.0], top_k=2)

    assert [result.doc_id for result in results] == ["doc-b", "doc-c"]
    assert [result.score for result in results] == [2.0, 1.0]


def test_build_results_rejects_invalid_inputs() -> None:
    documents = (Document(doc_id="doc-a", title="", text="Alpha"),)

    with pytest.raises(DataValidationError, match="counts"):
        build_results(documents, [], top_k=1)
    with pytest.raises(DataValidationError, match="finite"):
        build_results(documents, [float("inf")], top_k=1)
    with pytest.raises(ValueError, match="top_k"):
        build_results(documents, [1.0], top_k=0)


def test_validate_top_k_rejects_bool_and_non_positive_values() -> None:
    assert validate_top_k(3) == 3
    with pytest.raises(ValueError, match="top_k"):
        validate_top_k(True)
    with pytest.raises(ValueError, match="top_k"):
        validate_top_k(-1)


def test_query_text_accepts_query_or_string() -> None:
    query = Query(query_id="query-1", text="Does it work?")

    assert query_text(query) == "Does it work?"
    assert query_text("Does it work?") == "Does it work?"
    with pytest.raises(DataValidationError, match="Query"):
        query_text("   ")
    with pytest.raises(DataValidationError, match="Query"):
        query_text(1)  # type: ignore[arg-type]


def test_validate_documents_rejects_invalid_collections() -> None:
    document = Document(doc_id="doc-1", title="", text="Text.")

    assert validate_documents([document]) == (document,)
    with pytest.raises(DataValidationError, match="At least one"):
        validate_documents([])
    with pytest.raises(DataValidationError, match="Duplicate"):
        validate_documents([document, document])
    with pytest.raises(DataValidationError, match="only Document"):
        validate_documents([document, "not-a-document"])  # type: ignore[list-item]


def test_tokenize_and_document_text_are_deterministic() -> None:
    assert tokenize("Alpha, beta-2!") == ["alpha", "beta", "2"]
    plain = Document(doc_id="doc-1", title="", text="Body")
    titled = Document(doc_id="doc-2", title="Title", text="Body")

    assert document_text(plain) == "Body"
    assert document_text(titled) == "Title\nBody"
