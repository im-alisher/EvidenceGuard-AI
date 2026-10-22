from __future__ import annotations

from datetime import date

import pytest

from evidenceguard.data.models import (
    Document,
    EvidenceCondition,
    EvidenceExample,
    ExampleProvenance,
    Query,
    RelevanceJudgment,
    RetrievalDataset,
)
from evidenceguard.errors import DataValidationError
from evidenceguard.outcomes import DecisionOutcome


def _document(doc_id: str) -> Document:
    return Document(doc_id=doc_id, title="Title", text="Text.", published_at=date(2024, 1, 1))


def test_document_rejects_empty_text() -> None:
    with pytest.raises(DataValidationError, match="Document text"):
        Document(doc_id="doc-1", title="", text="   ")


def test_document_metadata_is_read_only() -> None:
    document = Document(
        doc_id="doc-1",
        title="",
        text="Text.",
        metadata={"source": "synthetic"},
    )

    with pytest.raises(TypeError):
        document.metadata["source"] = "changed"  # type: ignore[index]


def test_retrieval_dataset_rejects_duplicate_documents() -> None:
    document = _document("doc-1")
    query = Query(query_id="query-1", text="Question?")

    with pytest.raises(DataValidationError, match="duplicate document IDs"):
        RetrievalDataset(
            name="test",
            split="test",
            documents=(document, document),
            queries=(query,),
            qrels=(),
        )


def test_retrieval_dataset_rejects_unknown_qrel_document() -> None:
    query = Query(query_id="query-1", text="Question?")

    with pytest.raises(DataValidationError, match="unknown document"):
        RetrievalDataset(
            name="test",
            split="test",
            documents=(_document("doc-1"),),
            queries=(query,),
            qrels=(RelevanceJudgment(query_id="query-1", doc_id="doc-2", relevance=1),),
        )


def test_retrieval_dataset_returns_sorted_relevant_ids() -> None:
    dataset = RetrievalDataset(
        name="test",
        split="test",
        documents=(_document("doc-b"), _document("doc-a"), _document("doc-c")),
        queries=(Query(query_id="query-1", text="Question?"),),
        qrels=(
            RelevanceJudgment(query_id="query-1", doc_id="doc-b", relevance=1),
            RelevanceJudgment(query_id="query-1", doc_id="doc-a", relevance=2),
            RelevanceJudgment(query_id="query-1", doc_id="doc-c", relevance=0),
        ),
    )

    assert dataset.relevant_document_ids("query-1") == ("doc-a", "doc-b")
    assert dataset.relevant_document_ids("query-1", min_relevance=2) == ("doc-a",)


def test_evidence_example_rejects_relevant_documents_outside_pool() -> None:
    with pytest.raises(DataValidationError, match="outside its pool"):
        EvidenceExample(
            example_id="example-1",
            query=Query(query_id="query-1", text="Question?"),
            documents=(_document("doc-1"),),
            relevant_document_ids=("doc-2",),
            expected_outcome=DecisionOutcome.INSUFFICIENT_EVIDENCE,
            condition=EvidenceCondition.MISSING,
        )


def test_original_evidence_example_cannot_declare_transformation() -> None:
    with pytest.raises(DataValidationError, match="Original examples"):
        EvidenceExample(
            example_id="example-1",
            query=Query(query_id="query-1", text="Question?"),
            documents=(_document("doc-1"),),
            relevant_document_ids=("doc-1",),
            expected_outcome=DecisionOutcome.ANSWER_SUPPORTED,
            condition=EvidenceCondition.SUFFICIENT,
            provenance=ExampleProvenance.ORIGINAL,
            transformation="invented",
        )


def test_transformed_evidence_example_requires_source() -> None:
    with pytest.raises(DataValidationError, match="source_example_id"):
        EvidenceExample(
            example_id="example-1",
            query=Query(query_id="query-1", text="Question?"),
            documents=(_document("doc-1"),),
            relevant_document_ids=(),
            expected_outcome=DecisionOutcome.INSUFFICIENT_EVIDENCE,
            condition=EvidenceCondition.MISSING,
            provenance=ExampleProvenance.TRANSFORMED,
            transformation="remove_relevant_evidence",
        )
