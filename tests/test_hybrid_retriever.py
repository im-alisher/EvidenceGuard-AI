from __future__ import annotations

from collections.abc import Sequence

import pytest

from evidenceguard.data import Document, Query
from evidenceguard.errors import DataValidationError
from evidenceguard.retrieval.base import RetrievalResult
from evidenceguard.retrieval.hybrid import ReciprocalRankFusionRetriever


class StaticRetriever:
    def __init__(self, ranking: Sequence[str], documents: dict[str, Document]) -> None:
        self._ranking = tuple(ranking)
        self._documents = documents
        self.last_top_k: int | None = None

    @property
    def name(self) -> str:
        return "static"

    def retrieve(
        self,
        query: str | Query,
        *,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        self.last_top_k = top_k
        return [
            RetrievalResult(document=self._documents[doc_id], score=1.0, rank=rank)
            for rank, doc_id in enumerate(self._ranking[:top_k], start=1)
        ]


class ResultRetriever:
    def __init__(self, results: Sequence[object]) -> None:
        self._results = tuple(results)

    @property
    def name(self) -> str:
        return "results"

    def retrieve(
        self,
        query: str | Query,
        *,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        return list(self._results[:top_k])  # type: ignore[return-value]


def _documents(documents: tuple[Document, ...]) -> dict[str, Document]:
    return {document.doc_id: document for document in documents}


def test_rrf_fuses_rankings_by_reciprocal_rank(
    retrieval_documents: tuple[Document, ...],
) -> None:
    documents = _documents(retrieval_documents)
    lexical = StaticRetriever(["doc-a", "doc-b", "doc-c"], documents)
    dense = StaticRetriever(["doc-b", "doc-d"], documents)
    retriever = ReciprocalRankFusionRetriever([lexical, dense], rrf_k=1)

    results = retriever.retrieve("query", top_k=4)

    assert retriever.name == "hybrid_rrf"
    assert [result.doc_id for result in results] == ["doc-b", "doc-a", "doc-d", "doc-c"]
    assert results[0].score == pytest.approx(5.0 / 6.0)
    assert results[1].score == pytest.approx(1.0 / 2.0)
    assert results[2].score == pytest.approx(1.0 / 3.0)
    assert results[3].score == pytest.approx(1.0 / 4.0)
    assert [result.rank for result in results] == [1, 2, 3, 4]


def test_rrf_uses_doc_id_tie_breaking(
    retrieval_documents: tuple[Document, ...],
) -> None:
    documents = _documents(retrieval_documents)
    first = StaticRetriever(["doc-b"], documents)
    second = StaticRetriever(["doc-a"], documents)
    retriever = ReciprocalRankFusionRetriever([first, second], rrf_k=1)

    results = retriever.retrieve("query", top_k=2)

    assert [result.doc_id for result in results] == ["doc-a", "doc-b"]


def test_rrf_requests_configured_candidate_depth(
    retrieval_documents: tuple[Document, ...],
) -> None:
    documents = _documents(retrieval_documents)
    first = StaticRetriever(["doc-a", "doc-b"], documents)
    second = StaticRetriever(["doc-b", "doc-a"], documents)
    retriever = ReciprocalRankFusionRetriever(
        [first, second],
        candidate_k=3,
    )

    results = retriever.retrieve("query", top_k=1)

    assert first.last_top_k == 3
    assert second.last_top_k == 3
    assert len(results) == 1


def test_rrf_defaults_candidate_depth_to_top_k(
    retrieval_documents: tuple[Document, ...],
) -> None:
    documents = _documents(retrieval_documents)
    component = StaticRetriever(["doc-a", "doc-b"], documents)
    retriever = ReciprocalRankFusionRetriever([component])

    retriever.retrieve("query", top_k=1)

    assert component.last_top_k == 1


def test_rrf_rejects_duplicate_component_results(
    retrieval_documents: tuple[Document, ...],
) -> None:
    documents = _documents(retrieval_documents)
    duplicate = RetrievalResult(document=documents["doc-a"], score=1.0, rank=2)
    retriever = ReciprocalRankFusionRetriever(
        [ResultRetriever([RetrievalResult(documents["doc-a"], 2.0, 1), duplicate])]
    )

    with pytest.raises(DataValidationError, match="duplicate"):
        retriever.retrieve("query")


def test_rrf_rejects_non_result_component_output() -> None:
    retriever = ReciprocalRankFusionRetriever([ResultRetriever([object()])])

    with pytest.raises(DataValidationError, match="RetrievalResult"):
        retriever.retrieve("query")


def test_rrf_rejects_inconsistent_document_content(
    retrieval_documents: tuple[Document, ...],
) -> None:
    documents = _documents(retrieval_documents)
    conflicting = Document(doc_id="doc-a", title="", text="Different text.")
    first = ResultRetriever([RetrievalResult(documents["doc-a"], 1.0, 1)])
    second = ResultRetriever([RetrievalResult(conflicting, 1.0, 1)])
    retriever = ReciprocalRankFusionRetriever([first, second])

    with pytest.raises(DataValidationError, match="disagree"):
        retriever.retrieve("query")


def test_rrf_rejects_invalid_configuration() -> None:
    valid_component = ResultRetriever([])
    with pytest.raises(ValueError, match="At least one"):
        ReciprocalRankFusionRetriever([])
    with pytest.raises(ValueError, match="retrieve"):
        ReciprocalRankFusionRetriever([object()])  # type: ignore[list-item]
    for invalid_k in (-1, 1.5, True):
        with pytest.raises(ValueError, match="rrf_k"):
            ReciprocalRankFusionRetriever(
                [valid_component],
                rrf_k=invalid_k,  # type: ignore[arg-type]
            )
    for invalid_k in (0, 1.5, True):
        with pytest.raises(ValueError, match="candidate_k"):
            ReciprocalRankFusionRetriever(
                [valid_component],
                rrf_k=1,
                candidate_k=invalid_k,  # type: ignore[arg-type]
            )
