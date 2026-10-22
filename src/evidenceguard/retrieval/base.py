"""Shared retrieval interfaces, results, and deterministic ranking helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from evidenceguard.data.models import Document, Query
from evidenceguard.errors import DataValidationError


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked document returned for a query."""

    document: Document
    score: float
    rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.document, Document):
            raise DataValidationError("RetrievalResult document must be a Document.")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise DataValidationError("RetrievalResult score must be numeric.")
        numeric_score = float(self.score)
        if not math.isfinite(numeric_score):
            raise DataValidationError("RetrievalResult score must be finite.")
        object.__setattr__(self, "score", numeric_score)
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise DataValidationError("RetrievalResult rank must be an integer.")
        if self.rank < 1:
            raise DataValidationError("RetrievalResult rank must be at least 1.")

    @property
    def doc_id(self) -> str:
        """Return the retrieved document identifier."""

        return self.document.doc_id


class Retriever(Protocol):
    """A corpus-backed retriever independent of answer generation."""

    @property
    def name(self) -> str:
        """Return a stable retriever name for reports and logs."""

        ...

    def retrieve(
        self,
        query: str | Query,
        *,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Return up to ``top_k`` documents ranked by decreasing score."""

        ...


def validate_top_k(top_k: int) -> int:
    """Validate and return a requested result count."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    return top_k


def validate_documents(documents: Sequence[Document]) -> tuple[Document, ...]:
    """Validate a non-empty, uniquely identified document collection."""

    if isinstance(documents, (str, bytes)):
        raise DataValidationError("Documents must be a sequence of Document objects.")
    validated = tuple(documents)
    if not validated:
        raise DataValidationError("At least one document is required for retrieval.")
    seen: set[str] = set()
    for document in validated:
        if not isinstance(document, Document):
            raise DataValidationError("Documents must contain only Document objects.")
        if document.doc_id in seen:
            raise DataValidationError(
                f"Duplicate document ID in retrieval corpus: {document.doc_id!r}."
            )
        seen.add(document.doc_id)
    return validated


def query_text(query: str | Query) -> str:
    """Return query text from a Query object or non-empty string."""

    if isinstance(query, Query):
        return query.text
    if not isinstance(query, str):
        raise DataValidationError("Query must be a Query object or string.")
    if not query.strip():
        raise DataValidationError("Query text must be non-empty.")
    return query


def build_results(
    documents: Sequence[Document],
    scores: Sequence[float],
    top_k: int,
) -> list[RetrievalResult]:
    """Build deterministic ranked results with doc-ID tie-breaking."""

    limit = validate_top_k(top_k)
    if len(documents) != len(scores):
        raise DataValidationError("Document and score counts must match.")

    numeric_scores: list[float] = []
    for score in scores:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise DataValidationError("Retrieval scores must be numeric.")
        value = float(score)
        if not math.isfinite(value):
            raise DataValidationError("Retrieval scores must be finite.")
        numeric_scores.append(value)

    order = sorted(
        range(len(documents)),
        key=lambda index: (-numeric_scores[index], documents[index].doc_id),
    )
    return [
        RetrievalResult(
            document=documents[position],
            score=numeric_scores[position],
            rank=rank,
        )
        for rank, position in enumerate(order[:limit], start=1)
    ]
