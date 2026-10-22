"""Okapi BM25 lexical retrieval baseline."""

from __future__ import annotations

import math
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from evidenceguard.data.models import Document, Query
from evidenceguard.errors import DataValidationError
from evidenceguard.retrieval.base import (
    RetrievalResult,
    build_results,
    query_text,
    validate_documents,
    validate_top_k,
)
from evidenceguard.retrieval.text import document_text, tokenize


class BM25Retriever:
    """Rank documents with Okapi BM25 over deterministic lexical tokens."""

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        self._documents = validate_documents(documents)
        self._k1 = _validate_parameter(k1, "k1", minimum=0.0)
        self._b = _validate_parameter(b, "b", minimum=0.0, maximum=1.0)
        self._epsilon = _validate_parameter(epsilon, "epsilon", minimum=0.0)
        tokenized_corpus = [tokenize(document_text(document)) for document in self._documents]
        self._index = BM25Okapi(
            tokenized_corpus,
            k1=self._k1,
            b=self._b,
            epsilon=self._epsilon,
        )

    @property
    def name(self) -> str:
        """Return the stable retriever name."""

        return "bm25"

    @property
    def documents(self) -> tuple[Document, ...]:
        """Return the indexed corpus."""

        return self._documents

    @property
    def k1(self) -> float:
        """Return the BM25 term-frequency saturation parameter."""

        return self._k1

    @property
    def b(self) -> float:
        """Return the BM25 length-normalization parameter."""

        return self._b

    @property
    def epsilon(self) -> float:
        """Return the BM25 negative-IDF floor parameter."""

        return self._epsilon

    def retrieve(
        self,
        query: str | Query,
        *,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Return BM25-ranked documents for a query."""

        limit = validate_top_k(top_k)
        scores = self._index.get_scores(tokenize(query_text(query)))
        return build_results(self._documents, scores, limit)


def _validate_parameter(
    value: float,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be a finite number.")
    if numeric_value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and numeric_value > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return numeric_value
