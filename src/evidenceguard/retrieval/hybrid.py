"""Reciprocal rank fusion for combining independent retrievers."""

from __future__ import annotations

import math
from collections.abc import Sequence

from evidenceguard.data.models import Document, Query
from evidenceguard.errors import DataValidationError
from evidenceguard.retrieval.base import (
    RetrievalResult,
    Retriever,
    query_text,
    validate_top_k,
)


class ReciprocalRankFusionRetriever:
    """Fuse component rankings using reciprocal rank fusion."""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        rrf_k: int = 60,
        candidate_k: int | None = None,
    ) -> None:
        configured = tuple(retrievers)
        if not configured:
            raise ValueError("At least one component retriever is required.")
        for retriever in configured:
            if not callable(getattr(retriever, "retrieve", None)):
                raise ValueError("Each component must provide a callable retrieve method.")
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 0:
            raise ValueError("rrf_k must be a non-negative integer.")
        if candidate_k is not None:
            if isinstance(candidate_k, bool) or not isinstance(candidate_k, int):
                raise ValueError("candidate_k must be a positive integer or None.")
            if candidate_k < 1:
                raise ValueError("candidate_k must be a positive integer or None.")

        self._retrievers = configured
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k

    @property
    def name(self) -> str:
        """Return the stable retriever name."""

        return "hybrid_rrf"

    @property
    def retrievers(self) -> tuple[Retriever, ...]:
        """Return the component retrievers in fusion order."""

        return self._retrievers

    def retrieve(
        self,
        query: str | Query,
        *,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Return documents ranked by fused reciprocal ranks."""

        limit = validate_top_k(top_k)
        depth = limit if self._candidate_k is None else max(limit, self._candidate_k)
        normalized_query = query_text(query)
        documents: dict[str, Document] = {}
        component_scores: dict[str, list[float]] = {}

        for retriever in self._retrievers:
            seen: set[str] = set()
            for result in retriever.retrieve(normalized_query, top_k=depth):
                if not isinstance(result, RetrievalResult):
                    raise DataValidationError(
                        "Component retrievers must return RetrievalResult objects."
                    )
                doc_id = result.doc_id
                if doc_id in seen:
                    raise DataValidationError(
                        f"Component retriever returned duplicate document: {doc_id!r}."
                    )
                seen.add(doc_id)

                existing = documents.get(doc_id)
                if existing is not None and existing != result.document:
                    raise DataValidationError(
                        f"Component retrievers disagree on document content for {doc_id!r}."
                    )
                documents.setdefault(doc_id, result.document)
                component_scores.setdefault(doc_id, []).append(
                    1.0 / float(self._rrf_k + result.rank)
                )

        fused_scores = {
            doc_id: math.fsum(scores) for doc_id, scores in component_scores.items()
        }
        ranked_ids = sorted(fused_scores, key=lambda doc_id: (-fused_scores[doc_id], doc_id))
        return [
            RetrievalResult(
                document=documents[doc_id],
                score=fused_scores[doc_id],
                rank=rank,
            )
            for rank, doc_id in enumerate(ranked_ids[:limit], start=1)
        ]
