"""Exact dense retrieval over caller-supplied embedding encoders."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from evidenceguard.data.models import Document, Query
from evidenceguard.errors import DataValidationError
from evidenceguard.retrieval.base import (
    RetrievalResult,
    build_results,
    query_text,
    validate_documents,
    validate_top_k,
)
from evidenceguard.retrieval.embeddings import EmbeddingEncoder, FloatArray
from evidenceguard.retrieval.text import document_text


class DenseRetriever:
    """Rank documents by exact similarity over precomputed dense embeddings."""

    def __init__(
        self,
        documents: Sequence[Document],
        encoder: EmbeddingEncoder,
        *,
        normalize: bool = True,
    ) -> None:
        self._documents = validate_documents(documents)
        if not callable(getattr(encoder, "encode", None)):
            raise ValueError("encoder must provide a callable encode method.")
        if not isinstance(normalize, bool):
            raise ValueError("normalize must be a boolean.")

        self._encoder = encoder
        self._normalize = normalize
        raw_matrix = encoder.encode([document_text(document) for document in self._documents])
        matrix = _validate_embeddings(
            raw_matrix,
            expected_rows=len(self._documents),
            label="Document embeddings",
        )
        if normalize:
            matrix = _normalize_rows(matrix)
        matrix.setflags(write=False)
        self._matrix = matrix

    @property
    def name(self) -> str:
        """Return the stable retriever name."""

        return "dense"

    @property
    def documents(self) -> tuple[Document, ...]:
        """Return the indexed corpus."""

        return self._documents

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality."""

        return int(self._matrix.shape[1])

    def retrieve(
        self,
        query: str | Query,
        *,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Return documents ranked by exact embedding similarity."""

        limit = validate_top_k(top_k)
        query_matrix = _validate_embeddings(
            self._encoder.encode([query_text(query)]),
            expected_rows=1,
            label="Query embeddings",
        )
        if query_matrix.shape[1] != self.dimension:
            raise DataValidationError(
                "Query embedding dimension does not match the document embeddings."
            )
        if self._normalize:
            query_matrix = _normalize_rows(query_matrix)
        scores = self._matrix @ query_matrix[0]
        return build_results(self._documents, scores.tolist(), limit)


def _validate_embeddings(
    embeddings: Any,
    *,
    expected_rows: int,
    label: str,
) -> FloatArray:
    try:
        matrix = np.asarray(embeddings, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{label} must be a numeric array.") from exc
    if matrix.ndim != 2:
        raise DataValidationError(f"{label} must be a two-dimensional array.")
    if matrix.shape[0] != expected_rows:
        raise DataValidationError(
            f"{label} must contain {expected_rows} rows; received {matrix.shape[0]}."
        )
    if matrix.shape[1] < 1:
        raise DataValidationError(f"{label} must contain at least one column.")
    if not np.isfinite(matrix).all():
        raise DataValidationError(f"{label} must contain only finite values.")
    return matrix


def _normalize_rows(matrix: FloatArray) -> FloatArray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0.0)
