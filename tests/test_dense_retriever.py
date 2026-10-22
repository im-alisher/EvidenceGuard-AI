from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from evidenceguard.data import Document, Query
from evidenceguard.errors import DataValidationError
from evidenceguard.retrieval.dense import DenseRetriever


class MappingEncoder:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self._mapping[text] for text in texts], dtype=np.float64)


class ArrayEncoder:
    def __init__(self, value: object) -> None:
        self._value = value

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._value)


class QueryShapeEncoder(MappingEncoder):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if list(texts) == ["query-alpha"]:
            return np.asarray([1.0, 0.0])
        return super().encode(texts)


_MAPPING = {
    "alpha": [3.0, 0.0],
    "beta": [0.0, 2.0],
    "gamma": [1.0, 1.0],
    "delta": [0.0, 0.0],
    "query-long": [1.0, 0.0, 0.0],
}
_ENCODER = MappingEncoder(_MAPPING)


def test_dense_ranks_by_normalized_similarity(
    retrieval_documents: tuple[Document, ...],
) -> None:
    retriever = DenseRetriever(retrieval_documents, _ENCODER)

    results = retriever.retrieve(Query(query_id="query-1", text="alpha"), top_k=3)

    assert retriever.name == "dense"
    assert retriever.dimension == 2
    assert [result.doc_id for result in results] == ["doc-a", "doc-c", "doc-b"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(1.0 / np.sqrt(2.0))
    assert results[2].score == pytest.approx(0.0)


def test_dense_supports_unnormalized_inner_product(
    retrieval_documents: tuple[Document, ...],
) -> None:
    retriever = DenseRetriever(retrieval_documents, _ENCODER, normalize=False)

    results = retriever.retrieve("alpha", top_k=1)

    assert [result.doc_id for result in results] == ["doc-a"]
    assert results[0].score == pytest.approx(9.0)


def test_dense_rejects_invalid_document_embeddings(
    retrieval_documents: tuple[Document, ...],
) -> None:
    with pytest.raises(DataValidationError, match="two-dimensional"):
        DenseRetriever(retrieval_documents, ArrayEncoder([1.0, 0.0, 0.0, 0.0]))
    with pytest.raises(DataValidationError, match="4 rows"):
        DenseRetriever(retrieval_documents, ArrayEncoder(np.ones((2, 2))))
    with pytest.raises(DataValidationError, match="finite"):
        DenseRetriever(
            retrieval_documents,
            ArrayEncoder(np.asarray([[1.0, 0.0], [np.nan, 0.0], [0.0, 1.0], [1.0, 1.0]])),
        )


def test_dense_rejects_invalid_query_embeddings(
    retrieval_documents: tuple[Document, ...],
) -> None:
    one_dimensional = DenseRetriever(retrieval_documents, QueryShapeEncoder(_MAPPING))
    with pytest.raises(DataValidationError, match="two-dimensional"):
        one_dimensional.retrieve("query-alpha")

    wrong_dimension = DenseRetriever(retrieval_documents, _ENCODER)
    with pytest.raises(DataValidationError, match="dimension"):
        wrong_dimension.retrieve("query-long")


def test_dense_rejects_invalid_configuration(
    retrieval_documents: tuple[Document, ...],
) -> None:
    with pytest.raises(ValueError, match="encode"):
        DenseRetriever(retrieval_documents, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="normalize"):
        DenseRetriever(retrieval_documents, _ENCODER, normalize=1)  # type: ignore[arg-type]
