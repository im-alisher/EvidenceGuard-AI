"""Evaluate retrievers against dataset relevance judgments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from evidenceguard.data.models import RetrievalDataset
from evidenceguard.errors import EvaluationError
from evidenceguard.evaluation.metrics import (
    AggregateRetrievalMetrics,
    QueryRetrievalMetrics,
    aggregate_query_metrics,
    compute_query_metrics,
)
from evidenceguard.retrieval.base import RetrievalResult, Retriever, validate_top_k


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    """Per-query and aggregate retrieval metrics for one benchmark split."""

    dataset_name: str
    split: str
    retriever_name: str
    top_k: int
    cutoffs: tuple[int, ...]
    skipped_query_ids: tuple[str, ...]
    per_query: tuple[QueryRetrievalMetrics, ...]
    aggregates: tuple[AggregateRetrievalMetrics, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "retriever_name": self.retriever_name,
            "top_k": self.top_k,
            "cutoffs": list(self.cutoffs),
            "skipped_query_ids": list(self.skipped_query_ids),
            "aggregates": [asdict(metrics) for metrics in self.aggregates],
            "per_query": [asdict(metrics) for metrics in self.per_query],
        }


def evaluate_retriever(
    retriever: Retriever,
    dataset: RetrievalDataset,
    *,
    cutoffs: Sequence[int] = (1, 5, 10),
    top_k: int | None = None,
) -> RetrievalEvaluation:
    """Evaluate one retriever with macro-averaged retrieval metrics.

    Queries without any positive relevance judgment are excluded from metric
    calculation and reported in ``skipped_query_ids``.
    """

    normalized_cutoffs = _validate_cutoffs(cutoffs)
    maximum_cutoff = normalized_cutoffs[-1]
    depth = _resolve_top_k(top_k, maximum_cutoff)

    if not isinstance(dataset, RetrievalDataset):
        raise EvaluationError("dataset must be a RetrievalDataset.")
    retriever_name = getattr(retriever, "name", None)
    if not isinstance(retriever_name, str) or not retriever_name.strip():
        raise EvaluationError("retriever must expose a non-empty name.")
    if not callable(getattr(retriever, "retrieve", None)):
        raise EvaluationError("retriever must provide a callable retrieve method.")

    relevance_by_query: dict[str, dict[str, int]] = {
        query.query_id: {} for query in dataset.queries
    }
    for judgment in dataset.qrels:
        relevance_by_query[judgment.query_id][judgment.doc_id] = judgment.relevance

    document_ids = {document.doc_id for document in dataset.documents}
    per_query: list[QueryRetrievalMetrics] = []
    skipped_query_ids: list[str] = []
    metrics_by_cutoff: dict[int, list[QueryRetrievalMetrics]] = {
        cutoff: [] for cutoff in normalized_cutoffs
    }

    for query in dataset.queries:
        relevance = relevance_by_query[query.query_id]
        if not any(grade >= 1 for grade in relevance.values()):
            skipped_query_ids.append(query.query_id)
            continue

        retrieved_ids = _validate_retrieved_ids(
            retriever.retrieve(query, top_k=depth),
            known_document_ids=document_ids,
        )
        for cutoff in normalized_cutoffs:
            metrics = compute_query_metrics(
                query_id=query.query_id,
                relevance=relevance,
                retrieved_ids=retrieved_ids,
                cutoff=cutoff,
            )
            per_query.append(metrics)
            metrics_by_cutoff[cutoff].append(metrics)

    if not per_query:
        raise EvaluationError(
            "No queries with positive relevance judgments are available for evaluation."
        )

    aggregates = tuple(
        aggregate_query_metrics(metrics_by_cutoff[cutoff])
        for cutoff in normalized_cutoffs
    )
    return RetrievalEvaluation(
        dataset_name=dataset.name,
        split=dataset.split,
        retriever_name=retriever_name,
        top_k=depth,
        cutoffs=normalized_cutoffs,
        skipped_query_ids=tuple(skipped_query_ids),
        per_query=tuple(per_query),
        aggregates=aggregates,
    )


def _validate_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    if isinstance(cutoffs, (str, bytes)):
        raise ValueError("cutoffs must be a non-empty sequence of positive integers.")
    values = tuple(cutoffs)
    if not values:
        raise ValueError("cutoffs must be a non-empty sequence of positive integers.")
    for cutoff in values:
        if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 1:
            raise ValueError(
                "cutoffs must be a non-empty sequence of positive integers."
            )
    if len(set(values)) != len(values):
        raise ValueError("cutoffs must not contain duplicate values.")
    return tuple(sorted(values))


def _resolve_top_k(top_k: int | None, maximum_cutoff: int) -> int:
    if top_k is None:
        return maximum_cutoff
    try:
        depth = validate_top_k(top_k)
    except ValueError as exc:
        raise ValueError(
            "top_k must be a positive integer greater than or equal to every cutoff."
        ) from exc
    if depth < maximum_cutoff:
        raise ValueError(
            "top_k must be a positive integer greater than or equal to every cutoff."
        )
    return depth


def _validate_retrieved_ids(
    results: Sequence[RetrievalResult],
    *,
    known_document_ids: set[str],
) -> tuple[str, ...]:
    retrieved_ids: list[str] = []
    seen: set[str] = set()
    for position, result in enumerate(results, start=1):
        if not isinstance(result, RetrievalResult):
            raise EvaluationError("Retrievers must return RetrievalResult objects.")
        if result.rank != position:
            raise EvaluationError(
                "Retriever result ranks must match their returned order."
            )
        doc_id = result.doc_id
        if doc_id in seen:
            raise EvaluationError(f"Retriever returned duplicate document: {doc_id!r}.")
        if doc_id not in known_document_ids:
            raise EvaluationError(
                f"Retriever returned a document outside the dataset: {doc_id!r}."
            )
        seen.add(doc_id)
        retrieved_ids.append(doc_id)
    return tuple(retrieved_ids)


__all__ = ["RetrievalEvaluation", "evaluate_retriever"]
