"""Retrieval metrics with explicit binary and graded relevance semantics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evidenceguard.errors import DataValidationError, EvaluationError


@dataclass(frozen=True, slots=True)
class QueryRetrievalMetrics:
    """Retrieval metrics for one query at one cutoff."""

    query_id: str
    cutoff: int
    relevant_count: int
    recall: float
    precision: float
    reciprocal_rank: float
    average_precision: float
    ndcg: float
    hit: bool

    def __post_init__(self) -> None:
        _validate_query_id(self.query_id)
        _validate_cutoff(self.cutoff)
        if (
            isinstance(self.relevant_count, bool)
            or not isinstance(self.relevant_count, int)
            or self.relevant_count < 0
        ):
            raise DataValidationError("relevant_count must be a non-negative integer.")
        for name, value in (
            ("recall", self.recall),
            ("precision", self.precision),
            ("reciprocal_rank", self.reciprocal_rank),
            ("average_precision", self.average_precision),
            ("ndcg", self.ndcg),
        ):
            _validate_unit_interval(value, name)
        if not isinstance(self.hit, bool):
            raise DataValidationError("hit must be a boolean.")


@dataclass(frozen=True, slots=True)
class AggregateRetrievalMetrics:
    """Macro-averaged retrieval metrics across evaluable queries."""

    cutoff: int
    query_count: int
    mean_recall: float
    mean_precision: float
    mean_reciprocal_rank: float
    mean_average_precision: float
    mean_ndcg: float
    hit_rate: float

    def __post_init__(self) -> None:
        _validate_cutoff(self.cutoff)
        if (
            isinstance(self.query_count, bool)
            or not isinstance(self.query_count, int)
            or self.query_count < 1
        ):
            raise DataValidationError("query_count must be a positive integer.")
        for name, value in (
            ("mean_recall", self.mean_recall),
            ("mean_precision", self.mean_precision),
            ("mean_reciprocal_rank", self.mean_reciprocal_rank),
            ("mean_average_precision", self.mean_average_precision),
            ("mean_ndcg", self.mean_ndcg),
            ("hit_rate", self.hit_rate),
        ):
            _validate_unit_interval(value, name)


def compute_query_metrics(
    query_id: str,
    relevance: Mapping[str, int],
    retrieved_ids: Sequence[str],
    cutoff: int,
) -> QueryRetrievalMetrics:
    """Compute retrieval metrics for one ranked list.

    Binary metrics treat relevance grades >= 1 as relevant. Precision uses the
    cutoff as its denominator. Average precision uses ``min(relevant_count,
    cutoff)`` as its denominator. nDCG uses graded gains ``2**grade - 1``.
    """

    _validate_query_id(query_id)
    limit = _validate_cutoff(cutoff)
    grades = _validate_relevance(relevance)
    ranked_ids = _validate_ranked_ids(retrieved_ids)
    top_ids = ranked_ids[:limit]

    relevant_count = sum(1 for grade in grades.values() if grade >= 1)
    cumulative_hits = 0
    precision_sum = 0.0
    first_relevant_rank: int | None = None
    dcg = 0.0

    for rank, doc_id in enumerate(top_ids, start=1):
        grade = grades.get(doc_id, 0)
        if grade >= 1:
            cumulative_hits += 1
            precision_sum += cumulative_hits / rank
            if first_relevant_rank is None:
                first_relevant_rank = rank
            dcg += (2.0**grade - 1.0) / math.log2(rank + 1.0)

    recall = cumulative_hits / relevant_count if relevant_count else 0.0
    precision = cumulative_hits / limit
    reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
    ap_denominator = min(relevant_count, limit)
    average_precision = precision_sum / ap_denominator if ap_denominator else 0.0

    ideal_grades = sorted(
        (grade for grade in grades.values() if grade >= 1),
        reverse=True,
    )[:limit]
    idcg = math.fsum(
        (2.0**grade - 1.0) / math.log2(rank + 1.0)
        for rank, grade in enumerate(ideal_grades, start=1)
    )
    ndcg = min(1.0, dcg / idcg) if idcg > 0.0 else 0.0

    return QueryRetrievalMetrics(
        query_id=query_id,
        cutoff=limit,
        relevant_count=relevant_count,
        recall=recall,
        precision=precision,
        reciprocal_rank=reciprocal_rank,
        average_precision=average_precision,
        ndcg=ndcg,
        hit=first_relevant_rank is not None,
    )


def aggregate_query_metrics(
    metrics: Sequence[QueryRetrievalMetrics],
) -> AggregateRetrievalMetrics:
    """Macro-average query metrics that share one cutoff."""

    values = tuple(metrics)
    if not values:
        raise EvaluationError("At least one query metric is required for aggregation.")
    cutoff = values[0].cutoff
    if any(metric.cutoff != cutoff for metric in values):
        raise DataValidationError("All query metrics must use the same cutoff.")
    query_ids = [metric.query_id for metric in values]
    if len(set(query_ids)) != len(query_ids):
        raise DataValidationError("Duplicate query IDs cannot be aggregated.")

    count = len(values)
    return AggregateRetrievalMetrics(
        cutoff=cutoff,
        query_count=count,
        mean_recall=math.fsum(metric.recall for metric in values) / count,
        mean_precision=math.fsum(metric.precision for metric in values) / count,
        mean_reciprocal_rank=(
            math.fsum(metric.reciprocal_rank for metric in values) / count
        ),
        mean_average_precision=(
            math.fsum(metric.average_precision for metric in values) / count
        ),
        mean_ndcg=math.fsum(metric.ndcg for metric in values) / count,
        hit_rate=math.fsum(1.0 for metric in values if metric.hit) / count,
    )


def _validate_query_id(query_id: object) -> None:
    if not isinstance(query_id, str) or not query_id.strip():
        raise DataValidationError("query_id must be a non-empty string.")


def _validate_cutoff(cutoff: int) -> int:
    if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 1:
        raise DataValidationError("cutoff must be a positive integer.")
    return cutoff


def _validate_relevance(relevance: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(relevance, Mapping):
        raise DataValidationError("relevance must be a mapping of document IDs to grades.")
    validated: dict[str, int] = {}
    for doc_id, grade in relevance.items():
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise DataValidationError("Relevance document IDs must be non-empty strings.")
        if isinstance(grade, bool) or not isinstance(grade, int):
            raise DataValidationError("Relevance grades must be integers.")
        if grade < 0:
            raise DataValidationError("Relevance grades must be non-negative.")
        validated[doc_id] = grade
    return validated


def _validate_ranked_ids(retrieved_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(retrieved_ids, (str, bytes)):
        raise DataValidationError("retrieved_ids must be a sequence of document IDs.")
    ranked_ids = tuple(retrieved_ids)
    seen: set[str] = set()
    for doc_id in ranked_ids:
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise DataValidationError("Retrieved document IDs must be non-empty strings.")
        if doc_id in seen:
            raise DataValidationError(f"Duplicate retrieved document ID: {doc_id!r}.")
        seen.add(doc_id)
    return ranked_ids


def _validate_unit_interval(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataValidationError(f"{name} must be a finite number in [0, 1].")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
        raise DataValidationError(f"{name} must be a finite number in [0, 1].")


__all__ = [
    "AggregateRetrievalMetrics",
    "QueryRetrievalMetrics",
    "aggregate_query_metrics",
    "compute_query_metrics",
]
