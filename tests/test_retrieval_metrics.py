from __future__ import annotations

import math

import pytest

from evidenceguard.errors import DataValidationError, EvaluationError
from evidenceguard.evaluation.metrics import (
    aggregate_query_metrics,
    compute_query_metrics,
)


def test_compute_query_metrics_with_binary_and_graded_relevance() -> None:
    metrics = compute_query_metrics(
        query_id="query-1",
        relevance={"doc-a": 1, "doc-b": 2},
        retrieved_ids=["doc-b", "doc-x", "doc-a"],
        cutoff=3,
    )

    ideal_dcg = 3.0 + 1.0 / math.log2(3.0)
    assert metrics.query_id == "query-1"
    assert metrics.cutoff == 3
    assert metrics.relevant_count == 2
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(2.0 / 3.0)
    assert metrics.reciprocal_rank == pytest.approx(1.0)
    assert metrics.average_precision == pytest.approx(5.0 / 6.0)
    assert metrics.ndcg == pytest.approx(3.5 / ideal_dcg)
    assert metrics.hit is True


def test_compute_query_metrics_respects_cutoff() -> None:
    first = compute_query_metrics("query-1", {"doc-a": 1, "doc-b": 1}, ["doc-b", "doc-a"], 1)
    second = compute_query_metrics(
        "query-1",
        {"doc-a": 1, "doc-b": 1},
        ["doc-b", "doc-a"],
        2,
    )

    assert first.recall == pytest.approx(0.5)
    assert first.precision == pytest.approx(1.0)
    assert first.average_precision == pytest.approx(1.0)
    assert second.recall == pytest.approx(1.0)
    assert second.precision == pytest.approx(1.0)
    assert second.average_precision == pytest.approx(1.0)


def test_precision_uses_cutoff_as_denominator() -> None:
    metrics = compute_query_metrics("query-1", {"doc-a": 1}, ["doc-a"], cutoff=3)

    assert metrics.recall == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0 / 3.0)
    assert metrics.average_precision == pytest.approx(1.0)


def test_average_precision_uses_minimum_of_relevant_count_and_cutoff() -> None:
    metrics = compute_query_metrics(
        "query-1",
        {"doc-a": 1, "doc-b": 1},
        ["doc-a"],
        cutoff=2,
    )

    assert metrics.average_precision == pytest.approx(0.5)


def test_query_without_relevant_documents_has_zero_metrics() -> None:
    metrics = compute_query_metrics("query-1", {}, ["doc-a"], cutoff=1)

    assert metrics.relevant_count == 0
    assert metrics.recall == 0.0
    assert metrics.precision == 0.0
    assert metrics.reciprocal_rank == 0.0
    assert metrics.average_precision == 0.0
    assert metrics.ndcg == 0.0
    assert metrics.hit is False


def test_graded_ndcg_uses_exponential_gain() -> None:
    metrics = compute_query_metrics(
        "query-1",
        {"doc-a": 0, "doc-b": 1, "doc-c": 2},
        ["doc-b", "doc-c"],
        cutoff=2,
    )

    dcg = 1.0 + 3.0 / math.log2(3.0)
    idcg = 3.0 + 1.0 / math.log2(3.0)
    assert metrics.ndcg == pytest.approx(dcg / idcg)


def test_duplicate_retrieved_documents_are_rejected() -> None:
    with pytest.raises(DataValidationError, match="Duplicate"):
        compute_query_metrics("query-1", {"doc-a": 1}, ["doc-a", "doc-a"], 2)


@pytest.mark.parametrize("cutoff", [0, -1, True, 1.5])
def test_invalid_cutoff_is_rejected(cutoff: object) -> None:
    with pytest.raises(DataValidationError, match="cutoff"):
        compute_query_metrics("query-1", {"doc-a": 1}, ["doc-a"], cutoff)  # type: ignore[arg-type]


def test_invalid_relevance_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="non-negative"):
        compute_query_metrics("query-1", {"doc-a": -1}, ["doc-a"], 1)
    with pytest.raises(DataValidationError, match="integers"):
        compute_query_metrics("query-1", {"doc-a": True}, ["doc-a"], 1)
    with pytest.raises(DataValidationError, match="document IDs"):
        compute_query_metrics("query-1", {"": 1}, [], 1)


def test_invalid_ranked_list_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="sequence"):
        compute_query_metrics("query-1", {"doc-a": 1}, "doc-a", 1)
    with pytest.raises(DataValidationError, match="Retrieved document IDs"):
        compute_query_metrics("query-1", {"doc-a": 1}, [""], 1)


def test_aggregate_query_metrics_computes_macro_averages() -> None:
    perfect = compute_query_metrics("query-1", {"doc-a": 1}, ["doc-a"], 2)
    missed = compute_query_metrics("query-2", {"doc-b": 1}, ["doc-x"], 2)

    aggregate = aggregate_query_metrics([perfect, missed])

    assert aggregate.cutoff == 2
    assert aggregate.query_count == 2
    assert aggregate.mean_recall == pytest.approx(0.5)
    assert aggregate.mean_precision == pytest.approx(0.25)
    assert aggregate.mean_reciprocal_rank == pytest.approx(0.5)
    assert aggregate.mean_average_precision == pytest.approx(0.5)
    assert aggregate.mean_ndcg == pytest.approx(0.5)
    assert aggregate.hit_rate == pytest.approx(0.5)


def test_aggregate_query_metrics_rejects_empty_or_mixed_inputs() -> None:
    first = compute_query_metrics("query-1", {"doc-a": 1}, ["doc-a"], 1)
    second = compute_query_metrics("query-2", {"doc-a": 1}, ["doc-a"], 2)

    with pytest.raises(EvaluationError, match="At least one"):
        aggregate_query_metrics([])
    with pytest.raises(DataValidationError, match="same cutoff"):
        aggregate_query_metrics([first, second])
    with pytest.raises(DataValidationError, match="Duplicate query"):
        aggregate_query_metrics([first, first])
