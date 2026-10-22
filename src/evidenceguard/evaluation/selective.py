"""Risk-targeted threshold selection on development decision records."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from evidenceguard.errors import DataValidationError, EvaluationError
from evidenceguard.evaluation.decisions import (
    DecisionEvaluation,
    DecisionMetrics,
    DecisionRecord,
    apply_decision_threshold,
)
from evidenceguard.outcomes import DecisionOutcome


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    """A development-selected threshold and its measured risk-coverage trade-off."""

    maximum_risk: float
    threshold: float
    candidate_count: int
    metrics: DecisionMetrics

    def __post_init__(self) -> None:
        _validate_unit_interval(self.maximum_risk, "maximum_risk")
        _validate_unit_interval(self.threshold, "threshold")
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count < 1
        ):
            raise DataValidationError("candidate_count must be a positive integer.")
        if not isinstance(self.metrics, DecisionMetrics):
            raise DataValidationError("metrics must be a DecisionMetrics instance.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "maximum_risk": self.maximum_risk,
            "threshold": self.threshold,
            "candidate_count": self.candidate_count,
            "metrics": self.metrics.to_dict(),
        }


def selective_risk_curve(
    records: Sequence[DecisionRecord],
    thresholds: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[DecisionEvaluation, ...]:
    """Return thresholded evaluations at each requested confidence threshold."""

    if isinstance(thresholds, (str, bytes)):
        raise DataValidationError("thresholds must be a sequence of numbers.")
    try:
        values = tuple(thresholds)
    except TypeError as exc:
        raise DataValidationError(
            "thresholds must be a sequence of numbers."
        ) from exc
    if not values:
        raise DataValidationError("At least one threshold is required.")
    for value in values:
        _validate_unit_interval(value, "threshold")
    if len(set(values)) != len(values):
        raise DataValidationError("thresholds must not contain duplicates.")
    return tuple(
        apply_decision_threshold(records, threshold)
        for threshold in sorted(float(value) for value in values)
    )


def fit_abstention_threshold(
    records: Sequence[DecisionRecord],
    *,
    maximum_risk: float,
) -> ThresholdSelection:
    """Choose the highest-coverage development threshold under a risk budget.

    Candidate thresholds are ``0.0``, ``1.0``, and every confidence attached to
    a candidate answer. Ties prefer lower selective risk and then the higher,
    more conservative threshold. Thresholds must be fitted on development data.
    """

    _validate_unit_interval(maximum_risk, "maximum_risk")
    try:
        validated = tuple(records)
    except TypeError as exc:
        raise EvaluationError(
            "records must be a sequence of DecisionRecord objects."
        ) from exc
    if not validated:
        raise EvaluationError("At least one decision record is required.")
    if any(not isinstance(record, DecisionRecord) for record in validated):
        raise EvaluationError("records must contain only DecisionRecord objects.")
    example_ids = [record.example_id for record in validated]
    if len(set(example_ids)) != len(example_ids):
        raise EvaluationError("Decision records must have unique example IDs.")

    candidates = sorted(
        {
            0.0,
            1.0,
            *(
                record.confidence
                for record in validated
                if record.predicted_outcome is DecisionOutcome.ANSWER_SUPPORTED
            ),
        }
    )
    evaluations = tuple(
        apply_decision_threshold(validated, threshold) for threshold in candidates
    )
    feasible = tuple(
        evaluation
        for evaluation in evaluations
        if evaluation.metrics.selective_risk <= maximum_risk
    )
    if not feasible:
        raise EvaluationError("No candidate threshold satisfies the risk budget.")

    selected = min(
        feasible,
        key=lambda evaluation: (
            -evaluation.metrics.coverage,
            evaluation.metrics.selective_risk,
            -(evaluation.policy_threshold or 0.0),
        ),
    )
    assert selected.policy_threshold is not None
    return ThresholdSelection(
        maximum_risk=float(maximum_risk),
        threshold=selected.policy_threshold,
        candidate_count=len(candidates),
        metrics=selected.metrics,
    )


def _validate_unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataValidationError(f"{name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise DataValidationError(f"{name} must be between 0 and 1.")
    return numeric
