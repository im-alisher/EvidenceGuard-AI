"""Confidence calibration diagnostics for surfaced answers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from evidenceguard.errors import DataValidationError, EvaluationError
from evidenceguard.evaluation.decisions import DecisionRecord, _validate_records
from evidenceguard.outcomes import DecisionOutcome


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """Observed confidence and correctness within one probability interval."""

    lower_bound: float
    upper_bound: float
    count: int
    mean_confidence: float
    accuracy: float

    def __post_init__(self) -> None:
        _validate_unit_interval(self.lower_bound, "lower_bound")
        _validate_unit_interval(self.upper_bound, "upper_bound")
        if self.lower_bound >= self.upper_bound:
            raise DataValidationError(
                "Calibration bin lower_bound must be less than upper_bound."
            )
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise DataValidationError("Calibration bin count must be positive.")
        _validate_unit_interval(self.mean_confidence, "mean_confidence")
        _validate_unit_interval(self.accuracy, "accuracy")

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable representation."""

        return {
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "count": self.count,
            "mean_confidence": self.mean_confidence,
            "accuracy": self.accuracy,
        }


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    """Brier score and expected calibration error over answered predictions."""

    count: int
    brier_score: float
    expected_calibration_error: float
    bins: tuple[CalibrationBin, ...]

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise DataValidationError("Calibration count must be positive.")
        _validate_unit_interval(self.brier_score, "brier_score")
        _validate_unit_interval(
            self.expected_calibration_error,
            "expected_calibration_error",
        )
        object.__setattr__(self, "bins", tuple(self.bins))
        if any(not isinstance(item, CalibrationBin) for item in self.bins):
            raise DataValidationError(
                "bins must contain only CalibrationBin instances."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "count": self.count,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "bins": [item.to_dict() for item in self.bins],
        }


def evaluate_calibration(
    records: Sequence[DecisionRecord],
    *,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Evaluate answer confidence on records predicted to answer.

    Only records with ``predicted_outcome == ANSWER_SUPPORTED`` are included.
    A prediction is correct for calibration when the expected outcome is also
    ``ANSWER_SUPPORTED``. This isolates confidence in surfaced answers and does
    not measure calibration of abstention decisions.
    """

    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise DataValidationError("n_bins must be a positive integer.")
    validated = _validate_records(records)
    answered = tuple(record for record in validated if record.answered)
    if not answered:
        raise EvaluationError("No answered predictions are available for calibration.")

    outcomes = tuple(
        record.expected_outcome is DecisionOutcome.ANSWER_SUPPORTED
        for record in answered
    )
    brier_score = math.fsum(
        (record.confidence - float(correct)) ** 2
        for record, correct in zip(answered, outcomes, strict=True)
    ) / len(answered)

    bins: list[CalibrationBin] = []
    expected_calibration_error = 0.0
    for index in range(n_bins):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        members = tuple(
            (record, correct)
            for record, correct in zip(answered, outcomes, strict=True)
            if lower <= record.confidence < upper
            or (index == n_bins - 1 and record.confidence == 1.0)
        )
        if not members:
            continue
        count = len(members)
        mean_confidence = math.fsum(
            record.confidence for record, _ in members
        ) / count
        accuracy = math.fsum(float(correct) for _, correct in members) / count
        expected_calibration_error += (
            count / len(answered) * abs(mean_confidence - accuracy)
        )
        bins.append(
            CalibrationBin(
                lower_bound=lower,
                upper_bound=upper,
                count=count,
                mean_confidence=mean_confidence,
                accuracy=accuracy,
            )
        )

    return CalibrationMetrics(
        count=len(answered),
        brier_score=brier_score,
        expected_calibration_error=expected_calibration_error,
        bins=tuple(bins),
    )


def _validate_unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataValidationError(f"{name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise DataValidationError(f"{name} must be between 0 and 1.")
    return numeric
