"""Decision-level evaluation for evidence and abstention outputs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from evidenceguard.data.models import EvidenceCondition, EvidenceExample
from evidenceguard.errors import DataValidationError, EvaluationError
from evidenceguard.evidence.assessor import EvidenceAssessor
from evidenceguard.outcomes import DecisionOutcome
from evidenceguard.uncertainty.abstention import AbstentionPolicy, answer_confidence


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """One predicted decision with confidence in a surfaced answer."""

    example_id: str
    expected_outcome: DecisionOutcome
    predicted_outcome: DecisionOutcome
    confidence: float
    condition: EvidenceCondition | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not self.example_id.strip():
            raise DataValidationError("example_id must be a non-empty string.")
        if not isinstance(self.expected_outcome, DecisionOutcome):
            raise DataValidationError("expected_outcome must be a DecisionOutcome.")
        if not isinstance(self.predicted_outcome, DecisionOutcome):
            raise DataValidationError("predicted_outcome must be a DecisionOutcome.")
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise DataValidationError("confidence must be numeric.")
        numeric_confidence = float(self.confidence)
        if not math.isfinite(numeric_confidence) or not 0.0 <= numeric_confidence <= 1.0:
            raise DataValidationError("confidence must be between 0 and 1.")
        object.__setattr__(self, "confidence", numeric_confidence)
        if self.condition is not None and not isinstance(
            self.condition, EvidenceCondition
        ):
            raise DataValidationError("condition must be an EvidenceCondition or None.")

    @property
    def is_correct(self) -> bool:
        """Return whether the predicted and expected outcomes match."""

        return self.predicted_outcome is self.expected_outcome

    @property
    def answered(self) -> bool:
        """Return whether the decision surfaces an answer."""

        return self.predicted_outcome is DecisionOutcome.ANSWER_SUPPORTED

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "example_id": self.example_id,
            "condition": None if self.condition is None else self.condition.value,
            "expected_outcome": self.expected_outcome.value,
            "predicted_outcome": self.predicted_outcome.value,
            "confidence": self.confidence,
            "correct": self.is_correct,
            "answered": self.answered,
        }


@dataclass(frozen=True, slots=True)
class DecisionMetrics:
    """Aggregate answer coverage and selective risk for decision records."""

    total_count: int
    answered_count: int
    correct_count: int
    accuracy: float
    coverage: float
    selective_accuracy: float
    selective_risk: float
    abstention_rate: float

    def __post_init__(self) -> None:
        for name, value in (
            ("total_count", self.total_count),
            ("answered_count", self.answered_count),
            ("correct_count", self.correct_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DataValidationError(f"{name} must be a non-negative integer.")
        if self.total_count < 1:
            raise DataValidationError("total_count must be positive.")
        if self.answered_count > self.total_count:
            raise DataValidationError("answered_count cannot exceed total_count.")
        if self.correct_count > self.total_count:
            raise DataValidationError("correct_count cannot exceed total_count.")
        for name, value in (
            ("accuracy", self.accuracy),
            ("coverage", self.coverage),
            ("selective_accuracy", self.selective_accuracy),
            ("selective_risk", self.selective_risk),
            ("abstention_rate", self.abstention_rate),
        ):
            _validate_unit_interval(value, name)

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class DecisionEvaluation:
    """Decision records, aggregate metrics, and the applied threshold."""

    records: tuple[DecisionRecord, ...]
    metrics: DecisionMetrics
    policy_threshold: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "records",
            _validate_records(self.records),
        )
        if not isinstance(self.metrics, DecisionMetrics):
            raise DataValidationError("metrics must be a DecisionMetrics instance.")
        if self.policy_threshold is not None:
            _validate_unit_interval(self.policy_threshold, "policy_threshold")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "policy_threshold": self.policy_threshold,
            "metrics": self.metrics.to_dict(),
            "records": [record.to_dict() for record in self.records],
        }


def evaluate_decisions(records: Sequence[DecisionRecord]) -> DecisionMetrics:
    """Compute coverage, selective accuracy, and selective risk."""

    validated = _validate_records(records)
    total_count = len(validated)
    answered_count = sum(1 for record in validated if record.answered)
    correct_count = sum(1 for record in validated if record.is_correct)
    correct_answered = sum(
        1 for record in validated if record.answered and record.is_correct
    )

    accuracy = correct_count / total_count
    coverage = answered_count / total_count
    selective_accuracy = (
        correct_answered / answered_count if answered_count else 0.0
    )
    return DecisionMetrics(
        total_count=total_count,
        answered_count=answered_count,
        correct_count=correct_count,
        accuracy=accuracy,
        coverage=coverage,
        selective_accuracy=selective_accuracy,
        selective_risk=1.0 - selective_accuracy,
        abstention_rate=1.0 - coverage,
    )


def evaluate_evidence_assessor(
    assessor: EvidenceAssessor,
    examples: Sequence[EvidenceExample],
    *,
    policy: AbstentionPolicy | None = None,
) -> DecisionEvaluation:
    """Assess controlled examples and evaluate the resulting decisions."""

    if not isinstance(assessor, EvidenceAssessor):
        raise EvaluationError("assessor must be an EvidenceAssessor.")
    if isinstance(examples, (str, bytes)):
        raise EvaluationError("examples must be a sequence of EvidenceExample objects.")
    try:
        values = tuple(examples)
    except TypeError as exc:
        raise EvaluationError(
            "examples must be a sequence of EvidenceExample objects."
        ) from exc
    if not values:
        raise EvaluationError("At least one evidence example is required.")
    if any(not isinstance(example, EvidenceExample) for example in values):
        raise EvaluationError(
            "examples must contain only EvidenceExample objects."
        )

    applied_policy = AbstentionPolicy() if policy is None else policy
    if not isinstance(applied_policy, AbstentionPolicy):
        raise EvaluationError("policy must be an AbstentionPolicy or None.")

    records: list[DecisionRecord] = []
    for example in values:
        assessment = assessor.assess(example.query, example.documents)
        records.append(
            DecisionRecord(
                example_id=example.example_id,
                expected_outcome=example.expected_outcome,
                predicted_outcome=applied_policy.decide(assessment),
                confidence=answer_confidence(assessment),
                condition=example.condition,
            )
        )

    validated = _validate_records(records)
    return DecisionEvaluation(
        records=validated,
        metrics=evaluate_decisions(validated),
        policy_threshold=applied_policy.minimum_confidence,
    )


def apply_decision_threshold(
    records: Sequence[DecisionRecord],
    minimum_confidence: float,
) -> DecisionEvaluation:
    """Abstain on answer predictions below a confidence threshold."""

    _validate_unit_interval(minimum_confidence, "minimum_confidence")
    validated = _validate_records(records)
    thresholded: list[DecisionRecord] = []
    for record in validated:
        predicted = record.predicted_outcome
        if (
            predicted is DecisionOutcome.ANSWER_SUPPORTED
            and record.confidence < minimum_confidence
        ):
            predicted = DecisionOutcome.INSUFFICIENT_EVIDENCE
        thresholded.append(replace(record, predicted_outcome=predicted))
    final_records = _validate_records(thresholded)
    return DecisionEvaluation(
        records=final_records,
        metrics=evaluate_decisions(final_records),
        policy_threshold=float(minimum_confidence),
    )


def _validate_records(records: Sequence[DecisionRecord]) -> tuple[DecisionRecord, ...]:
    if isinstance(records, (str, bytes)):
        raise EvaluationError("records must be a sequence of DecisionRecord objects.")
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
    return validated


def _validate_unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataValidationError(f"{name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise DataValidationError(f"{name} must be between 0 and 1.")
    return numeric
