"""Threshold-based abstention over evidence assessments."""

from __future__ import annotations

import math
from dataclasses import dataclass

from evidenceguard.errors import DataValidationError
from evidenceguard.evidence.models import EvidenceAssessment
from evidenceguard.outcomes import DecisionOutcome


def answer_confidence(assessment: EvidenceAssessment) -> float:
    """Return the uncalibrated confidence that an answer is supported.

    Non-answer evidence assessments return ``0.0``. The value is the maximum
    supporting-document strength, not a calibrated probability.
    """

    if not isinstance(assessment, EvidenceAssessment):
        raise DataValidationError("assessment must be an EvidenceAssessment.")
    if assessment.decision is not DecisionOutcome.ANSWER_SUPPORTED:
        return 0.0
    return assessment.support_strength


@dataclass(frozen=True, slots=True)
class AbstentionPolicy:
    """Withhold supported answers below a confidence threshold.

    Conflicting assessments remain ``CONFLICTING_EVIDENCE``. Other assessments
    remain ``INSUFFICIENT_EVIDENCE``. The threshold must be fitted on
    development data and is not a calibrated probability cut-point by default.
    """

    minimum_confidence: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.minimum_confidence, bool) or not isinstance(
            self.minimum_confidence, (int, float)
        ):
            raise DataValidationError("minimum_confidence must be numeric.")
        threshold = float(self.minimum_confidence)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise DataValidationError(
                "minimum_confidence must be between 0 and 1."
            )
        object.__setattr__(self, "minimum_confidence", threshold)

    def decide(self, assessment: EvidenceAssessment) -> DecisionOutcome:
        """Return the thresholded decision for one evidence assessment."""

        if not isinstance(assessment, EvidenceAssessment):
            raise DataValidationError("assessment must be an EvidenceAssessment.")
        if assessment.decision is DecisionOutcome.CONFLICTING_EVIDENCE:
            return DecisionOutcome.CONFLICTING_EVIDENCE
        if (
            assessment.decision is DecisionOutcome.ANSWER_SUPPORTED
            and assessment.support_strength >= self.minimum_confidence
        ):
            return DecisionOutcome.ANSWER_SUPPORTED
        return DecisionOutcome.INSUFFICIENT_EVIDENCE
