"""Outcome labels for evidence-grounded decisions."""

from __future__ import annotations

from enum import Enum


class DecisionOutcome(str, Enum):
    """The decision returned by an evidence assessment policy."""

    ANSWER_SUPPORTED = "answer_supported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"

    @property
    def is_answered(self) -> bool:
        """Return whether the decision permits an answer to be surfaced."""
        return self is DecisionOutcome.ANSWER_SUPPORTED

    @property
    def is_abstention(self) -> bool:
        """Return whether the decision withholds an answer."""
        return not self.is_answered
