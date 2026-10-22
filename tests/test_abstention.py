from __future__ import annotations

import pytest

from evidenceguard.errors import DataValidationError
from evidenceguard.evidence import DocumentEvidence, EvidenceAssessment, EvidenceStance
from evidenceguard.outcomes import DecisionOutcome
from evidenceguard.uncertainty import AbstentionPolicy, answer_confidence


def _assessment(
    decision: DecisionOutcome,
    *,
    support_strength: float = 0.8,
) -> EvidenceAssessment:
    evidence: list[DocumentEvidence] = []
    if decision is DecisionOutcome.ANSWER_SUPPORTED:
        evidence.append(
            DocumentEvidence("doc-a", EvidenceStance.SUPPORTS, support_strength)
        )
    elif decision is DecisionOutcome.CONFLICTING_EVIDENCE:
        evidence.extend(
            (
                DocumentEvidence("doc-a", EvidenceStance.SUPPORTS, support_strength),
                DocumentEvidence("doc-b", EvidenceStance.CONTRADICTS, 0.7),
            )
        )
    else:
        evidence.append(DocumentEvidence("doc-a", EvidenceStance.NEUTRAL))
    return EvidenceAssessment(
        query_id="q1",
        decision=decision,
        document_evidence=tuple(evidence),
    )


def test_answer_confidence_returns_support_strength() -> None:
    assessment = _assessment(DecisionOutcome.ANSWER_SUPPORTED, support_strength=0.8)

    assert answer_confidence(assessment) == 0.8


@pytest.mark.parametrize(
    "decision",
    [
        DecisionOutcome.INSUFFICIENT_EVIDENCE,
        DecisionOutcome.CONFLICTING_EVIDENCE,
    ],
)
def test_answer_confidence_is_zero_for_abstentions(
    decision: DecisionOutcome,
) -> None:
    assert answer_confidence(_assessment(decision)) == 0.0


def test_answer_confidence_rejects_invalid_assessment() -> None:
    with pytest.raises(DataValidationError, match="EvidenceAssessment"):
        answer_confidence("assessment")  # type: ignore[arg-type]


def test_abstention_policy_answers_at_or_above_threshold() -> None:
    policy = AbstentionPolicy(0.8)

    assert (
        policy.decide(_assessment(DecisionOutcome.ANSWER_SUPPORTED, support_strength=0.8))
        is DecisionOutcome.ANSWER_SUPPORTED
    )
    assert (
        policy.decide(_assessment(DecisionOutcome.ANSWER_SUPPORTED, support_strength=0.9))
        is DecisionOutcome.ANSWER_SUPPORTED
    )


def test_abstention_policy_withholds_low_confidence_answer() -> None:
    policy = AbstentionPolicy(0.8)

    assert (
        policy.decide(_assessment(DecisionOutcome.ANSWER_SUPPORTED, support_strength=0.7))
        is DecisionOutcome.INSUFFICIENT_EVIDENCE
    )


def test_abstention_policy_preserves_conflict_and_insufficiency() -> None:
    policy = AbstentionPolicy(0.8)

    assert (
        policy.decide(_assessment(DecisionOutcome.CONFLICTING_EVIDENCE))
        is DecisionOutcome.CONFLICTING_EVIDENCE
    )
    assert (
        policy.decide(_assessment(DecisionOutcome.INSUFFICIENT_EVIDENCE))
        is DecisionOutcome.INSUFFICIENT_EVIDENCE
    )


def test_abstention_policy_normalizes_threshold() -> None:
    policy = AbstentionPolicy(1)

    assert policy.minimum_confidence == 1.0
    assert isinstance(policy.minimum_confidence, float)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, True, "0.5", float("nan"), float("inf")])
def test_abstention_policy_rejects_invalid_threshold(threshold: object) -> None:
    with pytest.raises(DataValidationError, match="minimum_confidence"):
        AbstentionPolicy(threshold)  # type: ignore[arg-type]


def test_abstention_policy_rejects_invalid_assessment() -> None:
    with pytest.raises(DataValidationError, match="EvidenceAssessment"):
        AbstentionPolicy().decide("assessment")  # type: ignore[arg-type]
