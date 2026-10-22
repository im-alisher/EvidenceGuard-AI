from __future__ import annotations

import pytest

from evidenceguard.outcomes import DecisionOutcome


def test_only_supported_answer_counts_as_answered() -> None:
    assert DecisionOutcome.ANSWER_SUPPORTED.is_answered is True
    assert DecisionOutcome.ANSWER_SUPPORTED.is_abstention is False

    assert DecisionOutcome.INSUFFICIENT_EVIDENCE.is_answered is False
    assert DecisionOutcome.INSUFFICIENT_EVIDENCE.is_abstention is True

    assert DecisionOutcome.CONFLICTING_EVIDENCE.is_answered is False
    assert DecisionOutcome.CONFLICTING_EVIDENCE.is_abstention is True


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("answer_supported", DecisionOutcome.ANSWER_SUPPORTED),
        ("insufficient_evidence", DecisionOutcome.INSUFFICIENT_EVIDENCE),
        ("conflicting_evidence", DecisionOutcome.CONFLICTING_EVIDENCE),
    ],
)
def test_outcome_serializes_as_stable_string(
    raw_value: str, expected: DecisionOutcome
) -> None:
    assert DecisionOutcome(raw_value) is expected
    assert expected.value == raw_value
