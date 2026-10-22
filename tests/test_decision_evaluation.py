from __future__ import annotations

import json

import pytest

from evidenceguard.data.models import (
    Document,
    EvidenceCondition,
    EvidenceExample,
    Query,
)
from evidenceguard.errors import DataValidationError, EvaluationError
from evidenceguard.evaluation import (
    DecisionEvaluation,
    DecisionMetrics,
    DecisionRecord,
    apply_decision_threshold,
    evaluate_calibration,
    evaluate_decisions,
    evaluate_evidence_assessor,
    fit_abstention_threshold,
    selective_risk_curve,
)
from evidenceguard.evidence import EvidenceAssessor, MetadataStanceClassifier
from evidenceguard.outcomes import DecisionOutcome
from evidenceguard.uncertainty import AbstentionPolicy


def _record(
    example_id: str,
    expected: DecisionOutcome,
    predicted: DecisionOutcome,
    confidence: float,
    *,
    condition: EvidenceCondition | None = None,
) -> DecisionRecord:
    return DecisionRecord(
        example_id=example_id,
        expected_outcome=expected,
        predicted_outcome=predicted,
        confidence=confidence,
        condition=condition,
    )


def _example(
    example_id: str,
    stances: tuple[tuple[str, str, str], ...],
    expected: DecisionOutcome,
    *,
    condition: EvidenceCondition = EvidenceCondition.SUFFICIENT,
) -> EvidenceExample:
    documents = tuple(
        Document(
            doc_id=doc_id,
            title="Title",
            text=f"Evidence for {doc_id}.",
            metadata={
                "evidence_stance": stance,
                "evidence_strength": strength,
            },
        )
        for doc_id, stance, strength in stances
    )
    relevant_ids = tuple(
        doc_id for doc_id, stance, _ in stances if stance == "supports"
    )
    return EvidenceExample(
        example_id=example_id,
        query=Query(query_id=f"query-{example_id}", text="Does it work?"),
        documents=documents,
        relevant_document_ids=relevant_ids,
        expected_outcome=expected,
        condition=condition,
    )


def test_decision_record_properties_and_serialization() -> None:
    record = _record(
        "ex1",
        DecisionOutcome.ANSWER_SUPPORTED,
        DecisionOutcome.INSUFFICIENT_EVIDENCE,
        0.4,
        condition=EvidenceCondition.MISSING,
    )

    assert not record.is_correct
    assert not record.answered
    assert record.to_dict() == {
        "example_id": "ex1",
        "condition": "missing_evidence",
        "expected_outcome": "answer_supported",
        "predicted_outcome": "insufficient_evidence",
        "confidence": 0.4,
        "correct": False,
        "answered": False,
    }


@pytest.mark.parametrize(
    ("example_id", "expected", "predicted", "confidence", "condition"),
    [
        ("", DecisionOutcome.ANSWER_SUPPORTED, DecisionOutcome.ANSWER_SUPPORTED, 0.8, None),
        ("ex1", "answer_supported", DecisionOutcome.ANSWER_SUPPORTED, 0.8, None),
        ("ex1", DecisionOutcome.ANSWER_SUPPORTED, "answer_supported", 0.8, None),
        ("ex1", DecisionOutcome.ANSWER_SUPPORTED, DecisionOutcome.ANSWER_SUPPORTED, True, None),
        (
            "ex1",
            DecisionOutcome.ANSWER_SUPPORTED,
            DecisionOutcome.ANSWER_SUPPORTED,
            -0.1,
            None,
        ),
        (
            "ex1",
            DecisionOutcome.ANSWER_SUPPORTED,
            DecisionOutcome.ANSWER_SUPPORTED,
            1.1,
            None,
        ),
        (
            "ex1",
            DecisionOutcome.ANSWER_SUPPORTED,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.8,
            "sufficient",
        ),
    ],
)
def test_decision_record_rejects_invalid_fields(
    example_id: str,
    expected: object,
    predicted: object,
    confidence: object,
    condition: object,
) -> None:
    with pytest.raises(DataValidationError):
        DecisionRecord(
            example_id=example_id,
            expected_outcome=expected,  # type: ignore[arg-type]
            predicted_outcome=predicted,  # type: ignore[arg-type]
            confidence=confidence,  # type: ignore[arg-type]
            condition=condition,  # type: ignore[arg-type]
        )


def test_evaluate_decisions_computes_coverage_and_selective_risk() -> None:
    records = (
        _record(
            "ex1",
            DecisionOutcome.ANSWER_SUPPORTED,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.9,
        ),
        _record(
            "ex2",
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.6,
        ),
        _record(
            "ex3",
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            0.0,
        ),
        _record(
            "ex4",
            DecisionOutcome.CONFLICTING_EVIDENCE,
            DecisionOutcome.CONFLICTING_EVIDENCE,
            0.0,
        ),
    )

    metrics = evaluate_decisions(records)

    assert metrics.total_count == 4
    assert metrics.answered_count == 2
    assert metrics.correct_count == 3
    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.coverage == pytest.approx(0.5)
    assert metrics.selective_accuracy == pytest.approx(0.5)
    assert metrics.selective_risk == pytest.approx(0.5)
    assert metrics.abstention_rate == pytest.approx(0.5)


def test_decision_metrics_serialization_and_validation() -> None:
    metrics = DecisionMetrics(
        total_count=2,
        answered_count=0,
        correct_count=1,
        accuracy=0.5,
        coverage=0.0,
        selective_accuracy=0.0,
        selective_risk=0.0,
        abstention_rate=1.0,
    )

    assert json.loads(json.dumps(metrics.to_dict()))["total_count"] == 2

    with pytest.raises(DataValidationError, match="total_count"):
        DecisionMetrics(
            total_count=0,
            answered_count=0,
            correct_count=0,
            accuracy=0.0,
            coverage=0.0,
            selective_accuracy=0.0,
            selective_risk=0.0,
            abstention_rate=0.0,
        )


@pytest.mark.parametrize(
    "records",
    [
        (),
        "records",
        (
            DecisionRecord(
                "ex1",
                DecisionOutcome.ANSWER_SUPPORTED,
                DecisionOutcome.ANSWER_SUPPORTED,
                0.8,
            ),
        )
        * 2,
        (object(),),
    ],
)
def test_evaluate_decisions_rejects_invalid_records(records: object) -> None:
    with pytest.raises(EvaluationError):
        evaluate_decisions(records)  # type: ignore[arg-type]


def test_decision_evaluation_serialization() -> None:
    records = (
        _record(
            "ex1",
            DecisionOutcome.ANSWER_SUPPORTED,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.8,
            condition=EvidenceCondition.SUFFICIENT,
        ),
    )
    evaluation = DecisionEvaluation(
        records=records,
        metrics=evaluate_decisions(records),
        policy_threshold=0.5,
    )

    payload = evaluation.to_dict()

    assert payload["policy_threshold"] == 0.5
    assert payload["metrics"]["coverage"] == 1.0
    assert payload["records"][0]["example_id"] == "ex1"
    json.dumps(payload)


def test_apply_decision_threshold_abstains_below_threshold() -> None:
    records = (
        _record(
            "ex1",
            DecisionOutcome.ANSWER_SUPPORTED,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.9,
        ),
        _record(
            "ex2",
            DecisionOutcome.ANSWER_SUPPORTED,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.4,
        ),
        _record(
            "ex3",
            DecisionOutcome.CONFLICTING_EVIDENCE,
            DecisionOutcome.CONFLICTING_EVIDENCE,
            0.0,
        ),
    )

    evaluation = apply_decision_threshold(records, 0.7)

    assert evaluation.policy_threshold == 0.7
    assert evaluation.metrics.coverage == pytest.approx(1.0 / 3.0)
    assert evaluation.metrics.selective_risk == 0.0
    assert [record.predicted_outcome for record in evaluation.records] == [
        DecisionOutcome.ANSWER_SUPPORTED,
        DecisionOutcome.INSUFFICIENT_EVIDENCE,
        DecisionOutcome.CONFLICTING_EVIDENCE,
    ]


def test_evaluate_evidence_assessor_applies_policy() -> None:
    examples = (
        _example(
            "ex1",
            (("doc-a", "supports", "0.9"),),
            DecisionOutcome.ANSWER_SUPPORTED,
        ),
        _example(
            "ex2",
            (("doc-a", "supports", "0.4"),),
            DecisionOutcome.ANSWER_SUPPORTED,
        ),
        _example(
            "ex3",
            (("doc-a", "contradicts", "0.8"),),
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            condition=EvidenceCondition.MISSING,
        ),
        _example(
            "ex4",
            (("doc-a", "supports", "0.8"), ("doc-b", "contradicts", "0.7")),
            DecisionOutcome.CONFLICTING_EVIDENCE,
            condition=EvidenceCondition.CONTRADICTORY,
        ),
    )

    evaluation = evaluate_evidence_assessor(
        EvidenceAssessor(MetadataStanceClassifier()),
        examples,
        policy=AbstentionPolicy(0.5),
    )

    assert evaluation.policy_threshold == 0.5
    assert evaluation.metrics.total_count == 4
    assert evaluation.metrics.answered_count == 1
    assert evaluation.metrics.correct_count == 3
    assert evaluation.metrics.coverage == pytest.approx(0.25)
    assert evaluation.metrics.selective_risk == 0.0
    assert [record.confidence for record in evaluation.records] == [0.9, 0.4, 0.0, 0.0]
    assert [record.condition for record in evaluation.records] == [
        EvidenceCondition.SUFFICIENT,
        EvidenceCondition.SUFFICIENT,
        EvidenceCondition.MISSING,
        EvidenceCondition.CONTRADICTORY,
    ]


def test_evaluate_evidence_assessor_rejects_invalid_inputs() -> None:
    assessor = EvidenceAssessor(MetadataStanceClassifier())
    example = _example(
        "ex1",
        (("doc-a", "supports", "0.8"),),
        DecisionOutcome.ANSWER_SUPPORTED,
    )

    with pytest.raises(EvaluationError, match="EvidenceAssessor"):
        evaluate_evidence_assessor(object(), (example,))  # type: ignore[arg-type]
    with pytest.raises(EvaluationError, match="At least one"):
        evaluate_evidence_assessor(assessor, ())
    with pytest.raises(EvaluationError, match="sequence"):
        evaluate_evidence_assessor(assessor, "examples")  # type: ignore[arg-type]
    with pytest.raises(EvaluationError, match="EvidenceExample"):
        evaluate_evidence_assessor(assessor, (object(),))  # type: ignore[arg-type]
    with pytest.raises(EvaluationError, match="AbstentionPolicy"):
        evaluate_evidence_assessor(assessor, (example,), policy=object())  # type: ignore[arg-type]


def test_evaluate_calibration_computes_brier_and_ece() -> None:
    records = (
        _record(
            "ex1",
            DecisionOutcome.ANSWER_SUPPORTED,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.9,
        ),
        _record(
            "ex2",
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.8,
        ),
        _record(
            "ex3",
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            0.2,
        ),
    )

    metrics = evaluate_calibration(records, n_bins=2)

    assert metrics.count == 2
    assert metrics.brier_score == pytest.approx(0.325)
    assert metrics.expected_calibration_error == pytest.approx(0.35)
    assert len(metrics.bins) == 1
    assert metrics.bins[0].count == 2
    assert metrics.bins[0].mean_confidence == pytest.approx(0.85)
    assert metrics.bins[0].accuracy == pytest.approx(0.5)
    json.dumps(metrics.to_dict())


def test_evaluate_calibration_places_one_in_final_bin() -> None:
    records = (
        _record(
            "ex1",
            DecisionOutcome.ANSWER_SUPPORTED,
            DecisionOutcome.ANSWER_SUPPORTED,
            1.0,
        ),
    )

    metrics = evaluate_calibration(records, n_bins=10)

    assert metrics.bins[-1].lower_bound == pytest.approx(0.9)
    assert metrics.bins[-1].accuracy == 1.0


def test_evaluate_calibration_requires_answered_predictions() -> None:
    records = (
        _record(
            "ex1",
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            0.0,
        ),
    )

    with pytest.raises(EvaluationError, match="No answered"):
        evaluate_calibration(records)


@pytest.mark.parametrize("n_bins", [0, True, 2.5, "10"])
def test_evaluate_calibration_rejects_invalid_bin_count(n_bins: object) -> None:
    records = (
        _record(
            "ex1",
            DecisionOutcome.ANSWER_SUPPORTED,
            DecisionOutcome.ANSWER_SUPPORTED,
            1.0,
        ),
    )

    with pytest.raises(DataValidationError, match="n_bins"):
        evaluate_calibration(records, n_bins=n_bins)  # type: ignore[arg-type]


def test_selective_risk_curve_applies_sorted_thresholds() -> None:
    records = (
        _record(
            "ex1",
            DecisionOutcome.ANSWER_SUPPORTED,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.9,
        ),
        _record(
            "ex2",
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.6,
        ),
        _record(
            "ex3",
            DecisionOutcome.CONFLICTING_EVIDENCE,
            DecisionOutcome.CONFLICTING_EVIDENCE,
            0.0,
        ),
    )

    curve = selective_risk_curve(records, (0.7, 0.0))

    assert [evaluation.policy_threshold for evaluation in curve] == [0.0, 0.7]
    assert curve[0].metrics.coverage == pytest.approx(2.0 / 3.0)
    assert curve[0].metrics.selective_risk == pytest.approx(0.5)
    assert curve[1].metrics.coverage == pytest.approx(1.0 / 3.0)
    assert curve[1].metrics.selective_risk == 0.0


def test_fit_abstention_threshold_chooses_max_coverage_under_risk() -> None:
    records = (
        _record(
            "ex1",
            DecisionOutcome.ANSWER_SUPPORTED,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.9,
        ),
        _record(
            "ex2",
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.6,
        ),
        _record(
            "ex3",
            DecisionOutcome.CONFLICTING_EVIDENCE,
            DecisionOutcome.CONFLICTING_EVIDENCE,
            0.0,
        ),
        _record(
            "ex4",
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            0.0,
        ),
    )

    selection = fit_abstention_threshold(records, maximum_risk=0.2)

    assert selection.threshold == 0.9
    assert selection.maximum_risk == 0.2
    assert selection.candidate_count == 4
    assert selection.metrics.coverage == pytest.approx(0.25)
    assert selection.metrics.selective_risk == 0.0
    json.dumps(selection.to_dict())


def test_fit_abstention_threshold_prefers_conservative_tie_break() -> None:
    records = (
        _record(
            "ex1",
            DecisionOutcome.ANSWER_SUPPORTED,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.9,
        ),
        _record(
            "ex2",
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.6,
        ),
    )

    selection = fit_abstention_threshold(records, maximum_risk=1.0)

    assert selection.threshold == 0.6
    assert selection.metrics.coverage == 1.0


def test_fit_abstention_threshold_rejects_invalid_inputs() -> None:
    records = (
        _record(
            "ex1",
            DecisionOutcome.ANSWER_SUPPORTED,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.9,
        ),
    )

    with pytest.raises(DataValidationError, match="maximum_risk"):
        fit_abstention_threshold(records, maximum_risk=1.1)
    with pytest.raises(EvaluationError, match="At least one"):
        fit_abstention_threshold((), maximum_risk=0.2)
    with pytest.raises(EvaluationError, match="unique"):
        fit_abstention_threshold((records[0], records[0]), maximum_risk=0.2)


def test_selective_risk_curve_rejects_invalid_thresholds() -> None:
    records = (
        _record(
            "ex1",
            DecisionOutcome.ANSWER_SUPPORTED,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.9,
        ),
    )

    for thresholds in ((), (0.5, 0.5), (-0.1,), (1.1,), "0.5"):
        with pytest.raises(DataValidationError):
            selective_risk_curve(records, thresholds)  # type: ignore[arg-type]
