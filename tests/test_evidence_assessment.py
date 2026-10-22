from __future__ import annotations

import json

import pytest

from evidenceguard.data.models import Document, Query
from evidenceguard.errors import DataValidationError
from evidenceguard.evidence import (
    STANCE_METADATA_KEY,
    STRENGTH_METADATA_KEY,
    DocumentEvidence,
    EvidenceAssessment,
    EvidenceAssessor,
    EvidenceStance,
    MetadataStanceClassifier,
)
from evidenceguard.outcomes import DecisionOutcome


def _document(
    doc_id: str,
    *,
    stance: str | None = None,
    strength: str | None = None,
    text: str = "evidence text",
) -> Document:
    metadata: dict[str, str] = {}
    if stance is not None:
        metadata[STANCE_METADATA_KEY] = stance
    if strength is not None:
        metadata[STRENGTH_METADATA_KEY] = strength
    return Document(doc_id=doc_id, title="Title", text=text, metadata=metadata)


def _query() -> Query:
    return Query(query_id="q1", text="Does treatment improve outcomes?")


class _StaticClassifier:
    def __init__(self, evidence: object) -> None:
        self.evidence = evidence

    def classify(self, query: Query, documents: object) -> object:
        return self.evidence


def test_evidence_stance_values() -> None:
    assert [stance.value for stance in EvidenceStance] == [
        "supports",
        "contradicts",
        "neutral",
        "unknown",
    ]


def test_document_evidence_normalizes_strength() -> None:
    evidence = DocumentEvidence(doc_id="doc-a", stance=EvidenceStance.SUPPORTS, strength=1)

    assert evidence.strength == 1.0
    assert isinstance(evidence.strength, float)


@pytest.mark.parametrize(
    ("doc_id", "stance", "strength"),
    [
        ("", EvidenceStance.SUPPORTS, 1.0),
        ("   ", EvidenceStance.SUPPORTS, 1.0),
        ("doc-a", "supports", 1.0),
        ("doc-a", EvidenceStance.SUPPORTS, True),
        ("doc-a", EvidenceStance.SUPPORTS, "0.8"),
        ("doc-a", EvidenceStance.SUPPORTS, -0.1),
        ("doc-a", EvidenceStance.SUPPORTS, 1.1),
        ("doc-a", EvidenceStance.SUPPORTS, float("nan")),
        ("doc-a", EvidenceStance.SUPPORTS, float("inf")),
    ],
)
def test_document_evidence_rejects_invalid_values(
    doc_id: str,
    stance: object,
    strength: object,
) -> None:
    with pytest.raises(DataValidationError):
        DocumentEvidence(doc_id=doc_id, stance=stance, strength=strength)  # type: ignore[arg-type]


def test_assessment_exposes_stance_groups_and_strengths() -> None:
    assessment = EvidenceAssessment(
        query_id="q1",
        decision=DecisionOutcome.CONFLICTING_EVIDENCE,
        document_evidence=(
            DocumentEvidence("doc-a", EvidenceStance.SUPPORTS, 0.4),
            DocumentEvidence("doc-b", EvidenceStance.SUPPORTS, 0.9),
            DocumentEvidence("doc-c", EvidenceStance.CONTRADICTS, 0.7),
            DocumentEvidence("doc-d", EvidenceStance.NEUTRAL, 0.2),
            DocumentEvidence("doc-e", EvidenceStance.UNKNOWN),
        ),
    )

    assert assessment.supporting_doc_ids == ("doc-a", "doc-b")
    assert assessment.contradicting_doc_ids == ("doc-c",)
    assert assessment.neutral_doc_ids == ("doc-d",)
    assert assessment.unknown_doc_ids == ("doc-e",)
    assert assessment.support_strength == 0.9
    assert assessment.contradiction_strength == 0.7


def test_assessment_to_dict_is_json_serializable() -> None:
    assessment = EvidenceAssessment(
        query_id="q1",
        decision=DecisionOutcome.ANSWER_SUPPORTED,
        document_evidence=(
            DocumentEvidence("doc-a", EvidenceStance.SUPPORTS, 0.8),
        ),
    )

    payload = assessment.to_dict()

    assert json.loads(json.dumps(payload)) == {
        "query_id": "q1",
        "decision": "answer_supported",
        "supporting_doc_ids": ["doc-a"],
        "contradicting_doc_ids": [],
        "neutral_doc_ids": [],
        "unknown_doc_ids": [],
        "support_strength": 0.8,
        "contradiction_strength": 0.0,
        "document_evidence": [
            {"doc_id": "doc-a", "stance": "supports", "strength": 0.8}
        ],
    }


def test_assessment_rejects_duplicate_documents() -> None:
    with pytest.raises(DataValidationError, match="duplicate"):
        EvidenceAssessment(
            query_id="q1",
            decision=DecisionOutcome.INSUFFICIENT_EVIDENCE,
            document_evidence=(
                DocumentEvidence("doc-a", EvidenceStance.NEUTRAL),
                DocumentEvidence("doc-a", EvidenceStance.UNKNOWN),
            ),
        )


@pytest.mark.parametrize(
    ("query_id", "decision", "document_evidence"),
    [
        ("", DecisionOutcome.INSUFFICIENT_EVIDENCE, ()),
        ("q1", "insufficient_evidence", ()),
        ("q1", DecisionOutcome.INSUFFICIENT_EVIDENCE, ("bad",)),
        ("q1", DecisionOutcome.INSUFFICIENT_EVIDENCE, "bad"),
        ("q1", DecisionOutcome.INSUFFICIENT_EVIDENCE, None),
    ],
)
def test_assessment_rejects_invalid_fields(
    query_id: str,
    decision: object,
    document_evidence: object,
) -> None:
    with pytest.raises(DataValidationError):
        EvidenceAssessment(
            query_id=query_id,
            decision=decision,  # type: ignore[arg-type]
            document_evidence=document_evidence,  # type: ignore[arg-type]
        )


def test_metadata_classifier_reads_stances_in_document_order() -> None:
    documents = (
        _document("doc-a", stance=" supports ", strength="0.8"),
        _document("doc-b", stance="CONTRADICTS"),
        _document("doc-c", stance="Neutral"),
        _document("doc-d", stance="unknown", strength="0"),
        _document("doc-e"),
    )

    evidence = MetadataStanceClassifier().classify(_query(), documents)

    assert [item.doc_id for item in evidence] == [
        "doc-a",
        "doc-b",
        "doc-c",
        "doc-d",
        "doc-e",
    ]
    assert [item.stance for item in evidence] == [
        EvidenceStance.SUPPORTS,
        EvidenceStance.CONTRADICTS,
        EvidenceStance.NEUTRAL,
        EvidenceStance.UNKNOWN,
        EvidenceStance.UNKNOWN,
    ]
    assert [item.strength for item in evidence] == [0.8, 1.0, 1.0, 0.0, 1.0]


def test_metadata_classifier_rejects_malformed_stance() -> None:
    with pytest.raises(DataValidationError, match="invalid evidence stance"):
        MetadataStanceClassifier().classify(
            _query(),
            (_document("doc-a", stance="maybe"),),
        )


@pytest.mark.parametrize("strength", ["", "strong", "nan", "inf", "-0.1", "1.1"])
def test_metadata_classifier_rejects_malformed_strength(strength: str) -> None:
    with pytest.raises(DataValidationError, match="strength"):
        MetadataStanceClassifier().classify(
            _query(),
            (_document("doc-a", stance="supports", strength=strength),),
        )


def test_metadata_classifier_accepts_empty_documents() -> None:
    assert MetadataStanceClassifier().classify(_query(), ()) == ()


def test_metadata_classifier_rejects_invalid_query() -> None:
    with pytest.raises(DataValidationError, match="Query"):
        MetadataStanceClassifier().classify("query", ())  # type: ignore[arg-type]


def test_metadata_classifier_rejects_non_document_values() -> None:
    with pytest.raises(DataValidationError, match="Document"):
        MetadataStanceClassifier().classify(
            _query(),
            (_document("doc-a"), "not a document"),  # type: ignore[arg-type]
        )


def test_metadata_classifier_rejects_document_string() -> None:
    with pytest.raises(DataValidationError, match="sequence"):
        MetadataStanceClassifier().classify(
            _query(),
            "document",  # type: ignore[arg-type]
        )


def test_metadata_classifier_rejects_none_documents() -> None:
    with pytest.raises(DataValidationError, match="sequence"):
        MetadataStanceClassifier().classify(
            _query(),
            None,  # type: ignore[arg-type]
        )


def test_assessor_marks_supported_evidence() -> None:
    assessment = EvidenceAssessor(MetadataStanceClassifier()).assess(
        _query(),
        (
            _document("doc-a", stance="supports", strength="0.4"),
            _document("doc-b", stance="supports", strength="0.9"),
            _document("doc-c", stance="neutral"),
        ),
    )

    assert assessment.decision is DecisionOutcome.ANSWER_SUPPORTED
    assert assessment.supporting_doc_ids == ("doc-a", "doc-b")
    assert assessment.support_strength == 0.9
    assert assessment.neutral_doc_ids == ("doc-c",)


def test_assessor_marks_empty_evidence_insufficient() -> None:
    assessment = EvidenceAssessor(MetadataStanceClassifier()).assess(_query(), ())

    assert assessment.decision is DecisionOutcome.INSUFFICIENT_EVIDENCE
    assert assessment.document_evidence == ()
    assert assessment.support_strength == 0.0
    assert assessment.contradiction_strength == 0.0


def test_assessor_marks_neutral_and_unknown_evidence_insufficient() -> None:
    assessment = EvidenceAssessor(MetadataStanceClassifier()).assess(
        _query(),
        (
            _document("doc-a", stance="neutral"),
            _document("doc-b"),
        ),
    )

    assert assessment.decision is DecisionOutcome.INSUFFICIENT_EVIDENCE
    assert assessment.neutral_doc_ids == ("doc-a",)
    assert assessment.unknown_doc_ids == ("doc-b",)


def test_assessor_marks_contradiction_only_insufficient() -> None:
    assessment = EvidenceAssessor(MetadataStanceClassifier()).assess(
        _query(),
        (_document("doc-a", stance="contradicts", strength="0.7"),),
    )

    assert assessment.decision is DecisionOutcome.INSUFFICIENT_EVIDENCE
    assert assessment.contradicting_doc_ids == ("doc-a",)
    assert assessment.contradiction_strength == 0.7


def test_assessor_marks_support_and_contradiction_as_conflicting() -> None:
    assessment = EvidenceAssessor(MetadataStanceClassifier()).assess(
        _query(),
        (
            _document("doc-a", stance="supports"),
            _document("doc-b", stance="contradicts"),
            _document("doc-c", stance="unknown"),
        ),
    )

    assert assessment.decision is DecisionOutcome.CONFLICTING_EVIDENCE
    assert assessment.supporting_doc_ids == ("doc-a",)
    assert assessment.contradicting_doc_ids == ("doc-b",)
    assert assessment.unknown_doc_ids == ("doc-c",)


def test_assessor_preserves_input_order_when_classifier_reorders() -> None:
    classifier = _StaticClassifier(
        (
            DocumentEvidence("doc-b", EvidenceStance.CONTRADICTS),
            DocumentEvidence("doc-a", EvidenceStance.SUPPORTS),
        )
    )

    assessment = EvidenceAssessor(classifier).assess(  # type: ignore[arg-type]
        _query(),
        (_document("doc-a", stance="supports"), _document("doc-b", stance="contradicts")),
    )

    assert [item.doc_id for item in assessment.document_evidence] == ["doc-a", "doc-b"]


def test_assessor_rejects_missing_classifier_method() -> None:
    with pytest.raises(DataValidationError, match="classify"):
        EvidenceAssessor(object())  # type: ignore[arg-type]


def test_assessor_rejects_invalid_query() -> None:
    with pytest.raises(DataValidationError, match="Query"):
        EvidenceAssessor(MetadataStanceClassifier()).assess("query", ())  # type: ignore[arg-type]


def test_assessor_rejects_document_string() -> None:
    with pytest.raises(DataValidationError, match="sequence"):
        EvidenceAssessor(MetadataStanceClassifier()).assess(
            _query(),
            "documents",  # type: ignore[arg-type]
        )


def test_assessor_rejects_none_documents() -> None:
    with pytest.raises(DataValidationError, match="sequence"):
        EvidenceAssessor(MetadataStanceClassifier()).assess(
            _query(),
            None,  # type: ignore[arg-type]
        )


def test_assessor_rejects_non_document_input() -> None:
    with pytest.raises(DataValidationError, match="Document"):
        EvidenceAssessor(MetadataStanceClassifier()).assess(
            _query(),
            ("not a document",),  # type: ignore[arg-type]
        )


def test_assessor_rejects_duplicate_input_documents() -> None:
    with pytest.raises(DataValidationError, match="Duplicate"):
        EvidenceAssessor(MetadataStanceClassifier()).assess(
            _query(),
            (_document("doc-a"), _document("doc-a")),
        )


@pytest.mark.parametrize(
    ("classified", "message"),
    [
        ("not evidence", "sequence"),
        (None, "sequence"),
        ((object(),), "DocumentEvidence"),
        (
            (
                DocumentEvidence("doc-a", EvidenceStance.SUPPORTS),
                DocumentEvidence("doc-a", EvidenceStance.CONTRADICTS),
            ),
            "duplicate",
        ),
        ((DocumentEvidence("doc-c", EvidenceStance.SUPPORTS),), "unknown"),
        ((), "omitted"),
    ],
)
def test_assessor_rejects_invalid_classifier_output(
    classified: object,
    message: str,
) -> None:
    classifier = _StaticClassifier(classified)
    documents = (_document("doc-a"), _document("doc-b"))

    with pytest.raises(DataValidationError, match=message):
        EvidenceAssessor(classifier).assess(_query(), documents)  # type: ignore[arg-type]
