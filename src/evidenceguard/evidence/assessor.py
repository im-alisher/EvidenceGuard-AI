"""Aggregate document stances into an explicit evidence decision."""

from __future__ import annotations

from collections.abc import Sequence

from evidenceguard.data.models import Document, Query
from evidenceguard.errors import DataValidationError
from evidenceguard.evidence.classifier import EvidenceStanceClassifier
from evidenceguard.evidence.models import (
    DocumentEvidence,
    EvidenceAssessment,
    EvidenceStance,
)
from evidenceguard.outcomes import DecisionOutcome


class EvidenceAssessor:
    """Apply a deterministic sufficiency policy to classifier outputs.

    Supporting evidence without contradiction yields ``ANSWER_SUPPORTED``.
    Support plus contradiction yields ``CONFLICTING_EVIDENCE``. Contradiction
    alone, neutral or unknown evidence, and an empty document list yield
    ``INSUFFICIENT_EVIDENCE`` because this component does not synthesize
    refutations. The policy is not a calibrated abstention model.
    """

    def __init__(self, classifier: EvidenceStanceClassifier) -> None:
        if not callable(getattr(classifier, "classify", None)):
            raise DataValidationError(
                "classifier must provide a callable classify method."
            )
        self._classifier = classifier

    def assess(
        self,
        query: Query,
        documents: Sequence[Document],
    ) -> EvidenceAssessment:
        """Assess evidence and return a validated decision record."""

        if not isinstance(query, Query):
            raise DataValidationError("query must be a Query instance.")
        validated_documents = _validate_documents(documents)

        classified = self._classifier.classify(query, validated_documents)
        if isinstance(classified, (str, bytes)):
            raise DataValidationError(
                "classifier must return a sequence of DocumentEvidence objects."
            )
        try:
            classified_values = tuple(classified)
        except TypeError as exc:
            raise DataValidationError(
                "classifier must return a sequence of DocumentEvidence objects."
            ) from exc
        evidence_by_id = _validate_classifications(
            classified_values,
            validated_documents,
        )
        ordered_evidence = tuple(
            evidence_by_id[document.doc_id] for document in validated_documents
        )

        has_support = any(
            evidence.stance is EvidenceStance.SUPPORTS
            for evidence in ordered_evidence
        )
        has_contradiction = any(
            evidence.stance is EvidenceStance.CONTRADICTS
            for evidence in ordered_evidence
        )
        if has_support and has_contradiction:
            decision = DecisionOutcome.CONFLICTING_EVIDENCE
        elif has_support:
            decision = DecisionOutcome.ANSWER_SUPPORTED
        else:
            decision = DecisionOutcome.INSUFFICIENT_EVIDENCE

        return EvidenceAssessment(
            query_id=query.query_id,
            decision=decision,
            document_evidence=ordered_evidence,
        )


def _validate_documents(documents: Sequence[Document]) -> tuple[Document, ...]:
    if isinstance(documents, (str, bytes)):
        raise DataValidationError("documents must be a sequence of Document objects.")
    try:
        validated = tuple(documents)
    except TypeError as exc:
        raise DataValidationError(
            "documents must be a sequence of Document objects."
        ) from exc
    seen: set[str] = set()
    for document in validated:
        if not isinstance(document, Document):
            raise DataValidationError("documents must contain only Document objects.")
        if document.doc_id in seen:
            raise DataValidationError(
                f"Duplicate document ID in evidence assessment: {document.doc_id!r}."
            )
        seen.add(document.doc_id)
    return validated


def _validate_classifications(
    classified: tuple[DocumentEvidence, ...],
    documents: tuple[Document, ...],
) -> dict[str, DocumentEvidence]:
    expected_ids = {document.doc_id for document in documents}
    evidence_by_id: dict[str, DocumentEvidence] = {}
    for evidence in classified:
        if not isinstance(evidence, DocumentEvidence):
            raise DataValidationError(
                "classifier must return only DocumentEvidence objects."
            )
        if evidence.doc_id in evidence_by_id:
            raise DataValidationError(
                f"Classifier returned duplicate evidence for {evidence.doc_id!r}."
            )
        if evidence.doc_id not in expected_ids:
            raise DataValidationError(
                f"Classifier returned evidence for unknown document "
                f"{evidence.doc_id!r}."
            )
        evidence_by_id[evidence.doc_id] = evidence

    missing_ids = expected_ids.difference(evidence_by_id)
    if missing_ids:
        missing = ", ".join(sorted(missing_ids))
        raise DataValidationError(
            f"Classifier omitted evidence for document(s): {missing}."
        )
    return evidence_by_id
