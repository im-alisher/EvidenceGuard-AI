"""Validated evidence stance and assessment result models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from evidenceguard.errors import DataValidationError
from evidenceguard.outcomes import DecisionOutcome


class EvidenceStance(str, Enum):
    """The relation between one document and a query or claim."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DocumentEvidence:
    """A stance label and uncalibrated strength for one document.

    ``strength`` is a caller-supplied signal, not a calibrated probability. The
    current assessment policy uses the stance label; strength is retained for
    later uncertainty analysis and reporting.
    """

    doc_id: str
    stance: EvidenceStance
    strength: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, str) or not self.doc_id.strip():
            raise DataValidationError("doc_id must be a non-empty string.")
        if not isinstance(self.stance, EvidenceStance):
            raise DataValidationError("stance must be an EvidenceStance.")
        if isinstance(self.strength, bool) or not isinstance(
            self.strength, (int, float)
        ):
            raise DataValidationError("Evidence strength must be numeric.")
        numeric_strength = float(self.strength)
        if not math.isfinite(numeric_strength) or not 0.0 <= numeric_strength <= 1.0:
            raise DataValidationError("Evidence strength must be between 0 and 1.")
        object.__setattr__(self, "strength", numeric_strength)


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """The stance distribution and rule-based decision for one query.

    Document order in ``document_evidence`` is preserved from the assessed
    input. Strength values are reported but do not currently alter the decision.
    """

    query_id: str
    decision: DecisionOutcome
    document_evidence: tuple[DocumentEvidence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise DataValidationError("query_id must be a non-empty string.")
        if not isinstance(self.decision, DecisionOutcome):
            raise DataValidationError("decision must be a DecisionOutcome.")
        if isinstance(self.document_evidence, (str, bytes)):
            raise DataValidationError(
                "document_evidence must be a sequence of DocumentEvidence objects."
            )
        try:
            evidence_values = tuple(self.document_evidence)
        except TypeError as exc:
            raise DataValidationError(
                "document_evidence must be a sequence of DocumentEvidence objects."
            ) from exc
        object.__setattr__(self, "document_evidence", evidence_values)
        if any(
            not isinstance(evidence, DocumentEvidence)
            for evidence in self.document_evidence
        ):
            raise DataValidationError(
                "document_evidence must contain only DocumentEvidence objects."
            )
        document_ids = [evidence.doc_id for evidence in self.document_evidence]
        if len(set(document_ids)) != len(document_ids):
            raise DataValidationError(
                "EvidenceAssessment contains duplicate document IDs."
            )

    @property
    def supporting_doc_ids(self) -> tuple[str, ...]:
        """Return document IDs labeled as supporting the query."""

        return self._doc_ids_for(EvidenceStance.SUPPORTS)

    @property
    def contradicting_doc_ids(self) -> tuple[str, ...]:
        """Return document IDs labeled as contradicting the query."""

        return self._doc_ids_for(EvidenceStance.CONTRADICTS)

    @property
    def neutral_doc_ids(self) -> tuple[str, ...]:
        """Return document IDs labeled as neutral."""

        return self._doc_ids_for(EvidenceStance.NEUTRAL)

    @property
    def unknown_doc_ids(self) -> tuple[str, ...]:
        """Return documents whose stance could not be established."""

        return self._doc_ids_for(EvidenceStance.UNKNOWN)

    @property
    def support_strength(self) -> float:
        """Return the maximum strength among supporting documents."""

        return self._maximum_strength(EvidenceStance.SUPPORTS)

    @property
    def contradiction_strength(self) -> float:
        """Return the maximum strength among contradicting documents."""

        return self._maximum_strength(EvidenceStance.CONTRADICTS)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "query_id": self.query_id,
            "decision": self.decision.value,
            "supporting_doc_ids": list(self.supporting_doc_ids),
            "contradicting_doc_ids": list(self.contradicting_doc_ids),
            "neutral_doc_ids": list(self.neutral_doc_ids),
            "unknown_doc_ids": list(self.unknown_doc_ids),
            "support_strength": self.support_strength,
            "contradiction_strength": self.contradiction_strength,
            "document_evidence": [
                {
                    "doc_id": evidence.doc_id,
                    "stance": evidence.stance.value,
                    "strength": evidence.strength,
                }
                for evidence in self.document_evidence
            ],
        }

    def _doc_ids_for(self, stance: EvidenceStance) -> tuple[str, ...]:
        return tuple(
            evidence.doc_id
            for evidence in self.document_evidence
            if evidence.stance is stance
        )

    def _maximum_strength(self, stance: EvidenceStance) -> float:
        return max(
            (
                evidence.strength
                for evidence in self.document_evidence
                if evidence.stance is stance
            ),
            default=0.0,
        )
