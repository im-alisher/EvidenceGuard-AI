"""Evidence stance classifier interfaces and an explicit-label adapter."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from evidenceguard.data.models import Document, Query
from evidenceguard.errors import DataValidationError
from evidenceguard.evidence.models import DocumentEvidence, EvidenceStance

STANCE_METADATA_KEY = "evidence_stance"
STRENGTH_METADATA_KEY = "evidence_strength"


class EvidenceStanceClassifier(Protocol):
    """Assign one query-specific stance to each input document."""

    def classify(
        self,
        query: Query,
        documents: Sequence[Document],
    ) -> Sequence[DocumentEvidence]:
        """Return exactly one DocumentEvidence result per input document."""

        ...


class MetadataStanceClassifier:
    """Read precomputed stance labels from document metadata.

    This adapter performs no natural-language inference. It is intended for
    controlled scenarios where stance labels were supplied by an annotator or
    an external classifier. Missing labels become ``UNKNOWN``; malformed labels
    are rejected.
    """

    def classify(
        self,
        query: Query,
        documents: Sequence[Document],
    ) -> tuple[DocumentEvidence, ...]:
        """Return metadata-backed stance labels in input document order."""

        if not isinstance(query, Query):
            raise DataValidationError("query must be a Query instance.")
        if isinstance(documents, (str, bytes)):
            raise DataValidationError("documents must be a sequence of Document objects.")
        try:
            values = tuple(documents)
        except TypeError as exc:
            raise DataValidationError(
                "documents must be a sequence of Document objects."
            ) from exc
        return tuple(self._classify_document(document) for document in values)

    def _classify_document(self, document: Document) -> DocumentEvidence:
        if not isinstance(document, Document):
            raise DataValidationError("documents must contain only Document objects.")

        raw_stance = document.metadata.get(STANCE_METADATA_KEY)
        if raw_stance is None:
            stance = EvidenceStance.UNKNOWN
        else:
            normalized = raw_stance.strip().casefold()
            try:
                stance = EvidenceStance(normalized)
            except ValueError as exc:
                accepted = ", ".join(item.value for item in EvidenceStance)
                raise DataValidationError(
                    f"Document {document.doc_id!r} has invalid evidence stance "
                    f"{raw_stance!r}; expected one of: {accepted}."
                ) from exc

        raw_strength = document.metadata.get(STRENGTH_METADATA_KEY)
        strength = 1.0 if raw_strength is None else self._parse_strength(
            raw_strength,
            document.doc_id,
        )
        return DocumentEvidence(
            doc_id=document.doc_id,
            stance=stance,
            strength=strength,
        )

    @staticmethod
    def _parse_strength(value: str, doc_id: str) -> float:
        try:
            strength = float(value)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"Document {doc_id!r} has non-numeric evidence strength {value!r}."
            ) from exc
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise DataValidationError(
                f"Document {doc_id!r} evidence strength must be between 0 and 1."
            )
        return strength
