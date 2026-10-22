"""Validated data models for retrieval datasets and evidence scenarios."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from types import MappingProxyType
from typing import Any

from evidenceguard.errors import DataValidationError
from evidenceguard.outcomes import DecisionOutcome
from evidenceguard.reproducibility import validate_seed


class EvidenceCondition(str, Enum):
    """Controlled evidence condition represented by a benchmark example."""

    SUFFICIENT = "sufficient"
    MISSING = "missing_evidence"
    IRRELEVANT = "irrelevant_documents"
    CONTRADICTORY = "contradictory_evidence"
    OUTDATED = "outdated_evidence"
    UNANSWERABLE = "unanswerable_question"


class ExampleProvenance(str, Enum):
    """Whether an example is an original benchmark item or a derived scenario."""

    ORIGINAL = "original"
    TRANSFORMED = "transformed"


@dataclass(frozen=True, slots=True)
class Document:
    """A retrievable text unit with optional temporal metadata."""

    doc_id: str
    title: str
    text: str
    published_at: date | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.doc_id, "doc_id")
        if not isinstance(self.title, str):
            raise DataValidationError("Document title must be a string.")
        _require_identifier(self.text, "Document text")
        if self.published_at is not None and not isinstance(self.published_at, date):
            raise DataValidationError("Document published_at must be a date or None.")
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class Query:
    """A retrieval query or scientific claim."""

    query_id: str
    text: str
    title: str = ""

    def __post_init__(self) -> None:
        _require_identifier(self.query_id, "query_id")
        _require_identifier(self.text, "Query text")
        if not isinstance(self.title, str):
            raise DataValidationError("Query title must be a string.")


@dataclass(frozen=True, slots=True)
class RelevanceJudgment:
    """A retrieval relevance label for one query-document pair."""

    query_id: str
    doc_id: str
    relevance: int

    def __post_init__(self) -> None:
        _require_identifier(self.query_id, "query_id")
        _require_identifier(self.doc_id, "doc_id")
        if isinstance(self.relevance, bool) or not isinstance(self.relevance, int):
            raise DataValidationError("Relevance must be an integer.")
        if self.relevance < 0:
            raise DataValidationError("Relevance must be non-negative.")


@dataclass(frozen=True, slots=True)
class RetrievalDataset:
    """A corpus, query set, and retrieval relevance judgments."""

    name: str
    split: str
    documents: tuple[Document, ...]
    queries: tuple[Query, ...]
    qrels: tuple[RelevanceJudgment, ...]
    source: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.name, "Dataset name")
        _require_identifier(self.split, "Dataset split")
        object.__setattr__(self, "documents", tuple(self.documents))
        object.__setattr__(self, "queries", tuple(self.queries))
        object.__setattr__(self, "qrels", tuple(self.qrels))

        if any(not isinstance(document, Document) for document in self.documents):
            raise DataValidationError("Dataset documents must be Document instances.")
        if any(not isinstance(query, Query) for query in self.queries):
            raise DataValidationError("Dataset queries must be Query instances.")
        if any(not isinstance(judgment, RelevanceJudgment) for judgment in self.qrels):
            raise DataValidationError(
                "Dataset qrels must be RelevanceJudgment instances."
            )

        document_ids = {document.doc_id for document in self.documents}
        if len(document_ids) != len(self.documents):
            raise DataValidationError("Dataset contains duplicate document IDs.")

        query_ids = {query.query_id for query in self.queries}
        if len(query_ids) != len(self.queries):
            raise DataValidationError("Dataset contains duplicate query IDs.")

        seen_qrels: set[tuple[str, str]] = set()
        for judgment in self.qrels:
            key = (judgment.query_id, judgment.doc_id)
            if key in seen_qrels:
                raise DataValidationError(
                    f"Duplicate relevance judgment for query {judgment.query_id!r} "
                    f"and document {judgment.doc_id!r}."
                )
            seen_qrels.add(key)
            if judgment.query_id not in query_ids:
                raise DataValidationError(
                    f"Relevance judgment references unknown query {judgment.query_id!r}."
                )
            if judgment.doc_id not in document_ids:
                raise DataValidationError(
                    f"Relevance judgment references unknown document {judgment.doc_id!r}."
                )

    def get_document(self, doc_id: str) -> Document:
        """Return a document by ID or raise a validation error."""
        for document in self.documents:
            if document.doc_id == doc_id:
                return document
        raise DataValidationError(f"Unknown document ID: {doc_id!r}.")

    def get_query(self, query_id: str) -> Query:
        """Return a query by ID or raise a validation error."""
        for query in self.queries:
            if query.query_id == query_id:
                return query
        raise DataValidationError(f"Unknown query ID: {query_id!r}.")

    def relevant_document_ids(
        self,
        query_id: str,
        *,
        min_relevance: int = 1,
    ) -> tuple[str, ...]:
        """Return retrieval-relevant document IDs sorted by document ID."""
        if isinstance(min_relevance, bool) or not isinstance(min_relevance, int):
            raise DataValidationError("min_relevance must be an integer.")
        if min_relevance < 0:
            raise DataValidationError("min_relevance must be non-negative.")
        self.get_query(query_id)
        return tuple(
            sorted(
                judgment.doc_id
                for judgment in self.qrels
                if judgment.query_id == query_id and judgment.relevance >= min_relevance
            )
        )


@dataclass(frozen=True, slots=True)
class EvidenceExample:
    """A controlled evidence scenario for one query.

    ``relevant_document_ids`` are retrieval-relevance labels. They do not imply
    that the retrieved evidence is sufficient or non-contradictory. The
    ``expected_outcome`` field carries the decision label for this scenario.
    """

    example_id: str
    query: Query
    documents: tuple[Document, ...]
    relevant_document_ids: tuple[str, ...]
    expected_outcome: DecisionOutcome
    condition: EvidenceCondition
    provenance: ExampleProvenance = ExampleProvenance.ORIGINAL
    source_example_id: str | None = None
    transformation: str | None = None
    seed: int | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.example_id, "example_id")
        if not isinstance(self.query, Query):
            raise DataValidationError("EvidenceExample query must be a Query instance.")
        if not isinstance(self.expected_outcome, DecisionOutcome):
            raise DataValidationError(
                "EvidenceExample expected_outcome must be a DecisionOutcome."
            )
        if not isinstance(self.condition, EvidenceCondition):
            raise DataValidationError(
                "EvidenceExample condition must be an EvidenceCondition."
            )
        if not isinstance(self.provenance, ExampleProvenance):
            raise DataValidationError(
                "EvidenceExample provenance must be an ExampleProvenance."
            )

        object.__setattr__(self, "documents", tuple(self.documents))
        if any(not isinstance(document, Document) for document in self.documents):
            raise DataValidationError(
                "EvidenceExample documents must be Document instances."
            )
        if any(
            not isinstance(doc_id, str) or not doc_id.strip()
            for doc_id in self.relevant_document_ids
        ):
            raise DataValidationError(
                "EvidenceExample relevant IDs must be non-empty strings."
            )
        object.__setattr__(
            self,
            "relevant_document_ids",
            tuple(sorted(self.relevant_document_ids)),
        )

        document_ids = {document.doc_id for document in self.documents}
        if len(document_ids) != len(self.documents):
            raise DataValidationError("EvidenceExample contains duplicate document IDs.")
        if len(set(self.relevant_document_ids)) != len(self.relevant_document_ids):
            raise DataValidationError(
                "EvidenceExample contains duplicate relevant document IDs."
            )
        unknown_relevant = set(self.relevant_document_ids).difference(document_ids)
        if unknown_relevant:
            unknown = ", ".join(sorted(unknown_relevant))
            raise DataValidationError(
                f"EvidenceExample references relevant documents outside its pool: {unknown}."
            )

        if self.provenance is ExampleProvenance.ORIGINAL:
            if self.source_example_id is not None or self.transformation is not None:
                raise DataValidationError(
                    "Original examples cannot declare a source example or transformation."
                )
        else:
            if not self.source_example_id:
                raise DataValidationError("Transformed examples require a source_example_id.")
            if not self.transformation:
                raise DataValidationError("Transformed examples require a transformation name.")

        if self.seed is not None:
            try:
                validate_seed(self.seed)
            except (TypeError, ValueError) as exc:
                raise DataValidationError(str(exc)) from exc
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    def get_document(self, doc_id: str) -> Document:
        """Return a document from this scenario's pool by ID."""
        for document in self.documents:
            if document.doc_id == doc_id:
                return document
        raise DataValidationError(f"Unknown document ID in example: {doc_id!r}.")


def _require_identifier(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{field_name} must be a non-empty string.")


def _normalize_metadata(metadata: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(metadata, Mapping):
        raise DataValidationError("Metadata must be a mapping of strings to strings.")
    normalized: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise DataValidationError("Metadata keys must be non-empty strings.")
        if not isinstance(value, str):
            raise DataValidationError("Metadata values must be strings.")
        normalized[key] = value
    return MappingProxyType(normalized)
