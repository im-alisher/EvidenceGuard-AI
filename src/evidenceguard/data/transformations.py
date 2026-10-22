"""Deterministic, non-destructive transformations of evidence scenarios."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from dataclasses import replace

from evidenceguard.data.models import (
    Document,
    EvidenceCondition,
    EvidenceExample,
    ExampleProvenance,
)
from evidenceguard.errors import DataValidationError
from evidenceguard.outcomes import DecisionOutcome
from evidenceguard.reproducibility import derive_seed, validate_seed


def remove_relevant_evidence(example: EvidenceExample) -> EvidenceExample:
    """Remove all retrieval-relevant evidence from a scenario."""
    if not example.relevant_document_ids:
        raise DataValidationError(
            "Cannot remove relevant evidence: the example has no relevant documents."
        )

    removed_ids = set(example.relevant_document_ids)
    remaining = tuple(
        document for document in example.documents if document.doc_id not in removed_ids
    )
    return _derive_example(
        example,
        documents=remaining,
        relevant_document_ids=(),
        expected_outcome=DecisionOutcome.INSUFFICIENT_EVIDENCE,
        condition=EvidenceCondition.MISSING,
        transformation="remove_relevant_evidence",
        metadata_updates={
            "removed_relevant_document_ids": ",".join(sorted(removed_ids)),
        },
    )


def add_irrelevant_documents(
    example: EvidenceExample,
    distractor_pool: Iterable[Document],
    *,
    count: int,
    seed: int,
) -> EvidenceExample:
    """Append a deterministic sample of caller-supplied distractor documents.

    The transformation assumes that the supplied pool is irrelevant to the
    query. It does not attempt to infer irrelevance from text.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise DataValidationError("count must be a positive integer.")
    try:
        valid_seed = validate_seed(seed)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(str(exc)) from exc

    existing_ids = {document.doc_id for document in example.documents}
    pool = _normalize_documents(distractor_pool, "distractor_pool")
    candidates = tuple(
        document for document in pool if document.doc_id not in existing_ids
    )
    if len(candidates) < count:
        raise DataValidationError(
            f"Not enough unused distractor documents: requested {count}, "
            f"available {len(candidates)}."
        )

    rng = random.Random(derive_seed(example.example_id, "irrelevant", valid_seed))
    selected_ids = set(rng.sample([document.doc_id for document in candidates], count))
    selected = tuple(
        sorted(
            (document for document in candidates if document.doc_id in selected_ids),
            key=lambda document: document.doc_id,
        )
    )
    added_ids = {document.doc_id for document in selected}

    return _derive_example(
        example,
        documents=(*example.documents, *selected),
        relevant_document_ids=example.relevant_document_ids,
        expected_outcome=example.expected_outcome,
        condition=EvidenceCondition.IRRELEVANT,
        transformation="add_irrelevant_documents",
        seed=valid_seed,
        metadata_updates={
            "added_irrelevant_document_ids": ",".join(sorted(added_ids)),
        },
    )


def add_contradictory_evidence(
    example: EvidenceExample,
    contradiction_documents: Iterable[Document],
) -> EvidenceExample:
    """Add explicit contradictory documents to a scenario with supporting evidence.

    The caller is responsible for establishing that the documents contradict the
    supporting evidence. No automatic textual contradiction inference is used.
    """
    if not example.relevant_document_ids:
        raise DataValidationError(
            "Contradictory evidence requires at least one relevant document."
        )
    additions = _normalize_documents(contradiction_documents, "contradiction_documents")
    if not additions:
        raise DataValidationError("At least one contradictory document is required.")

    existing_ids = {document.doc_id for document in example.documents}
    collisions = existing_ids.intersection(document.doc_id for document in additions)
    if collisions:
        raise DataValidationError(
            "Contradictory documents must use new IDs; collisions: "
            + ", ".join(sorted(collisions))
        )

    return _derive_example(
        example,
        documents=(*example.documents, *additions),
        relevant_document_ids=example.relevant_document_ids,
        expected_outcome=DecisionOutcome.CONFLICTING_EVIDENCE,
        condition=EvidenceCondition.CONTRADICTORY,
        transformation="add_contradictory_evidence",
        metadata_updates={
            "contradictory_document_ids": ",".join(
                sorted(document.doc_id for document in additions)
            ),
        },
    )


def replace_with_outdated_evidence(
    example: EvidenceExample,
    outdated_documents: Iterable[Document],
) -> EvidenceExample:
    """Replace supporting evidence with explicitly timestamped outdated documents.

    The replacement documents remain retrieval-relevant for evaluation, but the
    scenario label is insufficient evidence. Temporal invalidity is synthetic and
    must be established by the caller, not inferred from timestamps alone.
    """
    if not example.relevant_document_ids:
        raise DataValidationError(
            "Outdated-evidence replacement requires at least one relevant document."
        )
    replacements = _normalize_documents(outdated_documents, "outdated_documents")
    if not replacements:
        raise DataValidationError("At least one outdated document is required.")
    undated = [
        document.doc_id for document in replacements if document.published_at is None
    ]
    if undated:
        raise DataValidationError(
            "Outdated documents require published_at timestamps: " + ", ".join(undated)
        )

    removed_ids = set(example.relevant_document_ids)
    remaining = tuple(
        document for document in example.documents if document.doc_id not in removed_ids
    )
    remaining_ids = {document.doc_id for document in remaining}
    collisions = remaining_ids.intersection(document.doc_id for document in replacements)
    if collisions:
        raise DataValidationError(
            "Outdated document IDs collide with retained documents: "
            + ", ".join(sorted(collisions))
        )

    outdated_ids = tuple(sorted(document.doc_id for document in replacements))
    return _derive_example(
        example,
        documents=(*remaining, *replacements),
        relevant_document_ids=outdated_ids,
        expected_outcome=DecisionOutcome.INSUFFICIENT_EVIDENCE,
        condition=EvidenceCondition.OUTDATED,
        transformation="replace_with_outdated_evidence",
        metadata_updates={
            "removed_relevant_document_ids": ",".join(sorted(removed_ids)),
            "outdated_document_ids": ",".join(outdated_ids),
        },
    )


def _normalize_documents(
    documents: Iterable[Document],
    argument_name: str,
) -> tuple[Document, ...]:
    normalized = tuple(documents)
    for document in normalized:
        if not isinstance(document, Document):
            raise DataValidationError(f"{argument_name} must contain Document instances.")
    ids = {document.doc_id for document in normalized}
    if len(ids) != len(normalized):
        raise DataValidationError(f"{argument_name} contains duplicate document IDs.")
    return tuple(sorted(normalized, key=lambda document: document.doc_id))


def _derive_example(
    source: EvidenceExample,
    *,
    documents: tuple[Document, ...],
    relevant_document_ids: tuple[str, ...],
    expected_outcome: DecisionOutcome,
    condition: EvidenceCondition,
    transformation: str,
    metadata_updates: Mapping[str, str],
    seed: int | None = None,
) -> EvidenceExample:
    metadata = dict(source.metadata)
    metadata["source_condition"] = source.condition.value
    metadata.update(metadata_updates)
    example_id = _derive_example_id(
        source.example_id,
        transformation,
        condition,
        documents,
        seed,
    )
    return replace(
        source,
        example_id=example_id,
        documents=documents,
        relevant_document_ids=relevant_document_ids,
        expected_outcome=expected_outcome,
        condition=condition,
        provenance=ExampleProvenance.TRANSFORMED,
        source_example_id=source.example_id,
        transformation=transformation,
        seed=seed,
        metadata=metadata,
    )


def _derive_example_id(
    source_id: str,
    transformation: str,
    condition: EvidenceCondition,
    documents: tuple[Document, ...],
    seed: int | None,
) -> str:
    seed_component = "none" if seed is None else str(seed)
    fingerprint = derive_seed(
        source_id,
        transformation,
        condition.value,
        seed_component,
        *(document.doc_id for document in documents),
    )
    return f"{source_id}::{transformation}::{condition.value}::{fingerprint:08x}"
