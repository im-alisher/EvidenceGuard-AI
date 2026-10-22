from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from evidenceguard.data.beir import load_beir_dataset
from evidenceguard.data.examples import build_evidence_example
from evidenceguard.data.models import (
    Document,
    EvidenceCondition,
    EvidenceExample,
    ExampleProvenance,
)
from evidenceguard.data.transformations import (
    add_contradictory_evidence,
    add_irrelevant_documents,
    remove_relevant_evidence,
    replace_with_outdated_evidence,
)
from evidenceguard.errors import DataValidationError
from evidenceguard.outcomes import DecisionOutcome


@pytest.fixture
def answerable_example(mini_beir_dir: Path) -> EvidenceExample:
    dataset = load_beir_dataset(mini_beir_dir)
    return build_evidence_example(
        dataset,
        "q-answerable",
        DecisionOutcome.ANSWER_SUPPORTED,
    )


@pytest.fixture
def unanswerable_example(mini_beir_dir: Path) -> EvidenceExample:
    dataset = load_beir_dataset(mini_beir_dir)
    return build_evidence_example(
        dataset,
        "q-unanswerable",
        DecisionOutcome.INSUFFICIENT_EVIDENCE,
        condition=EvidenceCondition.UNANSWERABLE,
    )


def test_build_evidence_example_preserves_dataset_labels(
    answerable_example: EvidenceExample,
) -> None:
    assert answerable_example.example_id == "mini_beir:test:q-answerable"
    assert answerable_example.relevant_document_ids == ("doc-a",)
    assert answerable_example.condition is EvidenceCondition.SUFFICIENT
    assert answerable_example.provenance is ExampleProvenance.ORIGINAL


def test_remove_relevant_evidence_is_deterministic_and_non_destructive(
    answerable_example: EvidenceExample,
) -> None:
    original_ids = tuple(document.doc_id for document in answerable_example.documents)

    transformed = remove_relevant_evidence(answerable_example)
    repeated = remove_relevant_evidence(answerable_example)

    assert transformed.example_id == repeated.example_id
    assert transformed.relevant_document_ids == ()
    assert transformed.expected_outcome is DecisionOutcome.INSUFFICIENT_EVIDENCE
    assert transformed.condition is EvidenceCondition.MISSING
    assert transformed.provenance is ExampleProvenance.TRANSFORMED
    assert transformed.source_example_id == answerable_example.example_id
    assert transformed.transformation == "remove_relevant_evidence"
    assert "doc-a" not in {document.doc_id for document in transformed.documents}
    assert tuple(document.doc_id for document in answerable_example.documents) == original_ids


def test_remove_relevant_evidence_requires_relevant_documents(
    unanswerable_example: EvidenceExample,
) -> None:
    with pytest.raises(DataValidationError, match="no relevant documents"):
        remove_relevant_evidence(unanswerable_example)


def test_add_irrelevant_documents_is_seed_deterministic(
    answerable_example: EvidenceExample,
) -> None:
    distractors = (
        Document(doc_id="distractor-1", title="Unrelated", text="Synthetic distractor 1."),
        Document(doc_id="distractor-2", title="Unrelated", text="Synthetic distractor 2."),
    )

    first = add_irrelevant_documents(answerable_example, distractors, count=2, seed=21)
    second = add_irrelevant_documents(answerable_example, distractors, count=2, seed=21)

    assert first.example_id == second.example_id
    assert first.condition is EvidenceCondition.IRRELEVANT
    assert first.expected_outcome is answerable_example.expected_outcome
    assert first.relevant_document_ids == answerable_example.relevant_document_ids
    added = set(first.metadata["added_irrelevant_document_ids"].split(","))
    assert added == {"distractor-1", "distractor-2"}
    assert len(first.documents) == len(answerable_example.documents) + 2


def test_add_irrelevant_documents_excludes_existing_documents(
    answerable_example: EvidenceExample,
) -> None:
    existing = Document(doc_id="doc-c", title="", text="Already in the example.")
    new_distractor = Document(
        doc_id="distractor-new",
        title="",
        text="Not already in the example.",
    )

    transformed = add_irrelevant_documents(
        answerable_example,
        [existing, new_distractor],
        count=1,
        seed=21,
    )

    assert transformed.metadata["added_irrelevant_document_ids"] == "distractor-new"


def test_add_irrelevant_documents_rejects_insufficient_pool(
    answerable_example: EvidenceExample,
) -> None:
    with pytest.raises(DataValidationError, match="Not enough unused distractor"):
        add_irrelevant_documents(
            answerable_example,
            [Document(doc_id="doc-c", title="", text="Only one.")],
            count=2,
            seed=21,
        )


def test_add_irrelevant_documents_rejects_duplicate_pool_ids(
    answerable_example: EvidenceExample,
) -> None:
    duplicate = Document(doc_id="doc-c", title="", text="Synthetic distractor.")

    with pytest.raises(DataValidationError, match="duplicate document IDs"):
        add_irrelevant_documents(
            answerable_example,
            [duplicate, duplicate],
            count=1,
            seed=21,
        )


def test_add_irrelevant_documents_validates_seed(
    answerable_example: EvidenceExample,
) -> None:
    with pytest.raises(DataValidationError, match="Seed"):
        add_irrelevant_documents(
            answerable_example,
            [Document(doc_id="doc-c", title="", text="Synthetic distractor.")],
            count=1,
            seed=-1,
        )


def test_add_contradictory_evidence_sets_conflicting_label(
    answerable_example: EvidenceExample,
) -> None:
    contradiction = Document(
        doc_id="doc-contradiction",
        title="Contradictory result",
        text="A synthetic replication found no effect.",
    )

    transformed = add_contradictory_evidence(answerable_example, [contradiction])

    assert transformed.expected_outcome is DecisionOutcome.CONFLICTING_EVIDENCE
    assert transformed.condition is EvidenceCondition.CONTRADICTORY
    assert transformed.provenance is ExampleProvenance.TRANSFORMED
    assert transformed.get_document("doc-contradiction") == contradiction
    assert transformed.relevant_document_ids == answerable_example.relevant_document_ids


def test_add_contradictory_evidence_rejects_existing_id(
    answerable_example: EvidenceExample,
) -> None:
    with pytest.raises(DataValidationError, match="collisions"):
        add_contradictory_evidence(
            answerable_example,
            [Document(doc_id="doc-a", title="", text="Collision.")],
        )


def test_add_contradictory_evidence_requires_supporting_evidence(
    unanswerable_example: EvidenceExample,
) -> None:
    with pytest.raises(DataValidationError, match="at least one relevant"):
        add_contradictory_evidence(
            unanswerable_example,
            [Document(doc_id="doc-contradiction", title="", text="No effect.")],
        )


def test_replace_with_outdated_evidence_removes_current_support(
    answerable_example: EvidenceExample,
) -> None:
    outdated = Document(
        doc_id="doc-old",
        title="Old report",
        text="An old synthetic report claimed compound X reduced marker Y.",
        published_at=date(2015, 1, 1),
    )

    transformed = replace_with_outdated_evidence(answerable_example, [outdated])

    assert transformed.expected_outcome is DecisionOutcome.INSUFFICIENT_EVIDENCE
    assert transformed.condition is EvidenceCondition.OUTDATED
    assert transformed.relevant_document_ids == ("doc-old",)
    assert "doc-a" not in {document.doc_id for document in transformed.documents}
    assert transformed.get_document("doc-old") == outdated
    assert transformed.metadata["outdated_document_ids"] == "doc-old"


def test_replace_with_outdated_evidence_requires_timestamp(
    answerable_example: EvidenceExample,
) -> None:
    with pytest.raises(DataValidationError, match="published_at"):
        replace_with_outdated_evidence(
            answerable_example,
            [Document(doc_id="doc-old", title="", text="Undated replacement.")],
        )


def test_replace_with_outdated_evidence_rejects_retained_id_collision(
    answerable_example: EvidenceExample,
) -> None:
    with pytest.raises(DataValidationError, match="collide"):
        replace_with_outdated_evidence(
            answerable_example,
            [
                Document(
                    doc_id="doc-c",
                    title="",
                    text="Collision with retained document.",
                    published_at=date(2015, 1, 1),
                )
            ],
        )
