"""Build controlled evidence examples from retrieval datasets."""

from __future__ import annotations

from evidenceguard.data.models import (
    EvidenceCondition,
    EvidenceExample,
    ExampleProvenance,
    RetrievalDataset,
)
from evidenceguard.errors import DataValidationError
from evidenceguard.outcomes import DecisionOutcome


def build_evidence_example(
    dataset: RetrievalDataset,
    query_id: str,
    expected_outcome: DecisionOutcome,
    *,
    condition: EvidenceCondition = EvidenceCondition.SUFFICIENT,
    example_id: str | None = None,
) -> EvidenceExample:
    """Create an original evidence example using the dataset's full corpus.

    The caller supplies the expected decision label because retrieval relevance
    judgments alone do not establish whether evidence is sufficient.
    """
    if not isinstance(dataset, RetrievalDataset):
        raise DataValidationError("dataset must be a RetrievalDataset.")
    query = dataset.get_query(query_id)
    stable_example_id = example_id or f"{dataset.name}:{dataset.split}:{query_id}"

    return EvidenceExample(
        example_id=stable_example_id,
        query=query,
        documents=dataset.documents,
        relevant_document_ids=dataset.relevant_document_ids(query_id),
        expected_outcome=expected_outcome,
        condition=condition,
        provenance=ExampleProvenance.ORIGINAL,
        metadata={"dataset": dataset.name, "split": dataset.split},
    )
