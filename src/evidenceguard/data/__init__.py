"""Dataset loading, validation, and controlled evidence transformations."""

from evidenceguard.data.beir import load_beir_dataset
from evidenceguard.data.examples import build_evidence_example
from evidenceguard.data.models import (
    Document,
    EvidenceCondition,
    EvidenceExample,
    ExampleProvenance,
    Query,
    RelevanceJudgment,
    RetrievalDataset,
)
from evidenceguard.data.scifact import (
    SCIFACT_BEIR_LICENSE,
    SCIFACT_BEIR_MD5,
    SCIFACT_BEIR_URL,
    SCIFACT_CITATION,
    SCIFACT_ORIGINAL_LICENSE,
    compute_md5,
    load_scifact,
    verify_scifact_archive,
)
from evidenceguard.data.splits import QuerySplit, assign_query_splits
from evidenceguard.data.transformations import (
    add_contradictory_evidence,
    add_irrelevant_documents,
    remove_relevant_evidence,
    replace_with_outdated_evidence,
)

__all__ = [
    "Document",
    "EvidenceCondition",
    "EvidenceExample",
    "ExampleProvenance",
    "Query",
    "QuerySplit",
    "RelevanceJudgment",
    "RetrievalDataset",
    "SCIFACT_BEIR_LICENSE",
    "SCIFACT_BEIR_MD5",
    "SCIFACT_BEIR_URL",
    "SCIFACT_CITATION",
    "SCIFACT_ORIGINAL_LICENSE",
    "add_contradictory_evidence",
    "add_irrelevant_documents",
    "assign_query_splits",
    "build_evidence_example",
    "compute_md5",
    "load_beir_dataset",
    "load_scifact",
    "remove_relevant_evidence",
    "replace_with_outdated_evidence",
    "verify_scifact_archive",
]
