"""SciFact dataset metadata, validation helpers, and loader."""

from __future__ import annotations

import hashlib
from pathlib import Path

from evidenceguard.data.beir import load_beir_dataset
from evidenceguard.data.models import RetrievalDataset
from evidenceguard.errors import DataValidationError

SCIFACT_NAME = "scifact"
SCIFACT_ARCHIVE_FILENAME = "scifact.zip"
SCIFACT_BEIR_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)
SCIFACT_BEIR_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
SCIFACT_ORIGINAL_LICENSE = "CC BY-NC 2.0"
SCIFACT_BEIR_LICENSE = "CC BY-SA 4.0"
SCIFACT_CITATION = (
    "Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., "
    "& Hajishirzi, H. (2020). Fact or Fiction: Verifying Scientific Claims. "
    "EMNLP 2020. https://aclanthology.org/2020.emnlp-main.609"
)


def load_scifact(dataset_dir: str | Path, *, split: str = "test") -> RetrievalDataset:
    """Load a locally prepared SciFact corpus in BEIR layout."""
    return load_beir_dataset(dataset_dir, split=split, name=SCIFACT_NAME)


def compute_md5(path: str | Path) -> str:
    """Compute the lowercase MD5 digest of a file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise DataValidationError(f"File not found for checksum verification: {file_path}")
    digest = hashlib.md5(usedforsecurity=False)
    try:
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DataValidationError(f"Could not read file {file_path}: {exc}") from exc
    return digest.hexdigest()


def verify_scifact_archive(
    path: str | Path,
    *,
    expected_md5: str = SCIFACT_BEIR_MD5,
) -> str:
    """Verify a SciFact BEIR archive checksum and return the actual digest."""
    actual_md5 = compute_md5(path)
    if actual_md5 != expected_md5.lower():
        raise DataValidationError(
            f"SciFact archive checksum mismatch: expected {expected_md5}, "
            f"got {actual_md5}."
        )
    return actual_md5
