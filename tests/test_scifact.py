from __future__ import annotations

from pathlib import Path

import pytest

from evidenceguard.data.scifact import (
    SCIFACT_NAME,
    compute_md5,
    load_scifact,
    verify_scifact_archive,
)
from evidenceguard.errors import DataValidationError


def test_compute_md5_matches_known_value(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")

    assert compute_md5(path) == "900150983cd24fb0d6963f7d28e17f72"


def test_verify_archive_accepts_matching_checksum(tmp_path: Path) -> None:
    path = tmp_path / "scifact.zip"
    path.write_bytes(b"abc")

    actual = verify_scifact_archive(
        path,
        expected_md5="900150983CD24FB0D6963F7D28E17F72",
    )

    assert actual == "900150983cd24fb0d6963f7d28e17f72"


def test_verify_archive_rejects_checksum_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "scifact.zip"
    path.write_bytes(b"not the expected archive")

    with pytest.raises(DataValidationError, match="checksum mismatch"):
        verify_scifact_archive(path)


def test_compute_md5_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="File not found"):
        compute_md5(tmp_path / "missing.zip")


def test_load_scifact_uses_scifact_dataset_name(mini_beir_dir: Path) -> None:
    dataset = load_scifact(mini_beir_dir)

    assert dataset.name == SCIFACT_NAME
    assert dataset.split == "test"
