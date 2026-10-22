from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidenceguard.data.beir import load_beir_dataset
from evidenceguard.errors import DataValidationError


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_minimal_dataset(
    root: Path,
    *,
    qrels: str,
) -> None:
    (root / "qrels").mkdir(parents=True)
    _write_jsonl(
        root / "corpus.jsonl",
        [{"_id": "doc-1", "title": "Example", "text": "Example document text."}],
    )
    _write_jsonl(
        root / "queries.jsonl",
        [{"_id": "query-1", "title": "", "text": "Example query?"}],
    )
    (root / "qrels" / "test.tsv").write_text(qrels, encoding="utf-8")


def test_loads_synthetic_beir_fixture(mini_beir_dir: Path) -> None:
    dataset = load_beir_dataset(mini_beir_dir)

    assert dataset.name == "mini_beir"
    assert dataset.split == "test"
    assert len(dataset.documents) == 5
    assert len(dataset.queries) == 3
    assert len(dataset.qrels) == 2
    assert dataset.get_document("doc-a").title == "Compound X and marker Y"
    assert dataset.get_query("q-answerable").text == "Does compound X reduce marker Y?"
    assert dataset.relevant_document_ids("q-answerable") == ("doc-a",)
    assert dataset.relevant_document_ids("q-unanswerable") == ()


def test_missing_required_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="does not exist"):
        load_beir_dataset(tmp_path / "missing")


def test_missing_required_file_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "qrels").mkdir()
    _write_jsonl(tmp_path / "corpus.jsonl", [])

    with pytest.raises(DataValidationError, match="queries.jsonl"):
        load_beir_dataset(tmp_path)


def test_malformed_jsonl_reports_line_number(tmp_path: Path) -> None:
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text(
        '{"_id": "doc-1", "title": "", "text": "valid"}\n{invalid\n',
        encoding="utf-8",
    )
    _write_jsonl(
        tmp_path / "queries.jsonl",
        [{"_id": "query-1", "text": "query"}],
    )
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n",
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match=r"corpus\.jsonl:2"):
        load_beir_dataset(tmp_path)


def test_missing_jsonl_field_is_rejected(tmp_path: Path) -> None:
    _write_minimal_dataset(
        tmp_path,
        qrels="query-id\tcorpus-id\tscore\nquery-1\tdoc-1\t1\n",
    )
    _write_jsonl(
        tmp_path / "corpus.jsonl",
        [{"_id": "doc-1", "title": "Missing text"}],
    )

    with pytest.raises(DataValidationError, match="text must be a non-empty string"):
        load_beir_dataset(tmp_path)


def test_non_positive_qrels_are_filtered(tmp_path: Path) -> None:
    _write_minimal_dataset(
        tmp_path,
        qrels="query-id\tcorpus-id\tscore\nquery-1\tdoc-1\t0\n",
    )

    dataset = load_beir_dataset(tmp_path)

    assert dataset.qrels == ()
    assert dataset.relevant_document_ids("query-1") == ()


def test_unknown_qrel_document_is_rejected(tmp_path: Path) -> None:
    _write_minimal_dataset(
        tmp_path,
        qrels="query-id\tcorpus-id\tscore\nquery-1\tdoc-missing\t1\n",
    )

    with pytest.raises(DataValidationError, match="unknown document"):
        load_beir_dataset(tmp_path)


def test_duplicate_qrel_is_rejected(tmp_path: Path) -> None:
    _write_minimal_dataset(
        tmp_path,
        qrels=(
            "query-id\tcorpus-id\tscore\n"
            "query-1\tdoc-1\t1\n"
            "query-1\tdoc-1\t2\n"
        ),
    )

    with pytest.raises(DataValidationError, match="Duplicate relevance judgment"):
        load_beir_dataset(tmp_path)


def test_invalid_qrel_score_is_rejected(tmp_path: Path) -> None:
    _write_minimal_dataset(
        tmp_path,
        qrels="query-id\tcorpus-id\tscore\nquery-1\tdoc-1\tone\n",
    )

    with pytest.raises(DataValidationError, match="score must be an integer"):
        load_beir_dataset(tmp_path)
