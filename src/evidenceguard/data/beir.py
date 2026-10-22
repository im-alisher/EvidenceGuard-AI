"""Load retrieval datasets stored in the BEIR directory layout."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from evidenceguard.data.models import (
    Document,
    Query,
    RelevanceJudgment,
    RetrievalDataset,
)
from evidenceguard.errors import DataValidationError


def load_beir_dataset(
    dataset_dir: str | Path,
    *,
    split: str = "test",
    name: str | None = None,
) -> RetrievalDataset:
    """Load ``corpus.jsonl``, ``queries.jsonl``, and ``qrels/<split>.tsv``.

    Only positive relevance scores are retained as retrieval judgments. Extra
    fields in the JSONL records are ignored; missing required fields are errors.
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise DataValidationError(f"Dataset directory does not exist: {root}")

    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries.jsonl"
    qrels_path = root / "qrels" / f"{split}.tsv"
    for required_path in (corpus_path, queries_path, qrels_path):
        if not required_path.is_file():
            raise DataValidationError(f"Required dataset file not found: {required_path}")

    documents = tuple(_load_documents(corpus_path))
    queries = tuple(_load_queries(queries_path))
    qrels = tuple(_load_qrels(qrels_path))

    return RetrievalDataset(
        name=name or root.name,
        split=split,
        documents=documents,
        queries=queries,
        qrels=qrels,
        source=str(root),
    )


def _load_documents(path: Path) -> Iterator[Document]:
    for line_number, record in _read_jsonl(path):
        doc_id = _required_id(record, "_id", path, line_number)
        title = record.get("title", "")
        text = record.get("text")
        if title is None:
            title = ""
        if not isinstance(title, str):
            raise _record_error(path, line_number, "title must be a string or null")
        if not isinstance(text, str) or not text.strip():
            raise _record_error(path, line_number, "text must be a non-empty string")
        yield Document(doc_id=doc_id, title=title, text=text)


def _load_queries(path: Path) -> Iterator[Query]:
    for line_number, record in _read_jsonl(path):
        query_id = _required_id(record, "_id", path, line_number)
        title = record.get("title", "")
        text = record.get("text")
        if title is None:
            title = ""
        if not isinstance(title, str):
            raise _record_error(path, line_number, "title must be a string or null")
        if not isinstance(text, str) or not text.strip():
            raise _record_error(path, line_number, "text must be a non-empty string")
        yield Query(query_id=query_id, text=text, title=title)


def _load_qrels(path: Path) -> Iterator[RelevanceJudgment]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise DataValidationError(f"Qrels file has no header: {path}")
            columns = _resolve_qrel_columns(fieldnames, path)

            for line_number, row in enumerate(reader, start=2):
                if not any(value not in (None, "") for value in row.values()):
                    continue
                query_id = _required_tsv_value(row, columns["query_id"], path, line_number)
                doc_id = _required_tsv_value(row, columns["doc_id"], path, line_number)
                score_text = _required_tsv_value(row, columns["score"], path, line_number)
                try:
                    score = int(score_text)
                except ValueError as exc:
                    raise _record_error(
                        path,
                        line_number,
                        f"score must be an integer, got {score_text!r}",
                    ) from exc
                if score < 0:
                    raise _record_error(path, line_number, "score must be non-negative")
                if score > 0:
                    yield RelevanceJudgment(
                        query_id=query_id,
                        doc_id=doc_id,
                        relevance=score,
                    )
    except OSError as exc:
        raise DataValidationError(f"Could not read qrels file {path}: {exc}") from exc


def _resolve_qrel_columns(fieldnames: list[str], path: Path) -> dict[str, str]:
    normalized = {field.strip().lower().replace("_", "-"): field for field in fieldnames}
    aliases = {
        "query_id": ("query-id", "qid"),
        "doc_id": ("corpus-id", "doc-id", "docid"),
        "score": ("score", "relevance"),
    }
    resolved: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            if candidate in normalized:
                resolved[canonical] = normalized[candidate]
                break
        if canonical not in resolved:
            raise DataValidationError(
                f"Qrels file {path} is missing a column for {canonical!r}."
            )
    return resolved


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise _record_error(path, line_number, f"invalid JSON: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise _record_error(path, line_number, "record must be a JSON object")
                yield line_number, record
    except OSError as exc:
        raise DataValidationError(f"Could not read JSONL file {path}: {exc}") from exc


def _required_id(
    record: dict[str, Any],
    field_name: str,
    path: Path,
    line_number: int,
) -> str:
    value = record.get(field_name)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if not isinstance(value, str) or not value.strip():
        raise _record_error(
            path,
            line_number,
            f"{field_name} must be a non-empty string or integer",
        )
    return value


def _required_tsv_value(
    row: dict[str, str | None],
    column: str,
    path: Path,
    line_number: int,
) -> str:
    value = row.get(column)
    if not isinstance(value, str) or not value.strip():
        raise _record_error(path, line_number, f"column {column!r} must be non-empty")
    return value


def _record_error(path: Path, line_number: int, message: str) -> DataValidationError:
    return DataValidationError(f"{path}:{line_number}: {message}")
