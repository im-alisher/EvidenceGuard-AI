"""Text and CSV exports for validated experiment artifacts."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any
from uuid import uuid4

from evidenceguard.errors import ReportError
from evidenceguard.reporting.artifacts import RunArtifact, load_run_artifact
from evidenceguard.reporting.markdown import render_markdown

_RETRIEVAL_CSV_COLUMNS = (
    "cutoff",
    "query_count",
    "mean_recall",
    "mean_precision",
    "mean_reciprocal_rank",
    "mean_average_precision",
    "mean_ndcg",
    "hit_rate",
)


def write_markdown_report(
    artifact: RunArtifact | str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a Markdown report for an artifact and return the output path."""

    loaded = _coerce_artifact(artifact)
    destination = _prepare_path(output_path, ".md", overwrite=overwrite)
    _write_text_atomic(destination, render_markdown(loaded))
    return destination


def write_retrieval_csv(
    artifact: RunArtifact | str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write aggregate retrieval metrics as CSV and return the output path."""

    loaded = _coerce_artifact(artifact)
    retrieval = loaded.retrieval
    if retrieval is None:
        raise ReportError("Run artifact does not contain retrieval results.")
    destination = _prepare_path(output_path, ".csv", overwrite=overwrite)

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(_RETRIEVAL_CSV_COLUMNS),
        lineterminator="\n",
    )
    writer.writeheader()
    for aggregate in retrieval["aggregates"]:
        row: dict[str, Any] = {}
        for column in _RETRIEVAL_CSV_COLUMNS:
            value = aggregate[column]
            row[column] = value if column in {"cutoff", "query_count"} else f"{value:.6f}"
        writer.writerow(row)

    _write_text_atomic(destination, buffer.getvalue())
    return destination


def _coerce_artifact(artifact: RunArtifact | str | Path) -> RunArtifact:
    if isinstance(artifact, RunArtifact):
        return artifact
    if isinstance(artifact, (str, Path)):
        return load_run_artifact(artifact)
    raise ReportError("artifact must be a RunArtifact or an artifact path.")


def _prepare_path(
    output_path: str | Path,
    suffix: str,
    *,
    overwrite: bool,
) -> Path:
    if not isinstance(overwrite, bool):
        raise ReportError("overwrite must be a boolean.")
    destination = Path(output_path)
    if destination.suffix.lower() != suffix:
        raise ReportError(f"Output path must use the {suffix} suffix.")
    if destination.exists() and not overwrite:
        raise ReportError(
            f"Output already exists: {destination}. Pass overwrite=True to replace it."
        )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ReportError(f"Could not create output directory: {exc}") from exc
    return destination


def _write_text_atomic(destination: Path, text: str) -> None:
    temporary_path = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        temporary_path.replace(destination)
    except OSError as exc:
        raise ReportError(f"Could not write report {destination}: {exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
