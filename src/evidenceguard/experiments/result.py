"""Versioned, JSON-serializable experiment result artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidenceguard.errors import DataValidationError, ExperimentError
from evidenceguard.evaluation.calibration import CalibrationMetrics
from evidenceguard.evaluation.decisions import DecisionEvaluation
from evidenceguard.evaluation.retrieval import RetrievalEvaluation
from evidenceguard.reproducibility import validate_seed

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Measured outputs from one experiment run.

    At least one of retrieval, decision, or calibration results must be present.
    Optional sections are omitted as ``null`` in JSON when not measured.
    """

    name: str
    seed: int
    dataset_name: str
    split: str
    retriever_name: str | None = None
    retrieval: RetrievalEvaluation | None = None
    decisions: DecisionEvaluation | None = None
    calibration: CalibrationMetrics | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DataValidationError("Experiment name must be a non-empty string.")
        try:
            validate_seed(self.seed)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(str(exc)) from exc
        if not isinstance(self.dataset_name, str) or not self.dataset_name.strip():
            raise DataValidationError("dataset_name must be a non-empty string.")
        if not isinstance(self.split, str) or not self.split.strip():
            raise DataValidationError("split must be a non-empty string.")
        if self.retriever_name is not None and (
            not isinstance(self.retriever_name, str) or not self.retriever_name.strip()
        ):
            raise DataValidationError("retriever_name must be a non-empty string or None.")
        if self.retrieval is not None and not isinstance(
            self.retrieval, RetrievalEvaluation
        ):
            raise DataValidationError("retrieval must be a RetrievalEvaluation or None.")
        if self.decisions is not None and not isinstance(
            self.decisions, DecisionEvaluation
        ):
            raise DataValidationError("decisions must be a DecisionEvaluation or None.")
        if self.calibration is not None and not isinstance(
            self.calibration, CalibrationMetrics
        ):
            raise DataValidationError("calibration must be CalibrationMetrics or None.")
        if self.retrieval is None and self.decisions is None and self.calibration is None:
            raise DataValidationError(
                "ExperimentResult must include retrieval, decision, or calibration results."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable versioned artifact."""

        return {
            "schema_version": SCHEMA_VERSION,
            "experiment": {
                "name": self.name,
                "seed": self.seed,
            },
            "dataset": {
                "name": self.dataset_name,
                "split": self.split,
            },
            "retriever_name": self.retriever_name,
            "retrieval": None if self.retrieval is None else self.retrieval.to_dict(),
            "decisions": None if self.decisions is None else self.decisions.to_dict(),
            "calibration": (
                None if self.calibration is None else self.calibration.to_dict()
            ),
        }


def write_experiment_result(
    result: ExperimentResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a result artifact as deterministic, atomically replaced JSON."""

    if not isinstance(result, ExperimentResult):
        raise ExperimentError("result must be an ExperimentResult.")
    if not isinstance(overwrite, bool):
        raise ExperimentError("overwrite must be a boolean.")
    output_path = Path(path)
    if output_path.suffix.lower() != ".json":
        raise ExperimentError("Experiment result paths must use a .json suffix.")
    if output_path.exists() and not overwrite:
        raise ExperimentError(f"Experiment result already exists: {output_path}")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExperimentError(
            f"Could not create output directory {output_path.parent}: {exc}"
        ) from exc

    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_text(payload, encoding="utf-8")
        temporary_path.replace(output_path)
    except OSError as exc:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise ExperimentError(f"Could not write experiment result {output_path}: {exc}") from exc
    return output_path
