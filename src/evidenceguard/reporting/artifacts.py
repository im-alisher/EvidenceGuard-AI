"""Validation and loading for versioned experiment run artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from evidenceguard.errors import ReportError
from evidenceguard.experiments.result import SCHEMA_VERSION

_RETRIEVAL_AGGREGATE_KEYS = frozenset(
    {
        "cutoff",
        "query_count",
        "mean_recall",
        "mean_precision",
        "mean_reciprocal_rank",
        "mean_average_precision",
        "mean_ndcg",
        "hit_rate",
    }
)
_DECISION_METRIC_KEYS = frozenset(
    {
        "total_count",
        "answered_count",
        "correct_count",
        "accuracy",
        "coverage",
        "selective_accuracy",
        "selective_risk",
        "abstention_rate",
    }
)


@dataclass(frozen=True, slots=True)
class RunArtifact:
    """A validated view of one experiment artifact."""

    payload: Mapping[str, Any]
    source: Path | None = None

    @property
    def schema_version(self) -> int:
        """Return the artifact schema version."""

        return int(self.payload["schema_version"])

    @property
    def name(self) -> str:
        """Return the experiment name."""

        return str(self.payload["experiment"]["name"])

    @property
    def seed(self) -> int:
        """Return the configured seed."""

        return int(self.payload["experiment"]["seed"])

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""

        return str(self.payload["dataset"]["name"])

    @property
    def split(self) -> str:
        """Return the dataset split."""

        return str(self.payload["dataset"]["split"])

    @property
    def retriever_name(self) -> str | None:
        """Return the retriever name when one was recorded."""

        value = self.payload.get("retriever_name")
        return None if value is None else str(value)

    @property
    def retrieval(self) -> Mapping[str, Any] | None:
        """Return retrieval results when measured."""

        value = self.payload.get("retrieval")
        return None if value is None else _expect_mapping(value, "retrieval")

    @property
    def decisions(self) -> Mapping[str, Any] | None:
        """Return decision results when measured."""

        value = self.payload.get("decisions")
        return None if value is None else _expect_mapping(value, "decisions")

    @property
    def calibration(self) -> Mapping[str, Any] | None:
        """Return calibration results when measured."""

        value = self.payload.get("calibration")
        return None if value is None else _expect_mapping(value, "calibration")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-serializable copy of the artifact payload."""

        return cast(dict[str, Any], _deep_copy(self.payload))


def load_run_artifact(path: str | Path) -> RunArtifact:
    """Load and validate a JSON experiment artifact from disk."""

    artifact_path = Path(path)
    try:
        text = artifact_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ReportError(f"Run artifact not found: {artifact_path}") from exc
    except OSError as exc:
        raise ReportError(f"Could not read run artifact {artifact_path}: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReportError(f"Invalid JSON in run artifact {artifact_path}: {exc}") from exc
    return parse_run_artifact(payload, source=artifact_path)


def parse_run_artifact(
    payload: object,
    *,
    source: Path | None = None,
) -> RunArtifact:
    """Validate a decoded artifact mapping and return a read-only view."""

    mapping = _expect_mapping(payload, "run artifact")
    schema_version = mapping.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SCHEMA_VERSION
    ):
        raise ReportError(
            f"Unsupported artifact schema_version: {schema_version!r}; "
            f"expected {SCHEMA_VERSION}."
        )

    experiment = _expect_mapping(mapping.get("experiment"), "experiment")
    _expect_non_empty_string(experiment.get("name"), "experiment.name")
    _expect_int(experiment.get("seed"), "experiment.seed")

    dataset = _expect_mapping(mapping.get("dataset"), "dataset")
    _expect_non_empty_string(dataset.get("name"), "dataset.name")
    _expect_non_empty_string(dataset.get("split"), "dataset.split")

    retriever_name = mapping.get("retriever_name")
    if retriever_name is not None:
        _expect_non_empty_string(retriever_name, "retriever_name")

    retrieval = mapping.get("retrieval")
    decisions = mapping.get("decisions")
    calibration = mapping.get("calibration")
    if retrieval is None and decisions is None and calibration is None:
        raise ReportError("Run artifact contains no measured result sections.")

    if retrieval is not None:
        _validate_retrieval(retrieval)
    if decisions is not None:
        _validate_decisions(decisions)
    if calibration is not None:
        _validate_calibration(calibration)

    return RunArtifact(payload=mapping, source=source)


def _validate_retrieval(value: object) -> None:
    retrieval = _expect_mapping(value, "retrieval")
    cutoffs = retrieval.get("cutoffs")
    if not isinstance(cutoffs, list) or not cutoffs:
        raise ReportError("retrieval.cutoffs must be a non-empty list.")
    for cutoff in cutoffs:
        _expect_positive_int(cutoff, "retrieval.cutoffs item")

    aggregates = retrieval.get("aggregates")
    if not isinstance(aggregates, list) or not aggregates:
        raise ReportError("retrieval.aggregates must be a non-empty list.")
    for aggregate in aggregates:
        metrics = _expect_mapping(aggregate, "retrieval.aggregates item")
        missing = _RETRIEVAL_AGGREGATE_KEYS.difference(metrics)
        if missing:
            raise ReportError(
                "Retrieval aggregate is missing keys: " + ", ".join(sorted(missing))
            )
        _expect_positive_int(metrics["cutoff"], "retrieval.aggregates.cutoff")
        _expect_positive_int(
            metrics["query_count"],
            "retrieval.aggregates.query_count",
        )
        for key in _RETRIEVAL_AGGREGATE_KEYS.difference(
            {"cutoff", "query_count"}
        ):
            _expect_unit_interval(metrics[key], f"retrieval.aggregates.{key}")

    if not isinstance(retrieval.get("per_query"), list):
        raise ReportError("retrieval.per_query must be a list.")
    skipped = retrieval.get("skipped_query_ids")
    if not isinstance(skipped, list) or any(
        not isinstance(item, str) for item in skipped
    ):
        raise ReportError("retrieval.skipped_query_ids must be a list of strings.")


def _validate_decisions(value: object) -> None:
    decisions = _expect_mapping(value, "decisions")
    metrics = _expect_mapping(decisions.get("metrics"), "decisions.metrics")
    missing = _DECISION_METRIC_KEYS.difference(metrics)
    if missing:
        raise ReportError(
            "Decision metrics are missing keys: " + ", ".join(sorted(missing))
        )
    for key in _DECISION_METRIC_KEYS.difference(
        {"total_count", "answered_count", "correct_count"}
    ):
        _expect_unit_interval(metrics[key], f"decisions.metrics.{key}")
    total_count = _expect_positive_int(
        metrics["total_count"],
        "decisions.metrics.total_count",
    )
    answered_count = _expect_int(
        metrics["answered_count"],
        "decisions.metrics.answered_count",
    )
    correct_count = _expect_int(
        metrics["correct_count"],
        "decisions.metrics.correct_count",
    )
    if answered_count > total_count:
        raise ReportError("decisions.metrics.answered_count exceeds total_count.")
    if correct_count > total_count:
        raise ReportError("decisions.metrics.correct_count exceeds total_count.")
    policy_threshold = decisions.get("policy_threshold")
    if policy_threshold is not None:
        _expect_unit_interval(policy_threshold, "decisions.policy_threshold")
    if not isinstance(decisions.get("records"), list):
        raise ReportError("decisions.records must be a list.")


def _validate_calibration(value: object) -> None:
    calibration = _expect_mapping(value, "calibration")
    _expect_positive_int(calibration.get("count"), "calibration.count")
    _expect_unit_interval(calibration.get("brier_score"), "calibration.brier_score")
    _expect_unit_interval(
        calibration.get("expected_calibration_error"),
        "calibration.expected_calibration_error",
    )
    bins = calibration.get("bins")
    if not isinstance(bins, list):
        raise ReportError("calibration.bins must be a list.")
    for item in bins:
        calibration_bin = _expect_mapping(item, "calibration.bins item")
        _expect_unit_interval(
            calibration_bin.get("lower_bound"),
            "calibration.bins.lower_bound",
        )
        _expect_unit_interval(
            calibration_bin.get("upper_bound"),
            "calibration.bins.upper_bound",
        )
        _expect_positive_int(
            calibration_bin.get("count"),
            "calibration.bins.count",
        )
        _expect_unit_interval(
            calibration_bin.get("mean_confidence"),
            "calibration.bins.mean_confidence",
        )
        _expect_unit_interval(
            calibration_bin.get("accuracy"),
            "calibration.bins.accuracy",
        )


def _expect_mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportError(f"{where} must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise ReportError(f"{where} must use string keys.")
    return value


def _expect_non_empty_string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReportError(f"{where} must be a non-empty string.")
    return value


def _expect_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportError(f"{where} must be an integer.")
    if value < 0:
        raise ReportError(f"{where} must be non-negative.")
    return value


def _expect_positive_int(value: object, where: str) -> int:
    numeric = _expect_int(value, where)
    if numeric < 1:
        raise ReportError(f"{where} must be positive.")
    return numeric


def _expect_unit_interval(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{where} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ReportError(f"{where} must be between 0 and 1.")
    return numeric


def _deep_copy(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value
