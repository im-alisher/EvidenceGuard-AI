from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evidenceguard.config import (
    ExperimentConfig,
    ExperimentSpec,
    LoggingConfig,
    PathsConfig,
)
from evidenceguard.errors import ReportError
from evidenceguard.evaluation import (
    DecisionEvaluation,
    DecisionRecord,
    evaluate_calibration,
    evaluate_decisions,
)
from evidenceguard.experiments import (
    ExperimentResult,
    run_retrieval_experiment,
    write_experiment_result,
)
from evidenceguard.outcomes import DecisionOutcome
from evidenceguard.reporting import (
    load_run_artifact,
    parse_run_artifact,
    render_markdown,
    write_markdown_report,
    write_retrieval_csv,
)


def _config(tmp_path: Path, dataset: Path) -> ExperimentConfig:
    return ExperimentConfig(
        version=1,
        name="report-run",
        seed=13,
        paths=PathsConfig(
            datasets=tmp_path / "datasets",
            outputs=tmp_path / "reports",
        ),
        logging=LoggingConfig(),
        experiment=ExperimentSpec(
            dataset=dataset,
            retriever="bm25",
            cutoffs=(1, 3),
            top_k=3,
        ),
    )


def _retrieval_artifact(
    tmp_path: Path,
    mini_beir_dir: Path,
) -> tuple[ExperimentResult, Path]:
    result = run_retrieval_experiment(_config(tmp_path, mini_beir_dir))
    path = write_experiment_result(result, tmp_path / "run.json")
    return result, path


def _decision_result() -> ExperimentResult:
    records = (
        DecisionRecord(
            "ex1",
            DecisionOutcome.ANSWER_SUPPORTED,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.9,
        ),
        DecisionRecord(
            "ex2",
            DecisionOutcome.ANSWER_SUPPORTED,
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            0.2,
        ),
    )
    return ExperimentResult(
        name="decision-run",
        seed=7,
        dataset_name="synthetic",
        split="dev",
        decisions=DecisionEvaluation(
            records=records,
            metrics=evaluate_decisions(records),
            policy_threshold=0.8,
        ),
        calibration=evaluate_calibration(records, n_bins=2),
    )


def test_load_run_artifact(
    tmp_path: Path,
    mini_beir_dir: Path,
) -> None:
    result, path = _retrieval_artifact(tmp_path, mini_beir_dir)

    artifact = load_run_artifact(path)

    assert artifact.schema_version == 1
    assert artifact.name == result.name
    assert artifact.seed == 13
    assert artifact.dataset_name == "mini_beir"
    assert artifact.split == "test"
    assert artifact.retriever_name == "bm25"
    assert artifact.retrieval is not None
    assert artifact.decisions is None
    assert artifact.calibration is None
    assert artifact.source == path


def test_run_artifact_to_dict_is_detached(
    tmp_path: Path,
    mini_beir_dir: Path,
) -> None:
    _, path = _retrieval_artifact(tmp_path, mini_beir_dir)
    artifact = load_run_artifact(str(path))

    payload = artifact.to_dict()
    payload["experiment"]["name"] = "mutated"

    assert artifact.name == "report-run"


def test_render_markdown_for_retrieval(
    tmp_path: Path,
    mini_beir_dir: Path,
) -> None:
    _, path = _retrieval_artifact(tmp_path, mini_beir_dir)

    report = render_markdown(load_run_artifact(path))

    assert report.startswith("# EvidenceGuard AI run: report-run")
    assert "| 1 |" in report
    assert "| 3 |" in report
    assert "q-unanswerable" in report
    assert "No decision-level results" not in report
    assert "## Decisions\n\nNot measured in this artifact." in report
    assert "## Calibration\n\nNot measured in this artifact." in report
    assert "not answer correctness" in report


def test_render_markdown_for_decisions_and_calibration(tmp_path: Path) -> None:
    path = write_experiment_result(_decision_result(), tmp_path / "run.json")
    artifact = load_run_artifact(path)

    report = render_markdown(artifact)

    assert "Policy threshold: 0.8000" in report
    assert "| Coverage | 0.5000 |" in report
    assert "| Selective risk | 0.0000 |" in report
    assert "Brier score:" in report
    assert "Expected calibration error:" in report
    assert "| Confidence bin |" in report
    assert "## Retrieval\n\nNot measured in this artifact." in report


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema_version": 2},
        {
            "schema_version": 1,
            "experiment": {"name": "run", "seed": 1},
            "dataset": {"name": "data", "split": "test"},
            "retriever_name": "bm25",
            "retrieval": None,
            "decisions": None,
            "calibration": None,
        },
        {
            "schema_version": 1,
            "experiment": {"name": "", "seed": 1},
            "dataset": {"name": "data", "split": "test"},
            "retriever_name": "bm25",
            "retrieval": {"cutoffs": [1], "aggregates": [], "per_query": []},
        },
        {
            "schema_version": 1,
            "experiment": {"name": "run", "seed": 1},
            "dataset": {"name": "data", "split": "test"},
            "retriever_name": "bm25",
            "retrieval": {
                "cutoffs": [1],
                "aggregates": [{"cutoff": 1}],
                "per_query": [],
                "skipped_query_ids": [],
            },
        },
        {
            "schema_version": 1,
            "experiment": {"name": "run", "seed": 1},
            "dataset": {"name": "data", "split": "test"},
            "retriever_name": "bm25",
            "decisions": {"metrics": {}, "records": []},
        },
        {
            "schema_version": 1,
            "experiment": {"name": "run", "seed": 1},
            "dataset": {"name": "data", "split": "test"},
            "retriever_name": "bm25",
            "calibration": {
                "count": 1,
                "brier_score": 0.1,
                "expected_calibration_error": 0.1,
                "bins": [{"lower_bound": 0.0}],
            },
        },
    ],
)
def test_parse_run_artifact_rejects_invalid_payloads(payload: object) -> None:
    with pytest.raises(ReportError):
        parse_run_artifact(payload)


def test_load_run_artifact_rejects_missing_and_invalid_json(tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="not found"):
        load_run_artifact(tmp_path / "missing.json")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(ReportError, match="Invalid JSON"):
        load_run_artifact(invalid)


def test_write_markdown_report_and_overwrite(
    tmp_path: Path,
    mini_beir_dir: Path,
) -> None:
    _, artifact_path = _retrieval_artifact(tmp_path, mini_beir_dir)
    output = tmp_path / "reports" / "report.md"

    written = write_markdown_report(artifact_path, output)

    assert written == output
    first = output.read_text(encoding="utf-8")
    assert first.startswith("# EvidenceGuard AI run")
    with pytest.raises(ReportError, match="already exists"):
        write_markdown_report(artifact_path, output)

    write_markdown_report(artifact_path, output, overwrite=True)
    assert output.read_text(encoding="utf-8") == first


def test_write_markdown_report_accepts_artifact_and_validates_path(
    tmp_path: Path,
    mini_beir_dir: Path,
) -> None:
    _, artifact_path = _retrieval_artifact(tmp_path, mini_beir_dir)
    artifact = load_run_artifact(artifact_path)

    with pytest.raises(ReportError, match=".md"):
        write_markdown_report(artifact, tmp_path / "report.txt")
    with pytest.raises(ReportError, match="overwrite"):
        write_markdown_report(
            artifact,
            tmp_path / "report.md",
            overwrite="yes",  # type: ignore[arg-type]
        )
    with pytest.raises(ReportError, match="RunArtifact"):
        write_markdown_report(
            object(),  # type: ignore[arg-type]
            tmp_path / "report.md",
        )

    written = write_markdown_report(artifact, tmp_path / "report.md")
    assert written.exists()


def test_write_retrieval_csv(
    tmp_path: Path,
    mini_beir_dir: Path,
) -> None:
    _, artifact_path = _retrieval_artifact(tmp_path, mini_beir_dir)
    output = tmp_path / "reports" / "retrieval.csv"

    written = write_retrieval_csv(artifact_path, output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert written == output
    assert lines[0] == (
        "cutoff,query_count,mean_recall,mean_precision,mean_reciprocal_rank,"
        "mean_average_precision,mean_ndcg,hit_rate"
    )
    assert len(lines) == 3
    assert lines[1].startswith("1,")
    with pytest.raises(ReportError, match="already exists"):
        write_retrieval_csv(artifact_path, output)


def test_write_retrieval_csv_requires_retrieval(tmp_path: Path) -> None:
    artifact_path = write_experiment_result(
        _decision_result(),
        tmp_path / "decisions.json",
    )

    with pytest.raises(ReportError, match="does not contain retrieval"):
        write_retrieval_csv(artifact_path, tmp_path / "retrieval.csv")


def test_write_retrieval_csv_requires_csv_suffix(
    tmp_path: Path,
    mini_beir_dir: Path,
) -> None:
    _, artifact_path = _retrieval_artifact(tmp_path, mini_beir_dir)

    with pytest.raises(ReportError, match=".csv"):
        write_retrieval_csv(
            load_run_artifact(artifact_path),
            tmp_path / "retrieval.txt",
        )


def test_parse_run_artifact_can_omit_source() -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": {"name": "run", "seed": 1},
        "dataset": {"name": "synthetic", "split": "dev"},
        "retriever_name": None,
        "retrieval": None,
        "decisions": {
            "policy_threshold": None,
            "metrics": {
                "total_count": 1,
                "answered_count": 0,
                "correct_count": 0,
                "accuracy": 0.0,
                "coverage": 0.0,
                "selective_accuracy": 0.0,
                "selective_risk": 1.0,
                "abstention_rate": 1.0,
            },
            "records": [],
        },
        "calibration": None,
    }

    artifact = parse_run_artifact(payload)

    assert artifact.source is None
    assert artifact.retriever_name is None
    assert artifact.decisions is not None
    json.dumps(artifact.to_dict())
