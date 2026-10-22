from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from evidenceguard.config import (
    ExperimentConfig,
    ExperimentSpec,
    LoggingConfig,
    PathsConfig,
)
from evidenceguard.errors import DataValidationError, ExperimentError
from evidenceguard.evaluation import (
    DecisionEvaluation,
    DecisionRecord,
    evaluate_calibration,
    evaluate_decisions,
)
from evidenceguard.experiments import (
    SCHEMA_VERSION,
    ExperimentResult,
    experiment_output_path,
    run_configured_experiment,
    run_retrieval_experiment,
    write_experiment_result,
)
from evidenceguard.experiments.cli import main
from evidenceguard.outcomes import DecisionOutcome


def _config(
    tmp_path: Path,
    *,
    dataset: Path,
    retriever: str = "bm25",
    cutoffs: tuple[int, ...] = (1, 3),
    top_k: int = 3,
) -> ExperimentConfig:
    return ExperimentConfig(
        version=1,
        name="offline-run",
        seed=13,
        paths=PathsConfig(
            datasets=tmp_path / "datasets",
            outputs=tmp_path / "reports",
        ),
        logging=LoggingConfig(),
        experiment=ExperimentSpec(
            dataset=dataset,
            retriever=retriever,
            cutoffs=cutoffs,
            top_k=top_k,
        ),
    )


def test_run_retrieval_experiment_on_fixture(
    tmp_path: Path,
    mini_beir_dir: Path,
) -> None:
    result = run_retrieval_experiment(_config(tmp_path, dataset=mini_beir_dir))

    assert result.name == "offline-run"
    assert result.seed == 13
    assert result.dataset_name == "mini_beir"
    assert result.split == "test"
    assert result.retriever_name == "bm25"
    assert result.retrieval is not None
    assert result.retrieval.cutoffs == (1, 3)
    assert result.retrieval.skipped_query_ids == ("q-unanswerable",)


def test_experiment_result_serializes_optional_sections(tmp_path: Path) -> None:
    result = run_retrieval_experiment(
        _config(tmp_path, dataset=Path("tests/fixtures/mini_beir").resolve())
    )

    payload = result.to_dict()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["experiment"] == {"name": "offline-run", "seed": 13}
    assert payload["dataset"] == {"name": "mini_beir", "split": "test"}
    assert payload["retriever_name"] == "bm25"
    assert payload["decisions"] is None
    assert payload["calibration"] is None
    assert payload["retrieval"]["cutoffs"] == [1, 3]
    json.dumps(payload)


def test_experiment_result_can_include_decision_and_calibration() -> None:
    records = (
        DecisionRecord(
            "ex1",
            DecisionOutcome.ANSWER_SUPPORTED,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.9,
        ),
        DecisionRecord(
            "ex2",
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            DecisionOutcome.ANSWER_SUPPORTED,
            0.6,
        ),
    )
    metrics = evaluate_decisions(records)
    result = ExperimentResult(
        name="decision-run",
        seed=7,
        dataset_name="synthetic",
        split="dev",
        decisions=DecisionEvaluation(
            records=records,
            metrics=metrics,
            policy_threshold=0.7,
        ),
        calibration=evaluate_calibration(records, n_bins=2),
    )

    payload = result.to_dict()

    assert payload["retrieval"] is None
    assert payload["decisions"]["metrics"]["total_count"] == 2
    assert payload["calibration"]["count"] == 2
    json.dumps(payload)


def test_experiment_result_requires_measured_output() -> None:
    with pytest.raises(DataValidationError, match="retrieval"):
        ExperimentResult(
            name="empty",
            seed=13,
            dataset_name="synthetic",
            split="test",
        )


@pytest.mark.parametrize(
    ("name", "seed", "dataset_name", "split"),
    [
        ("", 13, "dataset", "test"),
        ("run", -1, "dataset", "test"),
        ("run", 13, "", "test"),
        ("run", 13, "dataset", ""),
    ],
)
def test_experiment_result_rejects_invalid_identity(
    name: str,
    seed: int,
    dataset_name: str,
    split: str,
) -> None:
    with pytest.raises(DataValidationError):
        ExperimentResult(
            name=name,
            seed=seed,
            dataset_name=dataset_name,
            split=split,
        )


def test_write_experiment_result_and_overwrite(
    tmp_path: Path,
    mini_beir_dir: Path,
) -> None:
    result = run_retrieval_experiment(_config(tmp_path, dataset=mini_beir_dir))
    output = tmp_path / "reports" / "run.json"

    written = write_experiment_result(result, output)
    first_text = output.read_text(encoding="utf-8")
    first_payload = json.loads(first_text)

    assert written == output
    assert first_payload["schema_version"] == SCHEMA_VERSION
    with pytest.raises(ExperimentError, match="already exists"):
        write_experiment_result(result, output)

    second = write_experiment_result(result, output, overwrite=True)
    assert second.read_text(encoding="utf-8") == first_text


def test_write_experiment_result_rejects_invalid_arguments(
    tmp_path: Path,
    mini_beir_dir: Path,
) -> None:
    result = run_retrieval_experiment(_config(tmp_path, dataset=mini_beir_dir))

    with pytest.raises(ExperimentError, match="ExperimentResult"):
        write_experiment_result(
            object(),  # type: ignore[arg-type]
            tmp_path / "run.json",
        )
    with pytest.raises(ExperimentError, match="overwrite"):
        write_experiment_result(
            result,
            tmp_path / "run.json",
            overwrite="yes",  # type: ignore[arg-type]
        )
    with pytest.raises(ExperimentError, match=".json"):
        write_experiment_result(result, tmp_path / "run.txt")


def test_experiment_output_path(tmp_path: Path, mini_beir_dir: Path) -> None:
    config = _config(tmp_path, dataset=mini_beir_dir)

    assert experiment_output_path(config) == tmp_path / "reports" / "offline-run.json"


def test_runner_requires_experiment_section(tmp_path: Path) -> None:
    config = ExperimentConfig(
        version=1,
        name="no-experiment",
        seed=13,
        paths=PathsConfig(tmp_path / "datasets", tmp_path / "reports"),
        logging=LoggingConfig(),
    )

    with pytest.raises(ExperimentError, match="does not define"):
        run_retrieval_experiment(config)
    with pytest.raises(ExperimentError, match="does not define"):
        experiment_output_path(config)


def test_runner_rejects_invalid_config_and_retriever(
    tmp_path: Path,
    mini_beir_dir: Path,
) -> None:
    with pytest.raises(ExperimentError, match="ExperimentConfig"):
        run_retrieval_experiment(object())  # type: ignore[arg-type]

    config = _config(tmp_path, dataset=mini_beir_dir, retriever="dense")

    with pytest.raises(ExperimentError, match="Unsupported"):
        run_retrieval_experiment(config)


def test_run_configured_experiment_writes_artifact(
    tmp_path: Path,
    mini_beir_dir: Path,
) -> None:
    config = _config(tmp_path, dataset=mini_beir_dir)

    path = run_configured_experiment(config)

    assert path == tmp_path / "reports" / "offline-run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["retrieval"]["cutoffs"] == [1, 3]
    with pytest.raises(ExperimentError, match="already exists"):
        run_configured_experiment(config)


def test_cli_runs_config_and_prints_artifact(
    tmp_path: Path,
    mini_beir_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_payload: dict[str, Any] = {
        "version": 1,
        "name": "cli-run",
        "seed": 13,
        "paths": {"datasets": "datasets", "outputs": "reports"},
        "logging": {"level": "INFO"},
        "experiment": {
            "dataset": str(mini_beir_dir),
            "retriever": "bm25",
            "cutoffs": [1, 3],
            "top_k": 3,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")

    exit_code = main([str(config_path), "--base-dir", str(tmp_path)])

    captured = capsys.readouterr()
    output_path = tmp_path / "reports" / "cli-run.json"
    assert exit_code == 0
    assert str(output_path) in captured.out
    assert output_path.exists()


def test_cli_reports_configuration_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main([str(tmp_path / "missing.yaml")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "error:" in captured.err
