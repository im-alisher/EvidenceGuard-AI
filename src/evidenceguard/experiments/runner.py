"""Configuration-driven execution of reproducible retrieval experiments."""

from __future__ import annotations

from pathlib import Path

from evidenceguard.config import ExperimentConfig, ExperimentSpec
from evidenceguard.data.beir import load_beir_dataset
from evidenceguard.errors import ExperimentError
from evidenceguard.evaluation.retrieval import evaluate_retriever
from evidenceguard.experiments.result import (
    ExperimentResult,
    write_experiment_result,
)
from evidenceguard.retrieval.bm25 import BM25Retriever


def experiment_output_path(config: ExperimentConfig) -> Path:
    """Return the configured output JSON path for an experiment."""

    _require_config(config)
    if config.experiment is None:
        raise ExperimentError("Configuration does not define an experiment.")
    return config.paths.outputs / f"{config.name}.json"


def run_retrieval_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Load a local BEIR dataset and evaluate the configured BM25 baseline."""

    _require_config(config)
    spec = config.experiment
    if spec is None:
        raise ExperimentError("Configuration does not define an experiment.")
    _validate_spec(spec)

    dataset = load_beir_dataset(spec.dataset)
    retriever = BM25Retriever(dataset.documents)
    retrieval = evaluate_retriever(
        retriever,
        dataset,
        cutoffs=spec.cutoffs,
        top_k=spec.top_k,
    )
    return ExperimentResult(
        name=config.name,
        seed=config.seed,
        dataset_name=dataset.name,
        split=dataset.split,
        retriever_name=retriever.name,
        retrieval=retrieval,
    )


def run_configured_experiment(
    config: ExperimentConfig,
    *,
    overwrite: bool = False,
) -> Path:
    """Run the configured experiment, write its artifact, and return the path."""

    result = run_retrieval_experiment(config)
    output_path = experiment_output_path(config)
    return write_experiment_result(result, output_path, overwrite=overwrite)


def _require_config(config: ExperimentConfig) -> None:
    if not isinstance(config, ExperimentConfig):
        raise ExperimentError("config must be an ExperimentConfig.")


def _validate_spec(spec: ExperimentSpec) -> None:
    if not isinstance(spec, ExperimentSpec):
        raise ExperimentError("experiment must be an ExperimentSpec.")
    if spec.retriever != "bm25":
        raise ExperimentError(
            f"Unsupported experiment retriever: {spec.retriever!r}. "
            "Only 'bm25' is currently supported."
        )
