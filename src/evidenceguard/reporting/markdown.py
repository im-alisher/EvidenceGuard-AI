"""Deterministic Markdown rendering for experiment artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from evidenceguard.errors import ReportError
from evidenceguard.reporting.artifacts import RunArtifact


def render_markdown(artifact: RunArtifact) -> str:
    """Render a compact research report without adding unmeasured results."""

    if not isinstance(artifact, RunArtifact):
        raise ReportError("artifact must be a RunArtifact.")

    lines = [
        f"# EvidenceGuard AI run: {_escape(artifact.name)}",
        "",
        "> Generated automatically from a machine-readable run artifact.",
        "> This report does not add results, infer missing sections, or establish",
        "> scientific validity. Verify dataset provenance before citing.",
        "",
    ]
    if artifact.source is not None:
        lines.extend([f"Artifact file: `{_escape(artifact.source.name)}`", ""])

    lines.extend(
        [
            "## Run metadata",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Experiment | {_escape(artifact.name)} |",
            f"| Seed | {artifact.seed} |",
            f"| Dataset | {_escape(artifact.dataset_name)} |",
            f"| Split | {_escape(artifact.split)} |",
            f"| Retriever | {_escape(artifact.retriever_name or 'not recorded')} |",
            "",
        ]
    )

    retrieval = artifact.retrieval
    if retrieval is None:
        lines.extend(["## Retrieval", "", "Not measured in this artifact.", ""])
    else:
        lines.extend(_render_retrieval(retrieval))

    decisions = artifact.decisions
    if decisions is None:
        lines.extend(["## Decisions", "", "Not measured in this artifact.", ""])
    else:
        lines.extend(_render_decisions(decisions))

    calibration = artifact.calibration
    if calibration is None:
        lines.extend(["## Calibration", "", "Not measured in this artifact.", ""])
    else:
        lines.extend(_render_calibration(calibration))

    lines.extend(
        [
            "---",
            "",
            "JSON artifacts remain the source of truth. Missing sections indicate",
            "unmeasured outputs, not zero-valued metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_retrieval(retrieval: Mapping[str, Any]) -> list[str]:
    aggregates = retrieval["aggregates"]
    lines = [
        "## Retrieval",
        "",
        "| Cutoff | Queries | Recall | Precision | MRR | MAP | nDCG | Hit rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for aggregate in aggregates:
        lines.append(
            "| "
            f"{aggregate['cutoff']} | "
            f"{aggregate['query_count']} | "
            f"{_number(aggregate['mean_recall'])} | "
            f"{_number(aggregate['mean_precision'])} | "
            f"{_number(aggregate['mean_reciprocal_rank'])} | "
            f"{_number(aggregate['mean_average_precision'])} | "
            f"{_number(aggregate['mean_ndcg'])} | "
            f"{_number(aggregate['hit_rate'])} |"
        )

    skipped = retrieval.get("skipped_query_ids", [])
    lines.append("")
    if skipped:
        noun = "query item" if len(skipped) == 1 else "query items"
        verb = "was" if len(skipped) == 1 else "were"
        pronoun = "it" if len(skipped) == 1 else "they"
        lines.append(
            f"{len(skipped)} {noun} {verb} excluded from retrieval metrics "
            f"because {pronoun} had no positive relevance judgments:"
        )
        lines.append("")
        lines.append(", ".join(f"`{_escape(query_id)}`" for query_id in skipped))
    else:
        lines.append("No queries were excluded from retrieval metrics.")
    lines.extend(
        [
            "",
            "Retrieval metrics measure ranking against relevance judgments, not "
            "answer correctness or evidence sufficiency.",
            "",
        ]
    )
    return lines


def _render_decisions(decisions: Mapping[str, Any]) -> list[str]:
    metrics = decisions["metrics"]
    threshold = decisions.get("policy_threshold")
    lines = [
        "## Decisions",
        "",
        f"Policy threshold: {_threshold(threshold)}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Examples | {metrics['total_count']} |",
        f"| Answered | {metrics['answered_count']} |",
        f"| Correct outcomes | {metrics['correct_count']} |",
        f"| Accuracy | {_number(metrics['accuracy'])} |",
        f"| Coverage | {_number(metrics['coverage'])} |",
        f"| Selective accuracy | {_number(metrics['selective_accuracy'])} |",
        f"| Selective risk | {_number(metrics['selective_risk'])} |",
        f"| Abstention rate | {_number(metrics['abstention_rate'])} |",
        "",
        "Selective risk is measured among answered predictions only. Conflicting "
        "and insufficient decisions count as abstentions.",
        "",
    ]
    return lines


def _render_calibration(calibration: Mapping[str, Any]) -> list[str]:
    lines = [
        "## Calibration",
        "",
        f"Answered predictions: {calibration['count']}",
        "",
        f"- Brier score: {_number(calibration['brier_score'])}",
        "- Expected calibration error: "
        f"{_number(calibration['expected_calibration_error'])}",
        "",
    ]
    bins: Sequence[Mapping[str, Any]] = calibration.get("bins", [])
    if bins:
        lines.extend(
            [
                "| Confidence bin | Count | Mean confidence | Accuracy |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for calibration_bin in bins:
            lines.append(
                "| "
                f"[{_number(calibration_bin['lower_bound'])}, "
                f"{_number(calibration_bin['upper_bound'])}) | "
                f"{calibration_bin['count']} | "
                f"{_number(calibration_bin['mean_confidence'])} | "
                f"{_number(calibration_bin['accuracy'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "Calibration diagnostics cover confidence in surfaced answers; they do "
            "not establish answer correctness or calibrated scientific truth.",
            "",
        ]
    )
    return lines


def _number(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError("Expected a numeric artifact value.")
    return f"{float(value):.4f}"


def _threshold(value: object) -> str:
    if value is None:
        return "not recorded"
    return _number(value)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")
