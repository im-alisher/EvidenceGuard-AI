"""Retrieval, decision, and calibration evaluation metrics."""

from evidenceguard.evaluation.calibration import (
    CalibrationBin,
    CalibrationMetrics,
    evaluate_calibration,
)
from evidenceguard.evaluation.decisions import (
    DecisionEvaluation,
    DecisionMetrics,
    DecisionRecord,
    apply_decision_threshold,
    evaluate_decisions,
    evaluate_evidence_assessor,
)
from evidenceguard.evaluation.metrics import (
    AggregateRetrievalMetrics,
    QueryRetrievalMetrics,
    aggregate_query_metrics,
    compute_query_metrics,
)
from evidenceguard.evaluation.retrieval import RetrievalEvaluation, evaluate_retriever
from evidenceguard.evaluation.selective import (
    ThresholdSelection,
    fit_abstention_threshold,
    selective_risk_curve,
)

__all__ = [
    "AggregateRetrievalMetrics",
    "CalibrationBin",
    "CalibrationMetrics",
    "DecisionEvaluation",
    "DecisionMetrics",
    "DecisionRecord",
    "QueryRetrievalMetrics",
    "RetrievalEvaluation",
    "ThresholdSelection",
    "aggregate_query_metrics",
    "apply_decision_threshold",
    "compute_query_metrics",
    "evaluate_calibration",
    "evaluate_decisions",
    "evaluate_evidence_assessor",
    "evaluate_retriever",
    "fit_abstention_threshold",
    "selective_risk_curve",
]
