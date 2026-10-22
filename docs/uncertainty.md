# Abstention, selective risk, and calibration

This module turns evidence assessments into thresholded decisions and evaluates
their coverage and risk. It does not fit a new evidence model, calibrate
scientific truth, or establish benchmark performance.

## Answer confidence

`answer_confidence(assessment)` returns:

- the maximum supporting-document strength when the assessment is
  `ANSWER_SUPPORTED`;
- `0.0` for `INSUFFICIENT_EVIDENCE` and `CONFLICTING_EVIDENCE`.

The value is an uncalibrated evidence signal. It is not the probability that an
answer is factually correct.

## Threshold policy

`AbstentionPolicy(minimum_confidence)` applies one confidence threshold:

| Assessment | Threshold condition | Returned decision |
| --- | --- | --- |
| `ANSWER_SUPPORTED` | confidence >= threshold | `ANSWER_SUPPORTED` |
| `ANSWER_SUPPORTED` | confidence < threshold | `INSUFFICIENT_EVIDENCE` |
| `CONFLICTING_EVIDENCE` | any | `CONFLICTING_EVIDENCE` |
| `INSUFFICIENT_EVIDENCE` | any | `INSUFFICIENT_EVIDENCE` |

The policy only withholds supported answers. It does not turn conflict or missing
evidence into an answer.

```python
from evidenceguard.evidence import EvidenceAssessor, MetadataStanceClassifier
from evidenceguard.uncertainty import AbstentionPolicy

assessor = EvidenceAssessor(MetadataStanceClassifier())
policy = AbstentionPolicy(minimum_confidence=0.7)
decision = policy.decide(assessor.assess(query, documents))
```

## Decision records

`DecisionRecord` stores:

- `example_id`;
- expected and predicted decision outcomes;
- uncalibrated answer confidence;
- optional controlled evidence condition.

Records require unique example IDs. `DecisionEvaluation` combines records,
aggregate metrics, and the applied threshold and provides a JSON-compatible
`to_dict()` output.

`evaluate_evidence_assessor(assessor, examples, policy=...)` runs the assessment
and abstention policy over controlled examples and compares predictions with
their explicit gold outcomes.

## Decision metrics

For a non-empty record set:

- `coverage` = answered predictions / all predictions;
- `abstention_rate` = 1 - coverage;
- `accuracy` = all correct predictions / all predictions;
- `selective_accuracy` = correct answered predictions / answered predictions;
- `selective_risk` = 1 - selective accuracy.

When no answer is surfaced, `selective_accuracy` and `selective_risk` are defined
as `0.0`; the abstention is represented by zero coverage, not by a risk claim.

Conflict and insufficiency predictions count as abstentions. Accuracy still
checks the exact three-way outcome, so predicting `INSUFFICIENT_EVIDENCE` for a
gold `CONFLICTING_EVIDENCE` example is incorrect.

## Risk-targeted threshold selection

`fit_abstention_threshold(records, maximum_risk=...)` evaluates candidate
thresholds on supplied development records. Candidates are `0.0`, `1.0`, and
each confidence attached to a candidate answer.

Feasible thresholds have selective risk no greater than `maximum_risk`. Among
them, selection maximizes coverage, then minimizes risk, then prefers the higher
threshold. A zero-coverage threshold is feasible and may be selected if no
answered prediction meets the risk budget.

`selective_risk_curve(records, thresholds)` returns the full coverage/risk points
for requested thresholds. `apply_decision_threshold(records, threshold)` applies
a threshold without fitting.

Threshold selection must use development data only. Reusing the test split for
threshold fitting leaks labels and invalidates reported test risk.

## Calibration diagnostics

`evaluate_calibration(records, n_bins=10)` includes only records predicted to
answer. A surfaced answer is treated as correct when the expected outcome is also
`ANSWER_SUPPORTED`.

It reports:

- Brier score over answer confidence and answer correctness;
- expected calibration error (ECE) over equal-width confidence bins;
- non-empty bins with count, mean confidence, and accuracy.

This is calibration of the confidence attached to **surfaced answers**. It does
not calibrate abstention confidence, evidence truth, or answer content. A small
ECE on synthetic labels would not demonstrate calibrated scientific fact
verification.

## Limitations

- Confidence signals are inherited from the evidence classifier and are not
  calibrated probabilities.
- Threshold fitting is empirical and can overfit small development sets.
- No statistical uncertainty intervals are reported yet.
- Controlled metadata labels are not automatic NLI predictions.
- No real benchmark thresholds, coverage, risk, or calibration results are
  reported in this phase.
