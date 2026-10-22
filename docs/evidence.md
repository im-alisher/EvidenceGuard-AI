# Evidence assessment

This module turns query-specific document stance labels into an explicit
evidence decision. It does not retrieve documents, generate answers, infer
natural-language entailment, or calibrate abstention thresholds.

## Scope

The Phase 5 implementation provides:

- a validated stance taxonomy;
- a classifier interface for batch document assessment;
- an adapter that reads precomputed labels from document metadata;
- a deterministic aggregation policy over document stances;
- JSON-serializable assessment output.

No stance model is shipped. In particular, this phase does **not** claim to
determine automatically whether scientific text supports or contradicts a claim.
The metadata adapter exists for controlled scenarios and externally supplied
classifier outputs. A later NLI or evidence-model adapter can implement the same
interface without changing the assessment result schema.

## Stance taxonomy

Each document receives exactly one stance relative to the query:

| Stance | Meaning |
| --- | --- |
| `supports` | The document supports the query or claim. |
| `contradicts` | The document contradicts the query or claim. |
| `neutral` | The document is relevant context but neither supports nor contradicts. |
| `unknown` | A stance could not be established or was not supplied. |

Stance labels are query-specific. For question-style queries, the annotator or
classifier must define what "supports" means for that question. The module does
not reinterpret the query.

`DocumentEvidence.strength` is an optional value in `[0, 1]`. It is retained for
later uncertainty analysis but is **not** a calibrated probability. The current
decision policy does not use strength to filter documents; it reports the
maximum supporting and contradicting strengths for auditability only.

## Metadata adapter

`MetadataStanceClassifier` reads two optional document metadata keys:

- `evidence_stance`: one of `supports`, `contradicts`, `neutral`, `unknown`,
  matched case-insensitively after trimming;
- `evidence_strength`: a finite number in `[0, 1]`, defaulting to `1.0`.

A missing stance becomes `unknown`. Malformed stance or strength values raise
`DataValidationError` rather than being silently ignored. The adapter performs
no text inference and does not read benchmark outcome labels.

```python
from evidenceguard.data import Document, Query
from evidenceguard.evidence import EvidenceAssessor, MetadataStanceClassifier

query = Query(query_id="q1", text="Does the treatment improve outcomes?")
documents = (
    Document(
        doc_id="d1",
        title="Trial",
        text="The treatment improved outcomes in the trial.",
        metadata={"evidence_stance": "supports", "evidence_strength": "0.9"},
    ),
    Document(
        doc_id="d2",
        title="Replication",
        text="A later trial found no improvement.",
        metadata={"evidence_stance": "contradicts", "evidence_strength": "0.8"},
    ),
)

assessment = EvidenceAssessor(MetadataStanceClassifier()).assess(query, documents)
print(assessment.decision.value)
print(assessment.supporting_doc_ids, assessment.contradicting_doc_ids)
```

The example labels are supplied inputs, not model predictions.

## Decision policy

`EvidenceAssessor` applies this deterministic policy:

| Supporting documents | Contradicting documents | Decision |
| --- | --- | --- |
| at least one | none | `ANSWER_SUPPORTED` |
| at least one | at least one | `CONFLICTING_EVIDENCE` |
| none | any | `INSUFFICIENT_EVIDENCE` |
| none | none | `INSUFFICIENT_EVIDENCE` |

Contradiction without support is treated as insufficient because this component
does not synthesize a refuting answer. Neutral and unknown documents do not
count as evidence. An empty document list is valid and yields
`INSUFFICIENT_EVIDENCE`.

The decision is a rule-based evidence status, not a calibrated abstention
decision. In particular:

- `ANSWER_SUPPORTED` does not establish answer correctness;
- support labels may come from an unvalidated external classifier;
- no temporal reasoning is performed, so outdated documents are not detected
  unless the classifier assigns a non-supporting stance;
- no minimum evidence count or strength threshold is applied.

## Output

`EvidenceAssessment` preserves the input document order and exposes:

- `decision`;
- `supporting_doc_ids`, `contradicting_doc_ids`, `neutral_doc_ids`, and
  `unknown_doc_ids`;
- `support_strength` and `contradiction_strength`, each the maximum strength in
  that stance group or `0.0` when absent;
- per-document stance and strength through `document_evidence`;
- `to_dict()` for JSON-compatible reporting.

Duplicate document IDs, duplicate classifier results, missing classifier
results, unknown document IDs, malformed fields, and non-document inputs are
rejected.

## Relationship to benchmark labels

`EvidenceExample.expected_outcome` is a gold scenario label. It is not an input
to evidence assessment and must not be read by a classifier. Assessment quality
will be evaluated against such labels in a later phase; the current module only
constructs predictions.

Retrieval relevance is also distinct from stance. A document may be relevant to
a query while remaining neutral, contradictory, or insufficient.

## Limitations

- No scientific NLI or fact-verification model is implemented.
- Metadata labels are only as reliable as their source.
- The synthetic transformations remain caller-asserted scenarios.
- Strength values are uncalibrated and do not affect the decision.
- Outdatedness and temporal validity are not assessed automatically.
- The decision policy is not fitted or evaluated on development data.
- No evidence-assessment benchmark results are reported.
