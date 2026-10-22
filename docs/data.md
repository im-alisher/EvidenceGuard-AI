# Dataset interface and controlled evidence scenarios

## Purpose

The data layer provides validated retrieval records and reproducible, controlled
evidence scenarios. It does not generate answers, retrieve documents, or
establish factual truth.

## Core records

| Record | Purpose |
| --- | --- |
| `Document` | A retrievable item with stable ID, title, text, optional `published_at` date, and string metadata. |
| `Query` | A stable query ID, query text, and optional title. |
| `RelevanceJudgment` | A non-negative retrieval relevance score for one query-document pair. |
| `RetrievalDataset` | A validated corpus, query set, qrels, name, split, and optional source description. |
| `EvidenceExample` | A per-query controlled scenario with a document pool, relevant document IDs, expected decision, condition, and provenance. |

`RetrievalDataset` rejects duplicate document IDs, duplicate query IDs, duplicate
qrels, and qrels that reference unknown queries or documents.

## BEIR and SciFact loading

`load_beir_dataset` expects:

```text
<dataset>/
  corpus.jsonl
  queries.jsonl
  qrels/<split>.tsv
```

- Corpus records require `_id`, `title`, and `text`.
- Query records require `_id` and `text`; `title` is optional.
- Qrels require query, document, and score columns. BEIR-style names
  (`query-id`, `corpus-id`, `score`) are supported.
- Only positive qrels are retained as retrieval relevance judgments.
- Unknown fields are ignored; missing required fields, malformed JSON, invalid
  scores, duplicate qrels, and unknown references are errors.

`load_scifact` is a thin wrapper that applies the dataset name and documented
SciFact checksum constants. It does not download data.

## Label semantics

`relevant_document_ids` are **retrieval relevance labels**. They do not prove that
evidence is sufficient, current, or non-contradictory.

`expected_outcome` is the decision label for the controlled scenario:

- `ANSWER_SUPPORTED`
- `INSUFFICIENT_EVIDENCE`
- `CONFLICTING_EVIDENCE`

`build_evidence_example` requires the caller to supply `expected_outcome`. This is
intentional: SciFact qrels alone do not establish evidence sufficiency or the
correct abstention decision. Benchmark labels must remain explicit.

## Controlled conditions

| Condition | Transformation | Label behavior |
| --- | --- | --- |
| `SUFFICIENT` | No perturbation. | Caller supplies expected outcome. |
| `MISSING` | `remove_relevant_evidence` removes all retrieval-relevant documents. | `INSUFFICIENT_EVIDENCE`; relevant IDs become empty. |
| `IRRELEVANT` | `add_irrelevant_documents` appends a deterministic sample from a caller-supplied distractor pool. | Expected outcome is unchanged. |
| `CONTRADICTORY` | `add_contradictory_evidence` appends caller-supplied contradictory documents. | `CONFLICTING_EVIDENCE`; original relevant IDs are retained. |
| `OUTDATED` | `replace_with_outdated_evidence` removes current relevant documents and adds explicitly dated replacements. | `INSUFFICIENT_EVIDENCE`; replacements remain retrieval-relevant. |
| `UNANSWERABLE` | No dedicated transformation; an original query with no supporting evidence is labeled explicitly. | `INSUFFICIENT_EVIDENCE`. |

### Transformation assumptions

- The caller asserts that distractor documents are irrelevant. The code does not
  infer irrelevance from text.
- The caller asserts that contradiction documents contradict the supporting
  evidence. No natural-language inference or NLI model is used.
- Outdated replacements must have explicit timestamps. The caller also establishes
  that they are temporally invalid for the question; a date alone does not prove
  real-world outdatedness.
- Outdated replacements remain retrieval-relevant so retrieval and evidence
  sufficiency can be evaluated separately.
- Added documents are appended without modifying the original example.
- Transformations are synthetic. They are controlled test conditions, not claims
  about real scientific evidence.

## Provenance

Original examples have `provenance=ORIGINAL`. Transformations produce
`provenance=TRANSFORMED` with:

- a stable `source_example_id`;
- the transformation name;
- the scenario condition;
- optional seed;
- metadata recording affected document IDs.

Example IDs are derived deterministically from the source ID, transformation,
condition, seed, and ordered document IDs. This makes repeated transformations
stable across processes and machines.

## Deterministic sampling

`add_irrelevant_documents` validates the random seed, sorts candidate documents
by ID, uses a local `random.Random` instance, and sorts selected documents before
inserting them. It does not mutate global random state.

## Query-level splits

`assign_query_splits` creates deterministic, disjoint train/validation/test query
groups. Queries are shuffled with a seed-derived local generator. Shared corpus
documents remain available across splits, which is standard for retrieval
evaluation; the guarantee is that no query appears in more than one split.

## Known limitations

- SciFact qrels are retrieval labels, not answer-sufficiency annotations.
- The public BEIR distribution does not itself provide the full original claim
  label set in this loader. Any answerability labels must be documented and
  validated separately before use.
- Controlled transformations do not replace expert annotation of real evidence.
- Irrelevance, contradiction, and outdatedness are caller-asserted in synthetic
  scenarios.
- The current interface does not implement temporal reasoning, NLI, or automatic
  corruption detection.
- The synthetic fixture is fictional and must never be presented as an empirical
  result.
- No train/test leakage guarantee is made for corpus documents; only query IDs
  are split.
