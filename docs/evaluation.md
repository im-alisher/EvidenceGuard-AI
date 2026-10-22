# Retrieval evaluation

## Purpose and scope

The evaluation layer measures retrieval rankings against dataset relevance
judgments. It does not generate answers, assess evidence sufficiency, calibrate
uncertainty, or run experiments. Reported retrieval quality is therefore distinct
from answer correctness and abstention quality.

All metrics are computed locally with the Python standard library. No external
evaluation service is used.

## Evaluating a retriever

```python
from evidenceguard.data import load_beir_dataset
from evidenceguard.evaluation import evaluate_retriever
from evidenceguard.retrieval import BM25Retriever

dataset = load_beir_dataset("datasets/scifact", split="test")
retriever = BM25Retriever(dataset.documents)
evaluation = evaluate_retriever(retriever, dataset, cutoffs=(1, 5, 10))

for aggregate in evaluation.aggregates:
    print(aggregate.cutoff, aggregate.mean_ndcg, aggregate.mean_recall)
```

`evaluate_retriever` retrieves once per evaluable query at a depth equal to the
largest requested cutoff. An explicit `top_k` may request a deeper ranking but
must be at least as large as every cutoff.

Each result must:

- be a `RetrievalResult`;
- use unique document IDs;
- reference documents in the evaluated dataset corpus;
- report ranks matching returned list order.

Cutoffs must be unique positive integers. The returned evaluation contains
per-query metrics and macro-averaged metrics for each cutoff.

## Queries without relevance judgments

A query with no positive relevance grade is **skipped** and listed in
`skipped_query_ids`. It is not counted as a retrieval success or failure.

This matters for SciFact questions with no supporting qrels. Those queries are
relevant to evidence sufficiency and abstention, but they do not provide a
positive target for standard retrieval metrics. Treating them as ordinary missed
retrievals would distort Recall, precision, MRR, MAP, and nDCG.

## Relevance semantics

`RelevanceJudgment.relevance` is a non-negative integer grade.

- Binary metrics treat every grade `>= 1` as relevant.
- Precision, Recall, MRR, Average Precision, and Hit use binary relevance.
- nDCG uses graded gains.
- Unknown retrieved documents are nonrelevant at metric level. The dataset
  evaluator rejects documents outside the evaluated corpus before computing
  metrics.

## Metrics

For one query and cutoff `k`, let `R` be the total number of positively relevant
documents in the full judgment set, and let `H` be the number of relevant
documents retrieved in the top `k`.

### Recall@k

```text
Recall@k = H / R
```

Returns `0` when `R = 0`. Evaluated queries normally have `R >= 1`; queries with
no positive labels are skipped by `evaluate_retriever`.

### Precision@k

```text
Precision@k = H / k
```

The denominator is always the requested cutoff, even when the retriever returns
fewer than `k` documents. This makes precision comparable across lists of
different available lengths.

### Hit@k / hit rate

A query is a hit when at least one relevant document appears in the top `k`.
Aggregate `hit_rate` is the fraction of evaluable queries that are hits.

### Reciprocal rank and MRR@k

```text
RR@k = 1 / rank_of_first_relevant_result    if a relevant result exists in top k
       0                                    otherwise
```

Aggregate `mean_reciprocal_rank` is the macro-average of per-query RR.

### Average Precision and MAP@k

At each relevant result in the top `k`, compute the precision at that rank. Sum
those precision values and divide by `min(R, k)`:

```text
AP@k = sum(precision_at_relevant_rank) / min(R, k)
```

Aggregate `mean_average_precision` is the macro-average of per-query AP, commonly
called MAP. Using `min(R, k)` makes the cutoff explicit. This repository does not
claim comparability with other tools unless their AP cutoff convention matches.

### nDCG@k

Graded gain is:

```text
gain(grade) = 2 ** grade - 1
```

Discounted cumulative gain is:

```text
DCG@k = sum(gain(grade_at_rank) / log2(rank + 1))
```

The ideal DCG sorts all positive judgment grades in descending order and applies
the same cutoff. nDCG is `DCG@k / IDCG@k`, clamped to a maximum of `1`. Queries
with no positive grades receive `0`.

Grades of `0` contribute no gain. Grades above `1` are used only by nDCG; binary
metrics still treat them as relevant.

## Aggregation

Aggregates are macro-averages over evaluable queries, not weighted by the number
of relevant documents per query. Every aggregate reports its cutoff and query
count.

`RetrievalEvaluation.to_dict()` returns a JSON-serializable structure containing
dataset identity, split, retriever name, retrieval depth, cutoffs, skipped query
IDs, per-query metrics, and aggregate metrics.

## Determinism

For fixed rankings and judgments, metric values are deterministic. The evaluator
uses retriever result order as the ranking and rejects result ranks that do not
match that order. Corpus and retriever-level determinism are covered in
`docs/retrieval.md`.

## Limitations

- These are retrieval metrics, not evidence-support or answer-correctness metrics.
- Relevance grades are taken as given; the evaluator does not infer relevance.
- No confidence intervals, significance tests, or multiple-comparison correction
  are implemented.
- No experiment runner, result table, plot, or dashboard is implemented yet.
- Skipping unjudged or unanswerable queries is appropriate for standard retrieval
  metrics but does not evaluate abstention.
- Cross-dataset comparisons require checking corpus, qrels, tokenization, model,
  cutoff, and AP conventions.
- The synthetic fixture is for software verification, not empirical claims.
