# Retrieval baselines

## Purpose and scope

The retrieval layer converts a query into a deterministic ranked list of corpus
documents. It is independent of answer generation, evidence assessment, and
abstention. Retrieval scores are ranking signals, not calibrated probabilities
that an answer is correct.

Retrieval metrics, experiment orchestration, and result reporting are not
implemented in this phase.

## Common interface

Every retriever implements:

```python
results = retriever.retrieve(query, top_k=10)
```

`query` may be a `Query` object or a non-empty string. `top_k` must be a positive
integer. If fewer than `top_k` documents are available, all available documents
are returned.

Each `RetrievalResult` contains:

| Field | Meaning |
| --- | --- |
| `document` | The retrieved `Document` object. |
| `doc_id` | Convenience property for `document.doc_id`. |
| `score` | Retriever-specific ranking score. |
| `rank` | One-based rank in the returned list. |

Results are sorted by decreasing score. Ties are broken by ascending `doc_id`,
which makes ranking deterministic for a fixed corpus and retriever. Component
retrievers must return unique document IDs for a single query.

Corpora are validated at retriever construction time. Corpora must be non-empty
and have unique document IDs.

## BM25 baseline

`BM25Retriever` uses Okapi BM25 from `rank_bm25` over a deterministic lexical
representation.

- Text is composed as `title + "\n" + text` when a title is present.
- Text is case-folded and tokenized with the ASCII alphanumeric pattern
  `[a-z0-9]+`.
- Punctuation and hyphens separate tokens; there is no stemming, stop-word
  removal, synonym expansion, or field weighting.
- Default parameters are `k1=1.5`, `b=0.75`, and `epsilon=0.25`.
- `k1`, `b`, and `epsilon` are validated at construction.

BM25 scores are dataset- and parameter-dependent and may be negative or zero.
They must not be interpreted as probabilities or compared directly with dense
similarities.

## Dense baseline

`DenseRetriever` performs exact similarity search over embeddings supplied by an
`EmbeddingEncoder`:

```python
class EmbeddingEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...
```

The returned array must be two-dimensional with one row per input text, finite
values, and at least one column. A single encoder must be used for both documents
and queries, and query/document dimensions must match.

By default, document and query rows are L2-normalized and scored by inner
product, which is equivalent to cosine similarity. Zero vectors remain zero and
receive zero similarity. Set `normalize=False` to use raw inner product instead.
The current index is an in-memory matrix and scores every document per query; it
does not use an approximate nearest-neighbor library.

`SentenceTransformerEncoder` is an optional adapter that imports
`sentence-transformers` lazily when instantiated. Constructing it can download a
model from Hugging Face unless the model is already cached. This repository does
not download models automatically; installation and model acquisition require
explicit user approval. The test suite uses deterministic stub encoders and does
not load a pretrained model.

## Hybrid reciprocal rank fusion

`ReciprocalRankFusionRetriever` combines component rankings with reciprocal rank
fusion:

```text
score(d) = sum over retrievers of 1 / (rrf_k + rank_retriever(d))
```

- `rrf_k` defaults to `60` and must be a non-negative integer.
- Components receive equal weight.
- `candidate_k` controls how many results are requested from each component.
  The actual depth is `max(top_k, candidate_k)`; when unset, it equals `top_k`.
- A document absent from a component contributes no score for that component.
- Fused ties are broken by ascending `doc_id`.
- Duplicate IDs within one component are rejected.
- If components return different content for the same document ID, fusion fails
  rather than silently choosing one version.
- Components are expected to index the same corpus. Fusion validates overlapping
  document content but cannot prove that component corpora are identical.

RRF uses ranks rather than raw scores, avoiding direct mixing of BM25 and dense
score scales. The fused score is still only a ranking signal.

## Determinism and reproducibility

For fixed documents, query text, parameters, component order, and encoder
outputs, ranking is deterministic. Sources of variation outside this layer
include model versions, tokenizer versions, numerical kernels, and hardware.
Dense experiments should record the encoder model identifier and version.

## Usage

```python
from evidenceguard.data import load_beir_dataset
from evidenceguard.retrieval import BM25Retriever, ReciprocalRankFusionRetriever

dataset = load_beir_dataset("datasets/scifact", split="test")
bm25 = BM25Retriever(dataset.documents)
hybrid = ReciprocalRankFusionRetriever([bm25], rrf_k=60)

results = hybrid.retrieve(dataset.queries[0], top_k=20)
```

A dense component requires an encoder implementation and explicit model
acquisition. Infrastructure beyond exact NumPy search, retrieval metrics, and
end-to-end evaluation is intentionally deferred to later phases.

## Limitations

- BM25 uses a simple ASCII tokenizer and no scientific-text normalization.
- Dense retrieval is only as meaningful as the supplied encoder; the tests use
  synthetic vectors to verify mechanics, not semantic quality.
- Exact dense search is linear in corpus size.
- Scores are not calibrated and are not comparable across retriever types.
- Hybrid fusion uses equal component weights; learned weighting is not
  implemented.
- Retrieval relevance is distinct from answer support and evidence sufficiency.
