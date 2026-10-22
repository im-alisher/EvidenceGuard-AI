# Mini BEIR fixture

This is a **synthetic** dataset created for offline tests. It contains fictional
scientific claims, documents, and relevance judgments. It is not derived from
SciFact and must not be presented as a research result.

Layout:

- `corpus.jsonl`: five fictional documents.
- `queries.jsonl`: three fictional queries, including one with no relevant
  documents.
- `qrels/test.tsv`: retrieval relevance judgments for the test split.

The fixture intentionally includes a supportive document (`doc-a`), a
contradictory document (`doc-b`), an irrelevant document (`doc-c`), an
old-report document (`doc-d`), and a directly relevant property document
(`doc-e`). Tests may use these as caller-supplied transformation inputs.
