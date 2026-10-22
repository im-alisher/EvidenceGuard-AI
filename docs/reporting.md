# Reporting and dashboard

Reporting consumes versioned JSON artifacts produced by
`evidenceguard.experiments`. The JSON artifact remains the source of truth.
Reports validate their inputs and never infer, backfill, or fabricate missing
sections.

## Artifact validation

`load_run_artifact(path)` reads JSON from disk. `parse_run_artifact(payload)`
validates an already-decoded mapping. Both return a `RunArtifact` with metadata
and optional retrieval, decision, and calibration sections.

Validation requires:

- `schema_version` exactly `1`;
- non-empty experiment name and dataset identity;
- a non-negative integer seed;
- at least one measured result section;
- complete retrieval aggregate, decision metric, and calibration-bin fields;
- numeric metric values within `[0, 1]` where applicable.

Malformed, unsupported, or empty artifacts raise `ReportError`. Optional
sections encoded as `null` are treated as unmeasured, not as zero-valued results.

## Python API

```python
from evidenceguard.reporting import (
    load_run_artifact,
    render_markdown,
    write_markdown_report,
    write_retrieval_csv,
)

artifact = load_run_artifact("reports/runs/retrieval-smoke.json")
markdown = render_markdown(artifact)
write_markdown_report(artifact, "reports/retrieval-smoke.md")
write_retrieval_csv(artifact, "reports/retrieval-smoke.csv")
```

Writers refuse to replace existing files unless `overwrite=True` is supplied.
Output suffixes are checked (`.md` and `.csv`), parent directories are created,
and writes use a temporary file followed by an atomic replacement.

## Markdown reports

Generated reports contain:

- run name, seed, dataset, split, and retriever;
- retrieval metrics by cutoff when measured;
- decision coverage, selective accuracy, selective risk, and abstention rate;
- Brier score, expected calibration error, and confidence bins;
- explicit "not measured" sections for absent outputs;
- warnings that retrieval relevance is not answer correctness and that
  calibration diagnostics do not establish scientific validity.

Reports are deterministic. They do not include plots, confidence intervals, or
claims absent from the artifact.

## Retrieval CSV

`write_retrieval_csv` exports one row per retrieval cutoff with:

```text
cutoff,query_count,mean_recall,mean_precision,mean_reciprocal_rank,
mean_average_precision,mean_ndcg,hit_rate
```

CSV export requires a measured retrieval section and uses six decimal places.

## Optional dashboard

`dashboard/app.py` is a local Streamlit viewer for uploaded JSON artifacts. It
shows run metadata, retrieval aggregates, decision metrics, calibration bins, a
Markdown report, and the validated JSON payload. It does not query a database,
run experiments, or download data.

```powershell
python -m pip install -e ".[dashboard]"
streamlit run dashboard/app.py
```

Streamlit is optional. Reporting and export functions work without it.

## Limitations

- No SciFact benchmark results or generated report artifacts are bundled.
- Synthetic-fixture runs verify workflow behavior, not scientific performance.
- Reports preserve whatever measured sections the artifact contains; they do not
  validate experimental design or statistical significance.
- Dashboard rendering is local and read-only.
