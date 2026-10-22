# Reproducible experiments

The experiment layer loads a local BEIR-format dataset, runs the configured BM25
baseline, evaluates retrieval metrics, and writes a versioned JSON artifact.
It does not download datasets or models and does not fabricate results.

## Configuration

The base configuration remains strict. An optional `experiment` mapping selects
a retrieval run:

```yaml
version: 1
name: retrieval-smoke
seed: 13
paths:
  datasets: datasets
  outputs: reports/runs
logging:
  level: INFO
experiment:
  dataset: tests/fixtures/mini_beir
  retriever: bm25
  cutoffs: [1, 3]
  top_k: 3
```

Rules:

- `experiment.dataset` is resolved relative to the configuration base directory
  and must point to a local BEIR-format dataset.
- `experiment.retriever` currently accepts only `bm25`.
- `cutoffs` must be unique positive integers and are sorted.
- `top_k` must be positive and at least the largest cutoff.
- Unknown keys and unsupported versions are rejected.

The optional section is absent from `configs/smoke.yaml`; it is present in
`configs/retrieval-smoke.yaml`, which uses the synthetic offline fixture.

## Running

From an installed environment:

```powershell
python -m evidenceguard.experiments configs/retrieval-smoke.yaml
```

The artifact is written to:

```text
<paths.outputs>/<config.name>.json
```

Existing artifacts are rejected unless `--overwrite` is supplied. Relative paths
resolve against the current working directory by default; use `--base-dir` to
choose another base.

For a manual SciFact run, point `experiment.dataset` at a locally acquired
BEIR-format directory. Dataset acquisition remains manual and subject to the
licenses documented in `datasets/README.md`.

## Artifact schema

Artifacts use `schema_version: 1`:

```json
{
  "schema_version": 1,
  "experiment": {"name": "retrieval-smoke", "seed": 13},
  "dataset": {"name": "mini_beir", "split": "test"},
  "retriever_name": "bm25",
  "retrieval": {"cutoffs": [1, 3], "aggregates": [], "per_query": []},
  "decisions": null,
  "calibration": null
}
```

`retrieval` contains per-query and aggregate metrics from
`evaluate_retriever`. Optional `decisions` and `calibration` sections allow
programmatic runs to attach Phase 6 outputs. A result must contain at least one
measured section.

Artifacts are written with stable JSON key ordering and a trailing newline.
Writes use a temporary file followed by an atomic replacement.

## Programmatic use

```python
from evidenceguard.config import load_experiment_config
from evidenceguard.experiments import run_configured_experiment

config = load_experiment_config("configs/retrieval-smoke.yaml")
output_path = run_configured_experiment(config)
```

To attach decision or calibration evaluations, construct `ExperimentResult`
directly and write it with `write_experiment_result`.

## Scope and limitations

- Only BM25 is configurable in this phase.
- Dense and hybrid retrievers remain available as Python components but require
  caller-supplied encoders.
- The runner does not generate evidence examples, fit thresholds, or assess
  scientific claims automatically.
- The synthetic fixture is not a benchmark result.
- No SciFact run is included in the repository.
- `seed` is recorded for provenance; BM25 execution itself is deterministic and
  does not use random sampling.
- Reporting and dashboard rendering are separate phases.
