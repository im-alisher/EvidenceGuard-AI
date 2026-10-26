# EvidenceGuard AI

**Uncertainty-aware evidence evaluation and abstention for retrieval-augmented generation.**

> **Project status: Phase 8 (reporting and dashboard).**
> Retrieval, evidence assessment, thresholded abstention, decision
> evaluation, reproducible retrieval runs, report exports, and an optional
> local dashboard are implemented. Baseline SciFact retrieval results (BM25, dense, and hybrid RRF) are included under reports/. No answer generation or verifier results exist yet.
> Claims in this repository distinguish measured findings from planned work.

## Research question

Can uncertainty-aware evidence evaluation reduce unsupported answers in RAG
systems while preserving useful answer coverage?

The project studies whether a retrieval-augmented system can recognize
insufficient, irrelevant, outdated, or contradictory evidence and abstain when
the available evidence does not support a reliable answer.

## What this project is

EvidenceGuard AI is a reproducible research benchmark, not a chatbot wrapper. The
planned system separates:

1. Data loading and controlled evidence scenarios.
2. Independent retrieval baselines (BM25, optional dense, and hybrid fusion).
3. Evidence assessment and explicit abstention decisions.
4. Calibration, selective-risk evaluation, and ablations.
5. A results-driven research dashboard and paper-style report.

The core benchmark is designed to run without paid APIs, and offline tests do
not require dataset or model downloads.

## Current implementation (Phases 1-8)

Phase 1 provides the project foundation:

- strict YAML experiment configuration with validation;
- consistent package logging;
- explicit decision outcomes:
  - `ANSWER_SUPPORTED`
  - `INSUFFICIENT_EVIDENCE`
  - `CONFLICTING_EVIDENCE`;
- deterministic seed derivation utilities.

Phase 2 provides the data layer:

- validated document, query, relevance, dataset, and evidence-scenario models;
- a strict BEIR loader and a SciFact wrapper;
- deterministic query-level train/validation/test splits;
- controlled transformations for missing, irrelevant, contradictory, and
  outdated evidence;
- stable provenance and example IDs for original versus transformed scenarios;
- a small fictional fixture for offline tests.

Phase 3 provides retrieval baselines:

- an Okapi BM25 baseline with deterministic tokenization;
- exact dense retrieval over caller-supplied embedding encoders;
- an optional, lazily imported sentence-transformers adapter;
- hybrid reciprocal rank fusion with configurable candidate depth;
- deterministic doc-ID tie-breaking and strict component-result validation.

Phase 4 provides retrieval evaluation:

- Recall@k, Precision@k, Hit@k, MRR@k, MAP@k, and graded nDCG@k;
- per-query and macro-averaged metrics at configurable cutoffs;
- explicit skipping of queries without positive relevance judgments;
- validation of result order, duplicate documents, and corpus membership;
- JSON-serializable evaluation output for later experiment reporting.

Phase 5 provides evidence assessment:

- a validated supporting, contradicting, neutral, and unknown stance taxonomy;
- a batch classifier interface plus an explicit metadata-label adapter;
- deterministic support/conflict/insufficiency decisions;
- strict validation of classifier output and document membership;
- JSON-serializable evidence assessments for later evaluation.

The phase does not perform automatic natural-language inference. Stance labels
must be supplied by an annotator or an external classifier, and decision
strengths are not calibrated.

Phase 6 provides abstention and decision evaluation:

- thresholded abstention over uncalibrated answer confidence;
- decision records with coverage, selective accuracy, and selective risk;
- risk-targeted threshold selection and risk-coverage curves for development
  data;
- Brier score and expected calibration error for surfaced answers;
- JSON-serializable decision and calibration outputs.

Threshold selection requires development data; no threshold has been fitted on
SciFact, and confidence values are not claimed to be calibrated probabilities.

Phase 7 provides reproducible experiment orchestration:

- an optional strict `experiment` configuration section for local BEIR runs;
- BM25 experiment execution over caller-provided local datasets;
- versioned JSON artifacts containing retrieval results and optional decision
  and calibration outputs;
- deterministic artifact writing with explicit overwrite protection;
- a `python -m evidenceguard.experiments` command-line entry point.

Phase 8 provides reporting and an optional local dashboard:

- strict loading and validation of versioned run artifacts;
- deterministic Markdown reports with measured and unmeasured sections;
- CSV export of aggregate retrieval metrics;
- an optional Streamlit viewer for uploaded JSON artifacts;
- no fabricated sections or inferred results.

See `docs/data.md` for dataset semantics, `docs/retrieval.md` for retriever
interfaces, `docs/evaluation.md` for retrieval metrics, `docs/evidence.md` for
evidence-assessment semantics, `docs/uncertainty.md` for abstention and
calibration limitations, `docs/experiments.md` for run configuration and
artifact schemas, and `docs/reporting.md` for validation, exports, and the
dashboard.

## Project layout

```text
EvidenceGuard-AI/
  configs/                  experiment configurations
  datasets/                 local dataset location (raw data is not committed)
  docs/                     setup and research documentation
  reports/                  generated experiment outputs and report material
  src/evidenceguard/
    data/                   dataset schemas, adapters, and transformations
    retrieval/              BM25, dense, and hybrid retrieval
    evidence/               support, conflict, and sufficiency signals
    uncertainty/            calibrated abstention methods
    evaluation/             retrieval, decision, and calibration metrics
    experiments/            configuration-driven experiment runner
    reporting/              machine-readable exports and research plots
  tests/                    offline unit tests and synthetic fixtures
  dashboard/                optional Streamlit research dashboard
  pyproject.toml
  README.md
  CONTRIBUTING.md
  LICENSE
```

## Requirements

- Python 3.11 or newer.
- No paid API or cloud service.
- Core offline path: Python standard library, NumPy, rank-bm25, PyYAML, pytest.
- Optional extras:
  - `dense`: sentence-transformers for dense retrieval experiments;
  - `analysis`: pandas, scikit-learn, and Matplotlib;
  - `dashboard`: Streamlit;
  - `dev`: pytest, coverage, Ruff, and mypy.

Dependency installation is intentionally manual. This repository does not create
virtual environments, install packages, or download datasets or models
automatically.

## Manual installation

Run these commands yourself from the repository root. The examples use Python
3.13 and PowerShell on Windows; adjust the interpreter if needed.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For the core package only:

```powershell
python -m pip install -e .
```

Optional extras can be combined:

```powershell
python -m pip install -e ".[dev,dense,analysis,dashboard]"
```

Install only the optional extras needed for your work. The reporting and export
functions do not require the dashboard extra.

## Running tests

From the repository root, with the environment already configured:

```powershell
python -m pytest
```

For coverage:

```powershell
python -m pytest --cov=evidenceguard --cov-report=term-missing
```

Static checks, once their tools are installed:

```powershell
python -m ruff check src tests dashboard
python -m ruff format --check src tests dashboard
python -m mypy src
```

## Configuration

The project includes a strict smoke configuration at `configs/smoke.yaml`:

```yaml
version: 1
name: smoke
seed: 13
paths:
  datasets: datasets
  outputs: reports/runs
logging:
  level: INFO
```

Relative paths are resolved against the working directory by default, or
against an explicit `base_dir` passed to `load_experiment_config`. Unknown keys
and unsupported versions are rejected rather than silently ignored.

`configs/retrieval-smoke.yaml` adds an optional retrieval experiment over the
synthetic fixture. Run it after installation with:

```powershell
python -m evidenceguard.experiments configs/retrieval-smoke.yaml
```

The command writes a JSON artifact under `reports/runs/`. It does not download
data or models.

## Reporting

Load an artifact, render Markdown, or export aggregate retrieval metrics:

```python
from evidenceguard.reporting import (
    load_run_artifact,
    write_markdown_report,
    write_retrieval_csv,
)

artifact = load_run_artifact("reports/runs/scifact-bm25.json")
write_markdown_report(artifact, "reports/scifact-bm25.md", overwrite=True)
write_retrieval_csv(artifact, "reports/scifact-bm25.csv", overwrite=True)
```

The optional dashboard reads uploaded artifacts only:

```powershell
python -m pip install -e ".[dashboard]"
streamlit run dashboard/app.py
```

## Baseline results

Macro-averaged retrieval metrics on the SciFact BEIR test split (300 evaluated
queries), rounded to three decimals from the committed reports:

| Retriever | Recall@10 | nDCG@10 | MRR@10 | Hit@10 | Full reports |
| --- | ---: | ---: | ---: | ---: | --- |
| BM25 | 0.776 | 0.652 | 0.618 | 0.797 | [Markdown](reports/scifact-bm25.md), [CSV](reports/scifact-bm25.csv), [JSON](reports/runs/scifact-bm25.json) |
| Dense (all-MiniLM-L6-v2) | 0.783 | 0.645 | 0.605 | 0.793 | [Markdown](reports/scifact-dense.md), [CSV](reports/scifact-dense.csv), [JSON](reports/runs/scifact-dense.json) |
| Hybrid RRF | 0.839 | 0.697 | 0.654 | 0.853 | [Markdown](reports/scifact-hybrid-rrf.md), [CSV](reports/scifact-hybrid-rrf.csv), [JSON](reports/runs/scifact-hybrid-rrf.json) |

Dense uses `sentence-transformers/all-MiniLM-L6-v2`; hybrid combines BM25 and
dense with RRF (`rrf_k=60`, candidate depth 10). All runs use cutoff 10 and
exclude the same 809 queries without positive test judgments. Retrieval relevance
does not establish answer correctness. No answer generation or verifier results
have been measured.

## Dataset and licensing

The public retrieval benchmark is **SciFact** (Wadden et al., EMNLP 2020):

- SciFact original dataset license: **CC BY-NC 2.0**.
- The BEIR redistribution is listed as **CC BY-SA 4.0**.
- Dataset terms are separate from the project's source-code license.
- No SciFact data is bundled in this repository.
- `datasets/README.md` contains manual acquisition instructions, the BEIR archive
  checksum, and expected layout. Nothing is downloaded automatically.
- The loader validates local files but does not fetch them.
- Offline tests use a small synthetic fixture and do not require the dataset.
- Users are responsible for confirming that their intended use complies with the
  applicable dataset license terms.

See `docs/data.md` for label semantics and controlled-scenario limitations.

Citation:

```bibtex
@inproceedings{wadden-etal-2020-fact,
  title = "Fact or Fiction: Verifying Scientific Claims",
  author = "Wadden, David and Lin, Shanchuan and Lo, Kyle and Wang, Lucy Lu and van Zuylen, Madeleine and Cohan, Arman and Hajishirzi, Hannaneh",
  booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
  month = nov,
  year = "2020",
  address = "Online",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2020.emnlp-main.609",
  doi = "10.18653/v1/2020.emnlp-main.609",
  pages = "7534--7550",
}
```

## Research integrity

- No benchmark result will be fabricated or hardcoded.
- Retrieval similarity is not treated as a calibrated probability of correctness.
- Baselines and experimental methods will be evaluated under identical
  conditions.
- Thresholds and calibrators will be fitted on development data, not test data.
- Unrun experiments and incomplete report sections will be marked explicitly.
- Statistical claims will require appropriate uncertainty estimates and analysis.

## License

The project source code is licensed under the **Apache License 2.0**. See
`LICENSE`. Dataset licenses and attribution remain separate and are not changed
by the code license.

## Contributing

See `CONTRIBUTING.md` for setup, testing, data-handling, and commit conventions.
