# EvidenceGuard AI

**Uncertainty-aware evidence evaluation and abstention for retrieval-augmented generation.**

> **Project status: Phase 2 (dataset layer).**
> No benchmark results, retrieval baselines, abstention models, or dashboard
> results exist yet. Claims in this repository will distinguish measured findings
> from planned work.

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

## Current implementation (Phases 1-2)

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

Retrieval, evidence assessment, experiments, results, and the dashboard are
intentionally not implemented yet. See `docs/data.md` for schema details, label
semantics, transformation assumptions, and limitations.

## Planned project layout

```text
EvidenceGuard-AI/
  configs/                  experiment configurations
  datasets/                 local dataset location (raw data is not committed)
  docs/                     setup and research documentation
  reports/                  generated experiment outputs and report material
  scripts/                  manual download and reproduction utilities
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

Modules are introduced only when their implementation phase begins.

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

Do not install optional extras until the corresponding phase is implemented and
reviewed.

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
python -m ruff check src tests
python -m ruff format --check src tests
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

## Dataset and licensing

The planned public benchmark is **SciFact** (Wadden et al., EMNLP 2020):

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
