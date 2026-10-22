# Contributing to EvidenceGuard AI

## Scope

EvidenceGuard AI is a research benchmark. Contributions should preserve
reproducibility, honest reporting, and a clear separation between measured
results and hypotheses.

## Local setup

Dependency installation is manual. From the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Do not commit virtual environments, secrets, downloaded datasets, or model
artifacts.

## Checks before committing

```powershell
python -m pytest
python -m ruff check src tests dashboard
python -m ruff format --check src tests dashboard
python -m mypy src
```

Run only the checks that are available in your environment, and report anything
that could not be run. Never bypass a failing test or suppress an error merely to
make a change appear complete.

## Data and model policy

- Do not commit SciFact or other third-party datasets.
- Do not commit model weights or caches.
- Use documented download instructions and verify checksums.
- Keep synthetic test fixtures small and clearly labeled as synthetic.
- Preserve the distinction between original and transformed examples.
- Do not add paid APIs or services.
- Do not add answer generation unless it is optional, locally runnable, and
  explicitly separated from evidence verification.

## Commit conventions

Use focused Conventional Commits, for example:

```text
feat(data): implement reproducible benchmark datasets
feat(retrieval): add BM25 and dense retrieval baselines
test: expand evaluation coverage
docs: document reproducibility and research findings
```

Do not bundle unrelated changes. Do not amend, squash, reset, or force-push
existing history. Do not add a remote or publish results without explicit
approval.

## Safety and environment restrictions

- Never create or run database migrations or initialize persistent databases.
- Never add secrets or `.env` files.
- Never run destructive commands or remove user data.
- Never install global packages or modify global configuration.
- Never start long or expensive experiments without approval.
