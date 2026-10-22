# Run dashboard

`app.py` is an optional local Streamlit viewer for versioned JSON artifacts. It
reads an uploaded artifact, validates its schema, displays measured retrieval,
decision, and calibration sections, and offers a generated Markdown download.

The dashboard does not query a database, download datasets, or run experiments.
No benchmark results are bundled.

## Run locally

Install the dashboard extra and the project, then start Streamlit:

```powershell
python -m pip install -e ".[dashboard]"
streamlit run dashboard/app.py
```

If Streamlit is not installed, the viewer cannot start; the reporting and export
functions remain usable from Python without it.

## Workflow

1. Generate an artifact:

   ```powershell
   python -m evidenceguard.experiments configs/retrieval-smoke.yaml
   ```

2. Upload `reports/runs/retrieval-smoke.json` in the dashboard.
3. Treat synthetic-fixture runs as workflow checks, not benchmark evidence.
