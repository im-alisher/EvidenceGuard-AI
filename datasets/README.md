# Dataset storage

No third-party dataset is committed to this repository. The `.gitignore`
excludes `datasets/raw/`, `datasets/downloads/`, and `datasets/scifact/`.

Offline tests use the small synthetic fixture in `tests/fixtures/mini_beir/` and
do not require any download.

## SciFact

The planned public benchmark is SciFact, distributed through BEIR:

- BEIR archive: `scifact.zip`
- URL: `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip`
- MD5: `5f7d1de60b170fc8027bb7898e2efca1`
- Original SciFact license: **CC BY-NC 2.0**
- BEIR redistribution license: **CC BY-SA 4.0**

Dataset terms are separate from this project's Apache-2.0 source-code license.
Confirm that your intended use complies with the applicable dataset terms before
downloading or using the data.

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

## Manual acquisition

These commands are documentation only. They are not executed by the project or
its tests. Run them yourself only after confirming the license terms.

PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path datasets\raw | Out-Null
Invoke-WebRequest `
  -Uri "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip" `
  -OutFile "datasets\raw\scifact.zip"
Get-FileHash -Algorithm MD5 "datasets\raw\scifact.zip"
Expand-Archive -Path "datasets\raw\scifact.zip" -DestinationPath "datasets\raw"
```

The MD5 output must be:

```text
5f7d1de60b170fc8027bb7898e2efca1
```

The project also provides a checksum helper after manual environment setup:

```powershell
python -c "from evidenceguard.data import verify_scifact_archive; verify_scifact_archive('datasets/raw/scifact.zip')"
```

Expected layout after extraction:

```text
datasets/raw/scifact/
  corpus.jsonl
  queries.jsonl
  qrels/
    test.tsv
```

Load it without downloading anything:

```powershell
python -c "from evidenceguard.data import load_scifact; ds = load_scifact('datasets/raw/scifact'); print(len(ds.documents), len(ds.queries), len(ds.qrels))"
```

The loader validates required files, JSONL fields, duplicate IDs, qrels columns,
scores, and references. See `docs/data.md` for schema and label limitations.
