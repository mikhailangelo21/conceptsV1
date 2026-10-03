# Inspect and reproduce the V3 evidence

This folder mirrors the relevant project paths. Run commands from this folder. It includes the generated data bank, code, saved results and interpretations. It does not include pretrained weights, the external activation cache or a Python virtual environment.

## Immediate inspection, with no installation

Open [reports/quality_suite_v3_findings/report.html](reports/quality_suite_v3_findings/report.html) in a browser for the complete illustrated report. Read [reports/quality_suite_v3_findings/findings.md](reports/quality_suite_v3_findings/findings.md) for the editable technical narrative. The raw primary outputs are in [runs/quality_suite_v3_laptop-88fad55b46b0](runs/quality_suite_v3_laptop-88fad55b46b0/). Check [FILE_MANIFEST.json](FILE_MANIFEST.json) to verify copied file hashes and original locations.

The run's `state.json` must say `complete` with no failures; `identity.json` pins the model revision and source hash. `source_snapshot.json` captures the precise code used for the scientific run. The visible Python files show the implementation at handoff time and may contain later reporting or CPU-resume improvements. When these differ, use the run snapshot to interpret what actually executed.

## Regenerate derived interpretation without a model

Create an environment with Python 3.11 or newer and the supplied [dependency lock](requirements-lock.txt), then install this project. On the machine that produced the report, a working `.venv` already exists in the original project; this copy intentionally has no virtual environment. The commands are:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e .
python populate_quality_suite.py --verify
pytest -m 'not integration' -q
python analysis/v3_findings.py
python analysis/v3_report.py
```

The last two Python commands recompute the derived CSVs, checks, `datasets.json`, `chart_map.json` and `artifact.json` from the copied fixtures and saved run results. They do not call the language model. They will rewrite derived files in this handoff folder; save another copy first if byte-for-byte preservation of this snapshot matters. The existing HTML is self-contained. Rebuilding that HTML uses the Data Analytics portable-report builder used at creation time, which is an external plugin and is not copied into this research package; the canonical `artifact.json` retains its full data and narrative.

The analytical validation record reports recomputation of 564 scalar prediction partitions with maximum MAE round-trip error `2.22e−16`, reconciliation of all 1,888 scored questions, and 48 matched underlying natural/A-B comparisons. The original offline software suite passed before the laptop run; the copied tests let an agent repeat it in a compatible environment.

## A fresh model run is a different operation

The original run used 8,254 forward passes, a 60-minute cumulative model-forward budget and local Hugging Face Qwen weights. Fresh extraction needs an MPS-capable machine, the exact model revision, enough disk space, network or cached weights, and time. Use [docs/V3_INTEGRATION.md](docs/V3_INTEGRATION.md) for `qd suite plan/run/report` semantics. The config requires validation of suite hashes and exact profile IDs. Do not point a new run at the copied completed run directory; use a new run location.

The copied `activation_index.json`, `activation_files.json` and `activation_cache.json` preserve the original cache map and identity, but some paths refer to the original machine's external cache and are not portable shard data. The raw readout vectors are not required to inspect or rederive this handoff's saved-result interpretation. To refit probes from hidden states without rerunning the model, retrieve the matching external activation shard cache from the original machine, then verify its keys against the included identity and indexes.
