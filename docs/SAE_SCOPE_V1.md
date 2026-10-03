# Qwen-Scope SAE collection, version 1

This branch collects representations for unchanged quality-suite V3 fixtures. It does not fit probes, perform PCA, label SAE features, or make scientific claims. It reads historical runs and caches without rewriting them. The 1-based V3 block maps to zero-based SAE layer `block_number - 1`; cached blocks 4, 8, 12, 16, 20, 24 and 28 map to layers 3, 7, 11, 15, 19, 23 and 27. Hooks capture `model.model.layers[N]` complete output before final RMS norm. The final `hidden_states` entry is not used as a substitute.

The primary checkpoint is `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_100` at revision `2756a49850d1d0288a18e561783deae0d810791e`. `top50` is separately trained (`ce1a79d9c5163932d65c417380e53230e1086370`) and needs a separate config/run. The Qwen3-1.7B-Base revision remains `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. Every checkpoint LFS SHA-256 is recorded in `sae_repository.json` and verified after download. Feature IDs are scoped to `(repository revision, checkpoint hash, layer)`.

## Commands

From the project root with `.venv` activated:

```bash
qd sae plan --profile smoke
qd sae collect --profile smoke --resume
qd sae validate --profile smoke --resume
qd sae plan --profile laptop
qd sae collect --profile laptop --resume
qd sae validate --profile laptop --resume
```

`plan` creates a versioned run; later invocations need `--resume`. All commands print the run directory. `collect` first encodes cache-ready chunks for up to ten minutes, then extracts missing raw states, then resumes encoding; the two latter stages can also be called separately as `qd sae extract` and `qd sae encode`. Resume with unchanged profile, config, layers and trace count. Model and encoding minute limits are cumulative; increase `--max-model-minutes` or `--max-encoding-minutes` after exhausting one. `--layers 3,7,11,15,19,23,27` creates a separate cached-layer run. Smoke defaults to layer 3 and four token traces; laptop defaults to all 28 layers and 128 traces; `extended` selects the full fixture bank. `plan` estimates raw bytes, sparse output bytes, downloads, free disk and model forwards.

The smoke layer is chosen because its raw states already exist in the V3 cache, not by predictive performance. The laptop layer list is fixed to all 28 blocks before looking at SAE outputs.

For the separately trained Top-50 release, copy `configs/sae_scope_v1.json` to a new config file, set `sae_variant` to `top50`, and use `--config`. Changing only K on a Top-100 checkpoint is not equivalent.

## Dataset files and joins

- `identity.json`, `config.json`, `source_v3.json`, `sae_repository.json`: immutable run, model, cache, SAE and numerical settings.
- `readouts.jsonl`: one row per unique `(token IDs, position)`, with exact text, IDs, offsets, token count, position and readout type. Cases have entity and final readouts. Comparisons have a final-prefix readout; V3 defines no target-entity span for them.
- `records.jsonl`: every case/readout and comparison/readout association, with original fixture metadata. Cases also carry contrast IDs, paired case IDs, hue triplet IDs and roles. Identical computational readouts can have multiple records. `target_registry.json` gives the physical units and label definitions for target keys.
- `traces.jsonl`, `trace_selection.json`: deterministic fixture-metadata-only sample, coverage, selected IDs and excluded case IDs. Each selected prompt contributes every non-padding token, separately from main readouts. `text_key` joins a trace to `readouts.jsonl` for exact token IDs and offsets.
- `raw/main/*.npz`, `raw/trace/*.npz`: original float32 residuals, zero-based layers, positions and run identity. Matching V3 shards remain in their original cache, referenced in `source_v3.json` and identity-checked on read.
- `features/{main,trace}/layerN/chunkXXXXXX.npz`: bounded sparse chunks. `keys` joins to `readout_id` for main rows or `trace_id:token_position` for trace rows. `ids` is int32 `[rows,K]`, `values` is float32 `[rows,K]`; all other coordinates are zero in the 32,768-wide vector. Every chunk carries layer, checkpoint hash, run identity, raw norm, error norm, relative error, cosine, nonzero count and flags.
- `feature_index.json`: stable main/trace key-to-row ordinals and chunk size for direct lookup.
- `manifest.json`: expected/completed/missing counts per layer and dataset, case-family coverage, reconstruction-error quantiles, schema, checksums, numerical settings, validation-code hashes and complete/partial status. `state.json` records cumulative model and encoding time.
- `collection_report.md`: regenerated factual status table after validation, including layer coverage, raw-shard counts, trace exclusions, budgets and resumption notes.

Flags bit 0 means zero raw norm, bit 1 zero reconstruction norm, and bit 2 a nonfinite diagnostic. Relative error is NaN when raw norm is zero. These are technical checks. A feature unobserved in this cohort is not thereby globally dead.

Encoding uses the [official full demo](https://huggingface.co/Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50/blob/main/app.py) convention `TopK(ReLU(h @ W_enc.T + b_enc), K)` and decodes with `sum_j z_j W_dec[:,feature_j] + b_dec`. Raw states receive no whitening, PCA, unit normalization or probe preprocessing. CPU float32 is the default SAE backend; batches and output chunks are bounded, and one checkpoint is loaded at a time. The language model is unloaded before encoding.

## Loader example

```python
from pathlib import Path
import json
from quality_dimensions.sae_collection import load_feature

run = Path('runs/sae_scope_v1_smoke_top100-cb4a52cc096f')
readout_id = json.loads(next((run / 'readouts.jsonl').open()))['readout_id']
row = load_feature(run, layer=3, readout_id=readout_id)
print(row['vector'].shape, row['ids'], row['values'])
print(row['records'])  # all case/question associations and labels/splits
print(row['diagnostics'], row['checkpoint_sha256'])
registry = json.loads((run / 'target_registry.json').read_text())
for record in row['records']:
    for target in record['metadata'].get('targets', {}):
        print(target, registry[target].get('unit', registry[target].get('physical_unit')))
```

Use the run path printed by `qd sae plan` for another checkout. Downloading one layer requires network access and about 537 MB. All 28 checkpoints exceed this machine's available disk, so collection can remain partial. Check `manifest.json` for factual coverage, not the command exit code.
