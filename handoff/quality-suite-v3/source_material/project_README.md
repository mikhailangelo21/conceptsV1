# quality-dimensions-llm

The versioned multi-quality suite is available through `qd suite`. See
[V3 integration and commands](docs/V3_INTEGRATION.md). Its smoke/laptop profiles,
grouped analyses, exact continuation scoring, and separately identified local
caches preserve the original pilot described below.

A reproducible dissertation pilot separating **linear accessibility**, **context sensitivity**, and **causal influence/selectivity** of an ordinal physical-size direction in `Qwen/Qwen3-1.7B-Base`. It uses native PyTorch/Transformers block hooks, plain text, and public pretrained weights. There is no LM fine-tuning, hosted API, chat template, quantization, or sampled generation.

**The included labels are `demo_ordinal`: assistant-authored provisional demonstration ranks, not human ratings or measured dimensions.** The software can produce a scientifically negative result. A successful pipeline is not evidence for a Gärdenforsian conceptual space.

## Install and inspect

Run from this directory. The supplied `.venv` links to a local environment at `~/.local/share/quality-dimensions-llm/venv`, created on the actual Apple Silicon Mac using its available Python 3.13.3. A Desktop-resident environment was replaced after macOS evicted dependency files (the filesystem marked them `dataless`), causing imports to stall. Keeping the environment outside cloud-managed Desktop avoids that failure. Python 3.11/3.12 are supported by the project requirement but were not separately tested. `requirements-lock.txt` is the exact tested macOS ARM64 dependency snapshot; a different platform may need a separate compatible snapshot.

```bash
python3 -m venv ~/.local/share/quality-dimensions-llm/venv
# For a fresh checkout only, create the project link:
ln -s ~/.local/share/quality-dimensions-llm/venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
qd doctor
pytest -m 'not integration'
```

Default backend selection is MPS float16 on this Mac, otherwise an explicitly recorded CPU float32 compatibility fallback. Explicit `backend: cuda` is supported without extra CUDA-only libraries. Precision/backend changes require a separate run. The checkpoint's stored bfloat16 conversion is recorded. All parameters are frozen; forwards use eval/inference mode, eager attention and no KV cache. Backend seeds do not imply bitwise identity across devices.

## Execute and resume

```bash
qd prepare-data --config configs/pilot_m2.yaml
qd run --config configs/smoke.yaml
qd run --config configs/pilot_m2.yaml --resume --max-model-minutes 60
qd report --run-dir runs/EXACT_RUN_ID
```

`prepare-data` prints the deterministic run path and exact workload. `--resume` can also create a run if none exists. Subsequent invocations require `--resume`. Copy the exact command printed in the run report:

```bash
qd run --config configs/pilot_m2.yaml --run-dir runs/EXACT_RUN_ID --resume --max-model-minutes 60
```

The budget covers synchronized model-forward time in that invocation, including warm-up/benchmark. Downloads, loading, CPU statistics and rendering are outside it. The current forward completes before stopping, so the limit can overrun by one prompt. Completed atomic rows persist. Every planned condition remains pending until it completes; a budget stop produces a visibly partial report. Another invocation grants another 60 minutes. An explicitly chosen larger budget, e.g. `--max-model-minutes 600`, permits a longer session.

You can execute stages separately, with the same configuration and run directory:

```bash
qd extract --config configs/pilot_m2.yaml --resume
qd analyze --config configs/pilot_m2.yaml --resume
qd intervene --config configs/pilot_m2.yaml --resume
qd report --run-dir runs/EXACT_RUN_ID
```

Analysis requires complete extraction; interventions require saved analysis. This avoids analyzing whichever concepts happened to finish first. Partial behavioral results remain labeled partial. CPU analysis does not reload the LM. Complete runs return immediately on resume. A changed configuration, data, prompt definition, or package source gets a new run ID; an explicit incompatible run path is rejected. Keep one writer per run/cache. Reports can be regenerated without rerunning inference.

The optional **independent** smaller-model run is `qd run --config configs/small_06b.yaml`. Nothing silently substitutes it. `configs/extended.yaml` demonstrates extra layers, random directions and behavioral wording; it is never launched automatically. Additional concepts can be supplied in a replacement CSV; geometry templates are explicitly defined in `prompts.py` and must preserve the fit/held-out separation.

Model and tokenizer use the same resolved Hugging Face commit; `model.json` pins it on resume. For an independent future replication, copy a configuration, change its profile/seed, explicitly set `revision` to the desired immutable commit, and supply independent labels. Leave old run directories intact. Ordinary Hugging Face caching is used; set `HF_HOME` before invoking the CLI to locate weights on a larger disk. The run preflight checks missing weight bytes and activation storage with a reserve. It does not delete files or disable MPS memory safeguards.

## Design and workload

The fixture has 60 concepts in six categories, with several ranks in each. Sorted within-category blocks of five are allocated 3/1/1 using a fixed RNG, producing 36/12/12 independent training/validation/test groups. Synonyms remain together. Small datasets cannot support exact crossed stratification; size-ordered blocks provide local rank coverage, not perfectly equal rank distributions. Size/category confounding remains, especially produce versus transport.

Two neutral templates enter fitting/validation; `t2` is reserved for final wording generalization. Neutral representations are averaged per synonym group before fitting. All 60 concepts have three neutral prompts and five additional conditions in `t0` and `t2`: **780 geometry prompts**. Contexts alter the subject, an unrelated object, or irrelevant Tuesday information. Naming prompts do not ask for size. Token IDs, lengths, final readout indices and rendered texts are saved. Anything over 256 tokens is rejected, never truncated.

Behavioral baselines include all 12 validation and all 12 test concepts: 192 prompts across size (two polarities × two mappings), subject identity and blue/red tag (two mappings each). Six test subjects are selected before inference by round-robin categories and alternating low/high ordinal rank. Their 48 prompts receive four directions × five doses: **960 intervention forwards**, including separately verified zero hooks. Baselines are reused. Total pilot workload is **1,944 forwards**, including 12 warm-up/benchmark forwards. Geometry caches shared with smoke can reduce actual new forwards. Exact counts are always in `plan.json`.

Tag assignments are seed/ID hashes independent of size labels. Counterbalancing correct-label positions prevents a fixed answer-token preference from constituting success. The baseline diagnostics are frozen after validation and before test questions; this implementation does not adaptively tune the predetermined question format. Reference reuse is logged, and interpretation remains conditional on the two reference-only objects.

## Hook and scoring semantics

Public block numbers are 1–28; module indices are 0–27. Each Qwen3 block performs attention with a residual addition, then MLP with a residual addition. Hooks read the complete block output, including the final block **before** the separate RMSNorm. The installed version returns a tensor, and the hook code also explicitly supports tensor-first tuple/list returns. It clones only the selected token to independent float32 CPU storage; it never retains sequence views. Hooks are removed in `finally` paths.

Extraction calls the transformer body without vocabulary logits. Behavioral scoring uses `logits_to_keep=1`, stable float32 full-vocabulary log-softmax and verified one-token A/B continuations at every actual prompt boundary. It tries a common leading-space encoding first. Multi-token answers are deliberately rejected rather than approximated. Scoring accuracy is the forced-choice A/B accuracy, not a claim that unrestricted generation would emit A or B.

Interventions clone the block output and replace only its final prompt-token slice with `h + λ s u`, exactly once. `u` is the train-only unit mean-difference direction; `s` is the sample SD (`ddof=1`) of its training concept projections. Remaining blocks and final RMSNorm run normally. Random unit directions use **the same `s`**, not their own projection variance. Requested and realized float16 perturbation norms, activation norms and ratios are saved.

## Analysis

- Within each training category, take lower/upper `floor(n/3)` rank groups, expand boundary ties, and skip categories whose cuts overlap or have fewer than three independent groups. Average category differences with equal weight; normalize and orient toward larger training labels. Save category vectors, cosine agreement, cuts and exclusions.
- Fit centered, unstandardized ridge using the declared `[0.1, 1, 10, 100, 1000]` grid. Validation Spearman chooses alpha; smaller alpha breaks ties. Highest validation direction Spearman selects the primary block; earliest block breaks ties. Save the selection before test evaluation.
- Fit centered, unwhitened PCA on training group means only, up to matrix rank. Keep its basis fixed for validation/test/context projections. PCA is exploratory.
- Report unseen concepts in seen templates separately from unseen concepts in held-out wording. Category means and a noun character/token-count ridge are lightweight baselines. MAE/R² apply only to calibrated ordinal predictors, never the uncalibrated direction.
- Context scores and paired changes use training mean/SD. Every held-out group is plotted; no successes-only filtering.
- Descriptive 95% intervals resample independent synonym groups after aggregating their related variants; degenerate Spearman replicates are counted. The 100 shuffled-label controls permute training groups **within category**, refit direction and validation-tuned ridge at the frozen primary block, and score test sets. They are a conditional reference distribution, not a selection-corrected global test.
- Behavioral effects use `D = log P(label meaning larger) − log P(label meaning smaller)`, correctly reversing both question polarity and option mapping. Accuracy need not improve under successful steering. Off-target identity/tag correctness and correct-label log odds are reported alongside size effects.

## Replace labels and preserve provenance

```bash
qd import-data --input independent_labels.csv --output data/independent_labels.csv \
  --references data/references.csv --pairs data/pairs.csv \
  --ratings independent_rater_observations.csv
```

`--ratings` is optional and preserved as a **separate** table, never synthesized. Required concept columns are `concept_id,lemma,synonym_group,category,size_score,label_kind,label_source,notes`. Optional columns such as `aliases` (pipe-separated), `units`, `aggregation` and `uncertainty` are preserved. Supported label kinds are `demo_ordinal`, `human_ordinal`, `sourced_measurement`; measurements require units and aggregation. Sources must be explicit. The importer validates concepts/references/pairs together; each target must have a pair and references must have separate IDs. Review whether old comparison pairs remain valid for a new dataset.

Copy the config and point `concepts` at the imported file. Identical prompt tokens, revision, backend, precision and implementation reuse activation shards; label changes create a new analysis identity and never reuse stale probes. Label uncertainty is **not** automatically incorporated into the bootstrap, even when observations are supplied. Modeling genuine rating uncertainty is an additional analytical step.

## Outputs and verification

Each run contains input snapshots, hashes, pinned config/revision, actual host information, splits, rendered/tokenized prompts, completion state, fitted arrays/metadata, raw predictions/context scores, per-forward intervention shards, summaries, PNG/SVG plots and `report.md`. Atomic activation shards are in `cache/activations/`; `activation_index.json` maps stable run row indices to their exact files. Copy that cache with a run for archival portability. RSS and MPS allocations are recorded separately because unified-memory measurements overlap.

`docs/VERIFICATION.md` records actual checks and execution status. Offline tests use tiny randomly initialized Qwen3 models, including native MPS float16 when available. Those fixtures are **software tests only** and do not populate the real-model reports. The separately marked integration test is:

```bash
QD_REAL_MODEL=1 pytest -m integration -q
```

The philosophical motivation and limits follow the supplied specification. Relevant primary sources are linked in each generated report and `docs/REFERENCES.md`. This project is an additive-steering pilot, not an exact replication of activation-patching papers.
