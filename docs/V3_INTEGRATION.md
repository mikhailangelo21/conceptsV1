# Quality-suite V3 integration

The suite uses its own loader, CLI, cache, run identity and analyses. It does not
adapt the old size-only CSV or overwrite original pilot runs. The input generator
and the model integration are separate programs.

## Commands

From this project directory:

```bash
source .venv/bin/activate
python populate_quality_suite.py --verify
pytest -m 'not integration' -q
qd suite plan --profile smoke
qd suite run --profile smoke --resume --max-model-minutes 60
qd suite run --profile laptop --max-model-minutes 60
```

`plan` creates a versioned run, so the following `run` needs `--resume`. If running
without `plan`, omit `--resume` for the first invocation. A completed run can be
reported again with `qd suite report --run-dir runs/quality_suite_v3_...`.
Use `--resume` with unchanged code/configuration to continue a partial run.
The budget counts synchronized model-forward seconds cumulatively across resumes;
loading, tokenization, CPU fitting, and report rendering are additional wall time.
An incomplete full run exits nonzero so shell command chains do not silently proceed.

Default V3 run folders in `runs/` are symlinks to
`~/.local/share/quality-dimensions-llm/runs/`. Its separate activation cache is under
`~/.local/share/quality-dimensions-llm/cache/quality_suite_v3/`. This avoids iCloud
evicting thousands of working shards from a Desktop project. Existing pilot caches
and runs remain at their original paths. Actual paths and identities are recorded
in each run. Explicit `--run-dir` retains the caller's chosen location.

## Extraction and provenance

The model revision comes from the original pinned Qwen3-1.7B-Base pilot. There is
no automatic model substitution. Inference uses MPS FP16, batch one, and blocks
4, 8, 12, 16, 20, 24, and 28. Character spans are validated against exact text and
tokenizer offsets. The entity readout is the last overlapping subtoken of the
final contextualized mention; the other readout is the final prompt token.

Both positions can be captured in one forward pass. The cache keys actual tokens
and position, with model/tokenizer revisions, dtype, backend, library versions,
selected blocks, and complete-block/pre-final-RMSNorm semantics in the identity.
Property labels do not cause duplicate activation storage.

Suite file hashes, IDs, foreign keys, source kinds, entity groups, wording roles,
factorial combination rules, and same-word-inventory binding pairs are checked.
Generated labels are authored fixtures, not human observations.

## Analyses

- Training requires training entities and wording; factorial training also requires
  training combinations. Numeric challenge rows never enter fitting.
- Lexical models have separate `lexical__...` names. Their provisional spelling
  groups do not establish synonym independence.
- Ridge preprocessing and alpha selection use grouped training folds. Vocabulary
  and IDF for unigram baselines are refitted in each fold. Layer/readout selection
  uses validation entities with training wording/combinations.
- Mean-difference directions, ridge, contrast-SVD rank curves, token/character
  length, unigram TF-IDF, and averaged input-embedding baselines are recorded.
  Contrast rank uses outer validation, with alpha inherited from the grouped
  full-linear fit. Rank is bounded by the numerical rank of training contrasts.
- Entity validation and entity test results are distinct. Case families and
  wording remain separate in reported partitions. Scalar PCA/SVD-to-ridge models
  remain single linear readouts; component count is not intrinsic dimensionality.
- Matched contrasts cover binding, controls, and factorial selectivity. Bootstrap
  uncertainty resamples entity-family means rather than treating prompt rows as
  independent. Matched local ranges/templates/domains have direction angles and
  bootstrap subspace overlap diagnostics.
- Physical-unit ridge models use stated training values and identity calibration,
  with numeric-literal baselines. Numeric sweeps, range roles, and equivalent-unit
  pairs remain distinct from ordinal codes.
- Hue uses coupled sine/cosine outputs, angular error, and circular near/far
  triplets. Factorial multi-output models use output-rank-bounded reduced-rank
  regression. Derived volume is not an independent target; HSV value is not
  emitted luminance.

## Behavior and optional follow-up

Behavior sums exact teacher-forced conditional log probabilities for all candidate
tokens. The tests compare this to token-by-token scoring. Candidate meanings,
answer mappings, printed order, raw scores, and entity splits are retained.
Single-token choices also compare the native margin with float32 selected output
head rows; the preceding model stays FP16. This is a precision diagnostic, not a
claim that the complete network ran in float32.

The supplied specification disables both nonlinear exploration and intervention.
No causal conclusion follows from the observational factorial selectivity table.
Optional CPU nonlinear code uses cached features, train-only preprocessing,
validation early stopping, logged restarts, and subspace stability. Restarts are
not interpreted as local semantic regions. There is no automatic escalation.
Optional causal code limits development-selected competent properties and matches
each factorial patch to its own baseline, identity and random controls. These
optional modes require a separately saved configuration and have not been included
in the default smoke/laptop verification.

## Independent ratings

```bash
qd suite import-ratings --input actual_ratings.csv --output data/independent/ratings.csv
```

Required columns are `concept_id,quality_id,rater_id,rating,source,license`.
Ratings must be finite and nonempty, concept/quality IDs must exist, and each
concept/quality/rater combination must be unique. The importer refuses to overwrite
an existing file or the generated fixture bank. It never fills missing ratings.

## Main outputs

Each run includes `report.md`, plots, `linear_metrics.csv`, `linear_predictions.csv`,
`linear_search.json`, `model_selection.json`, `paired_contrasts.csv`,
`paired_summary.csv`, `factorial_selectivity.csv`, `matched_local_alignment.csv`,
`local_subspace_stability.json`, `numeric_challenges.csv`, `unit_invariance.csv`,
`circular_neighborhoods.csv`, `multioutput_metrics.csv`, `multioutput_selection.json`,
and raw behavior shards plus score/bias summaries. Empty diagnostic files indicate
no applicable rows in that profile; coverage is recorded explicitly. Smoke is a
code check, and the laptop results are exploratory rather than a confirmatory test.
