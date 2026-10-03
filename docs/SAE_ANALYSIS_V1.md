# Qwen-Scope SAE analysis v1

This pipeline reads the existing [SAE collection](SAE_SCOPE_V1.md) and V3 case metadata. It never runs the language model, extracts new states, edits a collection shard, or fills missing SAE checkpoints. Analysis outputs live in a separate content-addressed run under `runs/sae_analysis_v1_*`. The collection manifest and repository checkpoint hashes are the authority for which layers exist.

## Execution

From the project root, with `.venv` activated:

```bash
qd sae-analyze run --profile smoke
qd sae-analyze run --profile core
ANALYSIS_RUN="runs/sae_analysis_v1_core-5fe3b13cdf8a"
qd sae-analyze enrich --run-dir "$ANALYSIS_RUN"
qd sae-analyze optional-methods --run-dir "$ANALYSIS_RUN"
qd sae-analyze context --run-dir "$ANALYSIS_RUN"
qd sae-analyze reconstruction --run-dir "$ANALYSIS_RUN"
qd sae-analyze baselines --run-dir "$ANALYSIS_RUN"
qd sae-analyze report --run-dir "$ANALYSIS_RUN"
```

Set `ANALYSIS_RUN` to the path printed by the core command if it differs from the already completed run shown here. Resume a stopped run with that path and `--resume`; stage names are `audit`, `methods`, `linear`, `nonlinear`, `tokens`, and `extended`. A completed stage is skipped on resume. A failed stage restarts from its beginning and rewrites only analysis outputs. `--profile extended` uses the same bounded core cohort plus optional method adapters. Exact bounds, seeds, representative layers and source path are in [the config](../configs/sae_analysis_v1.json). Changing config or analysis source changes the run identity.

`enrich` is a resumable post-core pass for PCA ranks 1/3/8/32, seed stability, original-space local PCA, decoder-direction PCA, matched-direction transfer, and train-fitted scale sensitivity. `optional-methods` runs available PaCMAP/TriMap/PHATE adapters, a bounded diffusion map, a separate supervised PLS comparison, prespecified UMAP/t-SNE sensitivity grids, and optional hue persistent homology. Missing libraries are recorded with reasons. `context` computes factorial target/off-target projections and joins frozen SAE summaries to the saved V3 behaviour scores, keeping natural continuations separate from A/B. These commands use existing collected states and feature chunks only.
`reconstruction` reuses frozen train-fitted PCA components to quantify rank-64 reconstruction separately on train, validation, test and withheld train challenges at every analysed layer/readout/view.
`baselines` fits prompt-length and unigram/bigram TF-IDF controls on the exact eligible training readouts used by the representation probes, with validation-only ridge selection. Prompt text often contains the authored quality level, so this is a lexical-shortcut control. It does not substitute for a static-embedding benchmark on a matched cohort; historical V3 summaries use different denominators.

The report command writes `artifact.json`, `findings.md`, `explorer.html`, and SVG/PNG figures. Package the canonical report as a self-contained `report.html` with the installed Data Analytics portable builder:

```bash
node /Users/popthornthanatit/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.10-13ceeea1f599/skills/build-report/scripts/deliver_portable_artifact.mjs --input "$ANALYSIS_RUN/artifact.json" --output "$ANALYSIS_RUN/report.html"
```

The builder validates source lineage and chart payloads, then checks the report at desktop and narrow viewports. The independent offline `explorer.html` has 2D/3D controls for layer, representation, readout, method, property, wording, split, and colour. It never changes a stored embedding when a colour or filter changes.

## Audit and representation grain

The audit reports expected and present records and independent entity-family counts by layer, property, case family, readout, wording and split. A case/readout association is distinct from a computational readout: multiple records may point to one prompt/token-position vector. All PCA and geometry matrices deduplicate by `readout_id`, while probes retain the case/label associations. Main entity and final-token sites remain separate. A comparison prefix lacks an entity readout by design. Tokens belong to a sampled prompt and are not independent family observations.

`H` is the original float32 block-output residual. `F` is a 32,768-column CSR code with original SAE feature IDs; active values come from the pinned Top-100 checkpoint. `H_hat` and `E=H-H_hat` are saved on a bounded aligned subset using the exact checkpoint decoder matrix and bias. Feature IDs have no common meaning across checkpoints. The pipeline checks run identity, layer, width, K, chunk keys, feature range, duplicate IDs and finite values before analysis. `audit/summary.json` records unencoded layers explicitly.

## Fitting and interpretation rules

Only cases satisfying the original V3 eligibility rule enter training: train entity family, familiar template wording, nonnumeric case, and training factorial combination. Validation is used only for ridge strength; test is frozen. Heldout wording, numeric challenges and heldout factorial combinations are never used to fit ordinary quality probes, PCA or feature candidates. Numeric rows have declared physical units in the source registry, but this analysis does not fit a physical calibration from withheld challenge examples. Ordinal values are authored design codes and should not be read as equal perceptual intervals.

The core run scans sparse activity on all completely encoded layers for entity, final and token-trace rows. It selects a fixed metadata-stratified, deduplicated cohort for expensive H/F PCA, prediction and original-space dimension diagnostics. Sample IDs are saved in `sample_ids.json`. A further fixed, split-balanced cohort supports nonlinear comparisons at network-position-selected layers 1, 11 and 27. No layer is selected on test performance.

Sparse F PCA uses a column-centred `LinearOperator` and PROPACK sparse SVD; it does not rename uncentred `TruncatedSVD` as PCA or build a feature covariance matrix. H PCA uses randomized centred PCA. Train means, components, singular values, variance ratios, transformed IDs and coordinates are in `linear/*_pca.npz`. Variance thresholds not reached by the first 64 components are null. The reported kNN/TwoNN estimators use original H or F vectors on a bounded subset, with exact duplicates removed. They are density- and cohort-sensitive complexity diagnostics, not the number of psychological dimensions.

The method registry is in `methods.csv`. Core linear baselines include centered PCA, uncentered F TruncatedSVD and nonadaptive random projection. UMAP is fitted on eligible train readouts and transforms heldout points. Standard sklearn t-SNE is marked descriptive all-splits. Three prespecified seeds are used for the 2D stochastic comparison; one is used for 3D display. Faithfulness reports local trustworthiness/continuity/neighbor overlap and sampled global distance-rank correlation/stress separately. The comparison is made to its exact 64-PC input; the PCA spectrum and heldout reconstruction diagnostic assess information lost before nonlinear embedding. Extended sklearn methods use a bounded 32-PC input; NMF alone uses original nonnegative uncentered F. Missing optional packages are recorded in `extended/status.csv`.

## Results map

| Output | Meaning |
|---|---|
| `audit/layers.csv`, `audit/coverage.csv` | Actual SAE coverage and denominators |
| `activity/*`, `features/candidates.csv`, `features/catalogue.csv` | Feature prevalence, magnitude, reconstruction association, training-selected hypotheses, checkpoint identities and activating prompts |
| `activity/*_coactivation.npz` | Bounded binary joint frequency, prevalence-adjusted lift/phi, and value correlation |
| `linear/*_pca.npz`, `linear/pca_summary.csv` | Train-fitted centered PCA and compressed coordinates; summary RMSE pools the sampled splits |
| `linear/partitioned_reconstruction.csv` | Frozen rank-64 reconstruction RMSE separately by fit/heldout partition |
| `linear/component_interpretation.csv` | Leading F-PC loadings and descriptive quality, wording, entity-family and prompt-length associations |
| `predictions/metrics.csv`, `predictions/rows.csv` | Partitioned ordinal/hue heldout scores and individual predictions |
| `baselines/metrics.csv`, `baselines/*_tfidf.joblib`, `baselines/*_fit.json` | Same-cohort lexical controls and their train-fitted vectorizers/IDs |
| `reconstruction/*_aligned.npz` | Bounded exact-decoder H, H_hat and E |
| `contrasts/matched.csv`, `hue/triplets.csv` | Prespecified matched feature changes and circular near/far tests |
| `dimension/estimates.csv` | Original-space TwoNN and kNN MLE sensitivity |
| `embeddings/*.npz`, `embeddings/faithfulness.csv` | 2D/3D coordinates and local/global fidelity |
| `tokens/summary.csv` | Prompt-grouped trace-role activity and causal-order retention |
| `provenance.json`, `identity.json`, `state.json` | Source hashes, dependencies, seeds, and resumable stage status |
| `predictions/additional_ranks.csv`, `predictions/transfer.csv` | Further compression ranks and test-entity withheld-wording results |
| `dimension/local_pca.csv`, `dimension/truncated_spectrum.csv` | Neighbourhood-scale and truncated-spectrum diagnostics |
| `dictionary/layer11_decoder_direction_pca.npz` | PCA on selected decoder columns as observations |
| `linear/sensitivity.csv`, `embeddings/stability.csv` | Row/feature scale sensitivity and three-seed reproducibility |
| `extended/optional_methods.csv`, `extended/nonlinear_sensitivity.csv` | Optional method status and prespecified parameter grids |
| `factorial/summary.csv`, `behavior/observational_summary.csv` | Off-target factorial sensitivity and format-separated behaviour links |

Candidate-feature ranks are hypotheses, with no uncorrected p-values or globally-dead labels. Top-K competition can induce coactivation patterns. PCA on feature activations describes variation in codes; dictionary geometry would require PCA on decoder directions and is a separate question. No PCA whitening is interpreted as Park et al.'s causal inner product. No pleasing 2D/3D arrangement by itself establishes a Gärdenfors quality dimension.

Core method choices follow [scikit-learn PCA](https://scikit-learn.org/stable/modules/decomposition.html), [scikit-learn manifold methods](https://scikit-learn.org/stable/modules/manifold.html), and [UMAP's heldout transform documentation](https://umap-learn.readthedocs.io/en/latest/transform.html). Linear CKA follows [Kornblith et al.](https://proceedings.mlr.press/v97/kornblith19a.html). These references explain the estimators; the saved outputs document this dataset's actual implementation and limits.
