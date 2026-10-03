# Covariance geometry extension of V3

This extension asks whether the covariance geometry of Park, Choe and Veitch better organizes the quality representations measured in V3. It adds a new CLI, configuration, metric cache and run namespace. It preserves the V3 fixture bank, completed runs, old activation caches and `handoff/quality-suite-v3/`.

Start with the completed run's `report.md`, then `ridge_summary.csv`, `output_pair_counts.csv`, `output_alignment_summary.csv`, `context_selectivity_comparison.csv` and `model_selection.json`. `source_snapshot.json` is the authoritative source for that new run. The original V3 handoff remains the entry point for the historical experiment.

## Mathematical contract

All mathematical vectors below are **columns** of length d. The model's actual output module is obtained with `get_output_embeddings()`. Its weight matrix has shape V × d; row v is gamma_v transpose. The implementation never substitutes averaged input embeddings for output vectors. The pinned Qwen head has no additive output bias; an output bias, if present in another architecture, would have to be retained separately when reproducing absolute logits.

For a uniform draw from **all V output-head rows**, including special and tokenizer-unassigned rows:

```
mu = (1/V) sum_v gamma_v
Sigma = (1/V) sum_v (gamma_v - mu)(gamma_v - mu)^T
<u,v>_output = u^T Sigma^-1 v
<a,b>_context = a^T Sigma b
g = Sigma^-1/2 gamma
l = Sigma^1/2 lambda
lambda^T gamma = l^T g
```

The covariance uses population denominator V, not V−1. Mean centering is used to estimate covariance. Output directions are differences and need no centering. Transforming centered output vectors subtracts the same `lambda^T mu` from every logit, preserving softmax probabilities and pairwise logit differences.

NumPy arrays store **rows**, with observations on axis zero: output `G_new = G @ Sigma^-1/2`; context `X_new = X @ Sigma^1/2`. Both square roots are symmetric. `Geometry.output`, `context`, `inner`, `norm`, `normalize`, `cosine` and `projection` implement these conventions. Projection returns a signed scalar along a unit direction in the declared metric. Zero directions cannot be normalized; cosine involving a zero vector is undefined, reported as missing.

The measurement pairing is always `lambda^T gamma_bar`, regardless of metric. Applying an inverse-covariance output inner product to two context vectors would be the wrong construction. The corresponding mapping from a measurement direction to an intervention direction is `s = Sigma^-1 gamma_bar`; the inverse map is `gamma_bar = Sigma s`. These identities are algebraic. Their semantic/causal interpretation requires the paper's assumptions, which this experiment does not establish.

## Representation-site audit and reuse

The original laptop run's saved `model.py` and `hooks.py` capture complete decoder block outputs **before** final RMSNorm. They are not the paper's final context representation. The new primary site is explicitly named `final_post_normalization_final_prompt_token`.

`Runner.final_output_readout()` registers a pre-hook on the actual output module and captures its final-token input. For cache reuse, the original final-token block-28 state passes through the frozen `model.model.norm` on the original MPS/FP16 backend. A deterministic selection of eight training prompts is forwarded again, spread across available case families. The cached residual, converted state and all reconstructed output logits are compared with that fresh forward.

Predeclared tolerances are residual/state max absolute error ≤0.002; native-head logits `allclose(atol=0.0625, rtol=0.001)`. These accommodate FP16 device rounding; every observed maximum and the number of logits is saved. A failing check aborts reuse. The preceding transformer is FP16: this is not a full-network float64 experiment. Metric calculations and probe analyses use float64 on CPU.

Model/tokenizer revision, transformer implementation hash, torch/transformers versions, backend, dtype, cache fingerprints, row keys, shape and finiteness are verified. Every reused shard gets a SHA-256 in `source_activation_ledger.json`. The normalized states and normalization weights have separate recorded hashes. Legacy cache files are never rewritten. Earlier-layer/entity results are copied as historical evidence with their original site metadata; the extension assigns no paper-grounded metric to those sites.

## Output-word analysis

Candidate pairs are generated deterministically from **existing** `concepts.csv` and `quality_labels.csv`. For each quality and original concept split, all low-level nouns (authored levels 1–2) are paired with all high-level nouns (levels 4–5). No labels, splits or lexicon entries are changed. There is no scalar hue output direction.

The completion format is exactly `Answer:` followed by one ASCII space and the original lemma. Tokenization must preserve the prefix tokens and append exactly one non-special token that decodes to the complete space-prefixed lemma. Multi-token, prefix-changing, special-token, identical-token and non-roundtripping candidates are excluded with reasons. This rule deliberately excludes many compound nouns; it does not select by alignment results. Candidate sets and eligibility are saved even for qualities lacking usable train or test pairs.

Each direction is the arithmetic mean of **training-pair** output differences, normalized in Euclidean or inverse-covariance geometry. Validation and test pairs are evaluated against the same frozen direction. Since the Cartesian pair set reuses nouns, pairs are not independent; results include distinct concept counts. Exact lexical forms remain in their original splits. A training-pair leave-one-out diagnostic is separate and still shares component nouns. The random baseline uses 512 seeded independent full-head row pairs, including special/unassigned rows, with the same draws across qualities/geometries. It is a broad geometric null, not a matched word-frequency/category baseline.

These noun pairs are **candidate lexical associations**, not asserted isolated counterfactuals. For example, two objects proposed to differ in mass usually also differ in size, category and identity. Provisional typicality labels, uneven single-token coverage and overlapping vocabulary across quality definitions constrain interpretation. No antonym purity or causal independence is assumed.

Context scores use the same raw final states and `lambda^T gamma_bar`. Geometry-specific normalization only rescales this output direction positively. Consequently raw score ordering, signs and affine-calibrated predictions are mathematically unchanged. Raw scores and their correlations are saved separately; MAE is only reported after training-only affine calibration to ordinal codes.

## Contextual contrast adaptation

For each non-lexical scalar quality, directions use matched training-family contrasts with positive target change: adjacent, full-range, binding and factorial one-axis increases. Both endpoints must satisfy the original V3 training filter: training entity, training wording, training combination for factorial rows, and not a numeric-challenge row. Nuisance invariants never define the semantic direction. Differences are averaged within an independent family, then family means are averaged equally. The exact contrast IDs and family IDs are saved.

The Euclidean arm uses `d / ||d||`; the covariance arm uses the context norm `sqrt(d^T Sigma d)`. Equivalently transform context differences by `Sigma^1/2` and use ordinary Euclidean tools. Signed projection and cosine are retained before calibration. A one-dimensional ordinary least-squares affine calibration uses eligible training rows only; its slope/intercept are logged. Binding signs, reordering, irrelevant additions and factorial changes are evaluated on matched endpoints. Factorial tables include every measured scalar property available at both endpoints, with an on-target flag.

These are **contextual contrast estimates**. A text edit controlling one authored property does not prove all other model concepts are fixed. Family-weighted summaries average each family's contrasts first. Detailed row-level tables retain pair IDs and original partitions. A contrast crossing a held-out factorial combination remains identifiable from both endpoint partitions and the original contrast's combination role.

## Ridge controls and coefficient transport

All new geometry comparisons use the same final readout, case IDs, training filters and evaluation partitions. Site and metric settings are fixed before examining test results.

Two ridge arms center X and y using training means, use an unpenalized intercept and minimize `sum squared residuals + alpha * ||B||_F^2` in their respective raw/transformed coordinates. Neither arm scales individual coordinates. Alpha comes from `[0.1,1,10,100,1000]`, selected by MSE in up to three grouped training folds, with centering refitted inside each fold. Ties use the smaller alpha. This preserves the geometry while making the changed penalty explicit.

V3's unchanged StandardScaler-plus-ridge helper is also fitted at this new site as a third control. The original selected-block ridge, mean-difference/SVD models, token-length, unigram TF-IDF, static-input-embedding, numeric-value and behavioral results are retained under `v3_historical_control/`; their original runs remain authoritative. These saved results are historical controls, not same-site covariance comparisons. The extension does not rerun generation behavior or physical-unit regressions. Lexical and constructed-context target fits remain separate.

An invertible transformation preserves the class of linear predictors. For coefficient column `w`, `w_new = Sigma^-1/2 w`; the raw-coordinate intercept is unchanged. If coefficients came from StandardScaler, first use `w_raw = w_scaled / scale` and `b_raw = b_scaled - mean^T w_raw`. Every new raw/standardized same-site ridge fit is transported and checked on all its evaluation rows at max absolute tolerance 1e-8. Refitting ridge in transformed coordinates may change performance because the penalty changes, not because information was created.

Hue remains a separate sine/cosine regression with circular angular error and near/far hue triplets. It does not enter scalar macro summaries. No PCA/SVD rank is interpreted as intrinsic semantic dimensionality.

## Numerical policy, memory and caching

The metric streams 2,048 output rows at a time through a float64 two-pass mean/covariance calculation. It avoids a full float64 vocabulary matrix. CPU `numpy.linalg.eigh` produces the eigensystem. Threads are limited to four during model preparation and analysis to suit the 16 GB laptop.

The configured `auto` policy uses the unregularized covariance if `min_eigenvalue > max_eigenvalue * 1e-10`. Otherwise it explicitly floors eigenvalues at `max_eigenvalue * rcond`. `unregularized` mode fails rather than hiding an unsafe inverse. Optional `ridge` mode uses `Sigma_effective = Sigma + loading I`, where `loading = max(ridge_relative * trace(Sigma)/d, max_eigenvalue * rcond)`. The complementary transforms use the **same effective matrix**. The saved diagnostics report raw/effective spectra, conditioning, formula and actual loading. No metric setting is selected by test performance.

The default smoke covariance uses a fixed-seed 8,192-row subsample and is integration evidence only. The laptop run uses the complete output vocabulary. Run identities distinguish these choices; `--vocabulary full` can also run the smoke case profile with a full metric. Special IDs and every tokenizer-unassigned row ID are recorded.

Metric caches live under `~/.local/share/quality-dimensions-llm/cache/quality_geometry_v1/metrics/`. Identity covers the exact model revision, actual output-head type/shape/dtype and streamed weight hash, vocabulary/tokenizer hash and selected-row hash, numerical settings and implementation version. Artifacts also have integrity hashes. Incompatible/corrupt reuse fails. New run directories under `runs/quality_geometry_v1_*` point to local non-iCloud storage, following the existing project's convention.

## Commands

Run from the original project root:

```bash
source .venv/bin/activate
pytest -m 'not integration' -q

# Optional standalone metric preparation. The following run automatically reuses it.
qd geometry metric --config configs/quality_geometry_v1.json --vocabulary full

# End-to-end integration: original 260-case smoke profile, sampled covariance.
qd geometry run --config configs/quality_geometry_v1.json --profile smoke

# Broad comparison: original 6,322-case laptop profile, full output vocabulary.
qd geometry run --config configs/quality_geometry_v1.json --profile laptop
```

An existing run requires `--resume`, unchanged source/config and its original `--profile`/`--vocabulary`. Changes require a new run directory. To split model preparation from cached CPU analysis:

```bash
qd geometry run --profile laptop --stage prepare --run-dir runs/quality_geometry_v1_my_run
qd geometry run --profile laptop --stage analyze --run-dir runs/quality_geometry_v1_my_run --resume
qd geometry report --run-dir runs/quality_geometry_v1_my_run
```

`--stage analyze` uses saved normalized states and the verified metric; it does not load model weights or spend forward budget. Reports and scientific PNGs are reproducible from saved tables. The model-forward verification budget is 60 minutes; only the fixed audit examples are newly forwarded when caches are present. Most work is CPU covariance and fitting.

If weights are missing, obtain the exact revision named in the config; model loading intentionally uses local-only mode. If source shards are missing, extract a **new** V3 run with the original configuration using `qd suite run --profile laptop --stage extract --run-dir runs/quality_suite_v3_geometry_source`, then set `source_run` in a separately saved geometry config to that new run. Do not direct extraction at a completed historical run. The legacy cache identity and frozen normalization still must pass audit.

## Optional steering, disabled by default

Neither default run invokes interventions. After measurement review, the explicit optional command is:

```bash
qd geometry steer --run-dir RUN --case-id CASE_ID --quality size --strength 0.1 --measurement-reviewed
```

This computes `lambda_new = lambda + strength * Sigma_effective^-1 gamma_bar` at the output head using saved final states and eligible output-token rows. It saves before/after algebraic scores in a new optional artifact, does not modify saved states, and does not rerun earlier transformer computation. The flag is required; omitting it fails. These float64 scores from FP16 state/weight values are not a new native FP16 generation experiment. No intervention has been run as part of the extension's default measurements.

## Verification and references

`tests/test_geometry.py` checks chunk covariance against a direct population calculation, direct metric versus transformed dots, complementary transforms, score/logit-difference invariance, positive-definite regularization, standardized coefficient transport, ridge objective equivalence, training exclusions, tokenizer handling, cache incompatibility/corruption and the steering gate. Existing V3/legacy tests remain applicable. Real smoke verification audits the actual output-head input and native logits, then runs all available comparison analyses.

The construction follows [Park, Choe and Veitch, Sections 2–4](https://arxiv.org/html/2311.03658v2), checked against the authors' [matrix construction](https://github.com/KihoPark/linear_rep_geometry/blob/main/store_matrices.py) and [analysis implementation](https://github.com/KihoPark/linear_rep_geometry/blob/main/linear_rep_geometry.py). This is an adaptation to the existing Qwen quality suite, not a replication of their LLaMA-2 concept bank. The paper's conditions for a causal interpretation are assumptions, not conclusions of this run.
