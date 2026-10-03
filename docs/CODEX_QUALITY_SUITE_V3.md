# Integrate the quality suite into the existing project

Read `docs/QUALITY_SUITE_V3.md`, `data/quality_suite_v3/README.md`,
`data/quality_suite_v3/manifest.json`, and `configs/quality_suite_v3.json`.
Implement support in the existing repository; preserve original pilot runs/caches.
This JSON is a SUITE SPECIFICATION, not a claim that the old size-only CLI already
accepts it. Inspect the real loaders, schemas and tests before changing them.

1. Add a versioned suite loader and a separate `qd suite ...` command family (or
   equally explicit compatible subcommands). Do not map every attribute into
   `size_score`, overwrite `data/concepts.csv`, or silently apply the old size prompt.
   Validate IDs, foreign keys, source kinds, group splits, prompt template roles and
   held-out combination cells. Reuse cache entries only when actual prompt tokens,
   model revision, dtype and hook/readout semantics agree.
2. Load selected case/question IDs from the named profile. `smoke` verifies code,
   `laptop` is the initial broad screen, and `extended` is optional. First print
   forward counts, independent group counts and expected activation storage.
   Use the same Qwen3-1.7B-Base, MPS FP16, batch 1, selective positions and resumable
   60-minute budget. No automatic multi-model run or exhaustive steering sweep.
3. Extract the complete-block residual after context, before final stack RMSNorm.
   Convert character readout spans to token spans using offsets from the pinned
   tokenizer; use the last overlapping entity subtoken and separately the final
   prompt token. Assert the spans are valid; the text files cannot provide token
   offsets before tokenization. Never pool an initial pre-context mention into
   the contextual readout. Capture one underlying activation per identical text/
   position/layer, attaching several property labels afterward.
4. Use training entities + training templates; for factorial fitting also require
   training combinations. Numeric challenge rows cannot enter fitting. Tune on
   validation entities, then report each held-out axis of generalization. Avoid
   random prompt-row splits. Preserve ambiguity about synonym grouping in lexical
   seeds and keep their claims separate from controlled synthetic cases.
5. Fit per-property mean differences, ridge, and train-only contrast SVD subspaces.
   For several independent outputs, implement multi-output/reduced-rank models.
   Respect sample/feature rank for PCA/contrast SVD, and also output rank for
   reduced-rank regression. Compare with token-length, unigram, static-embedding
   and numeric-literal baselines, as applicable. Fit preprocessing and regularization
   within grouped training folds.
   PCA-to-ridge for one scalar remains one linear readout; do not call its PC count
   the intrinsic dimension of the quality. Treat hue with sine/cosine outputs and
   circular error; never apply an ordinary increasing-hue Spearman criterion.
6. Compare the adjacent-level directions and their subspaces across ranges,
   templates and domains. Generate paired contrast, binding, control, interpolation,
   unit-invariance, circular-neighborhood and factorial-selectivity results.
   The `targets` design codes are ordinal. Use `specified_values` for physical-unit
   diagnostics, and do not mix numeric `__value` targets with ordinal codes.
   Volume is derived, and HSV value is not emitted luminance. Note the different
   scales and sample counts instead of pooling every score into one headline.
7. Add optional nonlinear/dictionary exploration on CPU-cached data only, disabled
   by default. Distinguish optimizer restarts from local regions of the representation.
   Log every restart; use validation selection, early stopping and subspace stability.
   Require a documented held-out improvement over linear baselines before escalating.
8. Evaluate per-property behavioral competence and answer bias before causal work.
   Some natural answers contain multiple tokens. Implement exact teacher-forced
   conditional sequence scoring and verify it, or mark that format unsupported;
   never use only the first token. Distinguish answer meaning from printed A/B order.
   Give matched baseline/steering comparisons on identical subjects and save raw
   scores. Verify float32 final-head log-odds on a small sample where FP16 effects
   are tiny. Keep interventions at validated sites with subsequent computational
   paths to the measured output.
9. Run targeted interventions only for a small development-selected set of
   properties with competent behavioral formats. Test the cross-property effect
   matrix using factorial one-axis changes, ordinary random controls and identity
   controls. Preserve negative findings and the v2 paired-analysis corrections.
10. Add focused tests for scientific invariants, load/filter joins, hue wrapping,
    multi-output targets, factorial isolation, train-only search, sequence scoring
    and cache reuse. Execute a real smoke run where available and then a bounded
    laptop screen. Do not require a positive scientific result in any test.
11. Produce raw CSV/JSON results, readable plots, an honest report with original
    versus v3 comparisons, label provenance, independent counts, tested hypotheses,
    all model-selection choices, failures and limits. A thousand constructed rows
    are not a thousand independent human observations. Add an importer for actual
    independent labels without inventing missing ratings.

Proceed with implementation and verification. Report exact commands and what
actually ran. Do not stop after another plan, fabricate model results, or claim
that the data-population script has already implemented this integration.
