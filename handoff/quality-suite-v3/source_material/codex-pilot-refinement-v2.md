# Codex prompt: refine the quality-dimensions pilot

Implement the following refinement in the existing `quality-dimensions-llm` repository. Preserve the original run `pilot_m2-792030855aa5`, its report, fitted objects, and caches as immutable pilot evidence. Inspect the actual code and raw records before making diagnoses. The observations below come from the supplied report and figures; the raw CSV files and implementation have not yet been independently audited by the author of this prompt.

Use the current Qwen3-1.7B-Base checkpoint on the 16 GB M2 Air. Keep ordinary model execution in the existing tested MPS configuration, with the targeted precision checks below. Do not upgrade the model, fine-tune it, increase steering strengths to obtain a desired outcome, or rewrite the working pipeline from scratch. Reuse compatible activations. Create separately identified audit and v2 runs with complete provenance.

The objective is to determine which evidence survives improved measurement, and why failures occur. A valid negative result is a successful experiment.

## Evidence motivating the changes

The report selects block 23 on validation data. For 12 held-out concepts, direction/ridge/noun-length Spearman correlations are approximately 0.6325/0.7209/0.5575. The direction's reported interval is very wide: 0.03224 to 0.9363. Do not infer a reliable advantage by comparing these point estimates or by inspecting overlap of separate confidence intervals.

Context means, in training projection SD units, imply these descriptive contrasts:

| Template | Subject enlarged minus shrunk | Other-object enlarged minus shrunk | Difference of those contrasts |
|---|---:|---:|---:|
| t0 | 0.6905 | 0.07977 | 0.61073 |
| t2 | 0.3326 | 0.02010 | 0.31250 |

These calculations use rounded report means. They have not supplied paired uncertainty or shown whether every object follows the pattern. Recompute them from raw records. Both subject-size contrasts are positive, even though enlargement is below the neutral baseline in t2. A wording-dependent offset is therefore an alternative to complete loss of size ordering; it is a hypothesis, not an established explanation. Preserve the original neutral-relative analysis alongside the new contrasts.

The report gives 26/48 correct baseline size judgments, but near-perfect identity/tag performance. The supplied narrative also reports positive steering favoring A in 19/24 size questions. Verify this from raw scores, including its dose and near-zero tolerance. At strength +1 the report gives mean size semantic log-odds change +0.01432, identity correct-log-odds change -0.0599, and tag change -0.08854. Different tasks need not have comparable sensitivity, but the off-target changes require explanation.

## Phase A: audit and reanalyse existing results, without new model forwards

### A1. Check definitions and scoring

Inspect `interventions.py`, `model.py`, `hooks.py`, `analysis.py`, `report.py`, prompt construction, and the raw behavior records. Trace a few concrete examples through the full scoring calculation.

- Deduplicate baseline rows and check which records represent independent subjects versus repeated mappings, polarities, doses, and directions.
- The report has 12 baseline test subjects and six intervention subjects. Report baseline competence on exactly the intervention subset as well as the full baseline set. Do not treat the 48 versus 24 size-question counts as an error without checking this distinction.
- Verify the A/B continuation token IDs and whitespace in full prompt context, and that no answer was already appended to the measured prefix.
- For single-token answers, verify `log P(A) - log P(B) == logit_A - logit_B` numerically in float32.
- Verify that the semantic sign follows which option means the subject is larger, including both question polarity and answer mapping. Separate semantic log odds from correct-answer log odds.
- Inspect the actual wording and labels of the two heavily reused reference objects. Flag ambiguous referents and invalid comparisons rather than assuming synthetic ordinal ranks establish physical truth.

For each fixed subject, reference, template, polarity, layer, and dose, pair the two label mappings. Let `delta_L_plus` be the intervention change in `log P(A)-log P(B)` when A means the subject is larger; let `delta_L_minus` be the change when B means the subject is larger. Compute:

```text
semantic_component = (delta_L_plus - delta_L_minus) / 2
common_A_component = (delta_L_plus + delta_L_minus) / 2
```

Keep individual mapping/polarity results visible. These are a descriptive decomposition, not proof of independent latent mechanisms. A constant A preference cancels in a balanced semantic average; do not claim that the presence of an A preference alone explains every residual semantic effect. Mapping-dependent interactions may remain. In future prompts vary the order in which A and B are printed separately from their meanings, so token preference and first-listed-option preference can be distinguished.

### A2. Paired context contrasts

For each concept and template calculate:

```text
subject_contrast = score(subject_enlarged) - score(subject_shrunk)
distractor_contrast = score(other_enlarged) - score(other_shrunk)
binding_contrast = subject_contrast - distractor_contrast
```

Report each concept's values, means, fractions above zero, and paired 95% bootstrap intervals resampling whole concept/synonym groups. Recompute the intervals from paired rows; do not subtract the published interval endpoints. Plot subjects paired across templates and retain neutral, irrelevant, and absolute context-shift plots.

Add an exploratory development-set analysis of whether template differences can be approximated by a score offset or positive affine rescaling. Fit any calibration using development/training concepts only and evaluate it on excluded concepts. Separately report raw fixed-direction performance and calibrated performance. A template-specific correction is additional flexibility, not evidence of one globally invariant coordinate. Never fix the observed t2 result by flipping its axis sign after looking at its test outcomes.

### A3. Probe advantage and category confounding

Using cached features, compare nuisance-only prediction against nuisance plus activation-derived prediction on identical excluded concepts. Nuisances initially include category, noun character count, and tokenizer token count. Fit all preprocessing and nuisance adjustments within training folds. Avoid residualizing features or labels on the complete dataset.

Report paired differences in predictive performance with concept-level uncertainty. A bootstrap of fixed test predictions is conditional on the trained models; label it accordingly. For development model selection, use grouped outer cross-validation with any layer/hyperparameter tuning confined to its inner training/validation data. Include within-category evaluation and a separately labeled leave-one-category-out generalization test where sample sizes permit. A category-only baseline for an unseen category needs an explicit fallback.

The original 60 objects and t2 wording have now informed methodological changes. Treat them as development data for v2. Preserve the old frozen split for historical reproduction, but do not call it a new untouched confirmatory test. Reserve new concepts and new wording families for the final evaluation after the v2 procedure is frozen.

Produce an audit report even if later phases cannot run. Distinguish verified errors, observed limitations, and untested explanations.

## Phase B: establish a behavioral assay that the model can perform

Create a small development-only comparison benchmark, initially about 30–50 distinct, unambiguous subject/reference pairs. Avoid inflated sample counts from swapping one pair repeatedly. Include both directions of size ordering, several reference objects, and explicit documentation of pair labels and provenance.

Compare a small prespecified set of formats:

1. The original A/B questions, for continuity.
2. Natural continuations appropriate for a pretrained base model, for example:

```text
Compared with a mouse, an elephant is usually much
```

Score the continuations ` larger` and ` smaller`, verifying actual tokenizer boundaries. Reverse subject and reference in matched examples. Do not use a prefix where the comparison reference only appears after the word being scored; a causal model cannot see future text.

3. The same continuation format with a small fixed set of disjoint, balanced demonstrations, if zero-shot competence is poor. Keep demonstration order and examples controlled; do not add item-specific hints.

For counterfactual descriptions use wording such as `In this scene, compared with ...` rather than `usually`, which could explicitly request ordinary category knowledge despite a described transformation.

Natural continuations remove arbitrary A/B encoding but do not remove every output-word bias. Retain paired comparisons and check lexical preferences. Score exact conditional continuations; use validated sequence scoring if an answer has multiple tokens. Do not replace this with sampled-text judging.

Report accuracy, pair-reversal consistency, semantic margins, and results by wording. Freeze format selection using development data. As an explicitly pragmatic, configurable pilot gate, require at least 80% balanced accuracy and 80% reversal consistency on unambiguous development comparisons before interpreting a new steering assay. Report uncertainty and sample sizes; these cutoffs are not universal scientific standards. Check contextual comprehension separately from ordinary-size knowledge.

If no tested format is competent, record that this model/assay cannot currently support the intended causal interpretation. Still finish all software, audits, and representational analyses. Do not silently change checkpoints or filter test examples until a gate passes.

## Phase C: separate typical size from size in a described situation

Maintain two explicitly named questions:

- **Typical size:** how large is this kind of object ordinarily? Use the original direction as a baseline, improving independent labels and nuisance controls.
- **Contextual instance size:** how large is the particular object described here? Learn a separate candidate from matched changes to the same subject.

The original pilot additionally assumed that a direction trained on typical object size would transfer to contextual instance size and from naming prompts to comparison prompts. Test these transfers explicitly; do not build them into the definition of success.

For a contextual direction at layer `l`, compute the training-only average of matched large-minus-small activation differences, balancing objects and wording families, then normalize. No question answers or test concepts may enter this fit. Compare it with the original typical-size direction; do not replace the original silently.

Hold out BOTH object identities and a wording family. Include tests that cannot be solved by detecting the mere presence of size-related words. For example, these two contexts have the same word inventory but bind the sizes to different entities:

```text
The rabbit is now the size of a horse. The statue is now the size of a pea.
The rabbit is now the size of a pea. The statue is now the size of a horse.
```

Keep the queried subject the rabbit in both. Counterbalance clause order and which entity is queried across examples. Hold this anchor-comparison wording out if training used `enlarged`/`shrunk`. Include unchanged-subject controls. State that even this test examines linguistic representation of described size, not perceptual grounding.

Add an entity readout after all relevant context, such as the last subtoken of the repeated noun in `The subject in this scene is the rabbit.` Compare this predefined location with the original final prompt marker. Locate tokens robustly and save offsets; do not search for whichever position works best on test data. A subject mention before its size description cannot already incorporate that later description in a causal decoder.

Separate within-task fitting/evaluation from transfer between naming and comparison tasks. Use the same readout convention and task family first, then report cross-task transfer as an additional test.

## Phase D: targeted precision checks and more informative interventions

### D1. Verify numerical resolution before interpreting tiny effects

The +0.01432 mean change is a roughly 1.0144-fold change in odds, not a 1.4 percentage-point change in accuracy. Inspect raw effect distributions, baseline margins, actual applied perturbation norms, and fraction of activation coordinates unchanged by rounding.

On a fixed small development subset, compare the existing score path with a float32 calculation of the candidate logit difference from the final normalized hidden vector and just the relevant output-weight rows. For two single-token answers:

```text
log_odds = dot(float32(w_answer1) - float32(w_answer2), float32(h_final))
```

Cast each weight row to float32 before subtraction. This avoids rounding the output dot product to float16; it does not undo float16 rounding in earlier layers. Do not cast an entire tied embedding/output module in place or retain a second full model unnecessarily.

If numerical sensitivity remains material, run a few complete float32 CPU forwards sequentially as a separate check, unloading the MPS model first. Compare baseline and intervened runs within each precision condition and state whether they use cast copies of the same original checkpoint. Confirm effective perturbation norms as well as requested norms. Do not infer a numerical bug simply because some results occur in quantized steps.

### D2. Establish a meaningful activation location

After behavioral competence is established, compare the original final-token site with the predefined context-informed subject-token site at a small development-selected layer set. Preserve the frozen test set. At a subject token earlier than the answer, an intervention after the FINAL decoder block has no remaining attention layer to carry the change to the later answer position; do not interpret that structurally ineffective case as evidence against size representation.

First test matched donor/recipient context patching at the chosen site, where the donor and recipient describe different subject sizes and the model's unmodified responses actually distinguish them. Full-vector patching tests whether information at that site mediates the contextual contrast; it does not isolate a size dimension. Then test replacing only the component along the fixed training direction:

```text
h_patched = h_recipient + u * dot(u, h_donor - h_recipient)
```

Align readout meaning, token identity, and prompt position carefully; record token-length differences. Include self-patches, reverse patches, other-object/distractor donors, and matched-norm random controls. Report raw semantic-margin changes. Avoid dividing by a tiny donor-recipient margin to manufacture a large recovery percentage.

Retain additive steering as a complementary test. Keep prespecified doses, counterbalanced outputs, identity/tag controls, and actual perturbation norms. Compare directions in the same task and coordinate system. Do not assume a probe's coefficients automatically specify the right intervention geometry.

## Labels and final evaluation

Keep demonstration labels visibly labeled. Add an importer for an appropriate independently collected rating dataset, such as the behavioral data released with Grand et al. (2022), after checking its schema, source, reuse terms, and which features/categories were actually rated. Do not invent missing measurements or silently map incomparable category-specific scales onto one global ruler. Use within-category analyses when the rating design only supports those comparisons.

For a later confirmatory study, reserve approximately 30–60 genuinely new, independently labeled concepts and new wording families, with the eventual sample size justified by the desired precision and available data. For now, implement the holdout mechanism and a clearly labeled data template if those labels are unavailable. Do not invent empirical evidence to complete a run.

## Deliverables and verification

Extend the existing modules where appropriate. Likely changes concern `analysis.py`, `report.py`, `prompts.py`, `interventions.py`, `hooks.py`, `probes.py`, and their tests; inspect rather than assume their responsibilities.

Provide a reproducible cache-only audit command and separately configured v2 stages. Retain the 60-minute bounded execution/resume behavior. Add focused tests for paired joins, semantic/A-bias decomposition, token scoring, post-context entity selection, train-only fitting, source/recipient patch alignment, and zero/self-patch invariance. No test should demand a positive scientific finding.

Generate: (1) audit tables and an honest revised pilot report, (2) behavioral-format diagnostics, (3) paired context/binding plots, and (4) conditional intervention results or a precise explanation of why the assay gate failed. Explain actual changes, tests executed, measured runtime, limitations, and exact continuation commands. Do not infer raw CSV findings solely from the report text.

Work in this order: cache-only audit, behavioral-format diagnostics, contextual representation experiment, then the small causal experiment when interpretable. Complete authorized work without repeatedly requesting routine implementation choices.

## Primary methodological references

- [Grand et al. (2022), Semantic projection recovers rich human knowledge of multiple object features from word embeddings](https://www.nature.com/articles/s41562-022-01316-8). The article links its behavioral data and code at [OSF](https://osf.io/5r2sz/).
- [Zhao et al. (2021), Calibrate Before Use](https://proceedings.mlr.press/v139/zhao21c.html).
- [Pezeshkpour and Hruschka (2024), sensitivity to option order](https://aclanthology.org/2024.findings-naacl.130/).
- [Park et al. (2024), linear representations and geometry](https://proceedings.mlr.press/v235/park24c.html).
- [Heimersheim and Nanda (2024), interpreting activation patching](https://arxiv.org/abs/2404.15255).

These are methodological foundations, not claims that the v2 design exactly replicates those papers. Proceed to inspect and refine the existing implementation.
