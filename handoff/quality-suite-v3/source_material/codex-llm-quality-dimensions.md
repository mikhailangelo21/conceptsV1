# Codex implementation prompt: LLM quality dimensions on an M2 Air with 16 GB

Copy the prompt below into Codex in the folder where you want the project implemented. This is an implementation request, including a bounded first run, rather than a request for another plan.

---

Implement a complete, reproducible Python research project called `quality-dimensions-llm` following this specification. My computer is an Apple M2 MacBook Air with 16 GB unified memory. Make sensible implementation decisions and proceed without asking about routine choices. Respect existing repository instructions and preserve unrelated files. If the current folder is already an appropriate repository, integrate there; otherwise create a project subdirectory.

Build the software, seed data, tests, documentation, command-line workflow, plots, and generated research report. Execute the tests and a real-model smoke test where the environment permits, then run the bounded pilot. Do not finish with a scaffold, pseudocode, unimplemented core functions, or an instruction that I must write the remaining code. Never fabricate measurements or claim a run happened when it did not.

## 1. Research question and scope

This is an undergraduate Computer Science and Philosophy dissertation investigating whether contextual representations in LLMs contain directions corresponding to interpretable quality dimensions. Gärdenfors' Conceptual Spaces is a motivation, not a conclusion the experiment must establish.

The first property is **physical size**, meaning the approximate overall spatial extent of the named physical object. Treat size labels as ordinal unless a supplied dataset establishes stronger measurement assumptions. Keep typical category size separate from the size of a particular object described in a context.

Implement three distinct experiments:

1. **Linear accessibility:** can a direction or linear probe fitted on training concepts predict size ranks for unseen concepts and an unseen wording template?
2. **Context sensitivity:** does the same fitted direction respond appropriately when the subject is described as shrunk or enlarged, while responding less to irrelevant information and size language about another object?
3. **Causal influence:** does changing the activation along that direction selectively alter size judgments, across reversed questions and answer-label orders?

Report these as separate claims. A PCA plot is exploratory evidence. Linear prediction alone does not establish a causally used feature. Successful steering alone does not establish a natural semantic mechanism, a privileged Euclidean metric, or a Gärdenforsian conceptual space. Failure in this small model does not settle what larger models represent.

## 2. Model, architecture, and runtime

Default model: `Qwen/Qwen3-1.7B-Base`.

- Use Hugging Face Transformers and PyTorch with native module hooks. This is a dense, pretrained, causal decoder model. Use plain text prompts; do not apply a chat template or substitute an instruction-tuned checkpoint.
- Its published configuration has 28 decoder blocks and residual width 2,048. Verify these from the loaded configuration and record them; derive tensor sizes from the configuration rather than hardcoding them.
- Document that each block contains attention and an MLP with residual additions, and that the model applies a final RMSNorm after the block stack. Record the relevant attention configuration from the loaded model.
- Resolve and record the exact model and tokenizer revision. Use the same pinned revision for both, and reuse it on resume.
- Freeze all model parameters, use evaluation mode and inference mode, and disable KV caching for these short complete-prompt forwards. No language-model fine-tuning or backpropagation is required. The small statistical probes are fitted separately on CPU.
- On an available MPS backend, load the weights explicitly as `torch.float16`. Record the conversion from the checkpoint's stored dtype. Do not silently switch precision during an experiment.
- Use CPU float32 as a clearly recorded compatibility fallback. Support explicit CUDA selection for later reuse, but require no CUDA-specific packages, FlashAttention installation, or NVIDIA hardware.
- Use eager attention initially for a simple, inspectable baseline. Any optional faster attention backend must be configured and recorded separately.
- Do not use quantization by default. Do not require Ollama, MLX, TransformerLens, a hosted model API, a web application, or a database.
- Supply an optional `Qwen/Qwen3-0.6B-Base` configuration for a smaller independent run. Do not substitute it silently or combine its activations with the 1.7B model.
- Detect the actual execution host. If Codex is running remotely on Linux, do not claim MPS was tested on my Mac.

Use a local virtual environment with a supported Python version, preferably 3.11 or 3.12, and a tested dependency lock or exact requirements snapshot. Inspect current official documentation and the installed Transformers implementation before relying on API details. Qwen3 requires a compatible Transformers release; do not guess a version or require an untested development branch.

Keep dependencies modest: PyTorch, Transformers, huggingface_hub, NumPy, SciPy, scikit-learn, pandas, matplotlib, PyYAML, tqdm, psutil, and pytest are sufficient, with loading helpers only if necessary. Prefer a simple CLI. Model downloads are authorized for this local setup; use public weights and ordinary caches. Do not start paid compute or upload data elsewhere.

## 3. Resource limits and run profiles

Default to batch size 1, one loaded model, and at most 256 input tokens. Reject overlength examples with a useful error; do not silently truncate away the readout marker or context.

Create these profiles:

- `smoke`: a small representative subset, including neutral and contextual examples, baseline scoring, and a nonzero intervention. It verifies mechanics and produces a report, but makes no scientific claims from its sample size.
- `pilot_m2`: the full dataset and analyses specified below, one selected intervention layer, six predetermined held-out intervention subjects, three fixed random control directions, and intervention strengths `[-1, -0.5, 0, 0.5, 1]` in the calibrated units defined below.
- `extended`: optional settings for more concepts, templates, random directions, and layers. Implement configuration support; do not launch this automatically.

The first automatic pilot run has a 60-minute model-execution budget, configurable by CLI. This is a stop-and-resume limit, not a claimed runtime. Save completed work and label the report partial if it expires. Provide the exact command to continue until all planned examples are complete. Do not silently drop conditions to fit the budget.

Benchmark approximately ten real prompts after warm-up, synchronizing MPS around timing. Report measured throughput and an explicitly approximate remaining-time estimate. Record process memory and available MPS allocation statistics separately; do not sum overlapping unified-memory measurements as though they were independent.

Keep selected activations on CPU or disk. Never accumulate all tokens' hidden states or attention matrices. Check for nonfinite activations and logits. On OOM, checkpoint and give an actionable error; do not disable MPS memory safeguards. Keep weights, caches, and generated runs out of Git.

## 4. Data and provenance

Provide an immediately runnable, inspectable pilot fixture containing about 60 unambiguous physical-object concepts from approximately six broad categories. Each category should contain several sizes. Aim for overlap in size across categories where possible, and explicitly describe remaining category-size confounding.

Examples to consider include an ant, mouse, rabbit, horse, elephant, blueberry, apple, watermelon, bead, mug, chair, bicycle, and bus. These are suggestions, not a scientifically validated list. Avoid ambiguous referents such as an unspecified species of tree or an object whose scale changes radically by interpretation.

The fixture can use manually authored or assistant-generated provisional ordinal size labels to make the engineering pipeline runnable. **Label these `demo_ordinal`, not human ratings or measured dimensions.** Never invent participants, rating reliability, physical measurements, citations, or data provenance. Figures and reports derived from these labels must prominently say that they use provisional demonstration labels.

Implement a CSV importer for later independent human ratings or sourced measurements. Preserve source, units, aggregation, and uncertainty when available. Missing genuine ratings must not prevent the demo pipeline from running, but the report must explain which conclusions still require independent labels.

Suggested concept fields:

`concept_id`, `lemma`, `synonym_group`, `category`, `size_score`, `label_kind`, `label_source`, `notes`.

Keep per-rater observations in a separate optional table rather than inventing them. Do not request an external LLM to generate experimental ground truth during a run.

Supply a separate, explicit comparison-pair table for behavioral tasks, including target, reference, expected relation, and provenance. Use clearly separated sizes rather than dubious ties. Reference-only objects should have separate IDs and be excluded from probe fitting. Log reuse of references; inference from this pilot is conditional on this small reference set.

The pipeline must validate unique IDs, required fields, finite labels, alias overlap, category coverage, pair validity, and label provenance before inference.

## 5. Splits and prompt construction

Split by concept/synonym group approximately 60/20/20 into train, validation, and test. Use a fixed seed and stratify by category and broad size bands as far as the small dataset permits. Save the split manifest. Every prompt and synonym for the same concept must stay in the same split.

Use two neutral wording templates for fitting and validation. Reserve a third template for the final template-generalization analysis. Do not use this held-out wording to choose a layer, probe, or hyperparameter.

Example neutral prompt:

```text
Context: The subject is the rabbit. It has its ordinary size.
Task: Name the subject.
Answer:
```

Create two genuine wording alternatives with the same final `Answer:` marker. Use grammatically appropriate noun forms. Geometry-extraction prompts should not explicitly ask for size, so that the primary result is not merely decoding a property the prompt directly requested.

For context experiments, create these matched conditions:

1. Neutral: the subject has its ordinary size.
2. Irrelevant information: the subject has its ordinary size and the observation takes place on a Tuesday.
3. Subject shrunk: the subject has been shrunk to one hundredth of its ordinary size.
4. Subject enlarged: the subject has been enlarged to one hundred times its ordinary size.
5. Shrinkage distractor: another object has been shrunk to one hundredth of its ordinary size; the subject is unchanged.
6. Enlargement distractor: another object has been enlarged to one hundred times its ordinary size; the subject is unchanged.

Use the same transformation wording in subject-change and distractor conditions where possible. Keep the subject noun and naming task fixed. These clauses test qualitative shifts, not whether a projection numerically represents a factor of 100. The extra wording and token positions remain possible confounds; record token counts and show the controls.

A manageable default is all 60 concepts in three neutral templates, plus the five additional conditions in one seen template and the held-out template: about 780 unique prompts. Reuse cached neutral prompts. Make counts configurable and print the exact planned workload before execution.

Save rendered text, prompt ID, concept ID, condition, template ID, split, token IDs, token count, and the selected readout index. The primary readout is the **last non-padding prompt token, before any answer is supplied**. Assert the final marker/tokenization is consistent. All relevant context must precede it. Never append the correct answer before extracting features.

## 6. Activation extraction: exact hook semantics

Capture the residual vector after each complete decoder block, after its attention and MLP residual additions, and **before the separate final RMSNorm** for the last block. In the usual Hugging Face layout these blocks are under `model.model.layers`, but verify the installed implementation.

Use direct block hooks and explicitly handle the actual output type. Do not assume `output[0]` is correct: an output may already be a tensor. Do not blindly interpret every entry of `output_hidden_states` as the required location; the final entry can have different normalization semantics across implementations.

Use block numbers 1 through `num_hidden_layers` in results, while documenting their zero-based module indices. If you additionally record the input embedding or final normalized state, give them separate names and never mix them into the block sequence.

Each hook must extract only the intended token vector, detach it, copy it to independent CPU storage, and avoid retaining a view of the full sequence tensor. Remove hooks reliably, including on exceptions. For extraction-only passes, call the transformer body without materializing vocabulary logits if appropriate.

Save float32 analysis copies, while recording that the forward computation was float16. Cache by a fingerprint including model/tokenizer revision, library version, backend, precision, attention implementation, prompt tokens, hook location, and readout position. Keep activation-cache identity separate from label/analysis identity, so new ratings can reuse identical model forwards without reusing stale probes.

Use resumable, bounded shards or equivalent simple storage with stable row indexing, completion metadata, and atomic writes. Do not silently reuse mismatched caches. Precompute the expected storage size from examples × layers × hidden width × bytes.

## 7. Fit directions and probes without leakage

Fit only on neutral training concepts in the two seen templates. Average those template activations per concept for fitting so repeated phrasings do not artificially increase the number of independent observations. Save every fitted transform and training ID list.

For each layer, implement:

**A. A size direction from differences of means.** Within each training category, identify its upper and lower size groups using a fixed, documented rule, such as upper and lower thirds. Compute the difference between their mean activations. Average these differences with equal category weight, then normalize to unit Euclidean length. Handle ties and insufficient groups explicitly. Orient the direction toward larger training examples. Save category-specific differences and their agreement, since a direction shared across domains is itself an empirical question. Optionally include the pooled large-minus-small direction as a descriptive comparison.

For training mean `mu_l` and unit direction `u_l`, the projection is:

```text
score_l(h) = dot(u_l, h - mu_l)
```

**B. A ridge regression probe.** Fit a centered linear predictor of size score. Select its regularization from a small declared grid using validation concepts only. Avoid a neural probe. Do not whiten or standardize every residual coordinate by default; such transformations change the geometry being investigated. If an optional transformation is implemented, fit it on training data and name the resulting coordinate system explicitly.

Use Spearman rank correlation as the primary predictive metric. Report MAE/R² only as secondary summaries with the limitations of ordinal labels. Evaluate separately on unseen concepts in seen templates and unseen concepts in the held-out template.

Add lightweight baselines: training-set category means, and a small linear model using noun character count and token count. These do not remove every linguistic confound, but make some easy alternatives visible.

Select the primary intervention layer using validation performance of the mean-difference direction, with a deterministic tie-break. Freeze that choice before test evaluation. The ridge readout coefficient is not automatically a validated steering direction; keep the primary intervention tied to the explicitly defined mean-difference vector.

## 8. PCA, context analysis, and uncertainty

- Fit centered, unwhitened PCA on neutral training concept vectors separately at each layer. Use only as many components as the sample rank permits. Keep each fitted PCA basis fixed when plotting its validation/test/context examples.
- Produce a layer-by-layer prediction plot and selected-layer PCA plots colored by size and marked by category. Include explained variance and distinguish training from test points. Do not claim that PC1 is a size axis or that low explained variance disproves a semantic feature.
- Apply the original fixed size direction to every context condition. Do not refit separate directions to shrunk and enlarged examples.
- For context plots, express projections relative to the neutral training mean and training projection standard deviation at that layer. This calibrates displays within layers; it does not establish one common physical coordinate system across layers.
- Compute paired changes relative to each concept's neutral prompt. Show whether enlargement tends to move scores upward, shrinkage downward, and distractor/irrelevant changes are smaller. Plot all held-out concepts, including failures.
- Bootstrap **concept groups**, preserving all their related prompt variants, for descriptive 95% intervals. Use approximately 1,000 bootstrap replicates, handle degenerate samples transparently, and state that this excludes label uncertainty unless real rating data supply it.
- Add a shuffled-training-label baseline, preserving concept groups and, for the main control, category structure. Refit the direction/probe for each permutation. A small default such as 100 permutations is sufficient for a pilot reference distribution. Do not describe this reference distribution as a fully corrected global significance test after selecting across layers/hyperparameters.
- Treat layer sweeps and secondary plots as exploratory. Report sample sizes in independent concepts, not just prompt rows. No uncorrected collection of significance claims across every layer.

## 9. Behavioral capability check and answer scoring

Before interpreting steering, check whether the model performs the relevant ordinary comparisons at baseline. Use validation examples to debug the format, then freeze it before test evaluation.

Example:

```text
Context: The subject is a rabbit of ordinary size.
Reference: A horse of ordinary size.
Question: Is the subject physically larger than the reference?
Options: A = yes; B = no.
Answer:
```

For each subject/reference pair, include both larger/smaller question formulations and both A/B mappings. Keep the identity of the subject explicit. Record the actual correct label separately from which label expresses the proposition that the subject is larger.

Use exact conditional answer log probabilities, preferably with verified single-token A/B continuations. Check tokenization in the full prompt boundary context, including leading spaces. Do not assume a string corresponds to one token. Prefer choosing a common valid single-token answer encoding and assert this throughout the pilot. If supporting multi-token answers, implement tested teacher-forced sequence scoring with correct causal shifting and explicit intervention-position semantics; do not approximate a multi-token answer using its first token.

For the main single-token case, obtain only the required final-position logits when the installed model supports this. Convert the small scoring tensor to float32 before stable log-softmax. No sampled text generation is needed for the main metric.

Report baseline accuracy, label bias, and performance by question polarity and option order. Retain all planned test examples; do not select only correctly answered test questions as the main analysis. A baseline-correct subset may be a clearly labeled secondary diagnostic. Poor task competence should qualify the scientific interpretation without being misreported as a software failure.

## 10. Activation interventions and selectivity controls

At the selected block output and final prompt position, implement:

```text
h_prime = h + lambda * s_l * u_l
```

Here `u_l` is the unit mean-difference direction and `s_l` is the standard deviation of its projections over neutral **training concept** vectors. This scale must come from training data, not test activations. The configured pilot strengths are `[-1, -0.5, 0, 0.5, 1]`.

Clone and replace only the selected output slice, preserving the module's return structure and dtype/device. Apply the change exactly once in that forward pass; all remaining layers and the final normalization then run normally. Do not modify weights, earlier token positions, another layer, or previously cached activations.

Define the behavioral score in semantic terms:

```text
D = log P(label meaning "subject is larger")
    - log P(label meaning "subject is smaller")
effect(lambda) = D_intervened - D_baseline
```

For a smaller-than question the mapping to yes/no reverses. For a swapped A/B mapping the answer-token mapping also reverses. Test both explicitly. Positive steering should be evaluated by a tendency to raise this semantic score, not by always increasing `P(A)`, `P(yes)`, or accuracy. A successful intervention can make a correct ordinary-size answer incorrect.

Use six held-out subjects selected deterministically by a prespecified category/size-balanced rule, without seeing model test performance. Evaluate both question polarities and both label mappings. Use one question template initially; make additional templates configurable.

Required controls:

1. An unmodified baseline and a zero-strength hook.
2. Positive and negative semantic steering.
3. Three seeded random unit directions for the pilot. Use the **same absolute perturbation norm** `abs(lambda) * s_l` as the semantic direction. Do not rescale each random vector by its own projection variance. These few directions give a descriptive control, not a strong significance claim over random directions.
4. Subject-identity questions and a context-given arbitrary property, such as a blue/red tag, with answer mappings counterbalanced. Apply the same layer/directions/strengths and report changes in correctness and correct-label log odds. Tag assignment must be independent of size and explicit in the prompt.

Reuse baseline forwards and evaluate directions sequentially. Show planned forward counts before running. Report perturbation norms relative to the original activation norms and any nonfinite outputs. Do not increase the dose until a desired effect appears. Larger doses, more random directions, or extra layers must be separately configured exploratory runs.

Keep representation extraction prompts and behavioral question prompts separate. Report this transfer as part of what is being tested. A negative result can mean the direction does not transfer between tasks, rather than that size is absent everywhere in the model.

## 11. CLI, reproducibility, and outputs

Provide a small installable package and documented commands equivalent to:

```bash
qd doctor
qd prepare-data --config configs/pilot_m2.yaml
qd run --config configs/smoke.yaml
qd run --config configs/pilot_m2.yaml --resume --max-model-minutes 60
qd report --run-dir runs/EXACT_RUN_ID
```

Expose extraction, analysis, and intervention stages individually as well. They must reuse compatible results. Document how to resume the same run, change the runtime budget, replace labels, switch to the smaller model, and conduct a separate future replication. A config change must not silently mutate a completed result.

Suggested project components: `pyproject.toml`, dependency lock, `README.md`, `configs/`, `data/`, a Python package containing data/splits/prompts/model/hooks/cache/probes/interventions/report modules, and `tests/`. Keep code clear and typed where useful. No mandatory notebook or GUI.

Each run should save:

- Resolved configuration, seeds, timestamps, actual hardware/backend/precision, environment versions, model/tokenizer revisions, dataset and template hashes, hook semantics, and any numerical/backend fallbacks.
- Input data, split manifest, prompt metadata, and an analysis-selection manifest created before test evaluation.
- Resumable activation shards, fitted directions/probes/PCA objects with their training metadata, and raw per-example predictions and intervention scores.
- CSV/JSON metric tables and static PNG/SVG plots: prediction by layer, selected-layer PCA, paired context shifts, semantic intervention dose response against random controls, and off-target effects.
- A generated Markdown report explaining what actually ran, label provenance, independent sample sizes, measured resource use, results, negative findings, limitations, and remaining experimental requirements. Embed the generated plots with working relative links.

All plots and claims must come from saved actual outputs. Mark demonstration labels and partial runs visibly. Do not fill missing stages with simulated results. Reports should separate software verification, predictive evidence, context evidence, causal/selectivity evidence, and future philosophical interpretation.

Set Python/NumPy/PyTorch seeds. Record any nondeterministic backend behavior rather than promising cross-device bitwise identity. Resuming must preserve the exact random control directions and data selections.

## 12. Tests that protect scientific validity

Use tiny local randomly initialized Qwen3 configurations for offline architecture tests, plus a separately marked real-model integration smoke test. Random-model fixtures verify code and must never appear as scientific findings.

Prioritize these meaningful tests:

- Concept/synonym groups cannot cross splits; held-out templates cannot enter fitting or selection. Changing test features/labels cannot change fitted preprocessing, directions, hyperparameters, or the selected layer.
- Hook shapes and token selection are correct, and cached slices do not retain full sequence storage. Hooks are removed even after exceptions.
- The captured final block output is the input to the separate final normalization; it is not mistakenly the normalized model output.
- A causal-prefix test: changing later tokens cannot change earlier-token activations. This catches mistaken context/readout reasoning.
- An observation-only hook and a zero-strength intervention preserve baseline logits within a declared numerical tolerance. The nonzero hook changes only the chosen position at its injection point, and weights remain unchanged.
- Interventions actually affect downstream computation; registering a hook that merely records values is not sufficient.
- Single-token conditional scoring matches direct logits. Any implemented multi-token scorer matches an independently checked teacher-forced calculation.
- Larger/smaller question reversal and A/B swapping produce the correct semantic log-odds mapping.
- Semantic and random perturbations have equal norms at matching strength.
- Train-only centering/PCA, grouped bootstrap resampling, and label permutations use the intended independent units.
- Interrupted runs resume without missing/duplicated rows, and incompatible caches are rejected.

Do not write tests that require a positive semantic result. Scientific failure is a valid outcome. The real-model smoke test should verify finite outputs, correct metadata, functioning hooks, and a readable report.

## 13. Implementation and execution order

1. Inspect the repository and actual host; create the environment and a short implementation plan.
2. Implement data, splits, prompts, and provenance validation.
3. Implement loading, precise hooks, scoring, and cache/resume behavior; verify them with offline tests.
4. Run the real-model smoke test and measure resource use. Fix actual defects before scaling up.
5. Implement and verify probes, PCA, grouped uncertainty, behavioral controls, and interventions.
6. Run the complete workflow within the automatic runtime budget, saving honest partial status if needed.
7. Inspect representative rendered plots and the generated report for legibility, correct axes/labels, provenance, and agreement with numeric outputs.
8. Finish with the exact commands I should run on my Mac, what you actually tested, the actual run path, and any genuine remaining blocker. If model access or MPS is unavailable in your environment, still finish the implementation and offline tests and identify precisely which integration checks remain unverified.

Do not ask me to select libraries or approve routine implementation choices. Do not quietly expand this into fine-tuning, a web product, a multi-model benchmark, or a literature survey. Larger models and broader philosophical claims are later stages.

## 14. Official implementation references and methodological reading

Consult these primary sources where relevant; verify installed-code behavior instead of treating a moving `main` branch as a pinned dependency:

- [Qwen3-1.7B-Base model card](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
- [Qwen3-1.7B-Base configuration](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/main/config.json)
- [Transformers Qwen3 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)
- [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [Park et al., The Linear Representation Hypothesis and the Geometry of Large Language Models](https://proceedings.mlr.press/v235/park24c.html)
- [Zhang and Nanda, Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042)
- [Heimersheim and Nanda, How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)

The specified experiment is a proposed pilot design, not an exact replication of these papers. Its main intervention is additive steering; do not mislabel it as a completed activation-patching replication.

Proceed to implementation now.
