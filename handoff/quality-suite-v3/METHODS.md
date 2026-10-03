# Experiment methods and setup

## Question and scope

The broad question was whether quality-like properties can be read from and reasoned about in a language model's internal states. The V3 screen separates four narrower questions: whether an explicitly stated level is linearly decodable, whether the model binds a property to the correct entity, whether a readout transfers across wording and combinations, and whether the model's answers reflect the described relation. An optional causal intervention and optional nonlinear exploration were implemented as gated paths but disabled in the completed smoke and laptop profiles. This run cannot settle causal necessity or intrinsic semantic dimensionality.

The starting size-only pilot is described in [the original project overview](source_material/project_README.md) and [docs/VERIFICATION.md](docs/VERIFICATION.md). The V3 design is in [docs/QUALITY_SUITE_V3.md](docs/QUALITY_SUITE_V3.md); the implementation instructions are preserved in [docs/CODEX_QUALITY_SUITE_V3.md](docs/CODEX_QUALITY_SUITE_V3.md). The earlier user specifications remain in [source_material](source_material/). The exact generated examples are in [data/quality_suite_v3](data/quality_suite_v3/), and the fixture [manifest](data/quality_suite_v3/manifest.json) carries file hashes.

## Data and labels

The full generated bank has 15 candidate qualities, 20 fictional entity families, 342 lexical concepts and seven factorial domains. The laptop profile selected 6,322 cases and 1,888 comparison questions. Case families include lexical knowledge, five-level graded descriptions, contextual controls, same-word binding swaps, numeric challenges, hue and factorial combinations. The profile's exact selected IDs are in [data/quality_suite_v3/profiles/laptop.json](data/quality_suite_v3/profiles/laptop.json), and every selected case has a stable ID.

All labels are author-generated examples or values stipulated in fictional scenes. They are not collected human ratings or measurements. Graded/control/binding/factorial targets generally use constructed ordinal codes `0, 0.25, 0.5, 0.75, 1`; a quarter-step is a design interval, not an equal psychological distance. Physical-value targets end in `__value` and use declared canonical units. Hue uses HSV convention and circular error. The empty `human_ratings_template.csv` does not imply any recruited raters. The importer for genuinely independent ratings is present but was not used in this run.

## Model and readouts

The model was frozen Qwen/Qwen3-1.7B-Base at commit `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. Inference used a Mac MPS device, FP16, batch size one, and model blocks 4, 8, 12, 16, 20, 24 and 28. For every input the pipeline saved the complete residual block output after context at two places: the last subtoken of the final contextual mention of the target entity, and the final prompt token. The entity character span was converted to token offsets using the pinned tokenizer; the final-token readout is separate. The model was never trained or fine-tuned for this suite.

Cache keys include actual prompt tokens and position, model/tokenizer revision, dtype, backend, library and hook semantics. Identical text/position readouts are shared across labels. The run contains 12,612 distinct cached readouts for 12,644 labelled readout requests. The cache is outside this handoff, while the run's identity, indexes and file mapping are copied for audit.

## Splits and fitting

All variants of one fictional entity family remain in one entity split. The three entity splits are train, validation and test. Plain/numeric wording may enter fitting; paraphrase/reordered wording is held out. Factorial fitting also requires training combinations. Numeric challenge rows are excluded even when their template is called `numeric`. The primary model site and settings were selected using validation entities under training wording and training combinations. Test rows were not used to select a winning layer.

Each scalar quality has a property-specific ridge probe, a mean-difference direction, a train-only contrast-SVD candidate subspace, and lightweight token-length, unigram TF-IDF and static input-embedding baselines. Lexical targets use separate `lexical__...` fits. Regularization and preprocessing are fitted inside grouped training folds. Physical-value regressors use declared values rather than ordinal codes and are compared with a numerical-text feature baseline. Hue uses sine/cosine targets and circular error. Factorial domains use multi-output reduced-rank regression. Ranks describe the predictive models tested, not the number of intrinsic mental or semantic dimensions.

The behavioral questions compare exact teacher-forced conditional likelihoods for both candidate continuations. Natural words can require several tokens; the implementation scores the whole sequence. A/B labels have independently counterbalanced meanings and printed order. The recorded precision check recomputes the selected output-head margin in float32 for single-token choices, while the model forward itself remains FP16. This is not a full-network float32 rerun.

## Independent units and limitations

The full laptop profile's independent entity-family counts are in [independent_counts_by_family.json](runs/quality_suite_v3_laptop-88fad55b46b0/independent_counts_by_family.json). Graded, controls and hue each have six training, two validation and two test families. Binding, numeric and factorial each have two training, one validation and one test family. Lexical test probes have only three to seven concepts per quality. Many rows are variants of the same small number of families; they are not independent human observations. Validation spans seven blocks, two readouts, regularization and some ranks, which limits certainty from one split. Synonyms beyond exact normalized spelling are not fully consolidated.

The exploratory V3 run makes no formal family-wise significance claim. The original size pilot used different data, labels and evaluation, so its numbers are contextual history rather than a control arm. Complete details and caveats are in [reports/quality_suite_v3_findings/findings.md](reports/quality_suite_v3_findings/findings.md).
