# What the V3 quality experiments show

Technical interpretation of the bounded laptop run · 28 September 2026

## Technical summary

**The model’s hidden states contain useful information about explicitly stated properties, including which entity a property belongs to. This run does not establish a stable, general-purpose geometric representation of those properties.** Familiar wording produces excellent ordering scores; changes in wording, numerical scale and combinations expose substantial weaknesses. The strongest positive finding is contextual binding beyond a bag of words. The strongest caution is that high correlation is often accompanied by poor calibration or weak transfer.

Across all 14 graded scalar properties, the average test Spearman correlation is **0.964 with familiar wording and 0.721 with held-out paraphrases**. Average absolute error rises from **0.121 to 0.316** on the constructed 0–1 target scale. These are equal-weight averages over properties, not population estimates or independent replications. Each property has just two test entity families.

The binding test is encouraging: every familiar-wording property has correctly signed changes for all tested matched swaps. The hidden-state probe distinguishes examples with the same words but different assignments, while static word averages and unigram features cannot. Yet there is only one independent test family, and the effect weakens under paraphrase.

Several harder claims remain unsupported. A numerical-text baseline has lower physical-value error on six of eight properties. Lexical generalization is mixed. Hue geometry is incomplete. Joint attribute prediction has serious calibration and wording-transfer problems. No causal intervention was run, so these results concern information that can be read from the model, not information demonstrated to control its answers.

## What was done, in simple terms

The experiment gave a frozen language model short descriptions of real or fictional items. Some descriptions varied one property through five levels, such as tiny to very large. Others swapped which item had a property, added distractions, changed the wording, expressed measurements in different units, or varied several properties independently.

At seven internal blocks—4, 8, 12, 16, 20, 24 and 28—the experiment recorded the model’s numerical state at the final mention of the target entity and at the final token. Small statistical readouts were trained to predict the supplied property labels from those states. The language model itself was not trained. Separate questions measured whether the model preferred the correct comparative answer, using either natural continuations or A/B labels.

The model was Qwen/Qwen3-1.7B-Base at revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, using FP16 inference on MPS. The final laptop run is `quality_suite_v3_laptop-88fad55b46b0`; it completed 6,322 labelled cases, 12,612 unique cached readouts and 1,888 questions. Recorded model time was 982.9 seconds, about 16.4 minutes. The separate smoke run checked the integration; it is not an independent scientific replication.

Labels are authored experimental fixtures, including provisional lexical ratings. They are not newly collected human judgments. A result can therefore validate tracking of the stipulated description without establishing agreement with human perception or real-world measurement.

## How to read the evidence

**Spearman correlation (rho)** asks whether predictions put items in the right order. One is perfect ordering, zero means no monotonic association, and a negative value means reversed ordering. It does not tell us whether predicted values are numerically correct. **Mean absolute error (MAE)** measures how far the predictions are from their labels; lower is better. A 0.25 error spans one step on a five-level 0–1 design scale. Those steps are experimental codes, not established equal perceptual distances.

**Generalization** here has distinct meanings: new entity families; held-out sentence templates; previously withheld attribute combinations; and challenges in numerical values or units. The figures keep these separate. Every displayed test score comes from a model selected using training and validation data, not the best-performing test layer.

Thousands of text rows do not provide thousands of independent replications. Graded, controls and hue each have six training, two validation and two test families. Binding, numerical challenges and factorial cases each have two training, one validation and one test family. Lexical test sizes vary from three to seven concepts per property. Multiple levels, wording variants and questions from the same family are correlated. Consequently this report uses descriptive comparisons and exact denominators, not row-wise confidence intervals or significance claims.

## 1. Familiar wording is easy to decode, but simple baselines are strong

All 14 familiar-wording graded properties have ridge correlations between 0.935 and 0.985. This shows that a linear readout can recover their ordering in this controlled setting. However, the average static-embedding correlation is already 0.930, and the unigram TF-IDF correlation is 0.922. Average TF-IDF MAE is 0.088, better than the hidden-state ridge MAE of 0.121. The descriptions contain highly informative property words, so high performance alone does not demonstrate an abstract internal axis.

Paraphrase exposes a different picture. Size correlation falls from 0.985 to 0.295; pitch from 0.935 to 0.320; hardness from 0.935 to 0.148. Mass and pleasantness preserve very strong ordering. Arousal retains a correlation of 0.788 but its MAE reaches 1.000 on the 0–1 target scale: substantial ordering information can coexist with a severely shifted numerical prediction. Ridge outputs are unconstrained, so errors exceeding the target range are possible and are not clipped in this report.

The first two figures therefore show correlation and error separately, with all 14 properties included. Each point represents ten test rows from two entity families, not ten independent entities. The numerical wording variant covers only eight properties and is kept in the supporting data rather than pooled with the 14-property comparison.

## 2. Binding provides evidence beyond word presence, with limited transfer

Binding pairs use the same word inventory while changing which entity has the high versus low property. A bag-of-words representation is identical across a swap; a contextual representation can distinguish the assignments. In familiar wording, all 14 hidden-state probes achieve rho 0.873 and MAE between 0.077 and 0.213. Static averages give constant predictions with rho zero; TF-IDF constant predictions have undefined rho, which must not be displayed as a measured zero. Both baselines have MAE 0.5.

The repeated rho of 0.873 is explained by tied binary targets and eight rows per property, not by a universal geometric constant. The more direct paired check shows positive predicted changes for every familiar-wording binding swap. Under paraphrase the equal-property mean fraction of correctly signed swaps falls to 94.6%, and absolute errors increase. Size falls to rho 0.436; hardness to 0.546. These are real successes and failures within this fixture, supported by only one test family.

Context tracking is also imperfect when the intended property stays fixed. Averaging property-specific paired absolute changes, clause reordering moves predictions by 0.138 with familiar wording and 0.241 with paraphrase. An irrelevant change moves them by 0.064 and 0.144 respectively. For comparison, a genuine adjacent level increase changes the familiar-wording prediction by 0.201 on average. These quantities use the same 0–1 prediction scale, although the contrast families are not identical samples. Nuisance wording effects can be a substantial fraction of a meaningful level change. Signed averages would hide some of these effects by cancellation; the figure uses absolute changes.

## 3. Lexical generalization is uneven and not a validation against humans

For the separate lexical probes, familiar-wording test correlations are negative for speed (−0.775), loudness (−0.632) and pitch (−0.667). Size, mass, temperature, elasticity and arousal look more promising, but their denominators are small. A correlation of one for mass rests on only three held-out concepts.

Wording matters here too: pleasantness changes from 0.949 to −0.158, and speed from −0.775 to 0.258. This suggests the readout is sensitive to how a concept is elicited. It does not establish that the model lacks the concept; it establishes that this selected linear probe and dataset do not consistently generalize. The authored ratings also prevent treating these scores as a replication of a human-norm study.

## 4. Recoverable information does not guarantee a correct answer

The behavioral task scores the relative likelihood of two supplied continuations. This is a constrained comparison, not free generation or a broad reasoning benchmark. Natural answers and A/B answers initially have different entity coverage, so an unmatched overall comparison would be misleading.

The refined comparison matches property, entity, template, level and reversal. It contains six properties, one test family and four underlying comparisons per property per wording. Each natural continuation is matched to the mean accuracy over four counterbalanced A/B variants. Across these matched comparisons, familiar-wording accuracy is **87.5% for natural continuations versus 63.5% for A/B**; with paraphrase it is **75.0% versus 59.4%**. This is evidence of format sensitivity in these examples, not a general causal estimate of an A/B penalty. Danger, for example, slightly improves under A/B while pitch gets substantially worse.

Some natural-continuation tasks exhibit a fixed semantic preference: danger selects the higher alternative and roughness selects the lower alternative across the test questions. Balanced labels then yield 50% accuracy. A positive average correct-answer log margin can still coexist with chance accuracy because a few confident correct answers outweigh several weaker incorrect ones. Accuracy and margins must be considered together.

Numerical precision is another small but real limitation. Recomputing the single-token output-head margin changes its sign on 15 of 1,568 questions, including four of 352 single-token test questions. The largest discrepancy is 0.030 nats. Margins close to zero should not be treated as robust preferences. The other 320 questions have multi-token continuations and do not receive the same single-token head check. This is not a full FP32 rerun of the model.

## 5. Numerical ordering is not an accurate physical measurement scale

Separate physical-value probes were trained only on eligible stated training values, then evaluated on excluded numerical challenges. The comparison baseline fits simple features extracted from numerical text; it is not a full unit-aware calculator. To compare properties without mixing grams, centimetres and other units, the refined figure divides each MAE by that property's physical training range. These are descriptive range-normalized errors, not percentage error relative to the true value.

The hidden-state probe loses to this baseline on six of eight properties: brightness, elasticity, loudness, mass, pitch and size. Its normalized MAE ranges from 0.157 for loudness to 0.648 for mass. It wins on speed (0.242 versus 0.649) and temperature (0.318 versus 0.390). The challenge mixture differs across properties; each has only one held-out family and 9–22 test cases. Supporting tables separate seen values, interpolation and extrapolation instead of claiming a universal numerical extrapolation result.

Equivalent-unit pairs reveal a partial success: the hidden-state probe is less sensitive to changing units than the numerical-text baseline in all five available unit tests. It is not invariant, however. Mean absolute prediction changes are about 1,182 canonical mass units, 176 pitch units, 20.2 size units, 10.7 speed units and 7.1 temperature units. Better unit robustness than a weak baseline does not imply accurate conversion or a linear physical magnitude code.

## 6. Multi-attribute generalization is fragile

The factorial experiment independently varies attributes in seven domains and fits joint reduced-rank readouts. This tests whether multiple outputs remain recoverable when attributes appear together, including combinations excluded from training. It is a stronger challenge than a single adjective varied in a fixed template.

Even familiar-wording test results are mixed. Speed in the motion/thermal domain has rho 0.945 and MAE 0.051, whereas the colour domain has weak saturation and value ordering (rho 0.074 and 0.092). Shape ordering looks strong but calibration is poor: length rho is 0.941 with MAE 1.552; width rho is 0.907 with MAE 1.350. Those are errors on the constructed 0–1 scale, not centimetres.

The figure changes wording while holding the combination category fixed at combinations eligible for training, and always evaluates a held-out entity. The supporting table additionally displays withheld combinations and the joint challenge. This prevents attributing every failure to combination novelty. Under the joint new-entity, new-wording, new-combination challenge, pleasantness retains rho 0.894 but MAE is 0.859, and elasticity retains rho 0.949 but MAE is 1.319. Predictions can preserve an ordering while being unusable as calibrated coordinates.

The most difficult partitions contain only two, five or ten rows per output from one entity family. Extreme negative R² values are unstable with such small target sets; MAE and exact counts are more informative. Predictive cross-property responses are observational readout diagnostics, not causal evidence that one property changes another inside the model.

## 7. Hue contains partial structure, not a demonstrated robust circle

Hue uses coupled sine/cosine outputs so that angles around the wrap point are treated as neighbours. At the validation-selected block 16 final-token site, test circular MAE is 56.3° for numerical hue descriptions and 71.1° for colour-word descriptions. On factorial colour cases it ranges from 70.5° to 91.5° across the entity/wording/combination partitions. A uniform random predicted angle would have expected absolute circular error 90°, but that is a reference calculation, not a fitted or significance-tested null for this dataset.

The separate neighbourhood diagnostic is also mixed. At that same selected site, the intended near hue is closer than the far hue in 19 of 24 numerical triplets (79.2%), but only 12 of 24 colour-word triplets (50%). Both sets contain two test families. This supports some numerical hue structure while falling short of a stable circular representation across forms of description.

There is a specific limitation in the subspace analysis: the six eligible training contrasts for hue are 0°/360° invariance pairs. Their differences describe variation between equivalent descriptions, not directions that move through hue. A six-dimensional SVD fit to these contrasts cannot estimate the intrinsic dimensionality of hue, and its poor prediction should not be interpreted as evidence against hue information generally.

## 8. What can be said about dimensions and geometry

Predictive rank, local direction alignment and intrinsic semantic dimensionality are different questions. Thirteen of the fourteen scalar controlled properties select contrast-SVD rank 16, the configured ceiling; elasticity selects eight. This means larger tested predictive subspaces were often useful within this search. It does not mean that those concepts have exactly sixteen dimensions. The contrast bank also mixes property-changing contrasts with nuisance or invariance contrasts, so its span is not a purified semantic subspace.

Matched local directions vary across ranges and templates. For size, adjacent familiar-wording training directions can have negative cosine similarity; one inspected comparison is approximately −0.807. This example is a diagnostic, not a selected headline test statistic. A single global straight direction is therefore not established. Bootstrap subspace stability is a training diagnostic: repeated samples of a small set of families do not establish generalization to a broad population. Some local fits have six families and others only two; rank and sampling coverage must be read alongside overlap scores.

Joint reduced-rank models select ranks one to three depending on the domain. These ranks reflect predictive compression and validation performance, not counts of psychological dimensions. In the colour model, sine/cosine outputs range from −1 to 1 while saturation/value use 0–1, so the joint squared-error objective also weights outputs unequally by scale. No causal patches or nonlinear exploratory models were executed in this profile. Neither causal necessity nor absence of nonlinear information can be inferred.

## Relation to the original pilot

The original size pilot selected block 23. Its ridge test correlation was 0.735 for seen templates and 0.721 for held-out templates, with broad recorded intervals (about 0.268–0.950 and 0.269–0.938). V3 greatly broadens the diagnostic questions: multiple properties, binding, numerical values, hue and factorial combinations.

It would be misleading to interpret V3's larger familiar-template correlations as an improvement over that pilot. The data, label scales, models of evaluation and test sizes differ. They are not paired before/after measurements. The useful progression is methodological: V3 reveals which easy successes survive stronger controls, rather than supplying a single replacement score. Earlier pilot data, runs and caches were preserved; the recorded preservation check is a metadata audit, not a before/after content-hash proof.

## Methodology, validation and reproducibility

The report reads the completed laptop CSV/JSON artifacts and joins question metadata by unique comparison ID. It does not call the model, refit probes or alter test selection. The companion script `analysis/v3_findings.py` regenerates the supporting tables. `provenance.json` records hashes of consumed result files; `datasets.json` contains the reviewed report data. Original `source_snapshot.json`, `model_selection.json`, `physical_model_selection.json`, fitted coefficients and run identity retain the experimental audit trail.

Probe fitting separates entity groups. Regularization uses grouped training cross-validation with preprocessing fitted inside folds; layer/readout choices use validation entities and training templates/combinations. Scalar selection prioritizes ordering, while physical prediction uses squared error, which helps explain why strong rank scores need not yield good numerical calibration. Seven blocks, two readouts and multiple regularization/rank settings constitute a substantial search relative to the small validation sets. A single fixed split and validation reuse limit certainty even without test-label leakage.

This interpretation checks exact case/question coverage, one-to-one metadata joins, four A/B counterbalances for every matched comparison, physical training-range denominators, finite exported values and test-family counts. It also recomputes scalar ridge metrics from saved predictions and reconciles behavioral accuracy with the run summaries. Undefined correlations remain missing. No rows are treated as independent merely because their sentences differ. The report's visuals display all relevant properties rather than a favorable subset, separate ordering from error, and show matched formats and comparable error units.

## Recommended next experiment

1. **Broaden independent coverage first.** Add genuinely varied entity families, independently authored templates and multiple prespecified split seeds. More permutations of the current wording are less valuable than new linguistic constructions. Keep a final test bank untouched by development.
2. **Use binding as the primary positive result to replicate.** Compare contextual probes with bag-of-words and order-sensitive text baselines on new assignments and held-out syntax. Report paired sign accuracy and calibrated MAE, with uncertainty clustered at the entity/template level.
3. **Separate prediction objectives.** Select rank-focused and calibration-focused probes independently on validation data. Report both on the same final tests. Preserve unclipped predictions so offsets and scale failures remain visible.
4. **Repair the geometry diagnostics before stronger claims.** Build hue-changing training contrasts; keep invariant contrasts as controls. Separate semantic from nuisance contrast banks. Examine rank curves beyond the current ceiling only with sufficient independent data, rather than interpreting a boundary optimum as a dimension count.
5. **Strengthen numerical and behavioral controls.** Add a unit-aware parsing baseline; balance challenge roles; retain matched natural/A/B questions. Audit near-zero margins and use a stronger precision check for any conclusion dependent on a few answers.
6. **Gate interventions on replicated predictive evidence.** Once a stable direction/subspace survives new templates, test targeted edits against random-direction, norm-matched and unrelated-property controls. Until then, use “decodable” or “predictive,” not “causal mechanism.”

## Further questions

Does the binding advantage persist with several independent sentence constructions rather than one held-out paraphrase? Is the paraphrase failure mainly an offset/scale shift that a validation-only calibration can correct, or does ordering itself fail? Can physical-value prediction beat a competent unit-aware baseline? Do independently constructed hue-changing contrasts yield a transferable circular subspace? Does any replicated direction selectively change comparative answers when intervened on?

**Dissertation-ready conclusion:** In this bounded, synthetic screening experiment, contextual hidden states supported linear decoding of many explicitly stated qualities and distinguished entity–property assignments beyond word-inventory baselines. However, template sensitivity, numerical calibration errors, mixed lexical transfer and weak multi-attribute generalization limit claims of stable, domain-general quality geometry. The findings motivate targeted replication and better-controlled geometric and causal tests; they do not establish universal semantic axes or causal mechanisms.
