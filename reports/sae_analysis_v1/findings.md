# Qwen-Scope SAE: quality structure and compression

## Technical summary

The collected Top-100 SAE dataset is partial across 28 requested layers: 9 layers have complete main and trace sparse codes, and 1 is partly encoded. This core analysis uses original residuals H and sparse codes F at available layers. Across 234 matched layer/readout/task evaluations from 14 quality tasks with at least 10 test records and two entity families, median rank correlation is 0.63 for H and 0.40 for F. The median within-pair H−F difference is 0.16, and H is higher in 78% of these comparisons. These evaluations reuse prompts and mostly have only two independent test families; they are not independent replications. The estimates describe this fixture bank, not human perceptual dimensions or causal model mechanisms.

## What was actually analysed

A unique vector is one text and token position. Case/readout records can share that vector, so PCA uses vectors once; prediction retains their labels. The zero-based SAE layer corresponds to the preceding one-based transformer block. The chart shows encoded main rows against the 14,500 requested readouts at each layer.

## Quality information and compression

The held-out ordering comparison uses probes fitted only on train entities with familiar wording and eligible factorial combinations. Validation selects ridge strength; test entities remain frozen. H is the original residual; F is the sparse SAE code. A compact PCA view is a train-fitted compression. At 16 PCs, the median within-task test rank-correlation drop is 0.31 for H and 0.20 for F, despite the leading components explaining substantial training variance. This measures information retained for this probe, not intrinsic quality dimensionality. The chart uses only tasks with at least ten test records and two entity families; all raw task results remain in the CSV.

## The wording itself is predictive

A word and two-word prompt-text baseline and a prompt-length baseline were fitted on the same eligible training readouts and scored on the same familiar-wording test tasks. At layer 11, compare them with H and F on matching entity and final-token cohorts. Prompt text can directly name the authored quality level, so its predictive score measures a lexical shortcut, not a latent representation. This control makes it harder to attribute all probe success to a conceptual quality axis.

## Wording transfer and contrast selectivity weaken

On the same eligible test tasks, changing from familiar to withheld wording reduces median rank correlation from 0.64 to 0.24 for H and from 0.39 to 0.10 for F. These are 225 matched layer/readout/task comparisons, not independent families. A mean low-to-high SAE contrast direction correctly signs a median 81% of training contrasts and 69% of test contrasts across the representative layers. Held-out factorial cells often have only one family, so they cannot establish robust independent factor geometry.

## Joint qualities are not cleanly independent here

In layer 11 final-token SAE codes, the median share of held-out one-factor contrasts whose target projection exceeds every off-target projection is 36% across 17 domain/quality summaries. Each summary has one test entity family. These fitted directions are not orthogonal by design, and these data do not support a strong claim that factorial qualities separate cleanly. Behavioural comparison margins were also linked to feature norm and reconstruction error by natural-continuation versus A/B format; those associations are descriptive and can reflect wording or difficulty.

## Geometry has to survive quantitative checks

Centered PCA measures training variance; its axes are not automatically semantic. The embedding comparison puts local neighbourhood trustworthiness beside sampled global distance-rank agreement. A visually clear UMAP or t-SNE layout alone cannot establish a manifold or a circular hue representation.

## Dimensionality estimates depend on neighbourhood scale

On bounded original-space cohorts, median TwoNN estimates are 2.63 for H and 3.41 for F; k-neighbour MLE at k=20 gives 5.06 and 6.58 respectively. The gap between methods and the local-PCA sensitivity to neighbourhood size are reasons to report a range of dataset-specific complexity diagnostics rather than a single count of human quality dimensions. Exact duplicates were removed and zero-distance exclusions are recorded in the source table.

## Hue is only partly ordered, not established as a circle

In the prespecified near/far hue triplets, layer 11 final-token F places the authored near colour closer than the far one in 59% of 240 triplets. Triplets reuse a small set of entity families; the validation and test rates are 48% and 69% over 48 and 48 triplets respectively. Angular prediction errors also vary greatly across layer and readout. This supports some local hue ordering in the constructed examples, but does not establish a stable circular manifold.

## Scope and limits

Labels are author-written design codes, not human ratings. The original V3 split and wording restrictions are kept for held-out probes; all-splits t-SNE is descriptive only. SAE feature IDs are checkpoint-local. The dataset has sparse codes on nine complete layers, one partial layer, and 18 layers without sparse codes. Median SAE relative reconstruction error is 0.37 at the entity readout and 0.23 at the final-token readout across complete layers. Some test conditions contain one or two independent entity families; their rows do not provide equally many independent replicates. Numeric challenges and physical values were withheld from ordinal fitting, so this report does not assert physical-unit calibration. No causal intervention or complete 28-layer SAE comparison is implied.

## Next checks

Inspect the per-quality and per-family CSVs before assigning a meaning to any SAE feature. Extend the remaining checkpoints only after adding storage, then rerun this analysis as a new versioned result. Independent human or external stimuli would be needed to test whether the constructed axes generalize beyond these prompts.
