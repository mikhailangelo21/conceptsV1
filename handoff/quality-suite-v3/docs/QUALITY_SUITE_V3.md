# Quality suite v3: design and interpretation

This is a reusable bank of ORIGINAL EXAMPLE DATA, not human norms, measurements,
model outputs, or evidence that every named attribute is a Gärdenforsian quality.
It extends the pilot while keeping typical knowledge, described properties,
referent binding, and causal use as separate questions.

## Labels and units

- `lexical`: author-proposed ordinary rankings, labeled `demo_ordinal`. All are
  provisional; inspect ambiguous, state-dependent, metaphorical and explicitly
  adjective-bearing entries. Do not cite them as independent ground truth.
- `graded`, `controls`, `binding`: the scene specifies an ordered state. Targets
  are DESIGN CODES 0, .25, .5, .75, 1. They do not mean equal psychological gaps.
  Word descriptions do not specify exact centimetres, temperatures, or masses.
  Only a numeric statement populates `specified_values` with physical values.
- `numeric`: the target key ends with `__value`, in the canonical physical unit.
  These are fictional stated measurements, not observations. Keep these targets
  separate from the normalized ordinal design codes. Unit-conversion statements
  may be rounded to eight significant digits; interpret equality to that precision.
- `factorial`: a complete grid of assigned attributes. Normalized design targets
  coexist with explicit physical values. The rectangular-block volume is derived
  from length, width and height, not an additional independent quality dimension.
- `hue`: standard HSV convention, not a perceptually uniform human color space.
  Predict sine/cosine or use circular distance. Two Cartesian output coordinates
  describe a one-dimensional circle; they do not establish two independent qualities.
  Color saturation/value are kept positive to avoid undefined hue at gray/black.
- Brightness versus luminance, loudness versus SPL, and elastic recovery versus
  elastic modulus are different constructs. The operational definitions in
  `qualities.json` matter. Affective/evaluative attributes specify a fictional
  observer or situation; they are not objective ratings or actual participant data.

Synthetic text with declared numbers tests whether a model represents and binds
stated attributes. It does not by itself show spontaneous world knowledge,
perceptual grounding, or a privileged semantic geometry. Report word-only and
numeric conditions separately, as well as transfer between them.

Fit inexpensive baselines in the same training folds: character/token counts,
unigram TF-IDF, averaged static input embeddings, and numeric-literal features for
numeric conditions. Explicit adjectives and numbers make many ordinary rows easy.
The same-word-inventory binding pairs are essential: a bag-of-words predictor
cannot distinguish their changed referent assignments. Compare context-free lexical
knowledge with context-bound tracking, rather than treating either as the other.

## Experimental families

1. Typical lexical associations, with noun-length and category controls.
2. Five ordered states of the same fictional item, across lexical/numeric forms.
3. Irrelevant additions, changes to another item, and explicitly denied foil claims.
4. Same-word-inventory role swaps, both clause orders and both queried entities.
5. Numeric interpolation, bounded range tests, selected extrapolation, equivalent units.
6. Independent factorial attributes: extent/mass; length/width/height;
   roughness/hardness/recovery; pitch/SPL; pleasantness/arousal/danger;
   hue/saturation/value; speed/temperature.
7. Circular hue neighborhoods and the equivalence of 0 and 360 degrees.
8. Natural-completion behavioral questions plus independent A/B meaning and
   printed-order counterbalancing. Reversing the queried pair reverses the truth.

## Splits and independent units

All variants of a synthetic entity family remain in one entity split, including
its companion item. The identities are invented names: held-out synthetic names
are not equivalent to held-out real-world object categories or people.
Plain/numeric wording is available for fitting; paraphrase/reordered wording is
reserved. Validate settings on validation ENTITIES using training wording first.
For fitting factorial models, also require `combination_split == 'train'`.
Report excluded entities, excluded wording, and excluded combinations separately
and jointly. None is a random prompt-row split.

Numeric sweep files are challenge-only; do not include them in fitting because
their template happens to be `numeric`. Fit a numeric predictor on stated values
in training graded/factorial rows, not on their design-code labels, before evaluating
the `__value` challenge. Declare any log/unit transform and fit calibration on train.

Lexical groups merge identical normalized spellings only. Review synonyms and
near-duplicates before empirical work. Exact names detected in the old pilot's
concept CSV are assigned to training and marked as overlaps; this does not detect
all semantic overlap. These example holdouts are exploratory, not independently
collected confirmation. Template and entity files are transparent, not concealed.

Bootstrap complete entity/concept families, with their variants, and acknowledge
the much smaller count of independent constructed scenarios than prompt rows.
Do not claim thousands of natural observations from thousands of generated sentences.
Use independently collected, appropriately licensed norms or new ratings later.
The empty human-rating template contains no invented participants or ratings.

## Wider search: axes, subspaces, and local behavior

Start with one direction per quality, then examine the span of MATCHED training
contrast vectors. Use SVD, rank curves, bootstrap subspace stability and held-out
semantic performance to test candidate subspace ranks [1,2,3,4,6,8,12,16]. Limit PCA
or contrast-SVD rank by sample/feature rank. Reduced-rank regression additionally
has an output-rank limit; many settings are inapplicable in small domains.

PCA can be solved with SVD; ridge has a convex quadratic objective. Repeated starts
do not discover additional meaningful local optima for these estimators. With a
single scalar target, PCA followed by linear regression is still one overall linear
functional of the original activation. More retained PCs do not prove that a
quality has that many intrinsic dimensions.

For several components, use multi-output prediction on the factorial grids,
reduced-rank regression or a shared low-dimensional subspace with separate heads.
Compare a shared subspace against property-specific directions and cross-domain
transfer. Use raw or train-centered residuals; document any altered metric.
Compare subspaces by principal angles/projectors, not arbitrary rotated columns.

To investigate LOCAL structure, fit prespecified adjacent-range contrasts (0→1,
1→2, 2→3, 3→4) on training entities, and compare their alignment and transfer.
Also compare domains and wording families. This asks whether a global axis bends
or becomes context-dependent; it is distinct from an optimizer having local minima.
With only five qualitative levels, do not claim a finely resolved manifold.
Use the numeric sweeps as explicit, separate interpolation/range diagnostics.

Only after linear baselines, optionally try a small signed dictionary-learning
model or a small nonlinear probe on cached features. Use 5 random initializations
for exploration, up to 10 for a specified replication; save all outcomes, convergence
and held-out validation scores. Do not choose a restart using test labels. Lower
reconstruction loss alone does not establish semantic interpretability. NMF assumes
nonnegative inputs and is not a default method for signed residual activations.

Use grouped nested validation for ranks, layers, regularization and nonlinear
choices. Start at blocks [4,8,12,16,20,24,28] for the current 28-block model, then
refine nearby layers on development data only. Broad searching increases selection
opportunities; log all tried settings and reserve a new final confirmatory sample.
Prefer interval estimates and held-out predictive improvement. Any formal screen
across many qualities/layers must define its testing family and correction.

## Causal follow-up

First verify behavioral competence for EACH property/format using development
data. Binary A/B words and natural continuations have different biases; score both
candidate continuations exactly, including multi-token answers, with validated
token boundaries. No answers appear after the prefix in these files.

Test an entity position after all context and the final prompt position separately.
Preserve pre-final-normalization block hook semantics. An earlier-token intervention
after the last attention block cannot propagate to a later answer position.

After competence and numerical-resolution checks, test matched-state patches and
projection replacement before enlarging additive doses. For multiple qualities,
measure a cross-property intervention matrix: intervention on quality j versus
predictions/judgments for quality k, with other factorial attributes held fixed.
Check changes against baseline variability rather than assuming raw scales match.
Keep identity/tag controls, matched-norm random directions, A-bias decomposition,
and precision checks. Prediction, described-state tracking and selective causal
influence remain separate outcomes.

## Primary references

- Grand et al., semantic projection and human feature judgments:
  https://www.nature.com/articles/s41562-022-01316-8
  The article links collected data at https://osf.io/5r2sz/ . These examples DO NOT
  reproduce or substitute for those collected data; category rating scales may differ.
- Park et al., linear representation geometry:
  https://proceedings.mlr.press/v235/park24c.html
- PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- Ridge: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- NMF: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
- Activation patching: https://arxiv.org/abs/2404.15255
