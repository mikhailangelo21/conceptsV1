"""Auditable Markdown report and standalone scientific figures, no model calls."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from .geometry import Geometry, sha256_file
from .util import atomic_json, read_json

BLUE, ORANGE, INK = "#2563A6", "#C66B2B", "#26313B"


def _table(frame):
    if frame.empty:
        return "No eligible observations."
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return "—" if not np.isfinite(value) else f"{value:.3f}"
        return str(value)
    return "| " + " | ".join(frame.columns) + " |\n| " + " | ".join(["---"] * len(frame.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))


def geometry_report(run_dir):
    run = Path(run_dir)
    identity = read_json(run / "identity.json")
    ref = read_json(run / "metric_reference.json")
    geometry = Geometry.load(ref["root"], ref["identity"])
    metrics = pd.read_csv(run / "metrics.csv")
    output = pd.read_csv(run / "output_alignment.csv")
    counts = pd.read_csv(run / "output_pair_counts.csv")
    site = read_json(run / "site_verification.json")
    plots = run / "plots"
    plots.mkdir(exist_ok=True)
    chart_map = []
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
    # Same-site comparisons: two distinct metrics; never collapse ranking and calibration.
    pair = metrics[metrics.method.isin(["ridge_euclidean", "ridge_covariance"])]
    compare = pair.pivot(index=["quality", "partition", "n_rows", "n_groups"], columns="method", values=["rho", "mae"])
    compare.columns = [f"{a}_{b.removeprefix('ridge_')}" for a, b in compare.columns]
    compare = compare.reset_index()
    compare["rho_delta"] = compare.rho_covariance - compare.rho_euclidean
    compare["mae_delta"] = compare.mae_covariance - compare.mae_euclidean
    compare.to_csv(run / "ridge_comparison.csv", index=False)
    headline = compare[compare.partition.isin(["entity_test|graded|plain", "entity_test+template|graded|paraphrase"])]
    summary = headline.groupby("partition").agg(n_qualities=("quality", "size"), n_rows_per_quality_min=("n_rows", "min"),
               n_rows_per_quality_max=("n_rows", "max"), n_families_min=("n_groups", "min"), n_families_max=("n_groups", "max"),
               rho_euclidean=("rho_euclidean", "mean"), rho_covariance=("rho_covariance", "mean"),
               mae_euclidean=("mae_euclidean", "mean"), mae_covariance=("mae_covariance", "mean"),
               rho_improved=("rho_delta", lambda x: int((x > 1e-6).sum())), rho_weakened=("rho_delta", lambda x: int((x < -1e-6).sum())),
               rho_unchanged=("rho_delta", lambda x: int((x.abs() <= 1e-6).sum())),
               mae_improved=("mae_delta", lambda x: int((x < -1e-6).sum())), mae_weakened=("mae_delta", lambda x: int((x > 1e-6).sum())),
               mae_unchanged=("mae_delta", lambda x: int((x.abs() <= 1e-6).sum()))).reset_index()
    summary.to_csv(run / "ridge_summary.csv", index=False)
    if not headline.empty:
        qualities = sorted(headline.quality.unique())
        fig, axes = plt.subplots(1, 2, figsize=(13, max(4.5, len(qualities) * .37 + 2)), layout="constrained")
        for ax, field, title in zip(axes, ["rho_delta", "mae_delta"], ["Change in ranking correlation", "Change in absolute prediction error"]):
            for offset, partition, label, color, marker in [(-.13, "entity_test|graded|plain", "Familiar wording", BLUE, "o"),
                    (.13, "entity_test+template|graded|paraphrase", "Held-out wording", ORANGE, "s")]:
                s = headline[headline.partition == partition].set_index("quality").reindex(qualities)
                ax.scatter(s[field], np.arange(len(qualities)) + offset, color=color, marker=marker, label=label, s=35)
            ax.axvline(0, color=INK, linewidth=1)
            ax.set_yticks(range(len(qualities)), qualities)
            ax.invert_yaxis()
            ax.grid(axis="y", alpha=.15)
            ax.set_title(title)
            ax.set_xlabel("Covariance − Euclidean; right is better" if field == "rho_delta" else "Covariance − Euclidean; left is better")
        axes[0].legend(loc="best", frameon=False)
        fig.suptitle(f"Same-site ridge comparison · {identity['profile']} profile\nEach point is one quality; {int(headline.n_groups.min())}–{int(headline.n_groups.max())} test families per point", fontsize=13)
        fig.savefig(plots / "ridge_changes.png", dpi=170)
        plt.close(fig)
        chart_map.append(dict(file="plots/ridge_changes.png", question="Does transformed ridge improve ranking or calibration?",
                               type="faceted signed dot plot", source="ridge_comparison.csv", palette="two roots, circle/square; zero reference"))
    with np.load(run / "output_directions.npz", allow_pickle=False) as a:
        directions = {k: a[k] for k in a.files}
    qualities = sorted({k.split("__")[0] for k in directions})
    heatmap_rows = []
    if qualities:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
        cmap = LinearSegmentedColormap.from_list("white_blue", ["#FFFFFF", BLUE])
        for ax, name in zip(axes, ["euclidean", "covariance"]):
            d = np.stack([directions[f"{q}__{name}"] for q in qualities])
            td = d if name == "euclidean" else geometry.output(d)
            td /= np.linalg.norm(td, axis=1)[:, None]
            similarity = np.abs(td @ td.T)
            im = ax.imshow(similarity, vmin=0, vmax=1, cmap=cmap)
            ax.set_xticks(range(len(qualities)), qualities, rotation=55, ha="right")
            ax.set_yticks(range(len(qualities)), qualities)
            ax.set_title(name.capitalize())
            for i, qa in enumerate(qualities):
                for j, qb in enumerate(qualities):
                    heatmap_rows.append(dict(geometry=name, quality_a=qa, quality_b=qb, absolute_cosine=similarity[i, j]))
                    if len(qualities) <= 10:
                        ax.text(j, i, f"{similarity[i,j]:.2f}", ha="center", va="center",
                                color="white" if similarity[i,j] > .6 else INK)
        fig.colorbar(im, ax=axes, shrink=.65, label="Absolute cosine: 0 perpendicular · 1 aligned")
        fig.suptitle("Similarity of training output-word directions\nEqual scales; perpendicularity alone does not establish causal separation", fontsize=13)
        fig.savefig(plots / "output_direction_similarity.png", dpi=170)
        plt.close(fig)
        chart_map.append(dict(file="plots/output_direction_similarity.png", question="How do relationships between output directions change?",
                               type="paired heatmaps, shared 0–1 scale", source="output_direction_similarity.csv", palette="single blue root"))
    pd.DataFrame(heatmap_rows).to_csv(run / "output_direction_similarity.csv", index=False)
    output_summary = output.groupby(["quality", "geometry", "evaluation"]).agg(n_pairs=("cosine", "size"),
        mean_cosine=("cosine", "mean"), mean_projection=("projection", "mean"),
        positive_fraction=("projection", lambda s: float((s > 0).mean()))).reset_index()
    output_summary.to_csv(run / "output_alignment_summary.csv", index=False)
    heldout = output_summary[output_summary.evaluation == "test"].merge(
        counts[counts.split == "test"][["quality", "independent_concepts"]], on="quality", validate="many_to_one")
    if not heldout.empty:
        q = sorted(heldout.quality.unique())
        fig, ax = plt.subplots(figsize=(10, max(2.6, len(q) * .48 + 1.8)), layout="constrained")
        for shift, name, color, marker in [(-.13, "euclidean", BLUE, "o"), (.13, "covariance", ORANGE, "s")]:
            s = heldout[heldout.geometry == name].set_index("quality").reindex(q)
            ax.scatter(s.mean_cosine, np.arange(len(q)) + shift, color=color, marker=marker, label=name.capitalize())
            for i, quality in enumerate(q):
                random = output_summary[(output_summary.quality == quality) & (output_summary.geometry == name) &
                                        (output_summary.evaluation == "uniform_random_output_pairs")]
                if len(random):
                    ax.scatter(random.mean_cosine.iloc[0], i + shift, color=color, marker="x", s=35)
        labels = []
        for quality in q:
            r = heldout[heldout.quality == quality].iloc[0]
            labels.append(f"{quality}  ({int(r.n_pairs)} pairs / {int(r.independent_concepts)} concepts)")
        ax.set_yticks(range(len(q)), labels)
        ax.invert_yaxis()
        ax.set_ylim(len(q) - .5, -.5)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Mean cosine to training direction · × marks random-pair mean")
        ax.legend(frameon=False)
        ax.set_title("Held-out output-word alignment\nOriginal test concepts; shared nouns make pairs dependent", pad=14)
        fig.savefig(plots / "output_heldout_alignment.png", dpi=170)
        plt.close(fig)
        chart_map.append(dict(file="plots/output_heldout_alignment.png", type="paired dot plot", question="Do unseen word pairs align with training directions?",
                               source="output_alignment_summary.csv + output_pair_counts.csv", palette="two roots, circle/square, random crosses"))
    alignment_path = run / "context_alignment.csv"
    context = pd.read_csv(alignment_path)
    if not context.empty:
        context["abs_projection"] = context.calibrated_delta.abs()
        keys = ["quality_id", "geometry", "kind", "entity_split", "template_id", "template_split"]
        family = context.groupby(keys + ["entity_group_id"], dropna=False).agg(cosine=("cosine", "mean"),
                    signed_change=("calibrated_delta", "mean"), absolute_change=("abs_projection", "mean"), n_pairs=("contrast_id", "size")).reset_index()
        cs = family.groupby(keys, dropna=False).agg(mean_cosine=("cosine", "mean"), signed_change=("signed_change", "mean"),
                    absolute_change=("absolute_change", "mean"), n_pairs=("n_pairs", "sum"), n_groups=("entity_group_id", "nunique")).reset_index()
        cs.to_csv(run / "context_alignment_summary.csv", index=False)
    else:
        cs = pd.DataFrame()
    # Machine-readable matched change comparisons include nuisance, binding and off-targets.
    paired = pd.read_csv(run / "paired_summary.csv")
    pc = paired[paired.method.isin(["context_projection_euclidean", "context_projection_covariance"])]
    keys = ["quality_id", "measured_quality", "kind", "entity_split", "template_id", "template_split", "combination_split", "on_target", "n_pairs", "n_groups"]
    pc = pc.pivot(index=keys, columns="method", values=["signed_delta", "absolute_delta", "positive_fraction"])
    pc.columns = [f"{a}_{b.removeprefix('context_projection_')}" for a, b in pc.columns]
    pc = pc.reset_index()
    pc.to_csv(run / "context_selectivity_comparison.csv", index=False)
    atomic_json(run / "chart_map.json", chart_map)
    vocab = ref["identity"]["vocabulary"]
    d = geometry.diagnostics
    testcounts = counts[counts.split == "test"][["quality", "eligible_pairs", "independent_concepts", "training_pairs"]]
    nuisance = pc[(pc.entity_split == "test") & pc.kind.isin(["binding_increase", "clause_order_invariant", "irrelevant_invariant", "factorial_one_axis_increase"])]
    ns = nuisance.groupby(["kind", "template_id", "on_target"]).agg(n_quality_comparisons=("quality_id", "size"),
         min_families=("n_groups", "min"), max_families=("n_groups", "max"),
         signed_euclidean=("signed_delta_euclidean", "mean"), signed_covariance=("signed_delta_covariance", "mean"),
         absolute_euclidean=("absolute_delta_euclidean", "mean"), absolute_covariance=("absolute_delta_covariance", "mean")).reset_index()
    ns.to_csv(run / "selectivity_summary.csv", index=False)
    numerical = read_json(run / "analysis_complete.json")
    historical_transport = read_json(run / "historical_coefficient_transport.json")
    # Renderer provenance is separate from the immutable source snapshot used for fitting.
    atomic_json(run / "rendering_source.json", {"file": __file__, "sha256": sha256_file(__file__),
                "source": Path(__file__).read_text(), "changes_analysis": False})
    text = f"""# Covariance geometry extension of V3

This is a paired comparison at the **final post-normalization, final-prompt-token state** of the pinned Qwen model. It tests a different geometry and ridge penalty, not whether the model acquired new information. Profile: **{identity['profile']}**. Covariance vocabulary: **{vocab['sampling']}**, {vocab['selected_rows']:,} of {vocab['output_rows']:,} output rows.

{'This smoke run verifies integration only; its vocabulary subsample and small profile are not the full scientific result.' if identity['profile'] == 'smoke' or identity['vocabulary'] != 'full' else 'This full-vocabulary laptop comparison reuses all 6,322 V3 cases and the original family/wording/combination splits.'}

## Same-site ridge transfer

{_table(summary)}

Correlation measures ordering; MAE measures calibration in authored ordinal-code units. “Improved”, “weakened” and “unchanged” are descriptive per-quality changes (tolerance 1e-6), not significance tests. Macro means weight qualities equally; repeated prompt rows do not increase the number of independent families. The transformed ridge fits use training-mean centering only. Their penalty differs from the Euclidean fit. V3's feature-standardized ridge is also fitted at this same site in `metrics.csv`; the original selected-block/entity results and all saved baseline metrics remain in `v3_historical_control/` and are not treated as a paired geometry effect.

![Ridge changes](plots/ridge_changes.png)

## Output-word consistency

{_table(heldout[['quality','geometry','n_pairs','independent_concepts','mean_cosine','positive_fraction']] if not heldout.empty else heldout)}

{_table(testcounts)}

Pairs come from the existing noun lexicon and provisional low/high typicality labels. Different objects also differ in category, identity, frequency and other properties. They are candidate associations, not isolated concept interventions. Zero eligible test pairs means no test evidence for that quality. Every excluded multi-token completion and its exact formatting is recorded in `output_pair_candidates.json`. No multi-token embedding was averaged. Training-pair leave-one-out retains shared concepts and is labelled separately from original test-split evaluation. Random pairs use {identity['config']['output_pairs']['random_pairs']} uniform full-head row pairs per quality (special/unassigned rows included).

![Output alignment](plots/output_heldout_alignment.png)

![Direction similarity](plots/output_direction_similarity.png)

Only the normalization of an output direction changes across metrics. Context measurement remains lambdaᵀ gamma_bar. Its sign/ranking and train-affine-calibrated predictions therefore remain unchanged up to rounding; a different normalization is not improved information. Smaller cross-quality cosine alone does not establish causal separability.

## Binding, nuisance edits and factorial selectivity

{_table(ns)}

These are macro summaries of context-projection changes after training-only affine calibration. Positive binding/target changes are expected; small absolute reorder, irrelevant-edit and off-target changes are preferred. On-target magnitude alone does not imply better selectivity. The detailed `context_selectivity_comparison.csv` retains each quality pair, split, family count and matched-pair count. `context_alignment_summary.csv` separately records geometric alignment; `paired_summary.csv` also contains ridge and output-measurement changes. One or two held-out families in many cells give limited evidence. Changing one authored sentence property need not leave every off-target model concept fixed.

## Numerical and implementation checks

- Population covariance accumulated in float64 chunks from `get_output_embeddings()`; full head includes {len(vocab['special_ids'])} special IDs and {len(vocab['unassigned_rows'])} rows not assigned by the tokenizer.
- Raw eigenvalues {d['raw_min']:.6g} to {d['raw_max']:.6g}; condition number {d['raw_condition']}. Effective formula: `{d['formula']}`; loading {d['loading']:.6g}.
- Directly verified {len(site['examples'])} training examples against all output logits. Maximum native-logit reconstruction error: {max(r['native_logit_max_error'] for r in site['examples']):.6g}; declared atol {site['tolerances']['native_logits_atol']}, rtol {site['tolerances']['native_logits_rtol']}. Full checks are in `site_verification.json`.
- Transporting fitted raw and standardized-ridge coefficients preserves predictions; maximum absolute error {numerical['max_coefficient_transport_error']:.3g}. This checks the same predictor, separately from refitting ridge in new coordinates.
- The {len(historical_transport)} actual saved V3 probe fits also passed coefficient transport (maximum error {max(r['max_absolute_error'] for r in historical_transport):.3g}). At their original earlier/entity sites this is an algebra check only, not a claim that the paper's metric applies there.
- Final states, source shard hashes, source snapshot hashes, metric identity, settings and source code are recorded. Original V3 caches/runs are read-only inputs.
- Hue uses sine/cosine regressions and angular error, plus circular near/far neighbourhoods in `hue_neighborhoods.csv`; it is excluded from scalar low→high directions. Historical numeric-value and behavioural results are retained, not rerun or relabelled as new geometry findings.
- No steering was run. The optional path is disabled unless measurement review is explicitly acknowledged; it edits the final state immediately at the output head.

## Scope

All labels are authored fixtures, with small independent test sets and confounded lexical comparisons. Results can support or weaken the usefulness of this geometry for these measurements. They do not establish causal quality axes, universal representations or Gärdenfors-style dimensions. See `docs/QUALITY_GEOMETRY_V1.md` in the project for equations, commands, cache rules and exact analysis choices.

References: [Park, Choe and Veitch, §§2–4](https://arxiv.org/html/2311.03658v2); [reference matrix construction](https://github.com/KihoPark/linear_rep_geometry/blob/main/store_matrices.py); [reference analyses](https://github.com/KihoPark/linear_rep_geometry/blob/main/linear_rep_geometry.py).
"""
    (run / "report.md").write_text(text)
    return run / "report.md"
