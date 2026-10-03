"""Source-backed report, export figures and offline embedding explorer for SAE analysis."""
from __future__ import annotations

import html
import json
import re
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from plotly.offline import get_plotlyjs

from .sae_analysis_core import jsonl, read_collection


def _scalar(value, digits=2):
    if value is None or not np.isfinite(value):
        return "unavailable"
    return f"{value:.{digits}f}"


def _write_figure(fig, run, name):
    plots = run / "plots"
    plots.mkdir(exist_ok=True)
    fig.savefig(plots / f"{name}.svg", bbox_inches="tight")
    fig.savefig(plots / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _catalogue_and_components(run, collection):
    candidates = pd.read_csv(run / "features" / "candidates.csv")
    catalogues = []
    component_rows = []
    readouts = {r["readout_id"]: r for r in collection["readouts"]}
    for layer in sorted(candidates["layer"].unique()):
        for readout in ("entity", "final"):
            subset = candidates[(candidates["layer"] == layer) & (candidates["readout"] == readout)]
            if subset.empty:
                continue
            activity_path = run / "activity" / f"layer{layer}_{readout}.npz"
            with np.load(activity_path) as data:
                fires = data["fires"].copy()
                frequency = data["frequency"].copy()
                conditional = data["mean_when_active"].copy()
                variance = data["variance"].copy()
                error_active = data["mean_error_when_active"].copy()
            checkpoint = collection["repository"]["checkpoints"][str(layer)]["sha256"]
            for row in subset.itertuples():
                feature = int(row.feature_id)
                catalogues.append({**row._asdict(),
                                   "sae_revision": collection["repository"]["revision"],
                                   "checkpoint_sha256": checkpoint,
                                   "total_readout_fires": int(fires[feature]),
                                   "readout_frequency": float(frequency[feature]),
                                   "mean_when_active": float(conditional[feature]),
                                   "activation_variance": float(variance[feature]),
                                   "mean_reconstruction_error_when_active": float(error_active[feature])})
            path = run / "linear" / f"layer{layer}_{readout}_F_pca.npz"
            with np.load(path) as data:
                ids = data["ids"].tolist()
                components = data["components"][:5].copy()
                scores = data["coordinates"][:, :5].copy()
            meta = []
            for rid in ids:
                record = next((r for r in collection["by_readout"][rid] if r["kind"] == "case"), None)
                meta.append(record["metadata"] if record else {})
            lengths = np.asarray([readouts[rid]["token_count"] for rid in ids])
            train_wording = np.asarray([m.get("template_split") == "train" for m in meta])
            families = [m.get("entity_group_id", "") for m in meta]
            for pc in range(5):
                loading = components[pc]
                positive = np.argsort(loading)[-5:][::-1]
                negative = np.argsort(loading)[:5]
                association = []
                for quality in sorted({q for m in meta for q in m.get("targets", {})
                                       if not q.endswith("__value")}):
                    indices = [i for i, m in enumerate(meta) if quality in m.get("targets", {})]
                    if len(indices) < 10:
                        continue
                    values = np.asarray([meta[i]["targets"][quality] for i in indices], dtype=float)
                    if len(np.unique(values)) < 2:
                        continue
                    rho = spearmanr(scores[indices, pc], values).statistic
                    if np.isfinite(rho):
                        association.append((quality, float(rho), len(indices)))
                association.sort(key=lambda item: abs(item[1]), reverse=True)
                total = float(np.sum((scores[:, pc] - scores[:, pc].mean()) ** 2))
                between = sum(len(group) * (np.mean(scores[[i for i, family in enumerate(families)
                                                      if family == group], pc]) - scores[:, pc].mean()) ** 2
                              for group in set(families))
                component_rows.append({
                    "layer": layer, "readout": readout, "component": pc + 1,
                    "checkpoint_sha256": checkpoint,
                    "positive_feature_loadings": [{"feature_id": int(feature), "loading": float(loading[feature]),
                                                   "frequency": float(frequency[feature])} for feature in positive],
                    "negative_feature_loadings": [{"feature_id": int(feature), "loading": float(loading[feature]),
                                                   "frequency": float(frequency[feature])} for feature in negative],
                    "top_quality_associations_descriptive": association[:5],
                    "prompt_length_spearman": float(spearmanr(scores[:, pc], lengths).statistic),
                    "heldout_wording_minus_familiar_score_mean":
                        float(scores[~train_wording, pc].mean() - scores[train_wording, pc].mean())
                        if (~train_wording).any() and train_wording.any() else np.nan,
                    "entity_family_eta_squared": float(between / total) if total > 0 else np.nan,
                    "fit_mode": "PCA train-only; component/label associations descriptive across sampled splits"})
    pd.DataFrame(catalogues).drop(columns=["Index"], errors="ignore").to_csv(
        run / "features" / "catalogue.csv", index=False)
    pd.DataFrame(component_rows).to_csv(run / "linear" / "component_interpretation.csv", index=False)


def _figures(run, layers, pca, predictive, faithfulness):
    colors = {"H": "#2265a7", "F": "#bf5b3d"}
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(layers["layer"], layers["main_completed"], color="#2265a7", width=.75, label="Encoded")
    ax.plot(layers["layer"], layers["main_expected"], color="#252d37", lw=1.4, label="Requested")
    ax.set(xlabel="SAE layer, zero-based", ylabel="Unique main readouts", title="Sparse encoding coverage")
    ax.legend(frameon=False)
    _write_figure(fig, run, "coverage")

    subset = pca[(pca["readout"] == "final") & (pca["view"].isin(["H", "F"]))]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for view in ("H", "F"):
        row = subset[(subset["view"] == view) & (subset["layer"] == 11)]
        if len(row):
            path = run / "linear" / f"layer11_final_{view}_pca.npz"
            with np.load(path) as data:
                cumulative = np.cumsum(data["explained_variance_ratio"])
            ax.plot(np.arange(1, len(cumulative) + 1), cumulative, color=colors[view], label=view, lw=2)
    for threshold in (.5, .8, .9, .95):
        ax.axhline(threshold, color="#c1c7ce", lw=.7)
    ax.set(xlabel="Number of centered PCs", ylabel="Fraction of training variance",
           ylim=(0, 1.02), title="Centered PCA spectrum, final readout, layer 11")
    ax.legend(frameon=False)
    _write_figure(fig, run, "pca_spectrum")

    test = predictive[(predictive["partition"] == "test_familiar") &
                      (predictive["records"] >= 10) & (predictive["entity_families"] >= 2) &
                      (predictive["view"].isin(["H", "F", "H_PCA16", "F_PCA16"])) &
                      (predictive["status"] == "complete")]
    if len(test):
        display = test.groupby(["layer", "view"], as_index=False)["spearman"].median()
        fig, ax = plt.subplots(figsize=(9, 3.8))
        for view, style in (("H", "-"), ("F", "-"), ("H_PCA16", "--"), ("F_PCA16", "--")):
            d = display[display["view"] == view]
            if len(d):
                ax.plot(d["layer"], d["spearman"], marker="o", lw=1.6,
                        ls=style, color=colors[view[0]], label=view)
        ax.axhline(0, color="#252d37", lw=.8)
        ax.set(xlabel="SAE layer, zero-based", ylabel="Median test Spearman across available qualities",
               title="Held-out ordering on familiar wording")
        ax.legend(frameon=False, ncol=4)
        _write_figure(fig, run, "heldout_ordering")

    baseline_path = run / "baselines" / "metrics.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        control = pd.concat([test[(test["layer"] == 11) & (test["view"].isin(["H", "F"]))],
                             baseline[(baseline["partition"] == "test_familiar") &
                                      (baseline["records"] >= 10) &
                                      (baseline["entity_families"] >= 2)]], ignore_index=True)
        fig, ax = plt.subplots(figsize=(8, 3.8))
        names = ["PROMPT_LENGTH", "TEXT_TFIDF", "F", "H"]
        labels = ["Length", "Prompt text", "SAE code F", "Residual H"]
        for readout, shift, color in (("entity", -.18, "#2265a7"),
                                      ("final", .18, "#bf5b3d")):
            values = [control[(control["view"] == name) &
                              (control["readout"] == readout)]["spearman"].median()
                      for name in names]
            ax.bar(np.arange(4) + shift, values, width=.34, label=readout, color=color)
        ax.axhline(0, color="#252d37", lw=.8)
        ax.set_xticks(np.arange(4), labels)
        ax.set(ylabel="Median test Spearman across eligible quality tasks",
               title="Same-cohort lexical controls versus layer 11 representations")
        ax.legend(frameon=False)
        _write_figure(fig, run, "lexical_controls")

    complete = faithfulness[(faithfulness["status"] == "complete") &
                            (faithfulness["dimensions"] == 2)]
    if len(complete):
        fig, ax = plt.subplots(figsize=(7.5, 4.3))
        for method, mark, label in (("UMAP", "o", "UMAP · train→heldout"),
                                     ("t-SNE", "^", "t-SNE · all splits")):
            d = complete[complete["method"] == method]
            ax.scatter(d["trustworthiness_k10"], d["pair_distance_spearman"],
                       label=label, marker=mark, alpha=.75, s=34)
        ax.set(xlabel="Local trustworthiness, k=10", ylabel="Sampled pair-distance rank correlation",
               title="Embedding fidelity relative to 64-PC input")
        ax.legend(frameon=False)
        _write_figure(fig, run, "embedding_faithfulness")

    hue = pd.read_csv(run / "hue" / "triplets.csv")
    grouped = hue.groupby(["layer", "readout"], as_index=False)["F_order_correct"].mean()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for readout, color in (("entity", "#2265a7"), ("final", "#bf5b3d")):
        d = grouped[grouped["readout"] == readout]
        ax.plot(d["layer"], d["F_order_correct"], marker="o", color=color, label=readout)
    ax.axhline(.5, color="#252d37", lw=.8, ls="--", label="50% reference")
    ax.set(xlabel="SAE layer, zero-based", ylabel="Fraction with near colour closer",
           ylim=(0, 1), title="Original-code hue near/far triplets")
    ax.legend(frameon=False)
    _write_figure(fig, run, "hue_triplets")

    transfer_path = run / "predictions" / "transfer.csv"
    if transfer_path.exists():
        transferred = pd.read_csv(transfer_path)
        familiar = predictive[(predictive["partition"] == "test_familiar") &
                              (predictive["records"] >= 10) &
                              (predictive["entity_families"] >= 2) &
                              (predictive["view"].isin(["H", "F"]))]
        withheld = transferred[(transferred["evaluation"] == "test_heldout_wording") &
                               (transferred["records"] >= 10) &
                               (transferred["entity_families"] >= 2) &
                               (transferred["view"].isin(["H", "F"]))]
        fig, ax = plt.subplots(figsize=(9, 3.8))
        for view in ("H", "F"):
            a = familiar[familiar["view"] == view].set_index(["layer", "readout", "quality"])
            b = withheld[withheld["view"] == view].set_index(["layer", "readout", "quality"])
            joined = a[["spearman"]].join(b[["spearman"]], how="inner",
                                          lsuffix="_familiar", rsuffix="_withheld").dropna().reset_index()
            for label, column, line in (("familiar", "spearman_familiar", "-"),
                                        ("withheld", "spearman_withheld", "--")):
                d = joined.groupby("layer")[column].median()
                ax.plot(d.index, d.values, color=colors[view], ls=line, marker="o",
                        label=f"{view} · {label}")
        ax.axhline(0, color="#252d37", lw=.8)
        ax.set(xlabel="SAE layer, zero-based", ylabel="Median test Spearman",
               title="Wording transfer on matched quality tasks")
        ax.legend(frameon=False, ncol=2)
        _write_figure(fig, run, "wording_transfer")

    factorial_path = run / "factorial" / "summary.csv"
    if factorial_path.exists():
        factorial = pd.read_csv(factorial_path)
        by_layer = factorial.groupby("layer").agg(
            selective=("target_exceeds_all_off_targets_fraction", "median"),
            correct_sign=("target_correct_sign_fraction", "median"))
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(by_layer.index, by_layer["correct_sign"], marker="o", color="#2265a7",
                label="Target direction correctly signed")
        ax.plot(by_layer.index, by_layer["selective"], marker="s", color="#bf5b3d",
                label="Target projection exceeds off-target")
        ax.set(xlabel="SAE layer, zero-based", ylabel="Median fraction across domain/quality summaries",
               ylim=(0, 1), title="Held-out factorial contrast selectivity")
        ax.legend(frameon=False)
        _write_figure(fig, run, "factorial_selectivity")

    dimension_path = run / "dimension" / "estimates.csv"
    if dimension_path.exists():
        estimates = pd.read_csv(dimension_path)
        fig, ax = plt.subplots(figsize=(8, 3.8))
        labels = ["TwoNN", "MLE k=5", "MLE k=10", "MLE k=20"]
        columns = ["twonn", "mle_k5", "mle_k10", "mle_k20"]
        positions = np.arange(len(labels))
        for view, shift in (("H", -.17), ("F", .17)):
            subset = estimates[estimates["view"] == view]
            values = [subset[column].median() for column in columns]
            ax.bar(positions + shift, values, width=.32, label=view, color=colors[view])
        ax.set_xticks(positions, labels)
        ax.set(ylabel="Median estimated dimension", title="Original-space dimension diagnostics")
        ax.legend(frameon=False)
        _write_figure(fig, run, "dimension_estimates")


def _canonical_artifact(run, audit, pca, predictions, faithfulness, profile):
    now = "2026-10-02T00:00:00Z"
    title = "Qwen-Scope SAE: quality structure and compression"
    complete = audit[audit["status"] == "complete"]
    partial = audit[audit["status"] == "partial"]
    n_layers = len(complete)
    test = predictions[(predictions["partition"] == "test_familiar") &
                       (predictions["status"] == "complete") &
                       (predictions["records"] >= 10) &
                       (predictions["entity_families"] >= 2) &
                       (predictions["view"].isin(["H", "F"]))]
    paired = test.pivot_table(index=["layer", "readout", "quality"], columns="view", values="spearman").dropna()
    h = paired["H"].median() if len(paired) else np.nan
    f = paired["F"].median() if len(paired) else np.nan
    difference = (paired["H"] - paired["F"]).median() if len(paired) else np.nan
    higher = float((paired["H"] > paired["F"]).mean()) if len(paired) else np.nan
    n_q = paired.reset_index()["quality"].nunique() if len(paired) else 0
    compression = predictions[(predictions["partition"] == "test_familiar") &
                              (predictions["status"] == "complete") &
                              (predictions["records"] >= 10) &
                              (predictions["entity_families"] >= 2) &
                              (predictions["view"].isin(["H", "H_PCA16", "F", "F_PCA16"]))]
    compressed = compression.pivot_table(index=["layer", "readout", "quality"],
                                         columns="view", values="spearman").dropna()
    h_drop = (compressed["H"] - compressed["H_PCA16"]).median() if len(compressed) else np.nan
    f_drop = (compressed["F"] - compressed["F_PCA16"]).median() if len(compressed) else np.nan
    baseline_path = run / "baselines" / "metrics.csv"
    has_baselines = baseline_path.exists()
    baseline = pd.read_csv(baseline_path) if has_baselines else pd.DataFrame()
    baseline_rows = []
    if has_baselines:
        baseline = baseline[(baseline["partition"] == "test_familiar") &
                            (baseline["records"] >= 10) &
                            (baseline["entity_families"] >= 2)]
        combined = pd.concat([test[test["layer"] == 11], baseline], ignore_index=True)
        for (view, readout), group in combined.groupby(["view", "readout"]):
            if view in {"H", "F", "PROMPT_LENGTH", "TEXT_TFIDF"}:
                baseline_rows.append({"view": view, "readout": readout,
                                      "median_spearman": float(group["spearman"].median()),
                                      "quality_tasks": int(group["quality"].nunique())})
    has_transfer = (run / "predictions" / "transfer.csv").exists()
    transfer = pd.read_csv(run / "predictions" / "transfer.csv") if has_transfer else pd.DataFrame()
    if has_transfer:
        transfer = transfer[(transfer["evaluation"] == "test_heldout_wording") &
                            (transfer["records"] >= 10) & (transfer["entity_families"] >= 2) &
                            (transfer["view"].isin(["H", "F"]))]
    matching = []
    if has_transfer:
        for view in ("H", "F"):
            familiar = test[test["view"] == view].set_index(["layer", "readout", "quality"])
            wording = transfer[transfer["view"] == view].set_index(["layer", "readout", "quality"])
            both = familiar[["spearman"]].join(wording[["spearman"]], how="inner",
                                               lsuffix="_familiar", rsuffix="_heldout").dropna()
            if len(both):
                matching.append((view, both))
    transfer_summary = {view: {"n": len(table),
                               "familiar": float(table["spearman_familiar"].median()),
                               "heldout": float(table["spearman_heldout"].median()),
                               "paired_drop": float((table["spearman_familiar"] -
                                                     table["spearman_heldout"]).median())}
                        for view, table in matching}
    if (run / "contrasts" / "direction_transfer.csv").exists():
        contrast_transfer = pd.read_csv(run / "contrasts" / "direction_transfer.csv")
        train_direction = contrast_transfer[contrast_transfer["partition"] == "train"]["fraction_correctly_signed"].median()
        test_direction = contrast_transfer[contrast_transfer["partition"] == "test"]["fraction_correctly_signed"].median()
    else:
        train_direction = test_direction = np.nan
    has_factorial = (run / "factorial" / "summary.csv").exists()
    factorial = pd.read_csv(run / "factorial" / "summary.csv") if has_factorial else pd.DataFrame()
    factorial_11 = factorial[factorial["layer"] == 11] if has_factorial else pd.DataFrame()
    factorial_selectivity = (factorial_11["target_exceeds_all_off_targets_fraction"].median()
                             if len(factorial_11) else np.nan)
    behavior_path = run / "behavior" / "observational_summary.csv"
    has_behavior = behavior_path.exists()
    dimension = pd.read_csv(run / "dimension" / "estimates.csv")
    twonn_h = dimension[dimension["view"] == "H"]["twonn"].median()
    twonn_f = dimension[dimension["view"] == "F"]["twonn"].median()
    mle20_h = dimension[dimension["view"] == "H"]["mle_k20"].median()
    mle20_f = dimension[dimension["view"] == "F"]["mle_k20"].median()
    hue = pd.read_csv(run / "hue" / "triplets.csv")
    hue_11 = hue[(hue["layer"] == 11) & (hue["readout"] == "final")]
    hue_rate = float(hue_11["F_order_correct"].mean()) if len(hue_11) else np.nan
    hue_validation = hue_11[hue_11["entity_split"] == "validation"]
    hue_test = hue_11[hue_11["entity_split"] == "test"]
    activity = pd.read_csv(run / "activity" / "summary.csv")
    error_entity = float(activity[activity["readout"] == "entity"]["median_relative_error"].median())
    error_final = float(activity[activity["readout"] == "final"]["median_relative_error"].median())
    if len(faithfulness) and "trustworthiness_k10" in faithfulness:
        umb = faithfulness[(faithfulness["method"] == "UMAP") & (faithfulness["dimensions"] == 2)]
        trust = umb["trustworthiness_k10"].median()
    else:
        trust = np.nan
    summary = (
        f"## Technical summary\n\nThe collected Top-100 SAE dataset is partial across 28 requested layers: "
        f"{n_layers} layers have complete main and trace sparse codes, and {len(partial)} is partly encoded. "
        f"This {profile} analysis uses original residuals H and sparse codes F at available layers. "
        f"Across {len(paired)} matched layer/readout/task evaluations from {n_q} quality tasks with at least "
        f"10 test records and two entity families, median rank correlation is {_scalar(h)} for H and "
        f"{_scalar(f)} for F. The median within-pair H−F difference is {_scalar(difference)}, and H is higher "
        f"in {_scalar(100 * higher, 0)}% of these comparisons. These evaluations reuse prompts and mostly "
        f"have only two independent test families; they are not independent replications. "
        f"The estimates describe this fixture bank, "
        f"not human perceptual dimensions or causal model mechanisms."
    )
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {"id": "summary", "type": "markdown", "body": summary},
        {"id": "coverage_text", "type": "markdown", "body":
         "## What was actually analysed\n\nA unique vector is one text and token position. "
         "Case/readout records can share that vector, so PCA uses vectors once; prediction retains their labels. "
         "The zero-based SAE layer corresponds to the preceding one-based transformer block. "
         "The chart shows encoded main rows against the 14,500 requested readouts at each layer."},
        {"id": "coverage_chart", "type": "chart", "chartId": "coverage"},
        {"id": "signal_text", "type": "markdown", "body":
         "## Quality information and compression\n\nThe held-out ordering comparison uses probes fitted "
         "only on train entities with familiar wording and eligible factorial combinations. Validation selects "
         "ridge strength; test entities remain frozen. H is the original residual; F is the sparse SAE code. "
         f"A compact PCA view is a train-fitted compression. At 16 PCs, the median within-task test "
         f"rank-correlation drop is {_scalar(h_drop)} for H and {_scalar(f_drop)} for F, despite the leading "
         "components explaining substantial training variance. This measures information retained for this "
         "probe, not intrinsic quality dimensionality. The chart uses only tasks with at least ten test records "
         "and two entity families; all raw task results remain in the CSV."},
        {"id": "prediction_chart", "type": "chart", "chartId": "prediction"},
        {"id": "lexical_text", "type": "markdown", "body":
         "## The wording itself is predictive\n\nA word and two-word prompt-text baseline and a prompt-length "
         "baseline were fitted on the same eligible training readouts and scored on the same familiar-wording "
         "test tasks. At layer 11, compare them with H and F on matching entity and final-token cohorts. "
         "Prompt text can directly name the authored quality level, so its predictive score measures a "
         "lexical shortcut, not a latent representation. This control makes it harder to attribute all "
         "probe success to a conceptual quality axis."},
        {"id": "lexical_chart", "type": "chart", "chartId": "lexical"},
        {"id": "transfer_text", "type": "markdown", "body":
         f"## Wording transfer and contrast selectivity weaken\n\nOn the same eligible test tasks, changing "
         f"from familiar to withheld wording reduces median rank correlation from "
         f"{_scalar(transfer_summary.get('H', {}).get('familiar'))} to "
         f"{_scalar(transfer_summary.get('H', {}).get('heldout'))} for H and from "
         f"{_scalar(transfer_summary.get('F', {}).get('familiar'))} to "
         f"{_scalar(transfer_summary.get('F', {}).get('heldout'))} for F. "
         f"These are {transfer_summary.get('H', {}).get('n', 0)} matched layer/readout/task comparisons, "
         "not independent families. A mean low-to-high SAE contrast direction correctly signs a median "
         f"{_scalar(100 * train_direction, 0)}% of training contrasts and "
         f"{_scalar(100 * test_direction, 0)}% of test contrasts across the representative layers. "
         "Held-out factorial cells often have only one family, so they cannot establish robust independent "
         "factor geometry."},
        {"id": "transfer_chart", "type": "chart", "chartId": "transfer"},
        {"id": "factorial_text", "type": "markdown", "body":
         f"## Joint qualities are not cleanly independent here\n\nIn layer 11 final-token SAE codes, "
         f"the median share of held-out one-factor contrasts whose target projection exceeds every off-target "
         f"projection is {_scalar(100 * factorial_selectivity, 0)}% across "
         f"{len(factorial_11)} domain/quality summaries. Each summary has one test entity family. "
         "These fitted directions are not orthogonal by design, and these data do not support a strong "
         "claim that factorial qualities separate cleanly. Behavioural comparison margins were also "
         "linked to feature norm and reconstruction error by natural-continuation versus A/B format; "
         "those associations are descriptive and can reflect wording or difficulty."},
        {"id": "factorial_chart", "type": "chart", "chartId": "factorial"},
        {"id": "geometry_text", "type": "markdown", "body":
         "## Geometry has to survive quantitative checks\n\nCentered PCA measures training variance; "
         "its axes are not automatically semantic. The embedding comparison puts local neighbourhood "
         "trustworthiness beside sampled global distance-rank agreement. A visually clear UMAP or t-SNE "
         "layout alone cannot establish a manifold or a circular hue representation."},
        {"id": "variance_chart", "type": "chart", "chartId": "variance"},
        {"id": "faithfulness_chart", "type": "chart", "chartId": "faithfulness"},
        {"id": "dimension_text", "type": "markdown", "body":
         f"## Dimensionality estimates depend on neighbourhood scale\n\nOn bounded original-space cohorts, "
         f"median TwoNN estimates are {_scalar(twonn_h)} for H and {_scalar(twonn_f)} for F; "
         f"k-neighbour MLE at k=20 gives {_scalar(mle20_h)} and {_scalar(mle20_f)} respectively. "
         "The gap between methods and the local-PCA sensitivity to neighbourhood size are reasons "
         "to report a range of dataset-specific complexity diagnostics rather than a single count "
         "of human quality dimensions. Exact duplicates were removed and zero-distance exclusions "
         "are recorded in the source table."},
        {"id": "dimension_chart", "type": "chart", "chartId": "dimension"},
        {"id": "hue_text", "type": "markdown", "body":
         f"## Hue is only partly ordered, not established as a circle\n\nIn the prespecified near/far hue "
         f"triplets, layer 11 final-token F places the authored near colour closer than the far one in "
         f"{_scalar(100 * hue_rate, 0)}% of {len(hue_11)} triplets. Triplets reuse a small set of entity "
         f"families; the validation and test rates are {_scalar(100 * hue_validation['F_order_correct'].mean(), 0)}% "
         f"and {_scalar(100 * hue_test['F_order_correct'].mean(), 0)}% over {len(hue_validation)} and "
         f"{len(hue_test)} triplets respectively. Angular prediction errors also vary greatly across "
         "layer and readout. This supports some local hue ordering in the constructed examples, "
         "but does not establish a stable circular manifold."},
        {"id": "hue_chart", "type": "chart", "chartId": "hue"},
        {"id": "limitations", "type": "markdown", "body":
         "## Scope and limits\n\nLabels are author-written design codes, not human ratings. "
         "The original V3 split and wording restrictions are kept for held-out probes; all-splits t-SNE "
         "is descriptive only. SAE feature IDs are checkpoint-local. The dataset has sparse codes on nine "
         "complete layers, one partial layer, and 18 layers without sparse codes. "
         f"Median SAE relative reconstruction error is {_scalar(error_entity)} at the entity readout and "
         f"{_scalar(error_final)} at the final-token readout across complete layers. "
         "Some test conditions contain one or two independent entity families; their rows do not provide "
         "equally many independent replicates. Numeric challenges and physical values were withheld from "
         "ordinal fitting, so this report does not assert physical-unit calibration. "
         "No causal intervention or complete 28-layer SAE comparison is implied."},
        {"id": "next", "type": "markdown", "body":
         "## Next checks\n\nInspect the per-quality and per-family CSVs before assigning a meaning to any SAE "
         "feature. Extend the remaining checkpoints only after adding storage, then rerun this analysis as a "
         "new versioned result. Independent human or external stimuli would be needed to test whether the "
         "constructed axes generalize beyond these prompts."},
    ]
    if not has_transfer:
        blocks = [block for block in blocks if block["id"] not in {"transfer_text", "transfer_chart"}]
    if not has_baselines:
        blocks = [block for block in blocks if block["id"] not in {"lexical_text", "lexical_chart"}]
    if not has_factorial:
        blocks = [block for block in blocks if block["id"] not in {"factorial_text", "factorial_chart"}]
    coverage_rows = [{"layer": int(r.layer), "encoded": int(r.main_completed),
                      "requested": int(r.main_expected)} for r in audit.itertuples()]
    pred_rows = []
    for (layer, view), d in test.groupby(["layer", "view"]):
        pred_rows.append({"layer": int(layer), "view": view,
                          "median_spearman": float(d["spearman"].median()),
                          "quality_tasks": int(d["quality"].nunique()),
                          "smallest_family_count": int(d["entity_families"].min())})
    variance_rows = []
    for view in ("H", "F"):
        path = run / "linear" / f"layer11_final_{view}_pca.npz"
        if path.exists():
            with np.load(path) as data:
                for rank, value in enumerate(np.cumsum(data["explained_variance_ratio"]), 1):
                    variance_rows.append({"rank": rank, "cumulative_variance": float(value), "view": view})
    faithful_rows = []
    if len(faithfulness):
        for row in faithfulness[(faithfulness["status"] == "complete") &
                                (faithfulness["dimensions"] == 2)].itertuples():
            faithful_rows.append({"method": f"{row.method} · {'train→heldout' if row.method == 'UMAP' else 'all splits'}",
                                  "trustworthiness": float(getattr(row, "trustworthiness_k10")),
                                  "distance_rank": float(row.pair_distance_spearman),
                                  "layer": int(row.layer), "view": row.view,
                                  "readout": row.readout})
    hue_rows = [{"layer": int(layer), "readout": readout, "near_before_far": float(group["F_order_correct"].mean()),
                 "triplets": len(group)} for (layer, readout), group in hue.groupby(["layer", "readout"])]
    transfer_rows = []
    for view, table in matching:
        for layer, layer_rows in table.reset_index().groupby("layer"):
            for wording, column in (("familiar", "spearman_familiar"), ("heldout", "spearman_heldout")):
                transfer_rows.append({"layer": int(layer), "series": f"{view} · {wording}",
                                      "median_spearman": float(layer_rows[column].median()),
                                      "matched_tasks": len(layer_rows)})
    factorial_rows = [{"layer": int(layer),
                       "median_target_selectivity": float(group["target_exceeds_all_off_targets_fraction"].median()),
                       "domain_quality_summaries": len(group),
                       "max_test_families": int(group["test_entity_families"].max())}
                      for layer, group in factorial.groupby("layer")] if has_factorial else []
    dimension_rows = [{"estimator": label, "view": view,
                       "median_estimated_dimension": float(dimension[dimension["view"] == view][column].median()),
                       "cohorts": int((dimension["view"] == view).sum())}
                      for view in ("H", "F")
                      for label, column in (("TwoNN", "twonn"), ("MLE k=5", "mle_k5"),
                                            ("MLE k=10", "mle_k10"), ("MLE k=20", "mle_k20"))]
    sql_path = run / "chart_data.sqlite"
    with sqlite3.connect(sql_path) as connection:
        pd.DataFrame(coverage_rows).to_sql("coverage_chart", connection, if_exists="replace", index=False)
        pd.DataFrame(pred_rows).to_sql("prediction_chart", connection, if_exists="replace", index=False)
        pd.DataFrame(baseline_rows, columns=["view", "readout", "median_spearman", "quality_tasks"]).to_sql(
            "lexical_chart", connection, if_exists="replace", index=False)
        pd.DataFrame(variance_rows).to_sql("variance_chart", connection, if_exists="replace", index=False)
        pd.DataFrame(faithful_rows).to_sql("faithfulness_chart", connection, if_exists="replace", index=False)
        pd.DataFrame(hue_rows).to_sql("hue_chart", connection, if_exists="replace", index=False)
        pd.DataFrame(transfer_rows, columns=["layer", "series", "median_spearman", "matched_tasks"]).to_sql(
            "transfer_chart", connection, if_exists="replace", index=False)
        pd.DataFrame(factorial_rows, columns=["layer", "median_target_selectivity",
                                              "domain_quality_summaries", "max_test_families"]).to_sql(
            "factorial_chart", connection, if_exists="replace", index=False)
        pd.DataFrame(dimension_rows).to_sql("dimension_chart", connection, if_exists="replace", index=False)
        connection.row_factory = sqlite3.Row
        queried = {name: [dict(row) for row in connection.execute(f"SELECT * FROM {name}_chart")]
                   for name in ("coverage", "prediction", "lexical", "variance", "faithfulness", "hue", "transfer", "factorial", "dimension")}
    source_specs = [
        {"id": "coverage_source", "label": "SAE collection and analysis coverage",
         "path": "chart_data.sqlite", "query": {"engine": "sqlite",
         "sql": "SELECT * FROM coverage_chart", "description": "Reviewed main SAE layer coverage",
         "tables_used": ["coverage_chart"]}},
        {"id": "prediction_source", "label": "Held-out prediction metrics",
         "path": "chart_data.sqlite", "query": {"engine": "sqlite",
         "sql": "SELECT * FROM prediction_chart", "description": "Reviewed familiar-wording test medians",
         "tables_used": ["prediction_chart"]}},
        {"id": "lexical_source", "label": "Same-cohort lexical controls",
         "path": "chart_data.sqlite", "query": {"engine": "sqlite",
         "sql": "SELECT * FROM lexical_chart", "description": "Prompt text and length controls versus layer 11 H and F",
         "tables_used": ["lexical_chart"]}},
        {"id": "variance_source", "label": "Train-fitted centered PCA",
         "path": "chart_data.sqlite", "query": {"engine": "sqlite",
         "sql": "SELECT * FROM variance_chart", "description": "Reviewed PCA cumulative variance by rank",
         "tables_used": ["variance_chart"]}},
        {"id": "faithfulness_source", "label": "Embedding faithfulness",
         "path": "chart_data.sqlite", "query": {"engine": "sqlite",
         "sql": "SELECT * FROM faithfulness_chart", "description": "Reviewed local and global fidelity",
         "tables_used": ["faithfulness_chart"]}},
        {"id": "hue_source", "label": "Matched hue triplet ordering",
         "path": "chart_data.sqlite", "query": {"engine": "sqlite",
         "sql": "SELECT * FROM hue_chart", "description": "Original-F near/far triplet success",
         "tables_used": ["hue_chart"]}},
        {"id": "transfer_source", "label": "Matched wording transfer metrics",
         "path": "chart_data.sqlite", "query": {"engine": "sqlite",
         "sql": "SELECT * FROM transfer_chart", "description": "Same-task familiar and withheld wording test scores",
         "tables_used": ["transfer_chart"]}},
        {"id": "factorial_source", "label": "One-factor contrast selectivity",
         "path": "chart_data.sqlite", "query": {"engine": "sqlite",
         "sql": "SELECT * FROM factorial_chart", "description": "Held-out target versus off-target direction projections",
         "tables_used": ["factorial_chart"]}},
        {"id": "dimension_source", "label": "Original-space dimension diagnostics",
         "path": "chart_data.sqlite", "query": {"engine": "sqlite",
         "sql": "SELECT * FROM dimension_chart", "description": "TwoNN and kNN MLE summaries on original H and F",
         "tables_used": ["dimension_chart"]}},
    ]
    charts = [
        {"id": "coverage", "title": "Encoded main readouts by SAE layer",
         "subtitle": "14,500 unique readouts requested at every zero-based layer",
         "type": "bar", "dataset": "coverage", "sourceId": "coverage_source",
         "encodings": {"x": {"field": "layer"}, "y": {"field": "encoded"}}},
        {"id": "prediction", "title": "Median held-out rank correlation",
         "subtitle": "Familiar wording; medians over quality tasks with unequal family counts",
         "type": "line", "dataset": "prediction", "sourceId": "prediction_source",
         "encodings": {"x": {"field": "layer"}, "y": {"field": "median_spearman"},
                       "color": {"field": "view"}}},
        {"id": "lexical", "title": "Layer 11 versus lexical controls",
         "subtitle": "Same familiar-wording test tasks; prompt text may contain the authored answer",
         "type": "bar", "dataset": "lexical", "sourceId": "lexical_source",
         "encodings": {"x": {"field": "view"}, "y": {"field": "median_spearman"},
                       "color": {"field": "readout"}}},
        {"id": "variance", "title": "Centered PCA training variance",
         "subtitle": "Final readout, layer 11; first 64 components",
         "type": "line", "dataset": "variance", "sourceId": "variance_source",
         "encodings": {"x": {"field": "rank"}, "y": {"field": "cumulative_variance"},
                       "color": {"field": "view"}}},
        {"id": "faithfulness", "title": "Embedding fidelity relative to 64-PC input",
         "subtitle": "UMAP train→heldout; t-SNE all splits; local k=10 versus global distance rank",
         "type": "scatter", "dataset": "faithfulness", "sourceId": "faithfulness_source",
         "encodings": {"x": {"field": "trustworthiness"}, "y": {"field": "distance_rank"},
                       "color": {"field": "method"}}},
        {"id": "hue", "title": "Matched hue near/far ordering in original F",
         "subtitle": "Fraction of authored triplets where near is closer; 240 triplets per layer/readout",
         "type": "line", "dataset": "hue", "sourceId": "hue_source",
         "encodings": {"x": {"field": "layer"}, "y": {"field": "near_before_far"},
                       "color": {"field": "readout"}}},
        {"id": "transfer", "title": "Test ordering under familiar and withheld wording",
         "subtitle": "Same eligible tasks, at least 10 test records and two entity families",
         "type": "line", "dataset": "transfer", "sourceId": "transfer_source",
         "encodings": {"x": {"field": "layer"}, "y": {"field": "median_spearman"},
                       "color": {"field": "series"}}},
        {"id": "factorial", "title": "One-factor SAE contrast selectivity",
         "subtitle": "Median target-exceeds-off-target fraction; one test family per domain/quality",
         "type": "bar", "dataset": "factorial", "sourceId": "factorial_source",
         "encodings": {"x": {"field": "layer"}, "y": {"field": "median_target_selectivity"}}},
        {"id": "dimension", "title": "Original-space dimension estimates",
         "subtitle": "Median across nine layers and two readouts; methods and k have different assumptions",
         "type": "bar", "dataset": "dimension", "sourceId": "dimension_source",
         "encodings": {"x": {"field": "estimator"}, "y": {"field": "median_estimated_dimension"},
                       "color": {"field": "view"}}},
    ]
    if not has_transfer:
        charts = [chart for chart in charts if chart["id"] != "transfer"]
    if not has_baselines:
        charts = [chart for chart in charts if chart["id"] != "lexical"]
    if not has_factorial:
        charts = [chart for chart in charts if chart["id"] != "factorial"]
    payload = {"surface": "report",
               "manifest": {"version": 1, "surface": "report", "title": title,
                            "generatedAt": now, "blocks": blocks, "charts": charts, "sources": source_specs},
               "snapshot": {"version": 1, "generatedAt": now, "status": "ready",
                            "datasets": queried},
               "sources": source_specs}
    (run / "artifact.json").write_text(json.dumps(payload, indent=2))
    (run / "findings.md").write_text("\n\n".join(block["body"] for block in blocks if block["type"] == "markdown") + "\n")
    return payload


def _explorer(run, collection):
    from .sae_analysis_core import feature_matrix
    embeddings = []
    metrics = pd.read_csv(run / "embeddings" / "faithfulness.csv")
    fidelity = {}
    for row in metrics.itertuples():
        if row.status == "complete" and row.seed == 1729:
            key = f"{row.layer}:{row.readout}:{row.view}:{row.method}:{row.dimensions}"
            fidelity[key] = {"trustworthiness_k10": float(row.trustworthiness_k10),
                             "pair_distance_spearman": float(row.pair_distance_spearman),
                             "normalized_stress": float(row.normalized_stress)}
    pattern = re.compile(r"layer(\d+)_(entity|final)_(H|F)_(UMAP|t-SNE)_(2|3)d_seed(\d+)\.npz")
    for path in sorted((run / "embeddings").glob("*.npz")):
        match = pattern.fullmatch(path.name)
        if not match or match.group(6) != "1729":
            continue
        layer, readout, view, method, dimensions, _ = match.groups()
        with np.load(path) as data:
            ids = data["ids"].tolist()
            coords = data["coordinates"].tolist()
        embeddings.append({"layer": int(layer), "readout": readout, "view": view,
                           "method": method, "dimensions": int(dimensions),
                           "ids": ids, "coordinates": coords})
    for layer in sorted({item["layer"] for item in embeddings}):
        for readout in ("entity", "final"):
            for view in ("H", "F"):
                pca_path = run / "linear" / f"layer{layer}_{readout}_{view}_pca.npz"
                if not pca_path.exists():
                    continue
                with np.load(pca_path) as data:
                    ids = data["ids"].tolist()
                    scores = data["coordinates"]
                    ratio = data["explained_variance_ratio"]
                for dimensions in (2, 3):
                    embeddings.append({"layer": layer, "readout": readout, "view": view,
                                       "method": "PCA", "dimensions": dimensions,
                                       "ids": ids, "coordinates": scores[:, :dimensions].tolist()})
                    fidelity[f"{layer}:{readout}:{view}:PCA:{dimensions}"] = {
                        "training_variance_fraction": float(np.sum(ratio[:dimensions]))}
    metadata = {}
    readout_by_id = {r["readout_id"]: r for r in collection["readouts"]}
    for rid, records in collection["by_readout"].items():
        case = next((r for r in records if r["kind"] == "case"), None)
        if case is None:
            continue
        meta = case["metadata"]
        targets = dict(meta.get("targets", {}))
        if "hue_sin" in targets and "hue_cos" in targets:
            targets["hue"] = float(np.degrees(np.arctan2(targets["hue_sin"], targets["hue_cos"])) % 360)
        metadata[rid] = {"text": readout_by_id[rid]["text"][:360], "case_id": meta["case_id"],
                         "quality_ids": meta.get("quality_ids", []), "targets": targets,
                         "wording": meta.get("template_split"), "split": meta.get("entity_split"),
                         "family": meta.get("entity_group_id"), "case_type": meta.get("case_type")}
    errors = {}
    for layer in sorted({item["layer"] for item in embeddings}):
        keys, _, diagnostic = feature_matrix(collection, layer)
        errors[str(layer)] = {key: float(error) for key, error in zip(keys, diagnostic["relative_error"])}
    payload = {"embeddings": embeddings, "metadata": metadata, "errors": errors, "fidelity": fidelity}
    script = get_plotlyjs()
    template = """<!doctype html><html lang="en"><meta charset="utf-8"><title>SAE embedding explorer</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:15px system-ui,sans-serif;color:#17212f;background:#f7f9fc;margin:0}
header{background:#fff;border-bottom:1px solid #d9e0e8;padding:20px 28px}h1{font-size:23px;margin:0 0 8px}
p{line-height:1.5}main{max-width:1300px;margin:auto;padding:18px 28px}.controls{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:12px}
label{font-size:13px;color:#435268}select{display:block;padding:7px;border:1px solid #bac6d5;border-radius:5px;background:#fff;color:#17212f}
#plot{height:650px;background:#fff;border:1px solid #d9e0e8;border-radius:8px}
.legend{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}.chip{font-size:12px;padding:3px 7px;border-radius:4px;border:1px solid #ccd6e2;background:#fff}
.note{font-size:13px;color:#526071;margin-top:12px}</style>
<header><h1>Qwen-Scope SAE embedding explorer</h1>
<p>Each point is one unique computational readout. Colours switch without refitting or moving coordinates.
UMAP fits on eligible training readouts and transforms the rest; t-SNE is an all-splits descriptive layout.
The fixed cohort is balanced across train, validation, test and withheld train wording.</p></header>
<main><div class="controls" id="controls"></div><div class="legend" id="legend"></div><div id="plot"></div>
<p class="note" id="denominator"></p><p class="note">A compact or circular-looking plot is not proof of a manifold. Use the faithfulness table and original-space hue triplets in the report.</p></main>
<script>__PLOTLY__</script><script>
const DATA=__DATA__;
const fields={layer:[...new Set(DATA.embeddings.map(x=>x.layer))],view:["H","F"],
 readout:["entity","final"],method:["PCA","UMAP","t-SNE"],dimensions:[2,3],
 property:["all",...new Set(Object.values(DATA.metadata).flatMap(x=>x.quality_ids))],
 wording:["all","train","heldout"],split:["all","train","validation","test"],
 colour:["quality level","wording","entity family","reconstruction error"]};
const state={layer:fields.layer[0],view:"H",readout:"final",method:"PCA",dimensions:2,
 property:"all",wording:"all",split:"all",colour:"quality level"};
const controls=document.getElementById("controls");
for(const [field,values] of Object.entries(fields)){const label=document.createElement("label");
 label.textContent=field;const select=document.createElement("select");select.id=field;
 for(const value of values){const option=document.createElement("option");option.value=value;option.textContent=value;select.appendChild(option)}
 select.value=state[field];select.onchange=()=>{state[field]=["layer","dimensions"].includes(field)?Number(select.value):select.value;draw()};
 label.appendChild(select);controls.appendChild(label)}
function hashColor(value){let h=0;for(const c of String(value)){h=(h*31+c.charCodeAt(0))>>>0}
 return "hsl("+(h%360)+" 58% 45%)"}
function draw(){const item=DATA.embeddings.find(x=>x.layer===state.layer&&x.view===state.view&&x.readout===state.readout&&
 x.method===state.method&&x.dimensions===state.dimensions);
 if(!item){Plotly.purge("plot");document.getElementById("denominator").textContent="No coordinates for this combination.";return}
 const rows=[];for(let i=0;i<item.ids.length;i++){const id=item.ids[i],meta=DATA.metadata[id];
 if(!meta)continue;if(state.property!=="all"&&!meta.quality_ids.includes(state.property))continue;
 if(state.wording!=="all"&&meta.wording!==state.wording)continue;if(state.split!=="all"&&meta.split!==state.split)continue;
 rows.push({id,meta,coord:item.coordinates[i],error:DATA.errors[state.layer]?.[id]})}
 const colors=rows.map(r=>state.colour==="reconstruction error"?r.error:
 state.colour==="wording"?hashColor(r.meta.wording):state.colour==="entity family"?hashColor(r.meta.family):
 (state.property==="all"?"#2265a7":r.meta.targets[state.property]));
 const numeric=state.colour==="reconstruction error"||state.colour==="quality level"&&state.property!=="all";
 const marker={size:state.dimensions===3?4:6,opacity:.72,color:colors};
 if(numeric){marker.colorscale="Viridis";marker.showscale=true;marker.colorbar={title:state.colour}}
 const legend=document.getElementById("legend");legend.innerHTML="";
 if(state.colour==="wording"||state.colour==="entity family"){const values=[...new Set(rows.map(r=>state.colour==="wording"?r.meta.wording:r.meta.family))].slice(0,18);
 for(const value of values){const chip=document.createElement("span");chip.className="chip";chip.textContent=value;
 chip.style.borderLeft="8px solid "+hashColor(value);legend.appendChild(chip)}}
 const hover=rows.map(r=>"<b>"+r.meta.case_id+"</b><br>"+r.meta.family+" · "+r.meta.case_type+
 "<br>"+r.meta.split+" · "+r.meta.wording+"<br>"+r.meta.text.replaceAll("<","&lt;").replaceAll(">","&gt;")+
 "<br>SAE reconstruction error: "+(r.error===undefined?"n/a":r.error.toFixed(3)));
 const trace=state.dimensions===3?{type:"scatter3d",mode:"markers",x:rows.map(r=>r.coord[0]),y:rows.map(r=>r.coord[1]),
 z:rows.map(r=>r.coord[2]),marker,text:hover,hoverinfo:"text"}:
 {type:"scatter",mode:"markers",x:rows.map(r=>r.coord[0]),y:rows.map(r=>r.coord[1]),marker,text:hover,hoverinfo:"text"};
 Plotly.newPlot("plot",[trace],{title:state.method+" · "+state.view+" · layer "+state.layer+" · "+state.readout,
 margin:{l:60,r:30,t:55,b:55},paper_bgcolor:"#fff",plot_bgcolor:"#fff",
 xaxis:{title:state.method==="PCA"?"principal component 1":"embedding coordinate 1"},
 yaxis:{title:state.method==="PCA"?"principal component 2":"embedding coordinate 2"}},{responsive:true,displaylogo:false});
 const metric=DATA.fidelity[state.layer+":"+state.readout+":"+state.view+":"+state.method+":"+state.dimensions]||{};
 const note=metric.trustworthiness_k10!==undefined?" Local trustworthiness k=10: "+metric.trustworthiness_k10.toFixed(3)+
 "; global distance-rank correlation: "+metric.pair_distance_spearman.toFixed(3)+".":
 metric.training_variance_fraction!==undefined?" Training variance in displayed PCs: "+(100*metric.training_variance_fraction).toFixed(1)+"%.":"";
 document.getElementById("denominator").textContent=rows.length+" visible readouts out of "+item.ids.length+
 " fixed coordinates. Layer "+state.layer+"; "+state.method+" "+state.dimensions+"D."+note+" Hover for exact prompt and provenance."}
draw();</script></html>"""
    output = template.replace("__PLOTLY__", script).replace("__DATA__", json.dumps(payload, separators=(",", ":")).replace("</", "<\\/"))
    (run / "explorer.html").write_text(output)


def build_report(run_dir):
    run = Path(run_dir).expanduser().resolve()
    config = json.loads((run / "config.json").read_text())
    source = (Path(__file__).resolve().parents[2] / config["source_run"]).resolve()
    collection = read_collection(source)
    _catalogue_and_components(run, collection)
    profile = json.loads((run / "identity.json").read_text())["profile"]
    audit = pd.read_csv(run / "audit" / "layers.csv")
    pca = pd.read_csv(run / "linear" / "pca_summary.csv")
    predictions = pd.read_csv(run / "predictions" / "metrics.csv")
    faithfulness = pd.read_csv(run / "embeddings" / "faithfulness.csv")
    _figures(run, audit, pca, predictions, faithfulness)
    _canonical_artifact(run, audit, pca, predictions, faithfulness, profile)
    _explorer(run, collection)
    return run / "artifact.json"
