"""Honest machine-readable and Markdown reporting for quality-suite v3."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .util import atomic_json, read_json


def _plots(run):
    plots = run / "plots"; plots.mkdir(exist_ok=True)
    metrics_path = run / "linear_metrics.csv"
    if metrics_path.exists():
        frame = pd.read_csv(metrics_path)
        view = frame[(frame.get("method", pd.Series(index=frame.index, dtype=str)) == "ridge") & (~frame.get("partition", pd.Series(index=frame.index, dtype=str)).fillna("").str.startswith("train")) & frame.get("rho", pd.Series(index=frame.index, dtype=float)).notna()].copy()
        if not view.empty:
            view["label"] = view.quality + " | " + view.partition
            # Do not select a flattering top-N of outcomes.
            view = view.sort_values(["quality","partition"])
            view = view[view.partition.str.startswith('entity_test|') & view.partition.str.endswith('|plain')]
        if not view.empty:
            fig, ax = plt.subplots(figsize=(9, max(4, .3 * len(view))))
            ax.barh(view.label, view.rho, color="#3974A8", edgecolor="#23313D")
            ax.axvline(0, color="#45515B", linewidth=.8)
            ax.set(xlim=(-1,1),xlabel="Spearman rho", title="Held-out linear readout performance")
            ax.set_title('Held-out linear readout performance',pad=30)
            ax.text(0, 1.01, f"Test entity groups per bar: {int(view.n_groups.min())}–{int(view.n_groups.max())}; constructed fixtures",
                    transform=ax.transAxes, fontsize=9, color="#45515B")
            fig.tight_layout(); fig.savefig(plots / "heldout_linear_rho.png", dpi=160); plt.close(fig)
    behavior_path = run / "behavior_summary.csv"
    if behavior_path.exists():
        frame = pd.read_csv(behavior_path)
        if 'entity_split' in frame:
            frame = frame[(frame.entity_split == 'test') & (frame.template_split == 'train')]
        if not frame.empty:
            labels = frame.quality_id + " | " + frame.format
            fig, ax = plt.subplots(figsize=(9, max(4, .3 * len(frame))))
            ax.barh(labels, frame.accuracy, color="#D39B2A", edgecolor="#23313D")
            ax.axvline(.5, color="#45515B", linestyle="--", linewidth=1, label="chance")
            ax.set(xlim=(0, 1), xlabel="Exact candidate-choice accuracy", title="Behavioral competence by property and format")
            ax.set_title('Behavioral competence by property and format',pad=30)
            ax.text(0, 1.01, f"Test questions per bar: {int(frame.n.min())}–{int(frame.n.max())}; exact sequence scoring",
                    transform=ax.transAxes, fontsize=9, color="#45515B")
            ax.legend(frameon=False); fig.tight_layout(); fig.savefig(plots / "behavior_accuracy.png", dpi=160); plt.close(fig)


def suite_report(run_dir):
    run = Path(run_dir)
    if read_json(run/'identity.json').get('schema_version') != '3.0.0':
        raise ValueError('Not a V3 run; refusing to write report')
    _plots(run)
    identity = read_json(run / "identity.json"); plan = read_json(run / "plan.json")
    state = read_json(run / "state.json"); selection = read_json(run / "model_selection.json") if (run / "model_selection.json").exists() else {}
    lines = [f"# Quality suite v3: {identity['profile']} profile", "",
             f"**Run status:** `{state['status']}`  ",
             f"**Suite fingerprint:** `{identity['suite_fingerprint']}`  ",
             f"**Model:** `{identity.get('model_id', 'not yet resolved')}` at `{identity.get('model_revision', 'not yet resolved')}`", "",
             "## What was tested", "",
             f"- {plan['activation_cases']:,} selected cases, two contextual readouts, and {len(plan['selected_blocks'])} complete-block residuals.",
             f"- {plan['behavior_questions']:,} exact candidate comparisons; multi-token candidates use teacher-forced sequence log probabilities.",
             f"- {plan['independent_counts']['activation_entity_groups']:,} activation entity families and {plan['independent_counts']['behavior_entity_groups']:,} behavioral entity families.",
             f"- Planned activation storage before text/position deduplication: {plan['expected_activation_bytes_before_deduplication']/2**20:.1f} MiB.", "",
             "The rows are assistant-authored constructed fixtures. Repeated sentences and variants are not independent human observations, and no positive result is assumed.", "",
             "## Model selection and held-out axes", ""]
    if selection:
        lines.extend(["| Property | Readout | Block | Ridge alpha | Train contrasts / rank |",
                      "|---|---:|---:|---:|---:|"])
        for quality, item in sorted(selection.items()):
            lines.append(f"| {quality} | {item['readout']} | {item['block']} | {item['ridge_alpha']:.4g} | {item['training_contrasts']} / {item['contrast_svd_rank']} |")
    else:
        lines.append("Analysis has not completed.")
    if (run/'linear_metrics.csv').exists():
        metrics=pd.read_csv(run/'linear_metrics.csv')
        if 'partition' in metrics:
            held=metrics[(metrics.method=='ridge') & metrics.partition.fillna('').str.startswith('entity_test')]
            lines.extend(['','### Test-entity results by condition','','| Property | Partition / family / wording | Rows / groups | Spearman rho | Circular MAE (degrees) |','|---|---|---:|---:|---:|'])
            for r in held.itertuples():
                rho=getattr(r,'rho',np.nan); circ=getattr(r,'circular_mae_degrees',np.nan)
                lines.append(f"| {r.quality} | {r.partition} | {r.n_rows:g} / {r.n_groups:g} | {rho:.3f} | {circ:.2f} |")
    if (run/'diagnostic_coverage.json').exists():
        coverage=read_json(run/'diagnostic_coverage.json')
        lines.extend(['',f"Matched diagnostic outputs: {coverage['paired_rows']} prediction differences, {coverage['numeric_predictions']} physical-unit challenge predictions, and {coverage['circular_rows']} circular-neighborhood comparisons. Each output retains its family and scale."])
    lines.extend(["", "Regularization and preprocessing were selected in grouped training-entity folds. The layer/readout choice used validation entities. Numeric challenge cases were excluded from ordinal fitting; factorial fits additionally required training combinations.", "",
                  "## Behavioral checks", ""])
    if (run / "behavior_summary.csv").exists():
        behavior = pd.read_csv(run / "behavior_summary.csv")
        if not behavior.empty:
            lines.extend(["| Property | Format | Questions | Accuracy | Mean correct margin | A/first-choice rate |",
                          "|---|---|---:|---:|---:|---:|"])
            for row in behavior.itertuples():
                split = f"{row.entity_split}/{row.template_split}" if hasattr(row,'entity_split') else 'pooled'
                lines.append(f"| {row.quality_id} ({split}) | {row.format} | {row.n} | {row.accuracy:.3f} | {row.mean_correct_margin:.3f} | {row.first_choice_rate:.3f} |")
        else: lines.append("No behavior rows completed.")
    else: lines.append("Behavioral evaluation has not completed.")
    lines.extend(["", "## Provenance, comparisons, and limits", "",
                  "- Label provenance is retained in every selected case and question and summarized in `label_provenance.json`.",
                  "- `comparison_to_original_pilot.json` records structural differences from the original size-only pilot without rewriting its data or results.",
                  "- Hue is evaluated with sine/cosine outputs and circular error. HSV value is not treated as luminance.",
                  "- Ordinal design codes and declared physical values remain separate. Rectangular volume is derived, not an independent fitted quality.",
                  "- Nonlinear/dictionary exploration and causal interventions remain disabled unless their documented validation/competence gates are met.",
                  "- Lexical synonym groups remain provisional; controlled synthetic results and lexical claims are reported separately.",
                  "- A screen can be null or negative. This pipeline does not require a positive scientific result.", "",
                  "- This is an exploratory multi-quality/multi-layer screen. No corrected confirmatory significance or intrinsic dimensionality claim is made. Validation-selected scores need fresh independent confirmation.",
                  "- Ridge standardizes features within training folds; contrast SVD uses raw residual differences. A scalar linear probe remains one linear functional regardless of subspace rank.",
                  "- A/first-choice rate refers to candidate index zero (A for AB questions), not the first option printed in the prompt. Printed order and semantic mappings are retained in raw rows and `answer_bias.csv`.",
                  "## Raw outputs", "",
                  "`linear_metrics.csv`, `linear_predictions.csv`, `adjacent_direction_alignment.csv`, `behavior_scores.csv`, `behavior_summary.csv`, `model_selection.json`, and cache/run manifests are the auditable outputs."])
    if (run/'original_pilot_reference.json').exists():
        original=read_json(run/'original_pilot_reference.json')
        lines.extend(['','## Original pilot reference','',original['comparison_caveat'],'',
                      f"Original selected block: {original['selected_layer']}. V3 choices are listed above.",'',
                      '| Original evaluation | Method | Groups | Spearman rho |','|---|---|---:|---:|'])
        for row in original['metrics']:
            lines.append(f"| {row['evaluation']} | {row['method']} | {row['n_groups']} | {row['rho']:.3f} |")
    if (run/'behavior_scores.csv').exists() and (run/'behavior_scores.csv').stat().st_size > 1:
        raw=pd.read_csv(run/'behavior_scores.csv')
        if not raw.empty and 'precision_disagreement' in raw:
            finite=raw.precision_disagreement.dropna()
            if len(finite):
                lines.extend(['','## Numerical resolution','',f"{len(finite)} single-token comparisons have float32 selected-head checks. Maximum native/head margin discrepancy: {finite.max():.6f} log-probability units. Effects of this order require caution; this is not a full float32-network comparison."])
    if state.get("failures"):
        lines.extend(["", "## Failures", "", *[f"- {value}" for value in state["failures"]]])
    (run / "report.md").write_text("\n".join(lines) + "\n")
    import hashlib
    atomic_json(run/'report_rendering.json',dict(renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                               source='Existing run CSV/JSON; no model inference during report regeneration'))
    return run / "report.md"
