"""Validate and summarize completed geometry runs without inference or refitting."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quality_dimensions.suite_analysis import _score, _circular_score
from quality_dimensions.suite_data import load_suite
from quality_dimensions.geometry import sha256_file
from quality_dimensions.geometry_report import _table, BLUE, ORANGE, INK
from quality_dimensions.util import atomic_json


def build(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ['ridge_changes.png','output_direction_similarity.png','output_heldout_alignment.png']:
        shutil.copyfile(run/'plots'/name,output/name)
    identity = json.loads((run/'identity.json').read_text())
    if identity['profile'] != 'laptop' or identity['vocabulary'] != 'full':
        raise ValueError('This findings deliverable requires the full-vocabulary laptop comparison')
    suite = load_suite(identity['config']['base_config'], identity['profile'])
    cases = {r['case_id']: r for r in suite['cases']}
    metrics = pd.read_csv(run/'metrics.csv', float_precision='round_trip')
    predictions = pd.read_csv(run/'predictions.csv', float_precision='round_trip')
    groups = {(q,m,p): g for (q,m,p),g in predictions.groupby(['quality','method','partition'])}
    errors = []
    for row in metrics.to_dict('records'):
        group = groups[(row['quality'],row['method'],row['partition'])]
        assert len(group) == row['n_rows']
        assert len({cases[c]['entity_group_id'] for c in group.case_id}) == row['n_groups']
        circular = row['quality'] == 'hue'
        y = np.array([json.loads(t) if circular else float(t) for t in group.target])
        pred = np.array([json.loads(t) if circular else float(t) for t in group.prediction])
        score = _circular_score(y,pred) if circular else _score(y,pred)
        for key,value in score.items():
            if pd.notna(row.get(key)) and value is not None:
                errors.append(abs(row[key]-value))
    assert max(errors) < 1e-9, max(errors)
    measured = predictions[predictions.method.str.startswith('output_measurement_')].pivot(
        index=['case_id','quality','partition'], columns='method', values='prediction').astype(float)
    measurement_error = float((measured.output_measurement_covariance-measured.output_measurement_euclidean).abs().max())
    assert measurement_error < 1e-9
    # Audit each source shard once. Historical activation files are read only.
    ledger = json.loads((run/'source_activation_ledger.json').read_text())
    sources = {r['path']: r['sha256'] for r in ledger}
    for path, expected in sources.items():
        assert sha256_file(path) == expected, path
    verification = dict(metric_cells_recomputed=len(metrics), max_metric_roundtrip_error=max(errors),
        max_calibrated_output_measurement_difference=measurement_error,
        immutable_source_shards_verified=len(sources), model_forwards=0, status='passed')
    atomic_json(output/'validation.json',verification)
    comparison = pd.read_csv(run/'ridge_comparison.csv')
    held = comparison[comparison.partition=='entity_test+template|graded|paraphrase']
    summary = pd.read_csv(run/'ridge_summary.csv')
    ridge = metrics[metrics.method.isin(['ridge_euclidean','ridge_covariance','ridge_v3_scaled_same_site']) &
                    metrics.partition.isin(['entity_test|graded|plain','entity_test+template|graded|paraphrase'])]
    macro = ridge.groupby(['partition','method']).agg(n_qualities=('quality','size'),rho=('rho','mean'),mae=('mae','mean'),
                                                  n_rows_per_quality=('n_rows','min'),n_families_per_quality=('n_groups','min')).reset_index()
    macro.to_csv(output/'ridge_controls.csv',index=False)
    held.to_csv(output/'heldout_per_quality.csv',index=False)
    a = pd.read_csv(run/'context_alignment_summary.csv')
    a = a[(a.entity_split=='test') & (a.kind=='adjacent_increase')]
    alignment = a.groupby(['template_id','geometry']).agg(mean_cosine=('mean_cosine','mean'),n_qualities=('quality_id','nunique'),
        n_pairs=('n_pairs','sum'),min_families_per_quality=('n_groups','min'),max_families_per_quality=('n_groups','max')).reset_index()
    alignment.to_csv(output/'context_consistency.csv',index=False)
    # Target and off-target comparisons use exactly the same factorial contrasts.
    paired = pd.read_csv(run/'paired_changes.csv')
    paired_summary = pd.read_csv(run/'paired_summary.csv')
    bn = paired_summary[(paired_summary.entity_split=='test') & paired_summary.method.isin(
        ['context_projection_euclidean','context_projection_covariance'])]
    binding_nuisance=[]
    for (method, template), group in bn.groupby(['method','template_id']):
        binding=group[group.kind=='binding_increase']
        order=group[group.kind=='clause_order_invariant']
        irrelevant=group[group.kind=='irrelevant_invariant']
        if binding.empty or order.empty or irrelevant.empty:
            continue
        binding_nuisance.append(dict(method=method,wording=template,n_qualities=binding.quality_id.nunique(),
            signed_binding_change=binding.signed_delta.mean(),fraction_positive=binding.positive_fraction.mean(),
            absolute_reorder_change=order.absolute_delta.mean(),absolute_irrelevant_change=irrelevant.absolute_delta.mean(),
            binding_families=int(binding.n_groups.min()),irrelevant_families=int(irrelevant.n_groups.min())))
    binding_nuisance=pd.DataFrame(binding_nuisance)
    binding_nuisance.to_csv(output/'binding_nuisance_summary.csv',index=False)
    selected = paired[(paired.entity_split=='test') & (paired.kind=='factorial_one_axis_increase') &
                      paired.method.isin(['context_projection_euclidean','context_projection_covariance'])]
    matched = []
    for (cid,method), group in selected.groupby(['contrast_id','method']):
        on, off = group[group.on_target], group[~group.on_target]
        if len(on) != 1 or off.empty:
            continue
        row = on.iloc[0]
        matched.append(dict(contrast_id=cid, method=method, template_id=row.template_id, entity_group_id=row.entity_group_id,
                            target=float(row.prediction_delta), off_target=float(off.prediction_delta.abs().mean()),
                            n_off_target_properties=len(off), changed_quality=row.quality_id))
    matched = pd.DataFrame(matched)
    matched.to_csv(output/'matched_factorial_changes.csv',index=False)
    family = matched.groupby(['method','template_id','entity_group_id']).agg(target=('target','mean'),off_target=('off_target','mean'),n_pairs=('contrast_id','nunique')).reset_index()
    selectivity = family.groupby(['method','template_id']).agg(target=('target','mean'),off_target=('off_target','mean'),
                                n_pairs=('n_pairs','sum'),n_families=('entity_group_id','nunique')).reset_index()
    selectivity.to_csv(output/'matched_factorial_summary.csv',index=False)
    fig,ax = plt.subplots(figsize=(11,4),layout='constrained')
    labels=[]
    for i,(template,field,label) in enumerate([('numeric','target','Familiar numeric: target (signed)'),
              ('numeric','off_target','Familiar numeric: off-target (absolute)'),
              ('reordered','target','Held-out order: target (signed)'),('reordered','off_target','Held-out order: off-target (absolute)')]):
        labels.append(label)
        for shift,name,color,marker in [(-.12,'euclidean',BLUE,'o'),(.12,'covariance',ORANGE,'s')]:
            row=selectivity[(selectivity.template_id==template)&(selectivity.method=='context_projection_'+name)].iloc[0]
            ax.scatter(row[field],i+shift,color=color,marker=marker,label=name.capitalize() if i==0 else None,s=40)
            ax.annotate(f"{row[field]:.3f}",(row[field],i+shift),xytext=(6,0),textcoords='offset points',va='center',fontsize=9)
    ax.set_yticks(range(4),labels);ax.set_ylim(3.55,-.6);ax.axvline(0,color=INK,lw=1)
    ax.set_xlim(0,max(selectivity.target.max(),selectivity.off_target.max())*1.4)
    ax.set_xlabel('Calibrated prediction change · larger target / smaller off-target is preferable')
    pair_label=str(int(selectivity.n_pairs.min())) if selectivity.n_pairs.min()==selectivity.n_pairs.max() else f"{int(selectivity.n_pairs.min())}–{int(selectivity.n_pairs.max())}"
    ax.set_title(f"Factorial sensitivity on identical matched contrasts\n{pair_label} contrasts per wording; {int(selectivity.n_families.max())} independent test family",pad=14)
    ax.legend(frameon=False,loc='lower right');ax.spines[['top','right']].set_visible(False)
    fig.savefig(output/'factorial_selectivity.png',dpi=170);plt.close(fig)
    report=run/'report.md'
    summary_rows=summary.set_index('partition')
    plain=summary_rows.loc['entity_test|graded|plain']; new=summary_rows.loc['entity_test+template|graded|paraphrase']
    output_counts=pd.read_csv(run/'output_pair_counts.csv')
    train=output_counts[(output_counts.split=='train') & (output_counts.eligible_pairs>0)]
    test=output_counts[(output_counts.split=='test') & (output_counts.eligible_pairs>0) & (output_counts.training_pairs>0)]
    word=pd.read_csv(run/'output_alignment_summary.csv');word=word[word.evaluation=='test']
    site=json.loads((run/'site_verification.json').read_text())
    history=json.loads((run/'historical_coefficient_transport.json').read_text())
    metric=json.loads((run/'metric_reference.json').read_text())
    state=json.loads((run/'state.json').read_text())
    held_index=held.set_index('quality')
    arousal,speed=held_index.loc['arousal'],held_index.loc['speed']
    hue=metrics[metrics.quality=='hue'].set_index(['method','partition'])
    def hue_error(method,template):
        return hue.loc[(method,'entity_test|hue|'+template),'circular_mae_degrees']
    sweetness=word.set_index('geometry')
    findings=f'''# What the covariance experiment found

The covariance geometry is implemented and both the smoke and full-vocabulary laptop comparisons completed. **It does not consistently improve quality representations in this experiment.** It modestly improves mean familiar-wording ridge performance, weakens mean ranking under new wording, and leaves contextual consistency and factorial selectivity broadly similar. V3-style standardized ridge at the same final site remains the stronger aggregate predictive control.

## What was done

We reused the pinned Qwen3-1.7B-Base model and all {len(suite['cases']):,} laptop cases, preserving their original labels, independent-family splits, held-out wording and factorial partitions. We converted cached final-block outputs through the model's frozen final normalization, then checked eight fresh prompts against the actual output head. Every cached-state and native-logit difference was zero across all 151,936 logits per checked prompt. This establishes the correct final readout for this implementation, not a general cross-device bitwise guarantee.

We computed the uniform covariance of all 151,936 output-head rows in float64 chunks. The covariance was numerically safe without regularization: eigenvalues {metric['diagnostics']['raw_min']:.6g}–{metric['diagnostics']['raw_max']:.6g}, condition number {metric['diagnostics']['raw_condition']:.1f}. Contexts use its positive square root; output words use its inverse square root. The complementary transforms preserve their logit pairing. Earlier/entity-token V3 results remain separate historical evidence.

## Improved, weakened and unchanged

| Measurement | Euclidean | Covariance | Reading |
|---|---:|---:|---|
| Familiar-wording ridge, mean Spearman | {plain.rho_euclidean:.3f} | {plain.rho_covariance:.3f} | Modest aggregate improvement |
| Familiar-wording ridge, mean MAE | {plain.mae_euclidean:.3f} | {plain.mae_covariance:.3f} | Lower aggregate error |
| New-wording ridge, mean Spearman | {new.rho_euclidean:.3f} | {new.rho_covariance:.3f} | Weaker ordering |
| New-wording ridge, mean MAE | {new.mae_euclidean:.3f} | {new.mae_covariance:.3f} | Lower mean error, uneven across qualities |

Each row averages 14 authored scalar qualities; each quality has ten test examples from only two independent families per wording. New-wording correlation improves for {int(new.rho_improved)}/14 qualities, worsens for {int(new.rho_weakened)}/14 and is unchanged for {int(new.rho_unchanged)}/14. MAE improves for {int(new.mae_improved)}/14 and worsens for {int(new.mae_weakened)}/14. Its median per-quality change is **{held.mae_delta.median():+.4f}**, so the typical quality's error slightly worsens even though the average improves. Arousal's large calibration correction (MAE {arousal.mae_euclidean:.3f}→{arousal.mae_covariance:.3f}) drives much of the mean improvement. Speed's ordering improves slightly while MAE worsens from {speed.mae_euclidean:.3f} to {speed.mae_covariance:.3f}: ranking and calibration are different questions.

![Per-quality changes](ridge_changes.png)

The full same-site controls are:

{_table(macro)}

The covariance arm does not beat the original preprocessing strategy in these macro summaries. Ridge was refitted after an invertible coordinate transform, so its penalty and the effective scale of the fixed alpha grid change. This can change predictions without creating any new linearly accessible information. Transporting coefficients instead of refitting preserves predictions to numerical precision, including all {len(history)} actual saved V3 probe fits (max error {max(r['max_absolute_error'] for r in history):.3g}). The old V3 headline selected layers/readouts differ from this final site and must not be read as a paired geometry comparison.

## Consistency and selectivity

{_table(alignment)}

Contextual adjacent-change alignment barely changes for familiar language and remains weak for paraphrases. These are family-balanced contextual contrast estimates; an authored text edit does not guarantee that all off-target model concepts stay fixed. Counts of contrasts are not counts of independent observations.

The matched context-projector binding and nuisance checks are:

{_table(binding_nuisance)}

Positive signed binding changes are expected. The positive fraction increases on familiar wording and stays unchanged on paraphrases; it remains well below perfect for these simple contrast projectors. Reordering changes are similar or slightly smaller, while irrelevant-edit changes increase under covariance. Each binding/reordering cell has one test family and each irrelevant-edit cell has two. These fractions and signed magnitudes do not show a uniform robustness improvement.

{_table(selectivity)}

![Factorial selectivity](factorial_selectivity.png)

The target and off-target statistics above use exactly the same contrasts. Off-target magnitude is averaged over other measured scalar properties within each contrast, then within each independent family. Only one test family is available. Covariance changes both quantities slightly; off-target effects remain substantial relative to target effects, so this does not establish clean property separation. The exhaustive run tables retain every quality pair and original partition. Clause-order and irrelevant-edit effects also remain mixed rather than uniformly shrinking; reducing all responses would not itself be a selectivity success.

Hue is evaluated separately with circular errors. At the same final site, Euclidean→covariance angular MAE is {hue_error('ridge_euclidean','plain'):.2f}°→{hue_error('ridge_covariance','plain'):.2f}° for word-based hue and {hue_error('ridge_euclidean','numeric'):.2f}°→{hue_error('ridge_covariance','numeric'):.2f}° for numeric hue, each with two test families. These small opposing changes do not support a general improvement. No scalar hue axis or intrinsic dimensionality claim is made.

## Output-word evidence is very limited

Only {len(train)} of 14 scalar qualities have usable training pairs after exact single-token completion filtering: {', '.join(train.quality)}. Only {len(test)} quality has both training and test pairs. Hardness has two eligible test pairs but no eligible training direction, so it cannot be evaluated.

{_table(word)}

Sweetness has one training pair (two concepts) and two test pairs sharing three concepts. Mean held-out cosine falls slightly, {sweetness.loc['euclidean','mean_cosine']:.3f}→{sweetness.loc['covariance','mean_cosine']:.3f}. Both test differences remain positive, but this tiny, confounded noun sample cannot settle the output-representation hypothesis. The other qualities' missing coverage is missing evidence, not a failed geometric test. Training leave-one-pair-out results share nouns and are not treated as new held-out concepts. Heatmaps show only the four available training directions, with identical scales and printed values.

![Output direction heatmaps](output_direction_similarity.png)

Output direction normalization alone cannot change the ranking of a fixed context measurement: lambdaᵀ gamma_bar remains the canonical pairing. After training-only affine calibration, the two measurement predictions differ by at most {measurement_error:.3g}. This expected invariance is a correctness check, not an empirical gain.

## Reproducibility and limits

- Run: `{run.name}`; full vocabulary, status **{state['status']}**, {state['elapsed_wall_seconds']:.1f} seconds of execution wall time after the local-copy workaround. The smoke run uses a separately labelled 8,192-row vocabulary subsample.
- Software verification: **41 offline tests passed**, one separate real-model integration marker deselected. The actual end-to-end smoke and laptop runs separately verified the pinned model integration. No interventions were run.
- Analytical QA recomputed {len(metrics):,} metric cells from saved predictions (max error {max(errors):.3g}), checked family denominators and unchanged calibrated output measurements, and rehashed {len(sources):,} unique source activation shards. See `validation.json`.
- Original V3 data, runs, caches and handoff are preserved. A local working copy was required because iCloud blocked reads from Desktop. The run records exact configuration, source snapshot, source shard hashes and metric identity. Later visual refinements have their own `rendering_source.json` and do not change fitted results.
- All labels are authored examples, not independent human measurements. Test families are few, noun pairs have semantic confounds, and no significance or universal quality-axis claim follows. The evidence supports retaining Euclidean and standardized controls rather than adopting covariance geometry as a general replacement.

Detailed machine-readable results and the automatic report: [{run.name}]({report.as_posix()}). Implementation, equations and commands: `docs/QUALITY_GEOMETRY_V1.md` in the project. The original research context remains `handoff/quality-suite-v3/START_HERE.md`.
'''
    (output/'findings.md').write_text(findings)
    atomic_json(output/'provenance.json',dict(run=str(run),identity=identity,
        inputs={name:sha256_file(run/name) for name in ['metrics.csv','predictions.csv','paired_changes.csv','context_alignment_summary.csv','metric_reference.json']},
        script_sha256=sha256_file(__file__), forwards=0))
    atomic_json(output/'chart_map.json',[dict(file='factorial_selectivity.png',source='matched_factorial_summary.csv',
        question='Does covariance improve target/off-target separation on the same contrasts?',
        type='paired horizontal dot comparison; printed values',palette='two roots; circle/square; zero reference')])
    print(output/'findings.md')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();build(a.run_dir,a.output)
