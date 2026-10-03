"""Reproduce the V3 interpretation tables without inference or modifying a run.

Run from project root: .venv/bin/python analysis/v3_findings.py
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from quality_dimensions.suite_data import load_suite

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/quality_suite_v3_laptop-88fad55b46b0'
OUT = ROOT / 'reports/quality_suite_v3_findings'
OUT.mkdir(parents=True, exist_ok=True)
sources = {}
datasets = {}

def read(name):
    path = RUN / name
    sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)

def save(name, frame):
    frame.to_csv(OUT / (name + '.csv'), index=False)
    datasets[name] = json.loads(frame.to_json(orient='records'))
    return frame

suite = load_suite(ROOT / 'configs/quality_suite_v3.json', 'laptop')
cases = {r['case_id']: r for r in suite['cases']}
linear = read('linear_metrics.csv')
test = linear[linear.partition.str.contains('entity_test')].copy()
test['family'] = test.partition.str.split('|').str[1]
test['wording'] = test.partition.str.split('|').str[2]
graded = save('graded', test[(test.family == 'graded') & test.method.isin(['ridge', 'static_embedding', 'unigram_tfidf'])])
save('lexical', test[(test.family == 'lexical') & (test.method == 'ridge')].assign(quality=lambda x:x.quality.str.replace('lexical__','')))
save('binding', test[(test.family == 'binding') & test.method.isin(['ridge', 'static_embedding', 'unigram_tfidf'])])
save('hue', test[(test.quality == 'hue') & (test.method == 'ridge')])

pair = read('paired_contrasts.csv')
pair = pair[pair.entity_split == 'test'].copy()
pair['absolute_delta'] = pair.prediction_delta.abs()
pair['positive'] = pair.prediction_delta > 0
paired = save('paired', pair.groupby(['quality','measured_quality','kind','template_id'], as_index=False).agg(n_pairs=('contrast_id','size'),n_groups=('entity_group_id','nunique'),mean_delta=('prediction_delta','mean'),mean_absolute_delta=('absolute_delta','mean'),fraction_positive=('positive','mean')))

behavior = read('behavior_scores.csv')
assert behavior.comparison_id.is_unique
assert set(behavior.comparison_id) == {q['comparison_id'] for q in suite['questions']}
meta = pd.DataFrame(suite['questions'])[['comparison_id','template_id','level_index','reversed_comparison']]
behavior = behavior.merge(meta, validate='one_to_one')
btest = behavior[behavior.entity_split == 'test'].copy()
save('behavior', btest.groupby(['quality_id','format','template_id'],as_index=False).agg(n=('correct','size'),n_groups=('entity_group_id','nunique'),accuracy=('correct','mean'),mean_correct_margin=('correct_margin','mean')))
single = behavior[behavior.single_token].copy()
single['sign_flip'] = (single.correct_margin > 0) != (single.precise_correct_margin > 0)
save('precision', single.groupby(['entity_split','format'],as_index=False).agg(n=('correct','size'),sign_flips=('sign_flip','sum'),max_disagreement=('precision_disagreement','max')))
keys = ['quality_id','entity_group_id','template_id','level_index','reversed_comparison']
natural = btest[btest.format == 'natural_continuation'][keys+['correct']].rename(columns={'correct':'natural_accuracy'})
ab = btest[btest.format != 'natural_continuation'].groupby(keys,as_index=False).agg(ab_accuracy=('correct','mean'),ab_variants=('correct','size'))
matched = natural.merge(ab, on=keys, validate='one_to_one')
assert (matched.ab_variants == 4).all()
save('matched_behavior',matched.groupby(['quality_id','template_id'],as_index=False).agg(n_comparisons=('natural_accuracy','size'),n_groups=('entity_group_id','nunique'),natural_accuracy=('natural_accuracy','mean'),ab_accuracy=('ab_accuracy','mean')))

numeric = read('numeric_challenges.csv')
models = json.loads((RUN/'physical_model_selection.json').read_text())
ranges = {}
for m in models:
    values = [cases[c]['specified_values'][m['quality']] for c in m['training_case_ids']]
    ranges[m['quality']] = max(values)-min(values)
assert all(v > 0 for v in ranges.values())
numeric['absolute_error'] = (numeric.prediction - numeric.target).abs()
numeric['normalized_absolute_error'] = numeric.absolute_error / numeric.quality.map(ranges)
n = numeric[numeric.entity_split == 'test']
save('numeric',n.groupby(['quality','method','value_role'],as_index=False).agg(n=('case_id','size'),n_groups=('entity_group_id','nunique'),mae=('absolute_error','mean'),range_normalized_mae=('normalized_absolute_error','mean')))
save('numeric_overall',n.groupby(['quality','method'],as_index=False).agg(n=('case_id','size'),n_groups=('entity_group_id','nunique'),mae=('absolute_error','mean'),range_normalized_mae=('normalized_absolute_error','mean')))
units = read('unit_invariance.csv')
contrasts = pd.DataFrame(suite['contrasts'])[['contrast_id','entity_split','entity_group_id']]
units = units.merge(contrasts,on='contrast_id',validate='many_to_one')
units['absolute_delta'] = units.prediction_delta.abs()
save('units', units[units.entity_split == 'test'].groupby(['quality','method'],as_index=False).agg(n_pairs=('contrast_id','size'),n_groups=('entity_group_id','nunique'),mean_absolute_delta=('absolute_delta','mean')))

multi = read('multioutput_metrics.csv')
save('factorial', multi[multi.partition.str.contains('entity_test')].copy())
circ = read('circular_neighborhoods.csv')
selection = json.loads((RUN/'model_selection.json').read_text())
hs = selection['hue']
circ = circ[(circ.entity_split=='test') & (circ.block==hs['block']) & (circ.readout==hs['readout'])]
save('hue_neighborhood',circ.groupby('template_id',as_index=False).agg(n_triplets=('triplet_id','size'),n_groups=('entity_group_id','nunique'),fraction_near_closer=('near_is_closer','mean')))
counts=[]
for (family,split),g in pd.DataFrame(suite['cases']).groupby(['case_type','entity_split']):
    counts.append(dict(family=family,split=split,rows=len(g),entity_groups=g.entity_group_id.nunique()))
save('counts',pd.DataFrame(counts))
assert len(suite['cases']) == 6322 and len(behavior)==1888
assert len(n)==270 and len(btest)==416
predictions = read('linear_predictions.csv')
checked = 0
max_error = 0.0
for (quality, partition), group in predictions.groupby(['quality','partition']):
    saved = linear[(linear.quality==quality) & (linear.partition==partition) & (linear.method=='ridge')]
    assert len(saved)==1
    mae = np.mean(np.abs(group.prediction-group.target))
    delta = abs(mae-saved.iloc[0].mae)
    max_error=max(max_error,float(delta))
    assert delta < 1e-6
    if pd.notna(saved.iloc[0].rho) and group.target.nunique()>1 and group.prediction.nunique()>1:
        assert abs(spearmanr(group.target,group.prediction).statistic-saved.iloc[0].rho)<1e-6
    checked+=1
bs = read('behavior_summary.csv')
for keys_,group in behavior.groupby(['quality_id','format','entity_split','template_split']):
    saved=bs
    for key,value in zip(['quality_id','format','entity_split','template_split'],keys_):
        saved=saved[saved[key]==value]
    assert len(saved)==1 and int(saved.iloc[0].n)==len(group)
    assert abs(saved.iloc[0].accuracy-group.correct.mean())<1e-9
(OUT/'validation.json').write_text(json.dumps({'status':'passed','scalar_partitions_recomputed':checked,'maximum_mae_roundtrip_difference':max_error,'behavioral_rows_reconciled':len(behavior),'matched_base_comparisons':len(matched),'test_single_token_sign_flips':int(single[single.entity_split=='test'].sign_flip.sum()),'checks':['profile coverage','unique IDs','one-to-one metadata joins','four A/B variants per matched comparison','positive training ranges','scalar MAE and nonconstant Spearman recomputation','all behavioral summary denominators and accuracies']},indent=2))
summary = {
 'graded_ridge_mean_rho':graded[graded.method=='ridge'].groupby('wording').rho.mean().to_dict(),
 'graded_ridge_mean_mae':graded[graded.method=='ridge'].groupby('wording').mae.mean().to_dict(),
 'precision_sign_flips':int(single.sign_flip.sum()),
 'matched_behavior':matched.groupby('template_id')[['natural_accuracy','ab_accuracy']].mean().to_dict(),
 'training_physical_ranges':ranges,
 'test_binding':datasets['binding'],
 'hue_neighborhood':datasets['hue_neighborhood'],
 'numeric_overall':datasets['numeric_overall'],
 'paired_kinds': sorted(pair.kind.unique()),
}
(OUT/'datasets.json').write_text(json.dumps(datasets,indent=2,allow_nan=False))
(OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
(OUT/'provenance.json').write_text(json.dumps({'run':RUN.name,'input_sha256':sources,'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'CPU-only analysis; original run artifacts not changed'},indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='test_binding'},indent=2))
