from pathlib import Path
import numpy as np
import pandas as pd
from .hooks import intervene
from .util import atomic_json,atomic_npz,read_json,digest
from .analysis import load_fit
from .probes import bootstrap_mean


def random_directions(width,count,seed):
    rng=np.random.default_rng(seed)
    vectors=rng.normal(size=(count,width)); return vectors/np.linalg.norm(vectors,axis=1,keepdims=True)


def semantic_score(logp,larger_label):
    return logp[larger_label]-logp['B' if larger_label=='A' else 'A']


def score_fields(row,logp):
    correct=row['correct_label']; other='B' if correct=='A' else 'A'
    return dict(logp_A=logp['A'],logp_B=logp['B'],correct_odds=logp[correct]-logp[other],
                correct=int(max(logp,key=logp.get)==correct),predicted_label=max(logp,key=logp.get),
                semantic_D=semantic_score(logp,row['larger_label']) if row['task']=='size' else None)


def collect_scores(run):
    run=Path(run); rows=[]
    for path in sorted((run/'behavior_shards').glob('*.json')): rows.append(read_json(path))
    if rows: pd.DataFrame(rows).to_csv(run/'behavior_scores.csv',index=False)
    return rows


def behavior(run,runner,rows,encoding,config,subjects,selected):
    run=Path(run); shards=run/'behavior_shards'; shards.mkdir(exist_ok=True)
    def baseline(row):
        path=shards/(row['prompt_id']+'_baseline.json')
        if path.exists(): return read_json(path)
        logp=runner.score(row,encoding)
        out=dict(**row,**score_fields(row,logp),direction='baseline',strength=0.,layer=0,effect=0.,correct_odds_change=0.,correctness_change=0.)
        atomic_json(path,out); return out
    # Frozen validation diagnostics precede any test behavioral scoring.
    for row in rows:
        if row['split']=='validation': baseline(row)
    validation=[read_json(shards/(r['prompt_id']+'_baseline.json')) for r in rows if r['split']=='validation']
    freeze=run/'behavior_format_selection.json'
    if not freeze.exists():
        atomic_json(freeze,dict(encoding=encoding,templates=config['behavior_templates'],validation_prompt_ids=[r['prompt_id'] for r in validation],
                    validation_accuracy=float(np.mean([r['correct'] for r in validation])),rule='Predetermined plain-text format, no adaptive prompt tuning; validation diagnostic recorded before test scoring'))
    for row in rows:
        if row['split']=='test': baseline(row)
    layers=list(dict.fromkeys([selected]+config.get('extra_intervention_layers',[])))
    for layer in layers:
        fit=load_fit(run,layer); u=fit['direction']; scale=fit['scale']
        random_path=run/'fits'/f'random_block_{layer:02d}.npz'
        if random_path.exists():
            with np.load(random_path,allow_pickle=False) as data: random=data['directions'].copy()
        else:
            random=random_directions(len(u),config['random_directions'],config['seed']+layer)
            atomic_npz(random_path,directions=random)
        directions={'semantic':u,**{f'random_{i+1}':v for i,v in enumerate(random)}}
        for name,direction in directions.items():
            for strength in config['strengths']:
                for row in rows:
                    if row['split']!='test' or row['concept_id'] not in subjects: continue
                    path=shards/(digest([row['prompt_id'],layer,name,strength])+'.json')
                    if path.exists(): continue
                    base=baseline(row)
                    with intervene(runner.blocks[layer-1],row['readout_index'],direction,scale,strength) as stats:
                        logp=runner.score(row,encoding)
                    scored=score_fields(row,logp)
                    if strength==0:
                        tol=2e-3 if runner.backend!='cpu' else 1e-5
                        if max(abs(scored[k]-base[k]) for k in ['logp_A','logp_B'])>tol: raise RuntimeError('Zero-strength hook fails baseline tolerance')
                    out=dict(**row,**scored,**stats,direction=name,strength=float(strength),layer=layer,
                             effect=scored['semantic_D']-base['semantic_D'] if row['task']=='size' else None,
                             correct_odds_change=scored['correct_odds']-base['correct_odds'],correctness_change=scored['correct']-base['correct'],baseline_correct=base['correct'])
                    atomic_json(path,out)
    summarize(run,config)
    atomic_json(run/'behavior_complete.json',dict(layers=layers,subjects=subjects))


def summarize(run,config):
    records=collect_scores(run)
    if not records: return
    run=Path(run); df=pd.DataFrame(records); base=df[df.direction.eq('baseline')]; summaries=[]
    for keys,g in base.groupby(['split','task','polarity','swapped']):
        grouped=g.groupby('synonym_group').correct.mean()
        summaries.append(dict(split=keys[0],task=keys[1],polarity=keys[2],swapped=keys[3],accuracy=g.correct.mean(),fraction_A=g.predicted_label.eq('A').mean(),n_prompts=len(g),**bootstrap_mean(grouped.to_numpy(),config['bootstrap_replicates'],config['seed'])))
    pd.DataFrame(summaries).to_csv(run/'baseline_metrics.csv',index=False)
    effects=[]
    for keys,g in df[~df.direction.eq('baseline')].groupby(['layer','task','direction','strength']):
        for metric in (['effect','correctness_change'] if keys[1]=='size' else ['correct_odds_change','correctness_change']):
            values=g.groupby('synonym_group')[metric].mean().to_numpy()
            effects.append(dict(layer=keys[0],task=keys[1],direction=keys[2],strength=keys[3],metric=metric,n_prompts=len(g),**bootstrap_mean(values,config['bootstrap_replicates'],config['seed'])))
    if effects: pd.DataFrame(effects).to_csv(run/'effect_summary.csv',index=False)
