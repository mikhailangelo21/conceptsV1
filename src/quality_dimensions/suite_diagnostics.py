"""Matched contrasts, physical challenges, circular and multi-output diagnostics."""
from collections import defaultdict
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .util import atomic_json


def reduced_rank_prediction(train_x, train_y, eval_x, alpha, rank):
    """Train-only output projection; rank bounded by centered samples/features/outputs."""
    limit = min(len(train_x) - 1, train_x.shape[1], train_y.shape[1], np.linalg.matrix_rank(train_y - train_y.mean(0)))
    if not 1 <= rank <= limit:
        raise ValueError('Reduced-rank regression exceeds the effective output/sample rank')
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='lsqr')).fit(train_x, train_y)
    fitted = model.predict(train_x); center = train_y.mean(0)
    _, _, vh = np.linalg.svd(fitted - center, full_matrices=False)
    projector = vh[:rank].T @ vh[:rank]
    return (model.predict(eval_x) - center) @ projector + center


def numeric_features(text):
    values = [float(v) for v in re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?', text)]
    values = (values + [0.] * 6)[:6]
    return values + [float(unit in text) for unit in [' cm', ' metre', ' mm', ' kg', ' g', 'Hz', 'dB', 'Celsius', 'Fahrenheit']]


def diagnostics(run, suite, activation, selections, models):
    from .suite_analysis import _eligible_fit, _partition, _ridge_cv, _score, _circular_score
    blocks = suite['config']['extraction_block_numbers']; alphas = suite['config']['ridge_alphas']
    cases = {r['case_id']: r for r in suite['cases']}
    paired = []; effects = []; missing = []
    def vector(case_id, quality):
        s = selections[quality]
        return activation[(case_id, s['readout'])][blocks.index(s['block'])]
    for c in suite['contrasts']:
        low, high = cases.get(c['low_case_id']), cases.get(c['high_case_id'])
        if low is None or high is None:
            missing.append(c['contrast_id']); continue
        q = c['quality_id']
        for output in models:
            if output.startswith('lexical__') or output == 'hue': continue
            if output != q and c['kind'] != 'factorial_one_axis_increase': continue
            if output not in low['targets'] or output not in high['targets']: continue
            pred = models[output].predict(np.stack([vector(low['case_id'], output), vector(high['case_id'], output)]))
            row = dict(contrast_id=c['contrast_id'], quality=q, measured_quality=output, kind=c['kind'],
                       entity_group_id=low['entity_group_id'], entity_split=low['entity_split'],
                       template_id=low['template_id'], template_split=low['template_split'],
                       combination_split=low['combination_split'], prediction_low=float(pred[0]),
                       prediction_high=float(pred[1]), prediction_delta=float(pred[1]-pred[0]),
                       target_delta=high['targets'][output]-low['targets'][output])
            paired.append(row)
            if c['kind'] == 'factorial_one_axis_increase': effects.append(row)
    pd.DataFrame(paired).to_csv(run/'paired_contrasts.csv', index=False)
    pd.DataFrame(effects).to_csv(run/'factorial_selectivity.csv', index=False)
    summary = []
    if paired:
        frame = pd.DataFrame(paired)
        for keys, group in frame.groupby(['quality', 'measured_quality', 'kind', 'entity_split', 'template_split', 'combination_split']):
            # Bootstrap whole family means: repeated templates/contrasts never become independent draws.
            values = group.groupby('entity_group_id').prediction_delta.mean().to_numpy()
            rng = np.random.default_rng(20260928)
            boot = rng.choice(values, size=(300, len(values)), replace=True).mean(1)
            summary.append(dict(zip(['quality','measured_quality','kind','entity_split','template_split','combination_split'], keys),
                                n_pairs=len(group), n_groups=len(values), mean_delta=float(values.mean()),
                                ci_low=float(np.quantile(boot,.025)) if len(values)>1 else None,
                                ci_high=float(np.quantile(boot,.975)) if len(values)>1 else None))
    pd.DataFrame(summary).to_csv(run/'paired_summary.csv', index=False)

    # Local directions are averages of matched pairs within a range/template/domain,
    # not differences between unmatched marginal means.
    local=defaultdict(list)
    for c in suite['contrasts']:
        a,b=cases.get(c['low_case_id']),cases.get(c['high_case_id']); q=c['quality_id']
        if a is None or b is None or q not in selections or q=='hue': continue
        if not _eligible_fit(a) or not _eligible_fit(b) or c['kind'] not in ['adjacent_increase','factorial_one_axis_increase']: continue
        key=(q,a['targets'][q],b['targets'][q],a['template_id'],a.get('factorial_domain','graded'))
        local[key].append((a['entity_group_id'],vector(b['case_id'],q)-vector(a['case_id'],q)))
    local_bases={}; local_stats=[]
    for key,pairs in local.items():
        groups=sorted({g for g,_ in pairs})
        matrix=np.stack([np.mean([d for g,d in pairs if g==group],axis=0) for group in groups])
        _,singular,vh=np.linalg.svd(matrix,full_matrices=False); rank=min(2,np.linalg.matrix_rank(matrix))
        if rank==0: continue
        direction=matrix.mean(0); direction/=np.linalg.norm(direction) or 1
        basis=vh[:rank]; local_bases[key]=(direction,basis)
        rng=np.random.default_rng(20260928); stability=[]
        for _ in range(30):
            sample=matrix[rng.integers(len(matrix),size=len(matrix))]
            _,_,boot=np.linalg.svd(sample,full_matrices=False)
            br=min(rank,np.linalg.matrix_rank(sample))
            if br: stability.append(float(np.linalg.norm(basis @ boot[:br].T,ord='fro')**2/min(rank,br)))
        local_stats.append(dict(quality=key[0],low=key[1],high=key[2],template=key[3],domain=key[4],n_groups=len(groups),
                                n_pairs=len(pairs),rank=rank,bootstrap_projector_overlap=float(np.mean(stability)),singular_values=singular.tolist()))
    align=[]
    for i,(ka,(da,ba)) in enumerate(local_bases.items()):
        for kb,(db,bb) in list(local_bases.items())[i+1:]:
            if ka[0]!=kb[0]: continue
            angles=np.degrees(np.arccos(np.clip(np.linalg.svd(ba@bb.T,compute_uv=False),0,1)))
            align.append(dict(quality=ka[0],range_template_domain_a=str(ka[1:]),range_template_domain_b=str(kb[1:]),
                              cosine=float(da@db),principal_angles_degrees=angles.tolist()))
    pd.DataFrame(align).to_csv(run/'matched_local_alignment.csv',index=False)
    atomic_json(run/'local_subspace_stability.json',local_stats)

    # Circular neighborhoods and 0/360 equivalence, per readout/block.
    circular = []
    for triplet in suite['triplets']:
        ids = [triplet[k] for k in ['anchor_case_id','near_case_id','far_case_id']]
        if not all(i in cases for i in ids): continue
        for readout in ['entity','final']:
            for li, block in enumerate(blocks):
                a,n,f = [activation[(i,readout)][li] for i in ids]
                dn,df = float(np.linalg.norm(a-n)), float(np.linalg.norm(a-f))
                circular.append(dict(triplet_id=triplet['triplet_id'], entity_group_id=triplet['entity_group_id'],
                                     entity_split=triplet['entity_split'], template_id=triplet['template_id'],
                                     readout=readout, block=block, near_distance=dn, far_distance=df, near_is_closer=dn<df))
    pd.DataFrame(circular).to_csv(run/'circular_neighborhoods.csv', index=False)

    # Physical-unit prediction is fitted on stated training values, never ordinal codes or challenge rows.
    physical = []; physical_choices = []
    qualities = sorted({q.removesuffix('__value') for r in cases.values() if r['case_type']=='numeric' for q in r['targets']})
    for q in qualities:
        train = [r for r in cases.values() if _eligible_fit(r) and q in r['specified_values']]
        valid = [r for r in cases.values() if r['case_type']!='numeric' and r['entity_split']=='validation'
                 and r['template_split']=='train' and r['combination_split']=='train' and q in r['specified_values']]
        challenge = [r for r in cases.values() if r['case_type']=='numeric' and q+'__value' in r['targets']]
        if len(train)<4 or len({r['entity_group_id'] for r in train})<2 or not valid: continue
        y = np.array([r['specified_values'][q] for r in train]); vy=np.array([r['specified_values'][q] for r in valid])
        choices=[]
        for readout in ['entity','final']:
            for li,block in enumerate(blocks):
                x=np.stack([activation[(r['case_id'],readout)][li] for r in train])
                vx=np.stack([activation[(r['case_id'],readout)][li] for r in valid])
                fit,alpha=_ridge_cv(x,y,[r['entity_group_id'] for r in train],alphas)
                loss=float(np.square(fit.predict(vx)-vy).mean())
                choices.append((loss,block,readout,alpha,fit))
        loss,block,readout,alpha,fit=min(choices,key=lambda v:v[:3])
        physical_choices.append(dict(quality=q,block=block,readout=readout,alpha=alpha,transform='identity; canonical physical units',validation_mse=loss,
                                     training_case_ids=[r['case_id'] for r in train]))
        literal,la=_ridge_cv(np.array([numeric_features(r['text']) for r in train]),y,[r['entity_group_id'] for r in train],alphas)
        for r in challenge:
            v=activation[(r['case_id'],readout)][blocks.index(block)]
            for method,pred in [('physical_ridge',fit.predict(v[None])[0]),('numeric_literal',literal.predict([numeric_features(r['text'])])[0])]:
                physical.append(dict(case_id=r['case_id'],quality=q,method=method,value_role=r['value_role'],condition=r['condition'],
                                     entity_group_id=r['entity_group_id'],entity_split=r['entity_split'],template_id=r['template_id'],
                                     target=r['targets'][q+'__value'],prediction=float(pred)))
    pd.DataFrame(physical).to_csv(run/'numeric_challenges.csv',index=False)
    atomic_json(run/'physical_model_selection.json',physical_choices)
    unit_pairs=[]
    pmap={(r['case_id'],r['method']):r for r in physical}
    for c in suite['contrasts']:
        if c['kind']!='unit_conversion_invariant': continue
        for method in ['physical_ridge','numeric_literal']:
            a=pmap.get((c['low_case_id'],method)); b=pmap.get((c['high_case_id'],method))
            if a and b: unit_pairs.append(dict(contrast_id=c['contrast_id'],quality=c['quality_id'],method=method,
                                               canonical_target=a['target'],prediction_delta=b['prediction']-a['prediction']))
    pd.DataFrame(unit_pairs).to_csv(run/'unit_invariance.csv',index=False)

    # Joint factorial outputs and reduced-rank models. Hue uses coupled sin/cos coordinates.
    multi=[]; multi_selection=[]
    domains=sorted({r['factorial_domain'] for r in cases.values() if r['case_type']=='factorial'})
    for domain in domains:
        rows=[r for r in cases.values() if r.get('factorial_domain')==domain]
        train=[r for r in rows if _eligible_fit(r)]
        valid=[r for r in rows if r['entity_split']=='validation' and r['template_split']=='train' and r['combination_split']=='train']
        if len(train)<4 or not valid or len({r['entity_group_id'] for r in train})<2: continue
        outputs=sorted(train[0]['targets']); y=np.array([[r['targets'][q] for q in outputs] for r in train])
        vy=np.array([[r['targets'][q] for q in outputs] for r in valid]); trials=[]
        for readout in ['entity','final']:
            for li,block in enumerate(blocks):
                x=np.stack([activation[(r['case_id'],readout)][li] for r in train]); vx=np.stack([activation[(r['case_id'],readout)][li] for r in valid])
                fit,alpha=_ridge_cv(x,y,[r['entity_group_id'] for r in train],alphas)
                limit=min(len(train)-1,x.shape[1],len(outputs),np.linalg.matrix_rank(y-y.mean(0)))
                for rank in range(1,limit+1):
                    pred=reduced_rank_prediction(x,y,vx,alpha,rank)
                    trials.append(dict(block=block,readout=readout,rank=rank,alpha=alpha,validation_mse=float(np.square(pred-vy).mean())))
        chosen=min(trials,key=lambda r:(r['validation_mse'],r['rank'],r['block'],r['readout']))
        multi_selection.append(dict(domain=domain,outputs=outputs,chosen=chosen,trials=trials,training_case_ids=[r['case_id'] for r in train]))
        li=blocks.index(chosen['block']); ro=chosen['readout']; x=np.stack([activation[(r['case_id'],ro)][li] for r in train])
        for partition in sorted({_partition(r) for r in rows}):
            ev=[r for r in rows if _partition(r)==partition]; ex=np.stack([activation[(r['case_id'],ro)][li] for r in ev])
            pred=reduced_rank_prediction(x,y,ex,chosen['alpha'],chosen['rank'])
            for j,q in enumerate(outputs):
                if q in ['hue_sin','hue_cos']: continue
                multi.append(dict(domain=domain,partition=partition,quality=q,rank=chosen['rank'],n_rows=len(ev),
                                  n_groups=len({r['entity_group_id'] for r in ev}),**_score(np.array([r['targets'][q] for r in ev]),pred[:,j])))
            if 'hue_sin' in outputs:
                inds=[outputs.index(q) for q in ['hue_sin','hue_cos']]
                truth=np.array([[r['targets'][q] for q in ['hue_sin','hue_cos']] for r in ev])
                multi.append(dict(domain=domain,partition=partition,quality='hue',rank=chosen['rank'],n_rows=len(ev),**_circular_score(truth,pred[:,inds])))
    pd.DataFrame(multi).to_csv(run/'multioutput_metrics.csv',index=False)
    atomic_json(run/'multioutput_selection.json',multi_selection)
    atomic_json(run/'diagnostic_coverage.json',dict(missing_contrast_endpoints=missing,paired_rows=len(paired),numeric_predictions=len(physical),
                                                  circular_rows=len(circular),factorial_domains=domains,
                                                  causal_status='disabled_by_suite_configuration; factorial_selectivity is observational, not causal'))
