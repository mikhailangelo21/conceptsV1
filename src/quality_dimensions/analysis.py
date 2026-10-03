from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .probes import fit_layer,choose_layer,spearman,bootstrap_rho,bootstrap_mean,shuffle_within_category,difference_direction,ridge_grid
from .util import atomic_json,atomic_npz,read_json,digest


def concept_vectors(rows,activations,concepts,split,templates,condition='neutral'):
    meta=concepts.set_index('concept_id'); groups={}
    for row,a in zip(rows,activations):
        if row['split']==split and row['template_id'] in templates and row['condition']==condition:
            group=meta.loc[row['concept_id'],'synonym_group']; groups.setdefault(group,[]).append(a)
    ids=sorted(groups)
    info=concepts.drop_duplicates('synonym_group').set_index('synonym_group').loc[ids]
    return np.stack([np.mean(groups[g],axis=0) for g in ids]),info,ids


def save_fit(path,fit):
    arrays={k:v for k,v in fit.items() if isinstance(v,np.ndarray)}
    atomic_npz(path.with_suffix('.npz'),**arrays)
    atomic_json(path.with_suffix('.json'),{k:v for k,v in fit.items() if k not in arrays})


def load_fit(run,layer):
    p=Path(run)/'fits'/f'block_{layer:02d}'
    out=read_json(p.with_suffix('.json'))
    with np.load(p.with_suffix('.npz'),allow_pickle=False) as f: out.update({k:f[k].copy() for k in f.files})
    return out


def analyze(run,config,concepts,rows,activations):
    run=Path(run); (run/'fits').mkdir(exist_ok=True)
    tr,ti,tids=concept_vectors(rows,activations,concepts,'train',config['seen_templates'])
    va,vi,vids=concept_vectors(rows,activations,concepts,'validation',config['seen_templates'])
    y=ti.size_score.to_numpy(float); vy=vi.size_score.to_numpy(float); cats=ti.category.to_numpy()
    fits={}
    # No held-out features or labels enter fitting/selection.
    for layer in range(1,tr.shape[1]+1):
        fits[layer]=fit_layer(tr[:,layer-1],y,cats,va[:,layer-1],vy,config['ridge_alphas'],tids)
        save_fit(run/'fits'/f'block_{layer:02d}',fits[layer])
    selected=choose_layer(fits)
    manifest=dict(selected_layer=selected,training_groups=tids,validation_groups=vids,fit_templates=config['seen_templates'],
                  heldout_template=config['heldout_template'],layer_rule='highest validation mean-difference Spearman; earliest block breaks ties',
                  alpha_rule='highest validation Spearman; smallest alpha breaks ties',coordinate_system='unwhitened raw residual; train-only centering',
                  size_groups='within-category floor(n/3), expand boundary ties; skip overlapping cuts or n<3',
                  bootstrap_unit='synonym group; all variants retained together; analysis aggregates variants before resampling',
                  permutation_scope='selected block, fixed after original validation selection; shuffle training group labels within category; refit direction and validation-tuned ridge')
    atomic_json(run/'selection.json',manifest)  # persisted BEFORE test evaluation
    evaluations=[('validation_seen','validation',config['seen_templates']),('test_seen','test',config['seen_templates']),('test_heldout','test',[config['heldout_template']])]
    metrics=[]; predictions=[]; context=[]; pca_rows=[]
    category_means=ti.groupby('category').size_score.mean().to_dict()
    def noun_features(info): return info[['noun_char_count','noun_token_count']].to_numpy(float)
    base=ridge_grid(noun_features(ti),y,noun_features(vi),vy,config['ridge_alphas'])
    atomic_npz(run/'fits'/'noun_baseline.npz',coef=base['coef'],mu=base['mu'])
    atomic_json(run/'fits'/'baselines.json',dict(training_groups=tids,category_means=category_means,noun_intercept=base['intercept'],noun_alpha=base['alpha']))
    for name,split,templates in evaluations:
        x,info,ids=concept_vectors(rows,activations,concepts,split,templates); truth=info.size_score.to_numpy(float)
        for layer,fit in fits.items():
            z=x[:,layer-1]-fit['mu']
            preds={'direction':z@fit['direction'],'ridge':z@fit['ridge_coef']+fit['ridge_intercept']}
            if layer==selected:
                preds.update(category=np.array([category_means.get(c,float(y.mean())) for c in info.category]),
                             noun=(noun_features(info)-base['mu'])@base['coef']+base['intercept'])
            for method,pred in preds.items():
                ci=bootstrap_rho(truth,pred,config['bootstrap_replicates'],config['seed'])
                metric=dict(evaluation=name,layer=layer,method=method,n_groups=len(ids),rho=spearman(truth,pred),**ci)
                if method!='direction':
                    metric.update(mae=float(abs(truth-pred).mean()),r2=float(1-((truth-pred)**2).sum()/((truth-truth.mean())**2).sum()) if np.ptp(truth)>0 else None)
                metrics.append(metric)
                for g,target,value in zip(ids,truth,pred): predictions.append(dict(evaluation=name,layer=layer,method=method,synonym_group=g,target=target,prediction=float(value)))
    # Projection changes: original fitted direction applied at every block.
    bykey={(r['concept_id'],r['template_id'],r['condition']):a for r,a in zip(rows,activations)}
    for row,a in zip(rows,activations):
        meta=concepts.set_index('concept_id').loc[row['concept_id']]
        neutral=bykey[(row['concept_id'],row['template_id'],'neutral')]
        for layer,fit in fits.items():
            score=float((a[layer-1]-fit['mu'])@fit['direction']/fit['scale'])
            change=float((a[layer-1]-neutral[layer-1])@fit['direction']/fit['scale'])
            context.append(dict(**{k:row[k] for k in ['prompt_id','concept_id','synonym_group','condition','template_id','split','token_count']},layer=layer,score=score,change=change))
        fit=fits[selected]; xy=(a[selected-1]-fit['pca_mean'])@fit['pca_components'][:2].T
        pca_rows.append(dict(**{k:row[k] for k in ['concept_id','condition','template_id','split']},category=meta.category,size_score=float(meta.size_score),pc1=float(xy[0]),pc2=float(xy[1]) if len(xy)>1 else 0.))
    pd.DataFrame(metrics).to_csv(run/'metrics.csv',index=False); pd.DataFrame(predictions).to_csv(run/'predictions.csv',index=False)
    context=pd.DataFrame(context); context.to_csv(run/'context.csv',index=False); pd.DataFrame(pca_rows).to_csv(run/'pca.csv',index=False)
    context_summary=[]
    for (layer,split,template,condition),g in context.groupby(['layer','split','template_id','condition']):
        means=g.groupby('synonym_group').change.mean()
        context_summary.append(dict(layer=layer,split=split,template_id=template,condition=condition,**bootstrap_mean(means.to_numpy(),config['bootstrap_replicates'],config['seed'])))
    pd.DataFrame(context_summary).to_csv(run/'context_summary.csv',index=False)
    # Descriptive category-preserving null at the frozen primary layer.
    rng=np.random.default_rng(config['seed']+201); null=[]; train=tr[:,selected-1]; valid=va[:,selected-1]
    for permutation in range(config['permutations']):
        shuffled=shuffle_within_category(y,cats,rng)
        try: direction=difference_direction(train,shuffled,cats)
        except ValueError:
            null.append(dict(permutation=permutation,valid=False)); continue
        ridge=ridge_grid(train,shuffled,valid,vy,config['ridge_alphas'])
        for name,split,templates in evaluations[1:]:
            x,info,ids=concept_vectors(rows,activations,concepts,split,templates); target=info.size_score.to_numpy(float); z=x[:,selected-1]-direction['mu']
            null.append(dict(permutation=permutation,valid=True,evaluation=name,layer=selected,direction_rho=spearman(target,z@direction['direction']),ridge_rho=spearman(target,(x[:,selected-1]-ridge['mu'])@ridge['coef']+ridge['intercept']),ridge_alpha=ridge['alpha']))
    pd.DataFrame(null).to_csv(run/'permutations.csv',index=False)
    atomic_json(run/'analysis_complete.json',dict(selected_layer=selected,analysis_fingerprint=digest([config,concepts.to_dict('records')]),n_training_groups=len(tids)))
    return selected
