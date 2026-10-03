"""Nested, grouped development validation of incremental activation information."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,KFold
from .probes import difference_direction,spearman
from .paired import paired_rho_difference
from .util import atomic_json,atomic_npz


def nuisance_matrix(info,training=None):
    if training is None:
        raw=info[['noun_char_count','noun_token_count']].to_numpy(float)
        training=dict(categories=sorted(info.category.unique()),mean=raw.mean(0),sd=np.maximum(raw.std(0),1e-8))
    numeric=(info[['noun_char_count','noun_token_count']].to_numpy(float)-training['mean'])/training['sd']
    category=np.array([[float(c==k) for k in training['categories']] for c in info.category])
    return np.column_stack([category,numeric]),training


def fit_predict(train_info,test_info,train_x,test_x,alpha,use_activation):
    a,pre=nuisance_matrix(train_info); b,_=nuisance_matrix(test_info,pre)
    direction=None
    if use_activation:
        direction=difference_direction(train_x,train_info.size_score.to_numpy(float),train_info.category.to_numpy())
        a=np.column_stack([a,(train_x-direction['mu'])@direction['direction']/direction['scale']])
        b=np.column_stack([b,(test_x-direction['mu'])@direction['direction']/direction['scale']])
    mu=a.mean(0);y=train_info.size_score.to_numpy(float);ym=y.mean();z=a-mu
    coef=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-ym))
    return (b-mu)@coef+ym,dict(nuisance=pre,mean=mu,coef=coef,intercept=ym,direction=direction,alpha=alpha,
                             training_groups=train_info.index.tolist(),unknown_category='all-zero category indicators; centered ridge intercept/numeric terms remain')


def folds(info,n,seed):
    if info.category.value_counts().min()>=n:
        return list(StratifiedKFold(n,shuffle=True,random_state=seed).split(info,info.category))
    return list(KFold(n,shuffle=True,random_state=seed).split(info))


def nested_evaluate(info,x,output,config):
    """info is already one row/group; x is group × block × width."""
    output.mkdir(parents=True,exist_ok=True); predictions=[]; selections=[]
    layers=config['audit_layers'];alphas=config['audit_alphas'];seed=config['seed']
    plans=[('grouped_outer_cv',str(i),tr,te) for i,(tr,te) in enumerate(folds(info,config['outer_folds'],seed))]
    for category in sorted(info.category.unique()):
        plans.append(('leave_one_category_out',category,np.flatnonzero(info.category.to_numpy()!=category),np.flatnonzero(info.category.to_numpy()==category)))
    for scheme,fold,tr,te in plans:
        inner=folds(info.iloc[tr],config['inner_folds'],seed+1)
        selected={}
        for augmented in [False,True]:
            candidates=[]
            for layer in layers if augmented else [layers[0]]:
                for alpha in alphas:
                    scores=[]
                    for it,iv in inner:
                        a=tr[it];b=tr[iv]
                        pred,_=fit_predict(info.iloc[a],info.iloc[b],x[a,layer-1],x[b,layer-1],alpha,augmented)
                        scores.append(spearman(info.iloc[b].size_score,pred))
                    score=float(np.nanmean(scores)) if np.isfinite(scores).any() else -np.inf
                    candidates.append((score,layer,alpha))
            best=max(candidates,key=lambda c:(c[0],-c[1],-c[2]));_,layer,alpha=best
            pred,fit=fit_predict(info.iloc[tr],info.iloc[te],x[tr,layer-1],x[te,layer-1],alpha,augmented)
            method='nuisance_plus_direction' if augmented else 'nuisance_only';selected[method]=pred
            stem=output/f'{scheme}_{fold}_{method}'
            atomic_json(stem.with_suffix('.json'),dict(layer=layer,alpha=alpha,fit=fit,inner_grid=candidates,training_groups=info.index[tr].tolist(),excluded_groups=info.index[te].tolist(),inner_splits=[dict(train=info.index[tr[a]].tolist(),validation=info.index[tr[b]].tolist()) for a,b in inner]))
            selections.append(dict(scheme=scheme,fold=fold,method=method,layer=layer,alpha=alpha))
        cat_means=info.iloc[tr].groupby('category').size_score.mean()
        for j,k in enumerate(te):
            row=info.iloc[k]
            predictions.append(dict(scheme=scheme,fold=fold,synonym_group=info.index[k],category=row.category,target=row.size_score,
                                    nuisance_only=selected['nuisance_only'][j],nuisance_plus_direction=selected['nuisance_plus_direction'][j],category_only=cat_means.get(row.category,info.iloc[tr].size_score.mean())))
    pred=pd.DataFrame(predictions);pred.to_csv(output.parent/'nuisance_predictions.csv',index=False)
    pd.DataFrame(selections).to_csv(output.parent/'nuisance_selections.csv',index=False)
    summaries=[]
    for scheme,g in pred.groupby('scheme'):
        for category,sub in [('all',g)]+list(g.groupby('category')):
            summaries.append(dict(scheme=scheme,category=category,baseline_rho=spearman(sub.target,sub.nuisance_only),augmented_rho=spearman(sub.target,sub.nuisance_plus_direction),category_only_rho=spearman(sub.target,sub.category_only),
                     **paired_rho_difference(sub.target,sub.nuisance_only,sub.nuisance_plus_direction,config['bootstrap_replicates'],seed)))
    pd.DataFrame(summaries).to_csv(output.parent/'nuisance_advantage.csv',index=False)
    return pred,summaries
