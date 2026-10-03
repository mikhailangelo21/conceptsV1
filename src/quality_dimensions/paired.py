"""Strict paired estimands; bootstrap independent groups, never condition rows."""
import numpy as np
import pandas as pd
from .probes import spearman,bootstrap_mean


def strict_pivot(frame,keys,condition,value,required):
    if frame.duplicated(keys+[condition]).any(): raise ValueError('Duplicate paired rows')
    out=frame.pivot(index=keys,columns=condition,values=value)
    if not set(required)<=set(out) or out[list(required)].isna().any().any(): raise ValueError('Incomplete paired conditions')
    return out


def context_contrasts(frame):
    keys=['concept_id','synonym_group','template_id']
    wide=strict_pivot(frame,keys,'condition','score',['enlarged','shrunk','distractor_enlarged','distractor_shrunk'])
    out=wide.reset_index()[keys].copy()
    out['subject_contrast']=(wide.enlarged-wide.shrunk).to_numpy()
    out['distractor_contrast']=(wide.distractor_enlarged-wide.distractor_shrunk).to_numpy()
    out['binding_contrast']=out.subject_contrast-out.distractor_contrast
    return out


def mapping_decomposition(frame):
    keys=['concept_id','synonym_group','reference','template_id','polarity','layer','strength','direction']
    wide=strict_pivot(frame,keys,'larger_label','delta_L',['A','B'])
    out=wide.reset_index()[keys].copy()
    out['delta_L_plus']=wide.A.to_numpy();out['delta_L_minus']=wide.B.to_numpy()
    out['semantic_component']=(out.delta_L_plus-out.delta_L_minus)/2
    out['common_A_component']=(out.delta_L_plus+out.delta_L_minus)/2
    return out


def paired_rho_difference(y,baseline,augmented,reps,seed):
    """All inputs one fixed prediction per independent concept group."""
    rng=np.random.default_rng(seed); y=np.asarray(y);a=np.asarray(baseline);b=np.asarray(augmented)
    diffs=[]
    for _ in range(reps):
        ix=rng.integers(len(y),size=len(y))
        value=spearman(y[ix],b[ix])-spearman(y[ix],a[ix])
        if np.isfinite(value): diffs.append(value)
    return dict(difference=spearman(y,b)-spearman(y,a),low=float(np.quantile(diffs,.025)) if diffs else None,
                high=float(np.quantile(diffs,.975)) if diffs else None,valid_replicates=len(diffs),degenerate_replicates=reps-len(diffs),n_groups=len(y),
                uncertainty='paired bootstrap of fixed excluded predictions, conditional on fitted models; excludes label and refitting uncertainty')


def group_summary(frame,columns,group='synonym_group',reps=1000,seed=1729):
    rows=[]
    for col in columns:
        values=frame.groupby(group)[col].mean().to_numpy()
        rows.append(dict(metric=col,fraction_above_zero=float((values>0).mean()),**bootstrap_mean(values,reps,seed)))
    return rows
