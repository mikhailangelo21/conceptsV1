from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED = ['concept_id','lemma','synonym_group','category','size_score','label_kind','label_source','notes']


def validate(concepts, references, pairs):
    for frame, fields in [(concepts, REQUIRED), (references, ['concept_id','lemma','label_source']),
                           (pairs, ['pair_id','target','reference','expected_relation','provenance'])]:
        if not set(fields) <= set(frame): raise ValueError(f'Missing fields: {set(fields)-set(frame)}')
        for field in set(fields)-{'notes'}:
            if frame[field].isna().any() or frame[field].astype(str).str.strip().eq('').any():
                raise ValueError(f'Missing values in {field}')
    if concepts.concept_id.duplicated().any() or references.concept_id.duplicated().any() or pairs.pair_id.duplicated().any():
        raise ValueError('IDs must be unique')
    if set(concepts.concept_id) & set(references.concept_id): raise ValueError('Reference-only IDs overlap fitting concepts')
    if not np.isfinite(pd.to_numeric(concepts.size_score, errors='coerce')).all(): raise ValueError('Nonfinite size labels')
    if not set(concepts.label_kind) <= {'demo_ordinal','human_ordinal','sourced_measurement'}: raise ValueError('Unknown label provenance kind')
    for kind, group in concepts.groupby('label_kind'):
        if kind == 'sourced_measurement':
            for col in ['units','aggregation']:
                if col not in group or group[col].fillna('').eq('').any(): raise ValueError(f'Measurements require {col}')
    owners = {}
    for frame in [concepts, references]:
        for r in frame.to_dict('records'):
            group = r.get('synonym_group', r['concept_id'])
            for alias in [r['lemma']] + str(r.get('aliases','')).split('|'):
                alias = ' '.join(alias.lower().split())
                if not alias or alias == 'nan': continue
                if alias in owners and owners[alias] != group: raise ValueError(f'Alias overlap: {alias}')
                owners[alias] = group
    for _, g in concepts.groupby('synonym_group'):
        if g.category.nunique()!=1 or g.size_score.nunique()!=1: raise ValueError('Synonyms must share category and label')
    coverage = concepts.drop_duplicates('synonym_group').groupby('category').size()
    if len(coverage)<2 or coverage.min()<5: raise ValueError('Need >=2 categories with >=5 independent groups each')
    if not set(pairs.target) <= set(concepts.concept_id) or not set(pairs.reference) <= set(references.concept_id): raise ValueError('Invalid comparison pair IDs')
    if not set(pairs.expected_relation) <= {'larger','smaller'}: raise ValueError('Comparison ties are not supported')
    if not set(concepts.concept_id) <= set(pairs.target): raise ValueError('Each concept needs an explicit comparison pair')


def load_data(config):
    c, r, p = [pd.read_csv(config[k], keep_default_na=False) for k in ['concepts','references','pairs']]
    validate(c,r,p)
    if config.get('category_limit'):
        cats=sorted(c.category.unique())[:int(config['category_limit'])]
        c=c[c.category.isin(cats)].copy(); p=p[p.target.isin(c.concept_id)].copy()
    return c.reset_index(drop=True),r,p


def split_concepts(concepts, seed):
    """Within category, ordered size blocks of five: seeded 3/1/1 allocation.

    Sorting ties by group ID gives an explicit deterministic rule. Remainders
    allocate to the largest deficit from 60/20/20. No synonym group is divided.
    """
    rng=np.random.default_rng(seed); mapping={}
    unique=concepts.sort_values('synonym_group').drop_duplicates('synonym_group')
    for _, g in unique.groupby('category',sort=True):
        ids=g.sort_values(['size_score','synonym_group']).synonym_group.tolist()
        counts=np.zeros(3,int)
        for start in range(0,len(ids),5):
            block=ids[start:start+5]; order=rng.permutation(len(block))
            labels=['train']*3+['validation','test'] if len(block)==5 else []
            for i,idx in enumerate(order):
                if labels: s=labels[i]; j=['train','validation','test'].index(s)
                else:
                    j=int(np.argmax(np.array([.6,.2,.2])*(counts.sum()+1)-counts)); s=['train','validation','test'][j]
                mapping[block[idx]]=s; counts[j]+=1
    return {r.concept_id:mapping[r.synonym_group] for r in concepts.itertuples()}


def select_subjects(concepts, splits, count):
    """Round-robin sorted categories, alternating smallest/largest test rank.
    Uses declared labels only, before any model evaluation.
    """
    test=concepts[concepts.concept_id.map(splits).eq('test')].drop_duplicates('synonym_group')
    queues=[]
    for i,(_,g) in enumerate(test.groupby('category',sort=True)):
        queues.append(g.sort_values(['size_score','concept_id'],ascending=[i%2==0,True]).concept_id.tolist())
    chosen=[]
    while any(queues) and len(chosen)<count:
        for queue in queues:
            if queue and len(chosen)<count: chosen.append(queue.pop(0))
    if len(chosen)!=count: raise ValueError('Not enough held-out subjects')
    return chosen


def import_data(source, output, references, pairs, ratings=None):
    """Lossless CSV import; optional separate observations are never synthesized."""
    from pathlib import Path
    c=pd.read_csv(source,keep_default_na=False)
    validate(c,pd.read_csv(references,keep_default_na=False),pd.read_csv(pairs,keep_default_na=False))
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True); c.to_csv(output,index=False)
    if ratings:
        r=pd.read_csv(ratings)
        if not {'concept_id','rater_id','rating'}<=set(r) or not set(r.concept_id)<=set(c.concept_id): raise ValueError('Invalid rater table')
        if not np.isfinite(r.rating).all(): raise ValueError('Nonfinite ratings')
        r.to_csv(output.with_name(output.stem+'_raters.csv'),index=False)
