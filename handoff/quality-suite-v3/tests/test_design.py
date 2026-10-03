import numpy as np
import pandas as pd
import pytest
from quality_dimensions.data import load_data,validate,split_concepts,select_subjects
from quality_dimensions.prompts import geometry_prompts,behavioral_prompts,semantic_labels,tokenize_prompt,answer_encoding
from quality_dimensions.interventions import semantic_score
from quality_dimensions.probes import fit_layer,choose_layer,grouped_bootstrap_indices,shuffle_within_category,bootstrap_rho
from quality_dimensions.analysis import concept_vectors


def test_data_splits_counts_and_aliases(fixture_config):
    c,r,p=load_data(fixture_config); splits=split_concepts(c,1729)
    assert len(c)==60 and len(p)==60
    assert pd.Series(splits).value_counts().to_dict()=={'train':36,'validation':12,'test':12}
    prompts=geometry_prompts(c,splits,fixture_config); assert len(prompts)==780
    assert all(row['text'].endswith('Answer:') and 'size?' not in row['text'] for row in prompts)
    subjects=select_subjects(c,splits,6); assert len(set(c.set_index('concept_id').loc[subjects].category))==6
    behavior=behavioral_prompts(c,r,p,splits,subjects,fixture_config); assert len(behavior)==192
    assert sum(row['concept_id'] in subjects for row in behavior)==48
    alias=c.iloc[[0]].copy(); alias['concept_id']='animal_alias'; alias['lemma']='small garden ant'; c2=pd.concat([c,alias],ignore_index=True)
    s2=split_concepts(c2,1729); assert s2['animal_alias']==s2[c.iloc[0].concept_id]
    bad=c.copy(); bad.loc[1,'lemma']=bad.loc[0,'lemma']
    with pytest.raises(ValueError,match='Alias overlap'): validate(bad,r,p)
    bad=c.copy(); bad.loc[0,'label_kind']='human ratings invented'
    with pytest.raises(ValueError,match='provenance'): validate(bad,r,p)
    bad=c.copy(); bad.loc[0,'size_score']=np.nan
    with pytest.raises(ValueError,match='Missing|Nonfinite'): validate(bad,r,p)


@pytest.mark.parametrize('polarity,swap,expected',[('larger',False,'A'),('larger',True,'B'),('smaller',False,'B'),('smaller',True,'A')])
def test_semantic_counterbalancing(polarity,swap,expected):
    larger,smaller=semantic_labels(polarity,swap)
    assert larger==expected and larger!=smaller
    assert semantic_score({larger:-1.,smaller:-3.},larger)==2.


def test_train_only_features_and_selection(fixture_config):
    c,_,_=load_data(fixture_config); splits=split_concepts(c,1729); rows=geometry_prompts(c,splits,fixture_config)
    rng=np.random.default_rng(9); acts=rng.normal(size=(len(rows),2,16))
    def fit(activations,concepts):
        x,info,ids=concept_vectors(rows,activations,concepts,'train',fixture_config['seen_templates'])
        v,vi,_=concept_vectors(rows,activations,concepts,'validation',fixture_config['seen_templates'])
        return {i:fit_layer(x[:,i-1],info.size_score.to_numpy(float),info.category.to_numpy(),v[:,i-1],vi.size_score.to_numpy(float),[1,10],ids) for i in [1,2]}
    first=fit(acts,c); modified=acts.copy()
    for i,row in enumerate(rows):
        if row['split']=='test' or row['template_id']==fixture_config['heldout_template'] or row['condition']!='neutral': modified[i]=rng.normal(size=(2,16))*999
    changed=c.copy(); changed.loc[changed.concept_id.map(splits).eq('test'),'size_score']=10000
    second=fit(modified,changed)
    assert choose_layer(first)==choose_layer(second)
    for layer in first:
        for key in ['mu','direction','pca_mean','pca_components','ridge_coef']:
            np.testing.assert_array_equal(first[layer][key],second[layer][key])
        assert first[layer]['ridge_alpha']==second[layer]['ridge_alpha']
        assert len(first[layer]['training_ids'])==36


def test_grouped_bootstrap_permutation_and_ties():
    groups=np.array(['a','a','b','b','b','c'])
    idx=grouped_bootstrap_indices(groups,np.random.default_rng(6))
    for group in set(groups):
        counts=[np.sum(idx==i) for i in np.flatnonzero(groups==group)]
        assert len(set(counts))==1
    y=np.array([1.,2.,3.,100.,200.,300.]); cats=np.array(['a']*3+['b']*3)
    shuffled=shuffle_within_category(y,cats,np.random.default_rng(2))
    for cat in set(cats): assert set(shuffled[cats==cat])==set(y[cats==cat])
    ci=bootstrap_rho(np.ones(3),np.ones(3),100,1); assert ci['degenerate_replicates']==100
