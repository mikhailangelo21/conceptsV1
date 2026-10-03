"""Phase A: immutable, cache-only audit. No language-model forward is called."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from .util import read_json,atomic_json,digest
from .probes import bootstrap_mean
from .paired import context_contrasts,mapping_decomposition,group_summary
from .nuisance import nested_evaluate
from .protection import forbid_original_write,verify_protected,evidence_hashes


def load_cached_concepts(source):
    concepts=pd.read_csv(source/'concepts_analysis.csv');prompts=read_json(source/'prompts.json')
    index={r['prompt_id']:r for r in read_json(source/'activation_index.json')}
    metadata=read_json(source/'activation_cache.json'); groups={}
    for row in prompts:
        if row['condition']!='neutral' or row['template_id'] not in ['t0','t1']: continue
        with np.load(index[row['prompt_id']]['cache_file'],allow_pickle=False) as shard:
            if str(shard['fingerprint'])!=digest(metadata['identity']): raise ValueError('Historical cache fingerprint mismatch')
            a=shard['activation'].copy()
        groups.setdefault(row['synonym_group'],[]).append(a)
    ids=sorted(groups);info=concepts.drop_duplicates('synonym_group').set_index('synonym_group').loc[ids]
    return info,np.stack([np.mean(groups[g],axis=0) for g in ids])


def score_audit(source,out,config):
    records=pd.read_csv(source/'behavior_scores.csv',keep_default_na=False)
    base=records[records.direction.eq('baseline')].copy()
    duplicate_count=int(base.prompt_id.duplicated().sum())
    if duplicate_count:
        for _,g in base.groupby('prompt_id'):
            if g[['logp_A','logp_B','correct_label','larger_label']].drop_duplicates().shape[0]!=1: raise ValueError('Conflicting duplicate baselines')
        base=base.drop_duplicates('prompt_id')
    subjects=read_json(source/'subjects.json')['ids'];summaries=[]
    for scope,subset in [('full_test',base[base.split.eq('test')]),('intervention_subjects',base[base.concept_id.isin(subjects)])]:
        for task,g in subset.groupby('task'):
            values=g.groupby('synonym_group').correct.mean().to_numpy()
            summaries.append(dict(scope=scope,task=task,n_subjects=g.synonym_group.nunique(),n_prompts=len(g),correct=int(g.correct.sum()),accuracy=g.correct.mean(),fraction_A=g.predicted_label.eq('A').mean(),**bootstrap_mean(values,config['bootstrap_replicates'],config['seed'])))
    pd.DataFrame(summaries).to_csv(out/'baseline_competence.csv',index=False)
    b=base[['prompt_id','logp_A','logp_B','semantic_D','correct_odds']].rename(columns={k:'baseline_'+k for k in ['logp_A','logp_B','semantic_D','correct_odds']})
    treated=records[records.direction.ne('baseline')].merge(b,on='prompt_id',how='left',validate='many_to_one',indicator=True)
    if not treated['_merge'].eq('both').all(): raise ValueError('Missing matched baseline')
    treated['delta_L']=(treated.logp_A-treated.logp_B)-(treated.baseline_logp_A-treated.baseline_logp_B)
    size=treated[treated.task.eq('size')].copy();sign=np.where(size.larger_label.eq('A'),1,-1)
    error=float(np.max(np.abs(size.delta_L*sign-pd.to_numeric(size.effect))))
    if error>2e-6: raise ValueError('Saved semantic effects disagree with mapping')
    correct_sign=np.where(treated.correct_label.eq('A'),1,-1)
    odds_error=float(np.max(abs((treated.logp_A-treated.logp_B)*correct_sign-treated.correct_odds)))
    if odds_error>2e-6: raise ValueError('Correct-label odds mismatch')
    size.to_csv(out/'mapping_rows.csv',index=False)
    decomposition=mapping_decomposition(size)
    decomposition.to_csv(out/'mapping_components.csv',index=False)
    component_summary=[]
    for (direction,dose),g in decomposition.groupby(['direction','strength']):
        for s in group_summary(g,['semantic_component','common_A_component'],reps=config['bootstrap_replicates'],seed=config['seed']):
            component_summary.append(dict(direction=direction,strength=dose,**s))
    pd.DataFrame(component_summary).to_csv(out/'mapping_component_summary.csv',index=False)
    tol=config['near_zero_tolerance'];positive=size[size.direction.eq('semantic')&size.strength.eq(1)]
    model=read_json(source/'model.json');encoding=read_json(source/'answer_encoding.json')
    tokenizer=AutoTokenizer.from_pretrained(model['model'],revision=model['tokenizer_revision'],local_files_only=True)
    for row in read_json(source/'behavior_prompts.json'):
        ids=tokenizer.encode(row['text'],add_special_tokens=False)
        if ids!=row['token_ids'] or not row['text'].endswith('Answer:'): raise ValueError('Historical prefix/tokenization mismatch')
        for label in 'AB':
            if tokenizer.encode(row['text']+encoding['prefix']+label,add_special_tokens=False)!=ids+[encoding['token_ids'][label]]: raise ValueError('Historical answer boundary mismatch')
    # Arithmetic test only: historical full logits were not persisted.
    logits=torch.tensor([12.25,-3.75,2.4],dtype=torch.float32);lp=logits.log_softmax(-1)
    numerical=float(abs((lp[0]-lp[1])-(logits[0]-logits[1])))
    evidence=dict(duplicate_baselines_removed=duplicate_count,unique_baseline_rows=len(base),semantic_reconstruction_max_error=error,correct_odds_max_error=odds_error,
                  plus_one_A_up=int((positive.delta_L>tol).sum()),plus_one_A_flat=int((positive.delta_L.abs()<=tol).sum()),plus_one_A_down=int((positive.delta_L < -tol).sum()),near_zero_tolerance=tol,
                  answer_encoding=encoding,tokenizer_checks='All historical behavioral prefixes and continuation boundaries verified locally; no answer appended to extraction prefix.',
                  float32_identity_unit_error=numerical,original_logit_identity='Original full logits were not saved; direct numerical verification on original forwards is unavailable in cache-only audit. Phase D saves both paths.',
                  independent_units='12 baseline test subjects; 6 intervention subjects. Polarities, mappings, doses and directions are repeated conditions, not independent subjects.')
    atomic_json(out/'scoring_audit.json',evidence)
    size[size.direction.eq('semantic')&size.strength.eq(1)].head(8).to_csv(out/'scoring_traces.csv',index=False)
    refs=pd.read_csv(source/'references.csv');refs['audit_note']='Reference scale varies, but named salt grain/container comparisons are widely separated ordinary-size claims; demo pair provenance is not independent physical validation.'
    refs.to_csv(out/'reference_review.csv',index=False)
    return evidence


def context_audit(source,out,config):
    layer=read_json(source/'selection.json')['selected_layer']
    raw=pd.read_csv(source/'context.csv');raw=raw[raw.layer.eq(layer)]
    # Preserve a copy of every original neutral-relative condition, all splits.
    raw.to_csv(out/'original_context_shifts.csv',index=False)
    paired=context_contrasts(raw);paired['historical_split']=paired.concept_id.map(read_json(source/'splits.json'))
    paired.to_csv(out/'paired_context_contrasts.csv',index=False);summaries=[]
    for (split,tid),g in paired.groupby(['historical_split','template_id']):
        for s in group_summary(g,['subject_contrast','distractor_contrast','binding_contrast'],reps=config['bootstrap_replicates'],seed=config['seed']): summaries.append(dict(historical_split=split,template_id=tid,**s))
    pd.DataFrame(summaries).to_csv(out/'paired_context_summary.csv',index=False)
    # Explicitly development-only calibration; frozen original train concepts fit it.
    keys=['concept_id','synonym_group','condition','split']
    wide=raw[raw.template_id.isin(['t0','t2'])].pivot(index=keys,columns='template_id',values='score').dropna().reset_index()
    tr=wide[wide.split.eq('train')];x=tr.t2.to_numpy();y=tr.t0.to_numpy()
    offset=float((y-x).mean());slope=max(1e-8,float(np.cov(x,y,ddof=0)[0,1]/np.var(x)));intercept=float(y.mean()-slope*x.mean())
    wide['raw']=wide.t2;wide['offset']=wide.t2+offset;wide['positive_affine']=slope*wide.t2+intercept
    wide.to_csv(out/'template_calibration_predictions.csv',index=False)
    atomic_json(out/'template_calibration_fit.json',dict(training_groups=sorted(tr.synonym_group.unique()),offset=offset,positive_slope=slope,intercept=intercept,
                  objective='Predict t0 score from t2 score over matched conditions, with equal concept/condition weight. Diagnostic of between-template agreement, not physical-size truth.',status='All original concepts/templates are v2 development evidence. No axis sign flip is allowed.'))
    calibration=[]
    for split,g in wide.groupby('split'):
        for method in ['raw','offset','positive_affine']:
            errors=(g[method]-g.t0)**2
            unit=errors.groupby(g.synonym_group).mean().to_numpy()
            summary=bootstrap_mean(unit,config['bootstrap_replicates'],config['seed'])
            calibration.append(dict(historical_split=split,method=method,rmse=float(np.sqrt(summary['mean'])),rmse_low=float(np.sqrt(summary['low'])),rmse_high=float(np.sqrt(summary['high'])),n_groups=len(unit)))
    pd.DataFrame(calibration).to_csv(out/'template_calibration_metrics.csv',index=False)
    return paired


def audit_plots(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=pd.read_csv(out/'paired_context_contrasts.csv');p=p[p.historical_split.eq('test')]
    fig,axes=plt.subplots(1,3,figsize=(11,4.5))
    for ax,metric in zip(axes,['subject_contrast','distractor_contrast','binding_contrast']):
        wide=p.pivot(index='synonym_group',columns='template_id',values=metric)
        for _,row in wide.iterrows(): ax.plot([0,1],row[['t0','t2']],'-o',color='#3465A4',alpha=.5,ms=3)
        ax.axhline(0,color='gray',lw=.8);ax.set_xticks([0,1],['t0','t2']);ax.set_title(metric.replace('_',' '));ax.set_ylabel('Training projection SD')
    fig.suptitle('Paired contrasts: 12 historical test concepts, now development\nPROVISIONAL DEMONSTRATION LABELS',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.88))
    for ext in ['png','svg']:fig.savefig(out/f'paired_context.{ext}',bbox_inches='tight')
    plt.close(fig)
    s=pd.read_csv(out/'mapping_component_summary.csv');s=s[s.direction.eq('semantic')]
    fig,ax=plt.subplots(figsize=(7,4.5))
    for metric,color in [('semantic_component','#3465A4'),('common_A_component','#B88720')]:
        g=s[s.metric.eq(metric)].sort_values('strength');ax.plot(g.strength,g['mean'],'-o',label=metric.replace('_',' '),color=color);ax.fill_between(g.strength,g.low,g.high,color=color,alpha=.15)
    ax.axhline(0,color='gray',lw=.8);ax.set(xlabel='Prespecified steering dose',ylabel='Paired log-odds change',title='6 subjects; descriptive group intervals\nPROVISIONAL DEMONSTRATION LABELS');ax.legend();fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(out/f'mapping_components.{ext}',bbox_inches='tight')
    plt.close(fig)


def audit_report(out):
    from .report import table
    evidence=read_json(out/'scoring_audit.json');context=pd.read_csv(out/'paired_context_summary.csv');adv=pd.read_csv(out/'nuisance_advantage.csv')
    text=['# Revised pilot interpretation — cache-only audit',
          '**PROVISIONAL DEMONSTRATION LABELS. All 60 original concepts and t2 are now development data, not a new untouched confirmatory test. No model forwards were used in this audit. The original run, report, fits and caches are unchanged.**',
          '## Scoring and experimental units',json.dumps(evidence,indent=2),table(pd.read_csv(out/'baseline_competence.csv'),['scope','task','n_subjects','n_prompts','correct','accuracy']),
          'No semantic-sign or baseline-join error was found in these raw records. The 48 full-test versus 24 intervention size questions arise from 12 versus 6 subjects, each with four repeated questions. Reference-only salt grain/container labels remain assistant-authored qualitative comparisons, not validated measurements. Scoring traces, boundaries and duplicate checks are saved.',
          '## Semantic and A-label components',
          'mapping_rows.csv retains every polarity and option mapping. mapping_components.csv pairs them at fixed subject/reference/template/polarity/layer/dose/direction. A common A shift cancels in a balanced semantic average. Its presence alone does not explain the residual semantic component; mapping-dependent interactions may remain. The decomposition does not prove independent latent mechanisms.',
          '![Mapping decomposition](mapping_components.png)',
          '## Paired subject and binding contrasts',table(context[context.historical_split.eq('test')],['template_id','metric','n_groups','mean','low','high','fraction_above_zero']),
          'Intervals resample paired whole concept groups, never subtract interval endpoints. Both templates can preserve enlarged-versus-shrunk ordering while having different neutral-relative offsets. An offset explanation is a hypothesis, assessed separately below; it does not establish global invariance. Every concept remains in paired_context_contrasts.csv, including failures.',
          '![Paired context](paired_context.png)',
          'Original neutral, irrelevant and absolute shifts are retained in original_context_shifts.csv and the unchanged original pilot plots.',
          '## Development-only template calibration',table(pd.read_csv(out/'template_calibration_metrics.csv'),['historical_split','method','n_groups','rmse','rmse_low','rmse_high']),
          'Offset and positive affine maps were fit using original training concepts only, then evaluated on excluded concepts. The target is agreement with t0 scores, not independently measured physical size. Extra template-specific flexibility is explicitly separated from raw fixed-direction results; no test-driven axis reversal was performed.',
          '## Paired predictive advantage and confounding',table(adv[adv.category.eq('all')],['scheme','category','n_groups','baseline_rho','augmented_rho','difference','low','high']),
          'Nuisances combine category indicators, noun character count and tokenizer token count. All preprocessing and size-direction fitting happen within training folds. Grouped outer folds use inner-only layer/penalty tuning. Leave-one-category-out is a separate, harder development test. Category-only predictions for an unseen category fall back to the training grand mean; nuisance ridge uses zero unseen-category indicators plus its trained intercept/numeric terms.',
          'Intervals are paired bootstraps of fixed excluded predictions, conditional on fitted models, not nested bootstrap refits or a global corrected significance test. Within-category rows are saved in nuisance_advantage.csv; small category sample sizes limit precision. Separate confidence-interval overlap is not used to establish an advantage.',
          '## Verified errors, limitations and untested explanations',
          '- Verified numerical/sign errors: none detected by the saved-record checks. Original full logits were not persisted, so direct historical float32 logit/log-probability equality cannot be recovered without new forwards; Phase D verifies this prospectively.',
          '- Observed limitations: low original behavioral competence, output-label sensitivity, reused references, small samples, synthetic labels, and nonzero off-target effects. Different tasks can have different log-odds sensitivity; magnitudes alone do not establish one common mechanism.',
          '- Untested explanations in Phase A: half-precision resolution, template-specific offsets as a causal explanation, output-position mechanisms, and transfer from typical to contextual instance size. Later v2 stages test these separately.',
          'See config.json, source_hashes.json, preservation_check.json, source snapshots, scoring_traces.csv and saved nested-fold fit metadata for reproducibility.']
    (out/'report.md').write_text('\n\n'.join(text)+'\n')


def run_audit(config):
    source=Path(config['source_run']);before=verify_protected()
    code={p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest() for p in ['audit.py','paired.py','nuisance.py']}
    key=digest([config,code,read_json(source/'identity.json')])[:12];out=Path(config['run_root'])/('audit-v2-'+key)
    forbid_original_write(out);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'config.json',config);atomic_json(out/'source_hashes.json',code)
    snapshots=out/'source';snapshots.mkdir(exist_ok=True)
    for name in code: (snapshots/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    print('Cache-only audit:',out,flush=True)
    score_audit(source,out,config);context_audit(source,out,config)
    info,x=load_cached_concepts(source);nested_evaluate(info,x,out/'nuisance_fits',config)
    audit_plots(out);audit_report(out)
    atomic_json(out/'preservation_check.json',dict(protected_files_before=before,protected_files_after=verify_protected(),unchanged=True,model_forwards=0))
    atomic_json(out/'state.json',dict(status='complete',model_forwards=0,development_only=True))
    print('Audit report:',out/'report.md',flush=True)
    return out
