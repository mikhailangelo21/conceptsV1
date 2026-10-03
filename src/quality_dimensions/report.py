from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .util import read_json
from .analysis import load_fit

BLUE='#3465A4'; GOLD='#B88720'; INK='#333333'


def plots(run):
    run=Path(run); output=run/'plots'; output.mkdir(exist_ok=True); names=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'figure.dpi':120})
    config=read_json(run/'config.json'); concepts=pd.read_csv(run/'concepts.csv')
    provenance='PROVISIONAL DEMONSTRATION LABELS' if 'demo_ordinal' in set(concepts.label_kind) else 'Independent supplied labels (see provenance)'
    def save(fig,name,title):
        fig.suptitle(title+'\n'+provenance,fontsize=12)
        fig.tight_layout(rect=(0,0,1,.91))
        for suffix in ['png','svg']: fig.savefig(output/(name+'.'+suffix),bbox_inches='tight')
        plt.close(fig); names.append(name)
    if (run/'analysis_complete.json').exists():
        selected=read_json(run/'selection.json')['selected_layer']; fit=load_fit(run,selected)
        metrics=pd.read_csv(run/'metrics.csv'); fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
        for ax,evaluation in zip(axes,['test_seen','test_heldout']):
            for method,color in [('direction',BLUE),('ridge',GOLD)]:
                g=metrics[(metrics.evaluation==evaluation)&(metrics.method==method)].sort_values('layer')
                ax.plot(g.layer,g.rho,label=method,color=color,linestyle='-' if method=='direction' else '--')
                ax.fill_between(g.layer,g.low,g.high,color=color,alpha=.12)
            ax.axhline(0,color='gray',lw=.7); ax.axvline(selected,color='gray',ls=':',label='validation-selected block')
            ax.set(xlabel='Complete decoder block (1-based)',ylabel='Spearman rank correlation',ylim=(-1.05,1.05),title=evaluation.replace('_',' ')); ax.legend(fontsize=8)
        save(fig,'prediction_by_layer','Unseen-concept prediction; shaded descriptive 95% group intervals')
        pca=pd.read_csv(run/'pca.csv'); neutral=pca[(pca.condition=='neutral')&(pca.template_id==config['heldout_template'])&(pca.split=='test')]
        training=pca[(pca.condition=='neutral')&(pca.template_id.isin(config['seen_templates']))&(pca.split=='train')].groupby(['concept_id','category','size_score','split'],as_index=False)[['pc1','pc2']].mean()
        points=pd.concat([training,neutral],ignore_index=True); fig,ax=plt.subplots(figsize=(8,5.5)); markers=['o','s','^','D','P','X','v','*']
        for i,(cat,g) in enumerate(points.groupby('category')):
            for split,sub in g.groupby('split'):
                sc=ax.scatter(sub.pc1,sub.pc2,c=sub.size_score,cmap='cividis',vmin=concepts.size_score.min(),vmax=concepts.size_score.max(),marker=markers[i%len(markers)],s=50 if split=='train' else 95,edgecolors=INK if split=='test' else 'none',linewidths=1.2,label=f'{cat} / {split}')
        fig.colorbar(sc,ax=ax,label='Ordinal size (not physical units)'); ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.25,1))
        ax.set(xlabel=f'PC1 ({fit["pca_variance"][0]:.1%} training variance)',ylabel=f'PC2 ({fit["pca_variance"][1]:.1%} training variance)')
        save(fig,'pca_selected',f'Block {selected}: fixed train-only PCA; larger outlined marks = test')
        context=pd.read_csv(run/'context.csv'); conditions=['shrunk','enlarged','irrelevant','distractor_shrunk','distractor_enlarged']
        fig,axes=plt.subplots(1,len(config['context_templates']),figsize=(12,5),squeeze=False)
        for ax,template in zip(axes.flat,config['context_templates']):
            subset=context[(context.layer==selected)&(context.split=='test')&(context.template_id==template)]
            for _,g in subset.groupby('synonym_group'):
                values=g.groupby('condition').change.mean().reindex(conditions)
                ax.plot(range(len(conditions)),values,color='gray',alpha=.4,lw=.8,marker='o',ms=3)
            means=subset.groupby('condition').change.mean().reindex(conditions)
            ax.scatter(range(len(conditions)),means,color=BLUE,s=60,zorder=5,label='mean')
            ax.axhline(0,color=INK,lw=.8); ax.set_xticks(range(len(conditions)),[c.replace('distractor_','other object\n') for c in conditions],rotation=30,ha='right')
            ax.set(title=template+'; every held-out concept',ylabel='Paired change / training projection SD'); ax.legend()
        save(fig,'context_shifts',f'Block {selected}: fixed-direction context response (neutral = zero)')
    if (run/'effect_summary.csv').exists():
        effects=pd.read_csv(run/'effect_summary.csv'); selected=read_json(run/'selection.json')['selected_layer']; effects=effects[effects.layer==selected]
        fig,ax=plt.subplots(figsize=(8,4.5))
        for name,g in effects[(effects.task=='size')&(effects.metric=='effect')].groupby('direction'):
            g=g.sort_values('strength'); color=BLUE if name=='semantic' else 'gray'
            ax.plot(g.strength,g['mean'],label=name,color=color,ls='-' if name=='semantic' else '--',marker='o')
            ax.fill_between(g.strength,g.low,g.high,color=color,alpha=.1)
        ax.axhline(0,color=INK,lw=.8); ax.set(xlabel='λ × training projection SD (same norm for all directions)',ylabel='Change in semantic log odds: subject larger'); ax.legend()
        save(fig,'steering_dose_response',f'Block {selected}: additive steering; descriptive 95% subject-group intervals')
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for ax,task in zip(axes,['identity','tag']):
            for name,g in effects[(effects.task==task)&(effects.metric=='correct_odds_change')].groupby('direction'):
                g=g.sort_values('strength'); ax.plot(g.strength,g['mean'],label=name,color=BLUE if name=='semantic' else 'gray',ls='-' if name=='semantic' else '--',marker='o')
                ax.fill_between(g.strength,g.low,g.high,color=BLUE if name=='semantic' else 'gray',alpha=.1)
            ax.axhline(0,color=INK,lw=.8); ax.set(title=task,xlabel='λ × training projection SD',ylabel='Change in correct-label log odds'); ax.legend(fontsize=8)
        save(fig,'off_target_effects',f'Block {selected}: identity and arbitrary-tag selectivity controls')
    return names


def table(frame,columns):
    # Avoid optional tabulate dependency.
    out=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in frame[columns].itertuples(index=False,name=None):
        out.append('| '+' | '.join(f'{v:.4g}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(out)


def report(run):
    from .protection import forbid_original_write
    forbid_original_write(run)
    run=Path(run); state=read_json(run/'state.json'); config=read_json(run/'config.json'); data=pd.read_csv(run/'concepts.csv'); images=plots(run)
    demo='demo_ordinal' in set(data.label_kind)
    text=[f'# Quality dimensions pilot — {state["status"].upper()}',
          '**PROVISIONAL DEMONSTRATION LABELS (`demo_ordinal`): no human ratings, measurements, or label reliability were collected.**' if demo else 'Supplied independent label provenance is preserved in concepts.csv.',
          f'Profile: `{config["profile"]}`. Run: `{run.name}`. '+('Smoke profile verifies mechanics only; no scientific conclusions from this sample.' if config['profile']=='smoke' else ''),
          '## What actually ran',f'Status: **{state["status"]}**. '+state.get('reason',''),
          '```json\n'+json.dumps({k:v for k,v in state.items() if k not in ['reason']},indent=2)+'\n```',
          '## Software verification',
          'Offline randomly initialized Qwen3 tests verify mechanics only; random-model outputs are excluded from this report. See docs/VERIFICATION.md for checks actually executed. A blocked real-model run does not verify pretrained-model performance.',
          '## Provenance and independent units',
          f'{len(data)} concept rows; {data.synonym_group.nunique()} independent synonym groups in {data.category.nunique()} categories. Labels are ordinal; MAE/R² are secondary and impose stronger numeric assumptions. Splits and prompt variants are saved. Sources: concepts.csv, references.csv, pairs.csv.',
          'Category/size confounding remains. In the full fixture, produce is predominantly small and transport predominantly large. Cross-category overlaps and category-balanced mean differences do not remove lexical familiarity, noun token length, polysemy, or other confounds. Ordinary category size is distinct from contextual size.',
          'Behavioral inference is conditional on the small reused reference set. Reference-only objects never enter probe fitting. Reference reuse counts are in plan.json.',
          '## Linear accessibility']
    if (run/'analysis_complete.json').exists():
        selected=read_json(run/'selection.json')['selected_layer']; metrics=pd.read_csv(run/'metrics.csv')
        text += [f'Primary block: {selected}, chosen from validation mean-difference performance before test evaluation. Coordinates are centered on neutral training concepts without whitening.',table(metrics[(metrics.layer==selected)&metrics.evaluation.str.startswith('test')],['evaluation','method','n_groups','rho','low','high']),
                 'Intervals resample independent concept/synonym groups, retaining related variants together. Degenerate replicate counts are in metrics.csv. Intervals exclude label uncertainty; supplied per-rater data are preserved but are not automatically modeled. Category-preserving shuffled-label controls refit the direction and validation-tuned ridge at the frozen selected block; permutations.csv is a descriptive reference, not a global corrected significance test.']
        primary=metrics[(metrics.layer==selected)&metrics.method.eq('direction')&metrics.evaluation.eq('test_heldout')].iloc[0]
        text += [f"Held-out wording: the direction has Spearman rho = {primary.rho:.3f} across {int(primary.n_groups)} unseen concept groups (descriptive 95% interval {primary.low:.3f} to {primary.high:.3f}). This is ordinal predictive evidence only; compare it with the nuisance baselines above."]
        null=pd.read_csv(run/'permutations.csv')
        if 'direction_rho' in null:
            text += ['Shuffled-label reference distribution (selected layer):',table(null.groupby('evaluation')[['direction_rho','ridge_rho']].agg(['mean','std']).reset_index().set_axis(['evaluation','direction_mean','direction_sd','ridge_mean','ridge_sd'],axis=1),['evaluation','direction_mean','direction_sd','ridge_mean','ridge_sd'])]
    else: text+=['Not evaluated: complete compatible activation extraction and train/validation fitting are required. No predictive result is claimed.']
    text+=['## Context sensitivity']
    if (run/'context_summary.csv').exists():
        s=pd.read_csv(run/'context_summary.csv'); selected=read_json(run/'selection.json')['selected_layer']
        text += [table(s[(s.layer==selected)&(s.split=='test')&s.condition.ne('neutral')],['template_id','condition','n_groups','mean','low','high']),
                 'Every held-out concept, including failures, is plotted. Scores use the fixed neutral-training direction and training projection SD. A qualitative shift does not measure a factor of 100. Clause length and token-position differences remain confounds; token counts are saved in prompts.json and context.csv.']
        raw=pd.read_csv(run/'context.csv')
        for tid,g in raw[(raw.layer==selected)&raw.split.eq('test')].groupby('template_id'):
            paired=g.groupby(['synonym_group','condition']).change.mean().unstack()
            if not {'shrunk','enlarged','irrelevant','distractor_shrunk','distractor_enlarged'}<=set(paired): continue
            shrink_fraction=float(paired.shrunk.lt(0).mean()); enlarge_fraction=float(paired.enlarged.gt(0).mean())
            target_abs=float(paired[['shrunk','enlarged']].abs().to_numpy().mean())
            control_abs=float(paired[['irrelevant','distractor_shrunk','distractor_enlarged']].abs().to_numpy().mean())
            text += [f"For {tid}, shrinkage moves downward in {shrink_fraction:.0%} of held-out groups and enlargement moves upward in {enlarge_fraction:.0%}. Mean absolute subject-change shift is {target_abs:.3f} training SD, versus {control_abs:.3f} for irrelevant/distractor controls. These proportions expose failures as well as successes; signed averages alone can hide them."]
    else: text+=['Not evaluated; no context response is claimed.']
    text+=['## Causal influence and selectivity']
    if (run/'baseline_metrics.csv').exists():
        text+=[table(pd.read_csv(run/'baseline_metrics.csv'),['split','task','polarity','swapped','n_groups','accuracy','fraction_A']),
               'Accuracy is restricted to the two verified answer tokens, with exact full-vocabulary conditional log probabilities saved. All predetermined test questions remain in the analysis, including baseline errors. Poor baseline competence limits causal interpretation. The semantic score reverses correctly for smaller questions and reversed A/B mappings.']
    else: text+=['Baseline behavioral competence and pretrained-model interventions have not been measured.']
    if (run/'effect_summary.csv').exists():
        effects=pd.read_csv(run/'effect_summary.csv'); text += [table(effects[(effects.direction=='semantic')],['layer','task','strength','metric','n_groups','mean','low','high']),
          'Effects are additive steering at the final prompt token, transferred from naming-task representations to behavioral prompts. Random directions use the same perturbation norm as the semantic direction; the configured control count is recorded in plan.json and these controls are descriptive. Raw records include actual float16 perturbation norms and activation-relative norms. Zero-dose invariance is checked automatically.']
        chosen=read_json(run/'selection.json')['selected_layer']
        size=effects[(effects.layer==chosen)&effects.task.eq('size')&effects.direction.eq('semantic')&effects.metric.eq('effect')].sort_values('strength')
        if len(size):
            negative=size[size.strength.lt(0)].iloc[0]; positive=size[size.strength.gt(0)].iloc[-1]
            expected=negative['mean']<0 and positive['mean']>0
            text += [f"At the extreme prespecified doses, mean semantic log-odds changes are {negative['mean']:.4f} (lambda={negative.strength:g}) and {positive['mean']:.4f} (lambda={positive.strength:g}). The endpoint signs {'follow' if expected else 'do not follow'} the hypothesized bidirectional pattern. This descriptive check is not a significance test; random controls, reversed questions, option-order performance and off-target effects remain essential to interpretation."]
    text+=['## Resources and reproducibility']
    for name in ['plan.json','benchmark.json','resources.json']:
        if (run/name).exists(): text += [f'`{name}`:', '```json\n'+json.dumps(read_json(run/name),indent=2)+'\n```']
    if (run/'model.json').exists():
        m=read_json(run/'model.json'); text += [f'Model `{m["model"]}`; model and tokenizer revision `{m["model_revision"]}`. Backend `{m["backend"]}`, forward dtype `{m["dtype"]}`, stored dtype `{m["stored_dtype"]}`. {m["num_hidden_layers"]} blocks × {m["hidden_size"]} residual width. Full attention configuration and environment are saved in model.json and environment.json.']
    text += ['Process RSS and MPS allocations overlap in unified memory and must not be summed. Approximate timing estimates are measured only when a real benchmark completes. Seeds do not guarantee cross-device bitwise identity.',
             '## Interpretation and remaining requirements',
             'The three experiments support separate claims. PCA is exploratory; PC1 is not assumed to be size and low PCA variance does not disprove a semantic feature. Prediction does not establish causal use. Steering does not establish a natural semantic mechanism, a privileged Euclidean metric, or a Gärdenforsian conceptual space. A negative result may reflect task transfer or small-model limitations.',
             'Independent labels and an independently specified replication are required before substantive dissertation claims. Layer sweeps and additional doses/layers are exploratory. This is a proposed additive-steering pilot, not an exact activation-patching replication.',
             '## Figures']
    text += [f'![{name.replace("_"," ")}](plots/{name}.png)' for name in images] or ['No empirical plots are available because the required model outputs have not been produced.']
    text += ['## Resume',f'```bash\nqd run --config {config["config_path"]} --run-dir {run.resolve()} --resume --max-model-minutes 60\n```',
             'Repeat the same command to add another bounded execution session; no planned conditions are discarded. Use `--max-model-minutes 600` for a longer explicitly requested session.',
             '## Implementation references',
             '- [Qwen model card](https://huggingface.co/Qwen/Qwen3-1.7B-Base) and [configuration](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/main/config.json).',
             '- [Transformers Qwen3](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py); installed source is recorded separately.',
             '- [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html).',
             '- [Park et al.](https://proceedings.mlr.press/v235/park24c.html), [Zhang and Nanda](https://arxiv.org/abs/2309.16042), [Heimersheim and Nanda](https://arxiv.org/abs/2404.15255).']
    (run/'report.md').write_text('\n\n'.join(text)+'\n')
    return run/'report.md'
