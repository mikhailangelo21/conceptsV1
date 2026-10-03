"""Small, explicitly enabled, development-gated factorial patch follow-up."""
import json
import numpy as np
import pandas as pd
from .hooks import capture, patch
from .scoring import encode_choices, score_choices
from .util import atomic_json, read_json


def causal_followup(run,suite,runner,tokenizer):
    cfg=suite['config']['intervention']
    if not cfg['enabled']:
        atomic_json(run/'causal_gate.json',dict(status='disabled_by_configuration',interventions_run=0,
                    reason='Initial smoke/laptop specification disables causal follow-up; factorial diagnostics are observational.'))
        return
    raw=pd.read_csv(run/'behavior_scores.csv')
    dev=raw[(raw.entity_split=='validation') & (raw.template_split=='train') & (raw.format=='natural_continuation')]
    eligible=[]
    for q,g in dev.groupby('quality_id'):
        if len(g)>=8 and g.entity_group_id.nunique()>=2 and g.correct.mean()>=.75:
            eligible.append((float(g.correct.mean()),q))
    selected=[q for _,q in sorted(eligible,key=lambda x:(-x[0],x[1]))[:cfg['maximum_initial_properties']]]
    selection=read_json(run/'model_selection.json'); cases={r['case_id']:r for r in suite['cases']}
    qualities={q['key']:q for q in json.loads((__import__('pathlib').Path(suite['config']['dataset_root'])/'qualities.json').read_text())}
    plan=[]
    for q in selected:
        if q not in selection: continue
        # Entity patch must have another attention block before the answer readout.
        layer=min(selection[q]['block'],runner.metadata['num_hidden_layers']-1)
        candidates=[c for c in suite['contrasts'] if c['kind']=='factorial_one_axis_increase' and c['quality_id']==q
                    and c['entity_split']=='test' and c['low_case_id'] in cases and c['high_case_id'] in cases]
        for c in candidates[:2]:
            low,high=cases[c['low_case_id']],cases[c['high_case_id']]
            for measured in sorted(set(low['primitive_axes']) & set(selected) & set(qualities)):
                plan.append(dict(contrast=c,quality=q,measured_quality=measured,layer=layer))
    atomic_json(run/'causal_gate.json',dict(status='selected_on_development_only',competence_threshold=.75,
                selected_properties=selected,planned_cells=len(plan),max_pairs_per_property=2,
                selection_rule='At least 8 validation questions and 2 validation entity groups; natural continuation accuracy >= .75; at most 3 properties'))
    rows=[]; root=run/'causal_shards'; root.mkdir(exist_ok=True)
    for item in plan:
        c=item['contrast']; low,high=cases[c['low_case_id']],cases[c['high_case_id']]; q=item['measured_quality']; spec=qualities[q]
        def prompt(case):
            # Canonical high-vs-low continuation for the stated recipient property.
            text=case['text']+f"\nThe {spec['quantity']} of {case['readout_entity']} is"
            row=encode_choices(tokenizer,dict(text=text,choices={'0':' '+spec['higher'],'1':' '+spec['lower']}),suite['config']['max_tokens'])
            start,end=case['readout_char_span']
            inds=[i for i,(a,b) in enumerate(row['token_offsets']) if a<end and b>start]
            if not inds: raise ValueError('Missing causal entity span')
            row['entity_index']=inds[-1]; return row
        recipient,donor=prompt(low),prompt(high); layer=item['layer']
        with capture([runner.blocks[layer-1]],recipient['entity_index']) as captured: runner.forward(recipient)
        original=captured[1].numpy()
        with capture([runner.blocks[layer-1]],donor['entity_index']) as captured: runner.forward(donor)
        replacement=captured[1].numpy()
        # Projection direction from the train-only fitted property at a validated matching block.
        with np.load(run/'fits'/f"{item['quality']}.npz") as fit:
            direction=fit['mean_direction'].copy() if int(fit['block'])==layer and str(fit['readout'])=='entity' else None
        rng=np.random.default_rng(20260928); random=rng.normal(size=len(original)); random/=np.linalg.norm(random)
        for condition in ['baseline','identity','full_patch','random_patch','component_patch']:
            if condition=='component_patch' and direction is None: continue
            shard=root/f"{c['contrast_id']}_{q}_{condition}.json"
            if shard.exists(): rows.append(read_json(shard)); continue
            def factory():
                return patch(runner.blocks[layer-1],recipient['entity_index'],
                             original if condition=='identity' else replacement,recipient_reference=original,
                             direction=direction if condition=='component_patch' else None,
                             random_direction=random if condition=='random_patch' else None)
            result=score_choices(runner,recipient,None if condition=='baseline' else factory)
            row=dict(contrast_id=c['contrast_id'],intervened_quality=item['quality'],measured_quality=q,
                     entity_group_id=low['entity_group_id'],layer=layer,condition=condition,native_margin=result['native_margin'],
                     precise_margin=result['precise_margin'],raw_log_probs=result['log_probs'],stats=result['stats'])
            atomic_json(shard,row); rows.append(row)
    frame=pd.DataFrame(rows)
    if not frame.empty:
        keys=['contrast_id','intervened_quality','measured_quality','entity_group_id','layer']
        baseline=frame[frame.condition=='baseline'][keys+['native_margin']].rename(columns={'native_margin':'baseline_margin'})
        frame=frame.merge(baseline,on=keys,validate='many_to_one'); frame['paired_delta']=frame.native_margin-frame.baseline_margin
    frame.to_csv(run/'causal_effect_matrix.csv',index=False)
    atomic_json(run/'causal_complete.json',dict(cells=len(plan),rows=len(rows),negative_findings_retained=True,
                interpretation='Matched within-pair patch minus baseline; rows share entity families. Candidate margin tests high/low semantics, not calibrated absolute physical values.'))
