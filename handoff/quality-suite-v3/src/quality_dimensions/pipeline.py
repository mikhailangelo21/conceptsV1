from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import yaml
from .util import atomic_json,read_json,digest
from .data import load_data,split_concepts,select_subjects
from .prompts import geometry_prompts,behavioral_prompts,tokenize_prompt,answer_encoding,TEMPLATES,CONDITIONS
from .model import environment,resolve,disk_preflight,Runner,BudgetExpired,seed_all
from .cache import ActivationCache
from .analysis import analyze
from .interventions import behavior,summarize,collect_scores
from .report import report


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()


def configuration(path):
    path=Path(path).resolve(); c=yaml.safe_load(path.read_text()); c['config_path']=str(path)
    for key in ['concepts','references','pairs','run_root','cache_root']:
        p=Path(c[key]); c[key]=str(p if p.is_absolute() else path.parent.parent/p)
    if c['batch_size']!=1: raise ValueError('Validated implementation requires batch_size=1')
    if not 1<=c['max_tokens']<=256: raise ValueError('max_tokens must be <=256')
    if len(c['seen_templates'])!=2 or c['heldout_template'] in c['seen_templates']: raise ValueError('Two seen and one distinct held-out template required')
    if not set(c['seen_templates']+[c['heldout_template']]+c['context_templates'])<=set(TEMPLATES): raise ValueError('Unknown template')
    if not set(c['context_templates'])<=set(c['seen_templates']+[c['heldout_template']]): raise ValueError('Context template outside extraction templates')
    if not set(c['behavior_templates'])<={'b0','b1'}: raise ValueError('Unknown behavior template')
    if 0 not in c['strengths'] or not any(s<0 for s in c['strengths']) or not any(s>0 for s in c['strengths']): raise ValueError('Need zero, negative, positive strengths')
    if min(c['ridge_alphas'])<=0: raise ValueError('Ridge penalties must be positive')
    if c['bootstrap_replicates']<1 or c['permutations']<1: raise ValueError('Positive replicate counts required')
    return c


def prepare(config,run_dir=None,resume=False,preparing=False):
    concepts,refs,pairs=load_data(config); splits=split_concepts(concepts,config['seed']); subjects=select_subjects(concepts,splits,config['intervention_subjects'])
    geometry=geometry_prompts(concepts,splits,config); behavioral=behavioral_prompts(concepts,refs,pairs,splits,subjects,config)
    code_hash=digest({p.name:p.read_text() for p in sorted(Path(__file__).parent.glob('*.py'))})
    identity=dict(config=config,data=concepts.to_dict('records'),references=refs.to_dict('records'),pairs=pairs.to_dict('records'),templates=TEMPLATES,conditions=CONDITIONS,code_hash=code_hash)
    fingerprint=digest(identity); run=Path(run_dir) if run_dir else Path(config['run_root'])/(config['profile']+'-'+fingerprint[:12])
    if (run/'identity.json').exists():
        if read_json(run/'identity.json')['fingerprint']!=fingerprint: raise ValueError('Run identity mismatch: config/data/templates/code changed. Start a separate run; completed results will not be mutated.')
        if not (resume or preparing): raise ValueError(f'Run exists: {run}; use --resume')
        return run,concepts,refs,pairs,splits,subjects,geometry,behavioral
    if run.exists() and any(run.iterdir()): raise ValueError('Refusing nonempty directory without a matching run manifest')
    run.mkdir(parents=True,exist_ok=True)
    atomic_json(run/'identity.json',dict(fingerprint=fingerprint,code_hash=code_hash,dataset_hash=digest(identity['data']),template_hash=digest([TEMPLATES,CONDITIONS])))
    atomic_json(run/'config.json',config); atomic_json(run/'environment.json',environment())
    concepts.to_csv(run/'concepts.csv',index=False); refs.to_csv(run/'references.csv',index=False); pairs.to_csv(run/'pairs.csv',index=False)
    atomic_json(run/'splits.json',splits); atomic_json(run/'subjects.json',dict(ids=subjects,rule='Round-robin categories; alternating smallest/largest held-out ordinal rank; pre-model selection'))
    atomic_json(run/'prompts_unencoded.json',geometry); atomic_json(run/'behavior_prompts_unencoded.json',behavioral)
    n_intervention=sum(r['concept_id'] in subjects and r['split']=='test' for r in behavioral)
    # Distinct extra layers counted conservatively until primary layer is selected.
    layers=1+len(set(config.get('extra_intervention_layers',[])))
    plan=dict(geometry_prompts=len(geometry),baseline_prompts=len(behavioral),intervention_prompts=n_intervention,
              directions=1+config['random_directions'],strengths=config['strengths'],intervention_layers_upper_bound=layers,
              intervention_forwards_upper_bound=n_intervention*(1+config['random_directions'])*len(config['strengths'])*layers,
              benchmark_forwards=12,split_group_counts={s:concepts[concepts.concept_id.map(splits).eq(s)].synonym_group.nunique() for s in ['train','validation','test']},
              reference_pair_reuse=pairs.reference.value_counts().to_dict(),reference_prompt_reuse=pd.Series([r['reference'] for r in behavioral]).value_counts().to_dict())
    plan['total_forwards_upper_bound']=plan['geometry_prompts']+plan['baseline_prompts']+plan['intervention_forwards_upper_bound']+12
    atomic_json(run/'plan.json',plan); atomic_json(run/'state.json',dict(status='prepared',created=now(),extraction_completed=0,behavior_rows_completed=0))
    return run,concepts,refs,pairs,splits,subjects,geometry,behavioral


def cache_identity(metadata):
    keys=['model','model_revision','tokenizer_revision','backend','dtype','attention','num_hidden_layers','hidden_size','versions','hook','implementation_hash']
    return {k:metadata[k] for k in keys}


def execute(config,stage='run',run_dir=None,resume=False,max_minutes=None):
    prepared=prepare(config,run_dir,resume); run,concepts,refs,pairs,splits,subjects,geometry,behavioral=prepared
    state=read_json(run/'state.json'); state.pop('error_type',None); runner=None; cache=None; start=now()
    if state['status']=='complete' and stage=='run': print(f'Already complete: {run}'); return run
    print('Run:',run,flush=True); print(json.dumps(read_json(run/'plan.json'),indent=2),flush=True)
    seed_all(config['seed'])
    try:
        if stage=='analyze':
            metadata=read_json(run/'model.json'); geometry=read_json(run/'prompts.json'); cache=ActivationCache(config['cache_root'],cache_identity(metadata))
            activations=[cache.get(r) for r in geometry]
            if any(a is None for a in activations): raise RuntimeError('Extraction incomplete; resume extract/run before analyze')
            concepts=pd.read_csv(run/'concepts_analysis.csv'); analyze(run,config,concepts,geometry,activations)
            state.update(status='partial',reason='Analysis complete; behavioral stage may remain.')
        else:
            previous=read_json(run/'model.json') if (run/'model.json').exists() else None
            cfg,tok,metadata=resolve(config,previous['model_revision'] if previous else None)
            # Inspection fingerprint is computed before loading weights, also for blocked runs.
            import inspect
            from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer,Qwen3Model
            metadata['implementation_hash']=digest([inspect.getsource(Qwen3DecoderLayer.forward),inspect.getsource(Qwen3Model.forward)])
            if previous and cache_identity(previous)!=cache_identity(metadata): raise ValueError('Backend/library/model cache identity changed on resume; start a separate run')
            atomic_json(run/'model.json',metadata)
            geometry=[tokenize_prompt(tok,r,config['max_tokens']) for r in geometry]
            behavioral=[tokenize_prompt(tok,r,config['max_tokens']) for r in behavioral]; encoding=answer_encoding(tok,behavioral)
            atomic_json(run/'prompts.json',geometry); atomic_json(run/'behavior_prompts.json',behavioral); atomic_json(run/'answer_encoding.json',encoding)
            concepts['noun_char_count']=concepts.lemma.str.len(); concepts['noun_token_count']=[len(tok.encode(noun,add_special_tokens=False)) for noun in concepts.lemma]
            concepts.to_csv(run/'concepts_analysis.csv',index=False)
            plan=read_json(run/'plan.json'); plan['activation_float32_bytes']=len(geometry)*cfg.num_hidden_layers*cfg.hidden_size*4
            plan['token_count_range']=[min(r['token_count'] for r in geometry+behavioral),max(r['token_count'] for r in geometry+behavioral)]
            atomic_json(run/'plan.json',plan)
            cache=ActivationCache(config['cache_root'],cache_identity(metadata)); atomic_json(run/'activation_cache.json',dict(root=str(cache.root),identity=cache.identity))
            atomic_json(run/'activation_index.json',[dict(row_index=i,prompt_id=r['prompt_id'],cache_file=str(cache.path(r))) for i,r in enumerate(geometry)])
            missing=sum(cache.get(r) is None for r in geometry)
            disk_preflight(metadata,missing*cfg.num_hidden_layers*cfg.hidden_size*4)
            runner=Runner.load(config,metadata,max_minutes if max_minutes is not None else config['max_model_minutes'])
            atomic_json(run/'model.json',metadata)
            if not (run/'benchmark.json').exists():
                runner.extract(geometry[0]); runner.score(behavioral[0],encoding)
                extraction_times=[]; scoring_times=[]
                for i in range(5):
                    start_t=runner.elapsed; runner.extract(geometry[i%len(geometry)]); extraction_times.append(runner.elapsed-start_t)
                    start_t=runner.elapsed; runner.score(behavioral[i%len(behavioral)],encoding); scoring_times.append(runner.elapsed-start_t)
                extraction_mean=float(np.mean(extraction_times)); scoring_mean=float(np.mean(scoring_times))
                atomic_json(run/'benchmark.json',dict(warmup_forwards=2,measured_forwards=10,extraction_seconds=extraction_times,scoring_seconds=scoring_times,
                     prompts_per_second=10/(sum(extraction_times)+sum(scoring_times)),approximate_remaining_minutes=(missing*extraction_mean+(plan['baseline_prompts']+plan['intervention_forwards_upper_bound'])*scoring_mean)/60,
                     caveat='Approximate: interventions, prompt lengths, thermal throttling and other applications can change throughput; synchronized device timings.'))
            if stage in ['run','extract']:
                for i,row in enumerate(geometry):
                    if cache.get(row) is None: cache.put(row,runner.extract(row))
                    state['extraction_completed']=i+1
                    atomic_json(run/'state.json',dict(state,status='running',updated=now()))
                    if (i+1)%25==0: print(f'Extracted {i+1}/{len(geometry)}; model seconds this session {runner.elapsed:.1f}',flush=True)
                atomic_json(run/'extraction_complete.json',dict(rows=len(geometry),cache_fingerprint=cache.key))
            if stage=='run' and not (run/'analysis_complete.json').exists():
                activations=[cache.get(r) for r in geometry]; analyze(run,config,concepts,geometry,activations); del activations
            if stage in ['run','intervene']:
                if not (run/'analysis_complete.json').exists(): raise RuntimeError('Run analysis before interventions')
                selected=read_json(run/'selection.json')['selected_layer']; layers=list(dict.fromkeys([selected]+config['extra_intervention_layers']))
                if any(l<1 or l>cfg.num_hidden_layers for l in layers): raise ValueError('Intervention block out of range')
                plan['intervention_layers']=layers; plan['intervention_forwards']=plan['intervention_prompts']*(1+config['random_directions'])*len(config['strengths'])*len(layers)
                plan['total_planned_forwards']=len(geometry)+len(behavioral)+plan['intervention_forwards']+12; atomic_json(run/'plan.json',plan)
                behavior(run,runner,behavioral,encoding,config,subjects,selected)
            complete=all((run/name).exists() for name in ['extraction_complete.json','analysis_complete.json','behavior_complete.json'])
            state.update(status='complete' if complete else 'partial',reason='All planned stages completed.' if complete else f'{stage} stage finished; other stages remain.')
    except BudgetExpired as exc:
        state.update(status='partial',reason=str(exc))
    except (Exception,KeyboardInterrupt) as exc:
        state['error_type']=type(exc).__name__
        state.update(status='blocked' if isinstance(exc,RuntimeError) and ('disk' in str(exc).lower() or 'out of memory' in str(exc).lower()) else 'partial',reason=f'{type(exc).__name__}: {exc}')
        if 'out of memory' in str(exc).lower(): state['reason']+=' Completed rows are preserved. Close memory-heavy applications or start a separate CPU/smaller-model run; MPS safeguards remain enabled.'
        print(state['reason'],flush=True)
    finally:
        if cache is not None: state['extraction_completed']=sum(cache.get(r) is not None for r in geometry)
        summarize(run,config); state['behavior_rows_completed']=len(collect_scores(run)); state['updated']=now()
        if runner is not None:
            previous_resources=read_json(run/'resources.json') if (run/'resources.json').exists() else {'sessions':[]}
            previous_resources['sessions'].append(dict(started=start,ended=now(),model_seconds=runner.elapsed,forwards=len(runner.timings),peak_measurements={key:max(r[key] for r in runner.resources) for key in runner.resources[0]} if runner.resources else {}))
            atomic_json(run/'resources.json',previous_resources)
        atomic_json(run/'state.json',state); report(run)
    print('Report:',run/'report.md',flush=True)
    return run
