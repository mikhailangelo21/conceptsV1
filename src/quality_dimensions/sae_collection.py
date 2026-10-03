"""Versioned, resumable Qwen-Scope SAE data collection for quality-suite V3."""
from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
from transformers import AutoTokenizer

from .hooks import output_tensor
from .model import BudgetExpired, Runner, resolve
from .sae_core import HIDDEN, WIDTH, encode_decode, load_checkpoint, sae_index, sha256_file
from .suite_cache import SuiteActivationCache
from .suite_data import load_suite, tokenize_case
from .util import atomic_json, atomic_npz, digest, read_json

VERSION = 'sae_scope_v1'
HOOK = 'model.model.layers[N] complete output before final RMSNorm'


def _jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def _write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as f:
        for row in rows: f.write(json.dumps(row, sort_keys=True) + '\n')
        f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def _suite_paths(config, config_path):
    base=Path(config_path).expanduser().resolve().parent.parent
    def resolved(value):
        path=Path(value).expanduser()
        return (path if path.is_absolute() else base/path).resolve()
    return resolved(config['suite_config']),resolved(config['source_v3_run'])


def _select_trace(cases, count):
    """Metadata-only round-robin over families, then quality/wording/splits."""
    families = collections.defaultdict(lambda: collections.defaultdict(list))
    for case in cases:
        key = (tuple(case['quality_ids']), case['template_id'], case['entity_split'],
               case['template_split'], case['combination_split'])
        families[case['case_type']][key].append(case)
    for buckets in families.values():
        for bucket in buckets.values(): bucket.sort(key=lambda c: digest(c['case_id']))
    family_order = sorted(families, key=digest)
    strata = {family: sorted(families[family], key=digest) for family in family_order}
    chosen = []; seen_text=set()
    while len(chosen) < min(count, len(cases)) and family_order:
        active = []
        for family in family_order:
            if len(chosen) >= count: break
            keys = strata[family]
            if keys:
                key = keys.pop(0)
                while families[family][key]:
                    candidate=families[family][key].pop(0)
                    if candidate['text'] not in seen_text:
                        chosen.append(candidate);seen_text.add(candidate['text']);break
                if families[family][key]: keys.append(key)
            if keys: active.append(family)
        family_order = active
    return chosen


def _trace_rows(cases, count, tokenizer):
    chosen = _select_trace(cases,count)
    traces=[]
    for c in chosen:
        ids=tokenizer(c['text'],add_special_tokens=False,return_offsets_mapping=True)
        traces.append({'trace_id':digest(['trace',ids['input_ids']]), 'case_id':c['case_id'],
                       'text_key':digest(ids['input_ids']), 'token_count':len(ids['input_ids']),
                       'selection_stratum':[c['case_type'],c['quality_ids'],c['template_id'],c['entity_split'],c['template_split'],c['combination_split']]})
    selection={'algorithm_version':'family_round_robin_v2','requested':count,'selected':len(traces),
        'case_families':dict(collections.Counter(c['case_type'] for c in chosen)),
        'qualities':dict(collections.Counter(q for c in chosen for q in c['quality_ids'])),
        'entity_splits':dict(collections.Counter(c['entity_split'] for c in chosen)),
        'template_splits':dict(collections.Counter(c['template_split'] for c in chosen)),
        'excluded_case_count':len(cases)-len(chosen),
        'selected_case_ids':[c['case_id'] for c in chosen],
        'excluded_case_ids':sorted({c['case_id'] for c in cases}-{c['case_id'] for c in chosen}),
        'exclusion_reason':'bounded deterministic metadata-stratified sample'}
    selection['trace_ids_sha256']=digest([t['trace_id'] for t in traces])
    return traces,selection


def _feature_index(run, config):
    main=[r['readout_id'] for r in _jsonl(run/'readouts.jsonl')]
    trace=[f"{t['trace_id']}:{pos}" for t in _jsonl(run/'traces.jsonl') for pos in range(t['token_count'])]
    if len(set(main))!=len(main) or len(set(trace))!=len(trace):
        raise ValueError('Nonunique SAE feature index key')
    index={'schema_version':VERSION,'chunk_rows':config['chunk_rows'],
           'main':{key:i for i,key in enumerate(main)},'trace':{key:i for i,key in enumerate(trace)}}
    path=run/'feature_index.json'
    if not path.exists() or read_json(path)!=index: atomic_json(path,index)


def _get_repo_info(config):
    variant = config['sae_variant']; repo = config['sae_repos'][variant]
    revision = config['sae_revisions'][variant]
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision: raise ValueError('SAE repository revision mismatch')
    files = {s.rfilename: s for s in info.siblings}
    checkpoints = {}
    for layer in range(28):
        name = f'layer{layer}.sae.pt'
        if name not in files or not files[name].lfs or not files[name].lfs.sha256:
            raise ValueError(f'Missing LFS hash for {name}')
        checkpoints[str(layer)] = {'filename': name, 'sha256': files[name].lfs.sha256,
                                   'bytes': files[name].size}
    return {'repo': repo, 'revision': revision, 'variant': variant,
            'k': 100 if variant == 'top100' else 50, 'checkpoints': checkpoints}


def _identity(config, suite, profile, layers, trace_count, source):
    model = read_json(source / 'model.json')
    cm = read_json(source / 'activation_cache.json')
    if model['model_revision'] != config['model_revision'] or cm['identity']['model_revision'] != config['model_revision']:
        raise ValueError('V3 model/cache revision differs from SAE configuration')
    if model['hook'] != cm['identity']['hook'] or cm['identity']['hidden_size'] != HIDDEN:
        raise ValueError('V3 hook or hidden width incompatible')
    if model['model'] != 'Qwen/Qwen3-1.7B-Base' or model['num_hidden_layers'] != 28:
        raise ValueError('Unexpected V3 model architecture')
    if config['feature_width'] != WIDTH or config['hidden_size'] != HIDDEN:
        raise ValueError('SAE dimensions mismatch')
    if any(not isinstance(i, int) or i < 0 or i > 27 for i in layers) or len(set(layers)) != len(layers):
        raise ValueError('Layers must be unique zero-based indices in 0..27')
    if config['encoding_dtype'] != 'float32' or config['encoding_backend'] not in {'cpu','mps','cuda'}:
        raise ValueError('Unsupported SAE numerical settings')
    base = {'version': VERSION, 'profile': profile, 'suite_fingerprint': suite['fingerprint'],
            'model_revision': model['model_revision'], 'tokenizer_revision': model['tokenizer_revision'],
            'legacy_cache_key': digest(cm['identity']), 'legacy_hook': cm['identity']['hook'],
            'layers': layers, 'trace_count': trace_count, 'variant': config['sae_variant'],
            'sae_revision': config['sae_revisions'][config['sae_variant']],
            'encoding_backend': config['encoding_backend'], 'encoding_dtype': config['encoding_dtype'],
            'chunk_rows': config['chunk_rows']}
    return base, model, cm


def _question_tokens(tokenizer, question, max_tokens):
    text = question['prefix']
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    if not ids or len(ids) > max_tokens: raise ValueError('Invalid comparison prefix length')
    return {'text': text, 'token_ids': ids, 'token_offsets': encoded['offset_mapping'],
            'readout_index': len(ids)-1, 'readout': 'final', 'token_count': len(ids)}


def prepare(config_path, profile='laptop', layers=None, trace_count=None, run_dir=None, resume=False):
    config = read_json(config_path)
    if config['schema_version'] != VERSION: raise ValueError('Unsupported SAE config')
    if profile not in {'smoke','laptop','extended'}: raise ValueError('Unknown profile')
    layers = sorted(config['layers'] if layers is None else layers)
    if profile == 'smoke' and layers == config['layers']: layers = [3]
    trace_count = (4 if profile == 'smoke' else config['trace_count']) if trace_count is None else trace_count
    if trace_count < 0: raise ValueError('Invalid trace count')
    suite_config, source = _suite_paths(config,config_path)
    suite = load_suite(suite_config, profile)
    identity, model, cm = _identity(config, suite, profile, layers, trace_count, source)
    name = f'{VERSION}_{profile}_{config["sae_variant"]}-{digest(identity)[:12]}'
    if run_dir:
        run = Path(run_dir).expanduser().resolve()
    else:
        run = Path(config_path).resolve().parent.parent / 'runs' / name
        if not run.exists():
            local = Path.home() / '.local/share/quality-dimensions-llm/runs' / name
            local.mkdir(parents=True, exist_ok=True); run.parent.mkdir(parents=True, exist_ok=True)
            run.symlink_to(local, target_is_directory=True)
    if (run/'identity.json').exists():
        if read_json(run/'identity.json') != identity: raise ValueError('SAE run identity mismatch')
        if not resume: raise ValueError(f'Run already exists; pass --resume: {run}')
    else:
        if run.exists() and any(run.iterdir()): raise ValueError('Nonempty run has no SAE identity')
        run.mkdir(parents=True, exist_ok=True)
        atomic_json(run/'identity.json', identity)
        atomic_json(run/'config.json', config)
        atomic_json(run/'source_v3.json', {'run': str(source), 'model': model,
                                         'activation_cache': cm, 'suite_fingerprint': suite['fingerprint']})
        repo = _get_repo_info(config)
        atomic_json(run/'sae_repository.json', repo)
        tokenizer = AutoTokenizer.from_pretrained(model['model'], revision=model['tokenizer_revision'], local_files_only=True)
        contrast_map = collections.defaultdict(list)
        for contrast in suite['contrasts']:
            for key, other in [('low_case_id','high_case_id'),('high_case_id','low_case_id')]:
                contrast_map[contrast[key]].append({'contrast_id': contrast['contrast_id'], 'paired_case_id': contrast[other], 'kind': contrast['kind']})
        triplet_map = collections.defaultdict(list)
        for triplet in suite['triplets']:
            for key in ['anchor_case_id','near_case_id','far_case_id']:
                triplet_map[triplet[key]].append({'triplet_id':triplet['triplet_id'],'role':key,
                                                  'paired_case_ids':[triplet[k] for k in ['anchor_case_id','near_case_id','far_case_id'] if k!=key]})
        readouts = {}; records = []
        def add(row, kind, metadata):
            rid = digest([row['token_ids'], row['readout_index']])
            text_key = digest(row['token_ids'])
            if rid not in readouts:
                readouts[rid] = {'readout_id': rid, 'text_key': text_key, 'text':row['text'],
                                 'token_ids':row['token_ids'], 'token_offsets':row['token_offsets'],
                                 'token_count':row['token_count'], 'position':row['readout_index'],
                                 'readout_type':row['readout']}
            records.append({'record_id':digest([kind, metadata.get('case_id',metadata.get('comparison_id')),row['readout']]),
                            'readout_id':rid,'kind':kind,'status':'included','metadata':metadata})
        for case in suite['cases']:
            metadata = dict(case)
            metadata['contrasts'] = contrast_map[case['case_id']]
            metadata['hue_triplets'] = triplet_map[case['case_id']]
            for row in tokenize_case(tokenizer,case,suite['config']['max_tokens']): add(row,'case',metadata)
        for q in suite['questions']:
            row = _question_tokens(tokenizer,q,suite['config']['max_tokens'])
            metadata = dict(q)
            metadata['entity_readout_exclusion'] = 'Comparison prefix has no V3 contextual-target entity span'
            add(row,'comparison',metadata)
        traces,selection=_trace_rows(suite['cases'],trace_count,tokenizer)
        _write_jsonl(run/'readouts.jsonl',readouts.values())
        _write_jsonl(run/'records.jsonl',records)
        _write_jsonl(run/'traces.jsonl',traces)
        atomic_json(run/'trace_selection.json',selection)
        atomic_json(run/'target_registry.json',suite['registry'])
        atomic_json(run/'state.json',{'status':'prepared','model_seconds':0.0,'encoding_seconds':0.0,'issues':[]})
    if not (run/'target_registry.json').exists():
        atomic_json(run/'target_registry.json',suite['registry'])
    selection=read_json(run/'trace_selection.json')
    if selection.get('algorithm_version')!='family_round_robin_v2':
        if any((run/'raw/trace').glob('*.npz')) or any((run/'features/trace').glob('layer*/*.npz')):
            # Completed older smoke data retain their original documented sample.
            pass
        else:
            tokenizer=AutoTokenizer.from_pretrained(model['model'],revision=model['tokenizer_revision'],local_files_only=True)
            traces,new_selection=_trace_rows(suite['cases'],trace_count,tokenizer)
            _write_jsonl(run/'traces.jsonl',traces)
            atomic_json(run/'trace_selection.json',new_selection)
            selection=new_selection
    if 'excluded_case_ids' not in selection:
        chosen_ids={t['case_id'] for t in _jsonl(run/'traces.jsonl')}
        selection['selected_case_ids']=[t['case_id'] for t in _jsonl(run/'traces.jsonl')]
        selection['excluded_case_ids']=sorted({c['case_id'] for c in suite['cases']}-chosen_ids)
        atomic_json(run/'trace_selection.json',selection)
    _feature_index(run,config)
    return run


def _legacy(run):
    cm = read_json(run/'source_v3.json')['activation_cache']
    root = Path(cm['root'])
    if not root.exists():
        # A collected run may be copied from another machine. Its original cache
        # path is provenance, while the identical cache shards can live locally.
        key = digest(cm['identity'])
        override = os.environ.get('QD_V3_CACHE_BASE')
        base = Path(override).expanduser() if override else Path(__file__).resolve().parents[2]/'cache'
        root = base/'quality_suite_v3'/key
        if override and not root.exists():
            raise FileNotFoundError(f'QD_V3_CACHE_BASE does not contain the pinned V3 cache: {root}')
    cache = SuiteActivationCache(root.parents[1], cm['identity'])
    if cache.root != root: raise ValueError('Legacy cache root mismatch')
    return cache


def _source_v3_run(run):
    """Locate a copied V3 run without rewriting its original provenance."""
    recorded = Path(read_json(run/'source_v3.json')['run'])
    if recorded.exists():
        return recorded
    override = os.environ.get('QD_V3_RUN_DIR')
    candidate = Path(override).expanduser() if override else Path(__file__).resolve().parents[2]/'runs'/recorded.name
    if not candidate.exists():
        raise FileNotFoundError(f'V3 run is missing; set QD_V3_RUN_DIR to its copied location: {candidate}')
    return candidate


def _legacy_row(readout):
    return {'token_ids':readout['token_ids'],'readout_index':readout['position']}


def _main_path(run, text_key): return run/'raw/main'/f'{text_key}.npz'
def _trace_path(run, text_key): return run/'raw/trace'/f'{text_key}.npz'


def _read_raw(path, identity):
    if not path.exists(): return None
    with np.load(path,allow_pickle=False) as d:
        if str(d['identity']) != identity: raise ValueError(f'Raw shard identity mismatch: {path}')
        return {k:d[k].copy() for k in d.files if k!='identity'}


def _raw_for(run, readout, layer, legacy, legacy_layer_map):
    if layer in legacy_layer_map:
        value = legacy.get(_legacy_row(readout))
        if value is not None: return value[legacy_layer_map[layer]]
    data = _read_raw(_main_path(run,readout['text_key']), digest(read_json(run/'identity.json')))
    if data is None: return None
    ls=data['layers'].tolist(); ps=data['positions'].tolist()
    if layer not in ls or readout['position'] not in ps: return None
    return data['values'][ls.index(layer),ps.index(readout['position'])]


def plan(run):
    run=Path(run); identity=read_json(run/'identity.json'); config=read_json(run/'config.json')
    readouts=list(_jsonl(run/'readouts.jsonl')); traces=list(_jsonl(run/'traces.jsonl'))
    cache=_legacy(run); legacy_layers={sae_index(i):n for n,i in enumerate(cache.identity['block_numbers'])}
    old_hits=sum(1 for r in readouts if cache.path(_legacy_row(r)).exists())
    main_missing=sum(len(identity['layers'])-int(cache.path(_legacy_row(r)).exists())*len(set(identity['layers'])&set(legacy_layers)) for r in readouts)
    trace_tokens=sum(t['token_count'] for t in traces)
    raw_estimate=main_missing*HIDDEN*4+trace_tokens*len(identity['layers'])*HIDDEN*4
    repo=read_json(run/'sae_repository.json')
    downloads=[]
    for layer in identity['layers']:
        file=repo['checkpoints'][str(layer)]
        cached=try_to_load_from_cache(repo['repo'],file['filename'],revision=repo['revision'])
        if not isinstance(cached,str): downloads.append({'layer':layer,'bytes':file['bytes']})
    sparse_estimate=(len(readouts)+trace_tokens)*len(identity['layers'])*repo['k']*8
    by_family=dict(collections.Counter(r['metadata']['case_type'] for r in _jsonl(run/'records.jsonl') if r['kind']=='case'))
    result={'run':str(run),'profile':identity['profile'],'layers':identity['layers'],
            'case_records':sum(by_family.values()),'case_family_records':by_family,
            'comparison_records':sum(1 for r in _jsonl(run/'records.jsonl') if r['kind']=='comparison'),
            'unique_readouts':len(readouts),'trace_prompts':len(traces),'trace_tokens':trace_tokens,
            'legacy_cache_readout_hits':old_hits,'legacy_cache_layers':sorted(legacy_layers),
            'main_layer_readouts_missing_before_new_extraction':main_missing,
            'raw_upper_bound_bytes':raw_estimate,'sparse_feature_estimate_bytes':sparse_estimate,
            'new_checkpoint_download_bytes':sum(r['bytes'] for r in downloads),
            'new_checkpoint_downloads':downloads,'free_disk_bytes':shutil.disk_usage(run).free,
            'estimated_new_text_forwards':len({r['text_key'] for r in readouts if not (set(identity['layers'])<=set(legacy_layers) and cache.path(_legacy_row(r)).exists())}|{t['text_key'] for t in traces}),
            'model_forward_budget_minutes':config['max_model_minutes'],
            'note':'Raw bytes are uncompressed upper estimates; CPU encoding time varies. Checkpoints download one layer on demand.'}
    atomic_json(run/'plan.json',result)
    return result


@contextmanager
def _capture_layers(runner, layers, positions=None):
    values={}; handles=[]
    def hook_for(layer):
        def hook(_module,_args,output):
            h=output_tensor(output)
            if h.ndim!=3 or h.shape[0]!=1: raise ValueError('Expected batch size one')
            v=h[0] if positions is None else h[0,positions,:]
            v=v.detach().to(device='cpu',dtype=torch.float32)
            values[layer]=v.clone().numpy()
        return hook
    try:
        for layer in layers: handles.append(runner.blocks[layer].register_forward_hook(hook_for(layer)))
        yield values
    finally:
        for h in handles: h.remove()


def extract(run, max_model_minutes=None):
    run=Path(run); identity=read_json(run/'identity.json'); config=read_json(run/'config.json')
    state=read_json(run/'state.json'); budget=(max_model_minutes or config['max_model_minutes'])*60
    readouts=list(_jsonl(run/'readouts.jsonl')); traces={t['text_key']:t for t in _jsonl(run/'traces.jsonl')}
    cache=_legacy(run); cached_layers={sae_index(i) for i in cache.identity['block_numbers']}
    by_text=collections.defaultdict(list)
    for r in readouts: by_text[r['text_key']].append(r)
    source=read_json(run/'source_v3.json')['model']
    model_config={'model':source['model'],'revision':source['model_revision'],'backend':source['backend'],'attention':source['attention']}
    runner=None; processed=0
    try:
        for text_key, rows in by_text.items():
            raw_main=_read_raw(_main_path(run,text_key),digest(identity))
            raw_trace=_read_raw(_trace_path(run,text_key),digest(identity)) if text_key in traces else None
            missing_main=[l for l in identity['layers'] if not (l in cached_layers and all(cache.path(_legacy_row(r)).exists() for r in rows)) and (raw_main is None or l not in raw_main['layers'])]
            missing_trace=[l for l in identity['layers'] if text_key in traces and (raw_trace is None or l not in raw_trace['layers'])]
            needed=sorted(set(missing_main+missing_trace))
            if not needed: continue
            if shutil.disk_usage(run).free < 1024**3:
                state['issues'].append('Less than 1 GiB free during extraction; resume after freeing disk')
                break
            if state['model_seconds']+(runner.elapsed if runner else 0)>=budget: raise BudgetExpired('SAE extraction budget exhausted')
            if runner is None:
                _,_,meta=resolve(model_config,source['model_revision'])
                if meta['model_revision']!=source['model_revision'] or meta['hidden_size']!=HIDDEN: raise ValueError('Model identity mismatch')
                runner=Runner.load(model_config,meta,(budget-state['model_seconds'])/60)
                if meta['implementation_hash']!=source['implementation_hash']: raise ValueError('Model implementation differs from V3 cache')
            positions=sorted({r['position'] for r in rows})
            capture_all_tokens=bool(missing_trace)
            with _capture_layers(runner,needed,None if capture_all_tokens else positions) as captured:
                runner.forward({'token_ids':rows[0]['token_ids']})
            if missing_main:
                vals=np.stack([captured[l][positions] if capture_all_tokens else captured[l] for l in missing_main]).astype('float32')
                if raw_main is not None:
                    if raw_main['positions'].tolist()!=positions: raise ValueError('Raw position identity mismatch')
                    vals=np.concatenate([raw_main['values'],vals],axis=0)
                    missing_main=raw_main['layers'].tolist()+missing_main
                atomic_npz(_main_path(run,text_key),identity=np.array(digest(identity)),layers=np.array(missing_main,dtype='int16'),positions=np.array(positions,dtype='int16'),values=vals)
            if missing_trace:
                vals=np.stack([captured[l] for l in missing_trace]).astype('float32')
                if raw_trace is not None:
                    vals=np.concatenate([raw_trace['values'],vals],axis=0)
                    missing_trace=raw_trace['layers'].tolist()+missing_trace
                atomic_npz(_trace_path(run,text_key),identity=np.array(digest(identity)),layers=np.array(missing_trace,dtype='int16'),values=vals)
            processed+=1
            if processed%250==0:
                print(f'SAE raw extraction: {processed} new texts; {state["model_seconds"]+(runner.elapsed if runner else 0):.1f} cumulative model seconds',flush=True)
            if processed%25==0:
                state.update(status='partial',model_seconds=state['model_seconds']+runner.elapsed,extraction_texts_this_session=processed)
                runner.elapsed=0
                atomic_json(run/'state.json',state)
    except BudgetExpired as e:
        state['issues'].append(str(e))
    finally:
        if runner is not None:
            state['model_seconds']+=runner.elapsed
            del runner
        state['status']='partial'; state['extraction_texts_this_session']=processed
        atomic_json(run/'state.json',state)
    return state


def _checkpoint(run,layer):
    repo=read_json(run/'sae_repository.json'); item=repo['checkpoints'][str(layer)]
    cached=try_to_load_from_cache(repo['repo'],item['filename'],revision=repo['revision'])
    if not isinstance(cached,str) and shutil.disk_usage(run).free < item['bytes']+1024**3:
        raise RuntimeError(f'Insufficient disk for SAE layer {layer}; need checkpoint plus 1 GiB reserve')
    path=hf_hub_download(repo['repo'],item['filename'],revision=repo['revision'])
    if sha256_file(path)!=item['sha256']: raise ValueError('SAE checkpoint hash mismatch')
    return path,item


def _chunk_path(run, kind, layer, chunk): return run/'features'/kind/f'layer{layer}'/f'chunk{chunk:06d}.npz'


def _write_chunk(path, identity, layer, checkpoint, keys, arrays):
    payload={'identity':np.array(digest(identity)),'layer':np.array(layer,dtype='int16'),
             'checkpoint_sha256':np.array(checkpoint['sha256']), 'keys':np.asarray(keys), **arrays}
    atomic_npz(path,**payload)


def _check_chunk(path, identity, layer, checkpoint, expected_keys):
    if not path.exists(): return False
    with np.load(path,allow_pickle=False) as d:
        if str(d['identity'])!=digest(identity) or int(d['layer'])!=layer or str(d['checkpoint_sha256'])!=checkpoint['sha256'] or d['keys'].tolist()!=expected_keys:
            raise ValueError(f'Feature chunk identity mismatch: {path}')
        if d['ids'].shape!=(len(expected_keys),read_json(path.parents[3]/'sae_repository.json')['k']): raise ValueError('Feature chunk shape mismatch')
    return True


def _trace_items(run):
    readouts={r['text_key']:r for r in _jsonl(run/'readouts.jsonl')}
    for t in _jsonl(run/'traces.jsonl'):
        row=readouts[t['text_key']]
        for pos in range(t['token_count']):
            yield {'readout_id':f"{t['trace_id']}:{pos}",'text_key':t['text_key'],'position':pos,
                   'case_id':t['case_id'],'token_ids':row['token_ids']}


def _raw_trace_for(run,item,layer,identity):
    data=_read_raw(_trace_path(run,item['text_key']),digest(identity))
    if data is None or layer not in data['layers'].tolist(): return None
    return data['values'][data['layers'].tolist().index(layer),item['position']]


def _encode_items(run, kind, layer, items, weights, checkpoint, budget_deadline):
    identity=read_json(run/'identity.json'); config=read_json(run/'config.json'); repo=read_json(run/'sae_repository.json')
    cache=_legacy(run); legacy_map={sae_index(b):i for i,b in enumerate(cache.identity['block_numbers'])}
    chunk_size=config['chunk_rows']; made=0
    for chunk in range((len(items)+chunk_size-1)//chunk_size):
        group=items[chunk*chunk_size:(chunk+1)*chunk_size]
        keys=[r['readout_id'] for r in group]
        path=_chunk_path(run,kind,layer,chunk)
        if _check_chunk(path,identity,layer,checkpoint,keys): continue
        if time.monotonic()>budget_deadline: break
        vectors=[]
        for row in group:
            h=_raw_for(run,row,layer,cache,legacy_map) if kind=='main' else _raw_trace_for(run,row,layer,identity)
            if h is None: break
            vectors.append(h)
        if len(vectors)!=len(group): continue
        parts=[]
        for start in range(0,len(vectors),config['encoding_batch_size']):
            if time.monotonic()>budget_deadline: break
            ids,values,diagnostics=encode_decode(np.stack(vectors[start:start+config['encoding_batch_size']]),weights,repo['k'],config['encoding_backend'])
            parts.append({'ids':ids,'values':values,**diagnostics})
        if sum(len(p['ids']) for p in parts)!=len(group): break
        arrays={key:np.concatenate([p[key] for p in parts]) for key in parts[0]}
        _write_chunk(path,identity,layer,checkpoint,keys,arrays); made+=len(group)
    return made


def encode(run, max_encoding_minutes=None):
    run=Path(run); identity=read_json(run/'identity.json'); config=read_json(run/'config.json')
    state=read_json(run/'state.json'); budget=(max_encoding_minutes or config['max_encoding_minutes'])*60
    remaining=budget-state.get('encoding_seconds',0)
    if remaining<=0:
        state['issues'].append('SAE encoding budget exhausted; increase --max-encoding-minutes to resume')
        atomic_json(run/'state.json',state); return state
    main=list(_jsonl(run/'readouts.jsonl')); trace=list(_trace_items(run)); start=time.monotonic(); deadline=start+remaining
    baseline=state.get('encoding_seconds',0)
    cached_layers={sae_index(b) for b in _legacy(run).identity['block_numbers']}
    ordered=sorted(identity['layers'],key=lambda layer:(layer not in cached_layers,layer))
    try:
        for layer in ordered:
            if time.monotonic()>=deadline: break
            if layer not in cached_layers and not any((run/'raw/main').glob('*.npz')):
                continue
            try:
                checkpoint_path,checkpoint=_checkpoint(run,layer)
            except (RuntimeError,OSError) as exc:
                if isinstance(exc,RuntimeError) and not str(exc).startswith('Insufficient disk for SAE layer'): raise
                state['issues'].append(str(exc)); break
            weights=load_checkpoint(checkpoint_path,checkpoint['sha256'])
            main_done=_encode_items(run,'main',layer,main,weights,checkpoint,deadline)
            trace_done=_encode_items(run,'trace',layer,trace,weights,checkpoint,deadline)
            del weights
            print(f'SAE layer {layer}: newly encoded {main_done} main and {trace_done} trace rows',flush=True)
            state['encoding_seconds']=baseline+time.monotonic()-start
            atomic_json(run/'state.json',state)
    finally:
        state['encoding_seconds']=baseline+time.monotonic()-start
        state['status']='partial'; atomic_json(run/'state.json',state)
    return state


def validate(run):
    run=Path(run); identity=read_json(run/'identity.json'); config=read_json(run/'config.json'); repo=read_json(run/'sae_repository.json')
    source_model=read_json(run/'source_v3.json')['model']
    prior=read_json(run/'manifest.json') if (run/'manifest.json').exists() else {'files':{}}
    def checked_sha(path):
        relative=str(path.relative_to(run)); actual=sha256_file(path)
        old=prior['files'].get(relative)
        if old and relative.startswith(('features/','raw/')) and old['sha256']!=actual:
            raise ValueError(f'Previously manifested shard checksum changed: {relative}')
        return actual
    main=list(_jsonl(run/'readouts.jsonl')); trace=list(_trace_items(run)); family_by_id=collections.defaultdict(set)
    case_family={}
    for record in _jsonl(run/'records.jsonl'):
        if record['kind']=='case':
            family_by_id[record['readout_id']].add(record['metadata']['case_type'])
            case_family[record['metadata']['case_id']]=record['metadata']['case_type']
    for item in trace: family_by_id[item['readout_id']].add(case_family[item['case_id']])
    manifest={'schema_version':VERSION,'run_identity':digest(identity),'feature_width':WIDTH,'hidden_size':HIDDEN,
              'k':repo['k'],'sae_repo':repo['repo'],'sae_revision':repo['revision'],
              'model_revision':identity['model_revision'],'hook':HOOK,
              'numerics':{'raw_extraction_backend':source_model['backend'],'raw_extraction_dtype':source_model['dtype'],
                          'raw_stored_dtype':'float32','encoding_backend':identity['encoding_backend'],
                          'encoding_dtype':identity['encoding_dtype'],'torch_version':torch.__version__,
                          'numpy_version':np.__version__},
              'validation_code_sha256':{'sae_core.py':sha256_file(Path(__file__).with_name('sae_core.py')),
                                        'sae_collection.py':sha256_file(Path(__file__))},
              'schema':{'features':'NPZ keys,ids,values,raw_norm,error_norm,relative_error,cosine,nnz,flags',
                        'feature_arrays':{'keys':['rows'],'ids':['rows',repo['k']],'values':['rows',repo['k']],
                                          'raw_norm':['rows'],'error_norm':['rows'],'relative_error':['rows'],
                                          'cosine':['rows'],'nnz':['rows'],'flags':['rows']},
                        'raw_main_values':['layers','positions',HIDDEN],
                        'raw_trace_values':['layers','non_padding_tokens',HIDDEN],
                        'main_join':'features/main/layerN/chunk*.npz keys -> readouts.jsonl readout_id -> records.jsonl readout_id',
                        'trace_join':'features/trace/layerN/chunk*.npz keys -> traces.jsonl trace_id:token_position -> cases via case_id',
                        'zero_handling':'relative_error NaN and flags bit 0 if raw norm zero; cosine NaN flagged bit 2 if degenerate',
                        'flags':'bit 0 zero raw norm; bit 1 zero reconstruction norm; bit 2 nonfinite diagnostic'},
              'layers':{},'files':{},'status':'partial'}
    all_complete=True
    for kind,items in [('main',main),('trace',trace)]:
        expected_families=collections.Counter()
        for row in items: expected_families.update(family_by_id.get(row['readout_id'],[]))
        for layer in identity['layers']:
            expected=len(items); complete=0; diagnostics=[]; families=collections.Counter(); flag_counts=collections.Counter(); nnz=[]
            for chunk in range((expected+config['chunk_rows']-1)//config['chunk_rows']):
                group=items[chunk*config['chunk_rows']:(chunk+1)*config['chunk_rows']]
                path=_chunk_path(run,kind,layer,chunk)
                if not path.exists(): continue
                ck=repo['checkpoints'][str(layer)]
                if not _check_chunk(path,identity,layer,ck,[r['readout_id'] for r in group]): raise ValueError('Missing chunk')
                with np.load(path,allow_pickle=False) as d:
                    if d['values'].dtype!=np.float32 or d['ids'].dtype!=np.int32: raise ValueError('Feature dtype mismatch')
                    if np.any(d['ids']<0) or np.any(d['ids']>=WIDTH): raise ValueError('Feature ID out of range')
                    if not np.isfinite(d['values']).all(): raise ValueError('Nonfinite sparse value')
                    diagnostics.extend(d['relative_error'][np.isfinite(d['relative_error'])].tolist())
                    nnz.extend(d['nnz'].tolist())
                    flag_counts.update(int(flag) for flag in d['flags'])
                complete+=len(group)
                for row in group: families.update(family_by_id.get(row['readout_id'],[]))
                manifest['files'][str(path.relative_to(run))]={'sha256':checked_sha(path),'rows':len(group),
                    'sparse_shape':[len(group),repo['k']]}
            manifest['layers'][f'{kind}:{layer}']={'expected':expected,'completed':complete,'missing':expected-complete,
                    'case_family_expected':dict(expected_families),'case_family_completed':dict(families),
                    'case_family_missing':{k:expected_families[k]-families[k] for k in expected_families},
                    'relative_error_p10':float(np.quantile(diagnostics,0.1)) if diagnostics else None,
                    'relative_error_p50':float(np.median(diagnostics)) if diagnostics else None,
                    'relative_error_p90':float(np.quantile(diagnostics,0.9)) if diagnostics else None,
                    'relative_error_max':float(max(diagnostics)) if diagnostics else None,
                    'mean_nonzero_features':float(np.mean(nnz)) if nnz else None,
                    'diagnostic_flag_counts':{str(k):v for k,v in flag_counts.items()}}
            if complete!=expected: all_complete=False
    for name in ['identity.json','config.json','sae_repository.json','source_v3.json','readouts.jsonl','records.jsonl','traces.jsonl','trace_selection.json','target_registry.json','feature_index.json']:
        manifest['files'][name]={'sha256':sha256_file(run/name),'bytes':(run/name).stat().st_size}
    for path in sorted((run/'raw').glob('*/*.npz')):
        with np.load(path,allow_pickle=False) as d:
            if str(d['identity'])!=digest(identity) or d['values'].dtype!=np.float32 or d['values'].shape[-1]!=HIDDEN:
                raise ValueError(f'Incompatible raw SAE shard: {path}')
            shape=list(d['values'].shape)
        manifest['files'][str(path.relative_to(run))]={'sha256':checked_sha(path),'shape':shape}
    manifest['status']='complete' if all_complete else 'partial'
    atomic_json(run/'manifest.json',manifest)
    state=read_json(run/'state.json'); state['status']=manifest['status'];atomic_json(run/'state.json',state)
    _collection_report(run,manifest,state)
    return manifest


def _collection_report(run, manifest, state):
    identity=read_json(run/'identity.json'); selection=read_json(run/'trace_selection.json')
    plan_data=read_json(run/'plan.json') if (run/'plan.json').exists() else {}
    cache_hits=plan_data.get('legacy_cache_readout_hits',0)
    trace_tokens=plan_data.get('trace_tokens',sum(t['token_count'] for t in _jsonl(run/'traces.jsonl')))
    raw_main_count=sum(k.startswith('raw/main/') for k in manifest['files'])
    raw_trace_count=sum(k.startswith('raw/trace/') for k in manifest['files'])
    lines=[f'# Qwen-Scope SAE collection: {manifest["status"]}',
           '',f'Run: `{run}`. Profile: `{identity["profile"]}`. SAE: `{manifest["sae_repo"]}` at `{manifest["sae_revision"]}`.',
           f'Model revision: `{manifest["model_revision"]}`. Feature width {WIDTH:,}; Top-K {manifest["k"]}.',
           '', 'These are collection and reconstruction diagnostics, not feature interpretations or performance results.',
           '',f'Input associations: {plan_data.get("case_records",0):,} case/readout records and '
           f'{plan_data.get("comparison_records",0):,} comparison-prefix records, '
           f'deduplicated to {plan_data.get("unique_readouts",0):,} unique main readouts.',
           '',f'Original V3 cache readout hits: {cache_hits:,}. '
           f'New raw main shards: {sum(k.startswith("raw/main/") for k in manifest["files"]):,}. '
           f'New raw trace shards: {raw_trace_count:,}.',
           f'Trace sample: {selection["selected"]} prompts, {trace_tokens:,} tokens; '
           f'{len(selection.get("excluded_case_ids",[])):,} case IDs excluded by the bounded sample.',
           f'Trace prompt families: {json.dumps(selection.get("case_families",{}),sort_keys=True)}.',
           '', '| SAE layer | Main completed / expected | Trace completed / expected | Main relative error p50 / p90 |',
           '|---:|---:|---:|---:|']
    for layer in identity['layers']:
        main=manifest['layers'][f'main:{layer}']; trace=manifest['layers'][f'trace:{layer}']
        diagnostic=(f'{main["relative_error_p50"]:.3f} / {main["relative_error_p90"]:.3f}'
                    if main['relative_error_p50'] is not None else '—')
        lines.append(f'| {layer} | {main["completed"]:,} / {main["expected"]:,} | '
                     f'{trace["completed"]:,} / {trace["expected"]:,} | {diagnostic} |')
    lines.extend(['',f'Cumulative model-forward time: {state["model_seconds"]:.1f} s. '
                  f'Cumulative SAE encoding wall time: {state["encoding_seconds"]:.1f} s. '
                  f'Free disk at validation: {shutil.disk_usage(run).free/1024**3:.2f} GiB.',
                  '', 'The complete per-family counts, flags, array shapes and checksums are in `manifest.json`.'])
    if state.get('issues'):
        lines.extend(['','Processing issues:']+[f'- {issue}' for issue in state['issues']])
    raw_complete=(raw_main_count==plan_data.get('estimated_new_text_forwards') and raw_trace_count==selection['selected'])
    if raw_complete:
        next_cap=int(state['encoding_seconds']//60)+30
        command=f'qd sae encode --profile {identity["profile"]} --run-dir "{run}" --resume --max-encoding-minutes {next_cap}'
    else:
        next_cap=int(state['model_seconds']//60)+30
        command=f'qd sae collect --profile {identity["profile"]} --run-dir "{run}" --resume --max-model-minutes {next_cap} --max-encoding-minutes {int(state["encoding_seconds"]//60)+30}'
    lines.extend(['',f'Resume command: `{command}`.',
                  'Provide enough checkpoint storage before resuming additional layers; the one-GiB download reserve remains enforced.'])
    path=run/'collection_report.md'; temporary=path.with_suffix('.tmp')
    temporary.write_text('\n'.join(lines)+'\n');os.replace(temporary,path)


def load_feature(run, layer, readout_id, kind='main'):
    """Load one 32,768-wide sparse vector and all matching experimental records."""
    from .sae_core import sparse_vector
    run=Path(run); index=read_json(run/'feature_index.json'); ordinal=index[kind][readout_id]
    path=_chunk_path(run,kind,layer,ordinal//index['chunk_rows'])
    if not path.exists(): raise KeyError(readout_id)
    if kind=='main':
        metadata=[r for r in _jsonl(run/'records.jsonl') if r['readout_id']==readout_id]
        trace_metadata=None
    else:
        trace_id,position_text=readout_id.rsplit(':',1)
        trace_metadata=next(t for t in _jsonl(run/'traces.jsonl') if t['trace_id']==trace_id)
        source_readout=next(r for r in _jsonl(run/'readouts.jsonl') if r['text_key']==trace_metadata['text_key'])
        position=int(position_text)
        trace_metadata=dict(trace_metadata,token_position=position,token_id=source_readout['token_ids'][position],
                            token_offset=source_readout['token_offsets'][position],text=source_readout['text'])
        metadata=[r for r in _jsonl(run/'records.jsonl') if r['kind']=='case' and r['metadata']['case_id']==trace_metadata['case_id']]
    with np.load(path,allow_pickle=False) as d:
        i=ordinal%index['chunk_rows']
        if str(d['keys'][i])!=readout_id: raise ValueError('Feature index/chunk mismatch')
        return {'vector':sparse_vector(d['ids'][i],d['values'][i]),'ids':d['ids'][i],
                'values':d['values'][i],'diagnostics':{k:d[k][i].item() for k in ['raw_norm','error_norm','relative_error','cosine','nnz','flags']},
                'records':metadata,'trace':trace_metadata,'checkpoint_sha256':str(d['checkpoint_sha256'])}
