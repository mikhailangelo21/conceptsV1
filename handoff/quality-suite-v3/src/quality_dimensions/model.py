from __future__ import annotations
import importlib.metadata
import inspect
import os
import platform
import random
import shutil
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from huggingface_hub import HfApi, try_to_load_from_cache, constants
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from .hooks import capture, finite
from .util import digest


class BudgetExpired(RuntimeError): pass


def environment():
    return dict(platform=platform.platform(),machine=platform.machine(),python=platform.python_version(),
                mps_available=torch.backends.mps.is_available(),cuda_available=torch.cuda.is_available(),
                physical_memory=psutil.virtual_memory().total,free_disk=shutil.disk_usage('.').free,
                versions={n:importlib.metadata.version(n) for n in ['torch','transformers','huggingface_hub','numpy','scipy','scikit-learn','pandas','matplotlib','PyYAML','psutil']})


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def backend_choice(request):
    backend=('mps' if torch.backends.mps.is_available() else 'cpu') if request=='auto' else request
    if backend not in ['cpu','mps','cuda']: raise ValueError('Choose auto, cpu, mps or cuda')
    if backend=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS requested but unavailable')
    if backend=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    return backend,torch.float32 if backend=='cpu' else torch.float16


def resolve(config, revision=None):
    info=HfApi().model_info(config['model'],revision=revision or config['revision'],files_metadata=True)
    sha=info.sha
    cfg=AutoConfig.from_pretrained(config['model'],revision=sha)
    if cfg.model_type!='qwen3': raise ValueError('Only dense Qwen3 is validated')
    if config['model']=='Qwen/Qwen3-1.7B-Base' and (cfg.num_hidden_layers,cfg.hidden_size)!=(28,2048): raise ValueError('Unexpected published 1.7B architecture')
    tok=AutoTokenizer.from_pretrained(config['model'],revision=sha)
    backend,dtype=backend_choice(config['backend'])
    metadata=dict(model=config['model'],model_revision=sha,tokenizer_revision=sha,backend=backend,
                  dtype=str(dtype),stored_dtype=str(getattr(cfg,'dtype','unknown')),
                  dtype_conversion=f'checkpoint -> {dtype}',attention=config['attention'],
                  num_hidden_layers=cfg.num_hidden_layers,hidden_size=cfg.hidden_size,
                  attention_config={k:getattr(cfg,k,None) for k in ['num_attention_heads','num_key_value_heads','head_dim','attention_bias','attention_dropout','rope_theta','max_position_embeddings','sliding_window']},
                  versions=environment()['versions'],hook='model.model.layers[i] output; i+1 block number; before final RMSNorm',
                  nondeterminism='Seeds fixed; no cross-device bitwise guarantee; native backend kernels may be nondeterministic',
                  config=cfg.to_dict(),fallback='CPU float32 compatibility fallback' if backend=='cpu' and config['backend']=='auto' else None)
    weights=[s for s in info.siblings if s.rfilename.endswith('.safetensors')]
    missing=[]
    for s in weights:
        cached=try_to_load_from_cache(config['model'],s.rfilename,revision=sha)
        if not isinstance(cached,str): missing.append((s.rfilename,s.size or 0))
    metadata['missing_weight_bytes']=sum(s for _,s in missing)
    metadata['weight_files']=[s.rfilename for s in weights]
    return cfg,tok,metadata


def disk_preflight(metadata,extra_bytes=0):
    location=Path(constants.HF_HUB_CACHE); location.mkdir(parents=True,exist_ok=True)
    available=shutil.disk_usage(location).free
    required=metadata['missing_weight_bytes']+extra_bytes+512*1024**2
    if available<required:
        raise RuntimeError(f'Insufficient disk: {available/2**30:.2f} GiB free in {location}; need at least {required/2**30:.2f} GiB for uncached public weights, activations and 0.5 GiB reserve. Free space or set HF_HOME to a larger volume, then resume. No model substitution performed.')


class Runner:
    def __init__(self,model,metadata,max_minutes=60):
        self.model=model.eval(); self.metadata=metadata; self.backend=metadata['backend']
        self.model.requires_grad_(False); self.blocks=model.model.layers
        if len(self.blocks)!=metadata['num_hidden_layers']: raise ValueError('Block count mismatch')
        self.limit=max_minutes*60; self.elapsed=0.; self.timings=[]; self.resources=[]

    @classmethod
    def load(cls,config,metadata,max_minutes):
        disk_preflight(metadata)
        model=AutoModelForCausalLM.from_pretrained(config['model'],revision=metadata['model_revision'],
              dtype=torch.float32 if metadata['backend']=='cpu' else torch.float16,
              attn_implementation=config['attention'])
        model.to(metadata['backend']); model.config.use_cache=False
        runner=cls(model,metadata,max_minutes)
        metadata['installed_block_forward']=inspect.getsource(type(model.model.layers[0]).forward)
        metadata['installed_model_forward']=inspect.getsource(type(model.model).forward)
        metadata['installed_lm_signature']=str(inspect.signature(model.forward))
        if 'logits_to_keep' not in inspect.signature(model.forward).parameters: raise RuntimeError('Installed LM lacks verified logits_to_keep API')
        metadata['implementation_hash']=digest([metadata['installed_block_forward'],metadata['installed_model_forward']])
        return runner

    def sync(self):
        if self.backend=='mps': torch.mps.synchronize()
        elif self.backend=='cuda': torch.cuda.synchronize()

    def resource(self):
        result={'rss_bytes':psutil.Process().memory_info().rss}
        if self.backend=='mps':
            result.update(mps_current_allocated_bytes=torch.mps.current_allocated_memory(),mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),mps_recommended_max_bytes=torch.mps.recommended_max_memory())
        return result

    def forward(self,row,logits=False):
        if self.elapsed>=self.limit: raise BudgetExpired('Model-execution budget exhausted; completed rows are saved')
        ids=torch.tensor([row['token_ids']],device=self.backend)
        self.sync(); start=time.monotonic()
        try:
            with torch.inference_mode():
                if logits:
                    out=self.model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0,-1].float()
                    finite(out,'logits'); result=out.detach().cpu().clone()
                else:
                    self.model.model(input_ids=ids,use_cache=False)
                    result=None
            self.sync()
            return result
        finally:
            duration=time.monotonic()-start; self.elapsed+=duration; self.timings.append(duration)
            self.resources.append(self.resource())

    def extract(self,row):
        with capture(self.blocks,row['readout_index']) as values: self.forward(row)
        if len(values)!=len(self.blocks): raise RuntimeError('Missing block activations')
        return np.stack([values[i+1].numpy() for i in range(len(self.blocks))])

    def extract_blocks(self, row, block_numbers):
        """Capture only named complete-block residuals at one token position."""
        blocks = [self.blocks[number - 1] for number in block_numbers]
        with capture(blocks, row['readout_index']) as values:
            self.forward(row)
        if len(values) != len(blocks):
            raise RuntimeError('Missing selected block activations')
        return np.stack([values[i + 1].numpy() for i in range(len(blocks))])

    def extract_readouts(self, rows, block_numbers):
        """Two selective positions in one batch-one text forward; independent snapshots."""
        from contextlib import ExitStack
        if not rows or any(r['token_ids'] != rows[0]['token_ids'] for r in rows):
            raise ValueError('Shared extraction requires identical prompt tokens')
        blocks = [self.blocks[number - 1] for number in block_numbers]
        with ExitStack() as stack:
            captures = [stack.enter_context(capture(blocks, r['readout_index'])) for r in rows]
            self.forward(rows[0])
        return [np.stack([values[i+1].numpy() for i in range(len(blocks))]) for values in captures]

    def score(self,row,encoding):
        logits=self.forward(row,logits=True)
        logprobs=torch.log_softmax(logits.float(),dim=-1)
        return {label:float(logprobs[token]) for label,token in encoding['token_ids'].items()}
