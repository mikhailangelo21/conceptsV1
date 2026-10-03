"""Exact conditional continuations and targeted output-row float32 precision."""
from contextlib import nullcontext
import time
import numpy as np
import torch
from .hooks import finite
from .model import BudgetExpired


def encode_choices(tokenizer,row,max_tokens=256):
    row=dict(row);encoded=tokenizer(row['text'],add_special_tokens=False,return_offsets_mapping=True)
    ids=encoded['input_ids'];offsets=encoded['offset_mapping']
    if len(ids)>max_tokens: raise ValueError('Overlength prefix; truncation is forbidden')
    row.update(token_ids=ids,token_count=len(ids),readout_index=len(ids)-1,token_offsets=offsets)
    if 'entity_span' in row:
        start,end=row['entity_span']
        selected=[i for i,(a,b) in enumerate(offsets) if a<end and b>start]
        if not selected or not row['text'][start:end]==row['subject']: raise ValueError('Entity span/token alignment mismatch')
        row['entity_index']=selected[-1];row['entity_token_id']=ids[selected[-1]]
        row['entity_after_context']=start>=row['context_end']
        if not row['entity_after_context']: raise ValueError('Entity readout precedes relevant context')
    if 'choices' in row:
        candidates={}
        for label,answer in row['choices'].items():
            full=tokenizer.encode(row['text']+answer,add_special_tokens=False)
            if full[:len(ids)]!=ids or len(full)==len(ids): raise ValueError('Continuation changes prefix tokenization or is empty')
            tail=full[len(ids):]
            if len(ids)+len(tail)-1>max_tokens: raise ValueError('Teacher-forced continuation exceeds token limit')
            candidates[label]=tail
        if len(candidates)!=2: raise ValueError('Exactly two candidate continuations required')
        row['candidate_ids']=candidates
    return row


def _sequence_forward(runner,ids,positions):
    if runner.elapsed>=runner.limit: raise BudgetExpired('Model-execution budget exhausted')
    runner.sync();start=time.monotonic()
    try:
        with torch.inference_mode():
            x=torch.tensor([ids],device=runner.backend)
            keep=torch.tensor(positions,device=runner.backend)
            logits=runner.model(input_ids=x,use_cache=False,logits_to_keep=keep).logits[0].float()
            finite(logits,'teacher-forced logits'); result=logits.detach().cpu().clone()
        runner.sync();return result
    finally:
        runner.elapsed+=time.monotonic()-start;runner.timings.append(time.monotonic()-start);runner.resources.append(runner.resource())


def score_choices(runner,row,intervention_factory=None):
    """Multi-token scores sum correctly shifted full-vocabulary log probabilities.

    Each candidate teacher-forced forward applies an intervention once, at the
    fixed PREFIX position, never at answer tokens. For one-token candidates a
    shared forward also captures the final normalized hidden vector and compares
    the two selected head rows in float32 on CPU. No full module is recast.
    """
    candidates=row['candidate_ids']; labels=list(candidates); lengths=[len(candidates[k]) for k in labels]
    stats=[];native={};single=max(lengths)==1
    if single:
        hidden=[]
        def capture_norm(_m,_a,out):
            hidden.append(out[0,-1].detach().to('cpu',torch.float32).clone())
        handle=runner.model.model.norm.register_forward_hook(capture_norm)
        try:
            with intervention_factory() if intervention_factory else nullcontext({}) as stat:
                logits=runner.forward(row,logits=True)
            stats.append(dict(stat))
        finally:handle.remove()
        ids=[candidates[k][0] for k in labels];lp=torch.log_softmax(logits.float(),dim=-1)
        native={k:float(lp[t]) for k,t in zip(labels,ids)}
        head=runner.model.lm_head
        with torch.inference_mode():
            # Slice BEFORE copying/casting, never convert a tied module in place.
            w1=head.weight[ids[0]].detach().to('cpu',torch.float32).clone()
            w2=head.weight[ids[1]].detach().to('cpu',torch.float32).clone()
            precise=float(torch.dot(w1-w2,hidden[0]))
            if getattr(head,'bias',None) is not None:
                precise+=float(head.bias[ids[0]].float().cpu()-head.bias[ids[1]].float().cpu())
        native_logits=[float(logits[t]) for t in ids]
        identity_error=abs((native[labels[0]]-native[labels[1]])-(native_logits[0]-native_logits[1]))
    else:
        for label in labels:
            tail=candidates[label];n=len(row['token_ids']);ids=row['token_ids']+tail[:-1]
            with intervention_factory() if intervention_factory else nullcontext({}) as stat:
                logits=_sequence_forward(runner,ids,list(range(n-1,n+len(tail)-1)))
            stats.append(dict(stat))
            lp=torch.log_softmax(logits,dim=-1)
            native[label]=float(lp[torch.arange(len(tail)),torch.tensor(tail)].sum())
        precise=None;native_logits=None;identity_error=None
    margin=native[labels[0]]-native[labels[1]]
    return dict(log_probs=native,native_margin=margin,precise_margin=precise,native_candidate_logits=native_logits,
                logprob_logit_identity_error=identity_error,single_token=single,stats=stats,
                precision_note='Float32 selected output-row difference; preceding network remains in its declared dtype.' if single else 'Exact teacher-forced multi-token scores; no selective-head precision approximation.')
