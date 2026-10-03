"""End-to-end RANDOM MODEL software QA only; never a research result."""
import re
from pathlib import Path
import numpy as np
import pytest
import torch
from quality_dimensions import pipeline
from quality_dimensions.model import Runner,BudgetExpired
from quality_dimensions.util import read_json,digest
from quality_dimensions.prompts import tokenize_prompt,answer_encoding


class TinyTokenizer:
    def encode(self,text,add_special_tokens=False):
        return [10 if t=='A' else 11 if t=='B' else 12+int(digest(t)[:8],16)%52 for t in re.findall(r'\w+|[^\w\s]',text)]


def test_token_boundary_and_overlength():
    tok=TinyTokenizer(); row=tokenize_prompt(tok,dict(prompt_id='x',text='Context: rabbit.\nAnswer:'),256)
    encoding=answer_encoding(tok,[row]); assert encoding['token_ids']=={'A':10,'B':11}
    assert row['readout_index']==len(row['token_ids'])-1
    with pytest.raises(ValueError,match='truncation prohibited'): tokenize_prompt(tok,dict(prompt_id='x',text='word '*300+'\nAnswer:'),256)
    class Broken(TinyTokenizer):
        def encode(self,text,add_special_tokens=False):
            result=super().encode(text,add_special_tokens)
            if text.endswith(('A','B')): result+= [22,23]
            return result
    with pytest.raises(ValueError,match='single-token'): answer_encoding(Broken(),[row])


def test_full_workflow_and_resume(tmp_path,monkeypatch,tiny):
    config=pipeline.configuration(Path(__file__).parents[1]/'configs'/'smoke.yaml')
    config.update(cache_root=str(tmp_path/'cache'),run_root=str(tmp_path/'runs'),permutations=3,bootstrap_replicates=25)
    def resolve(c,revision=None):
        metadata=dict(model='RANDOM TEST FIXTURE ONLY',model_revision='random-local-fixture',tokenizer_revision='random-local-fixture',backend='cpu',dtype='torch.float32',stored_dtype='random initialization',attention='eager',num_hidden_layers=3,hidden_size=24,versions={'test':'fixture'},hook='complete block',missing_weight_bytes=0)
        return tiny.config,TinyTokenizer(),metadata
    monkeypatch.setattr(pipeline,'resolve',resolve); monkeypatch.setattr(pipeline,'disk_preflight',lambda *a:None)
    monkeypatch.setattr(Runner,'load',classmethod(lambda cls,c,m,b:Runner(tiny,m,b)))
    first=pipeline.execute(config,run_dir=tmp_path/'run',max_minutes=.00001)
    assert read_json(first/'state.json')['status']=='partial'
    assert 'budget' in read_json(first/'state.json')['reason']
    run=pipeline.execute(config,run_dir=first,resume=True,max_minutes=10)
    state=read_json(run/'state.json'); assert state['status']=='complete',state
    assert state['extraction_completed']==260
    assert state['behavior_rows_completed']==160
    assert len(list((run/'plots').glob('*.png')))==5
    assert len(list((run/'plots').glob('*.svg')))==5
    before={p.name:p.read_bytes() for p in (run/'behavior_shards').glob('*.json')}
    pipeline.execute(config,run_dir=run,resume=True,max_minutes=1)
    assert before=={p.name:p.read_bytes() for p in (run/'behavior_shards').glob('*.json')}
    changed=dict(config,seed=123)
    with pytest.raises(ValueError,match='identity mismatch'): pipeline.prepare(changed,run_dir=run,resume=True)
    print('RANDOM TEST FIXTURE plot QA path:',run)


def test_tiny_mps_float16(tiny):
    if not torch.backends.mps.is_available(): pytest.skip('MPS unavailable on actual host')
    from quality_dimensions.hooks import capture,intervene
    tiny=tiny.to(device='mps',dtype=torch.float16)
    runner=Runner(tiny,dict(backend='mps',num_hidden_layers=3),1)
    row=dict(token_ids=[1,2,3],readout_index=2)
    a=runner.extract(row); assert a.dtype==np.float32 and np.isfinite(a).all()
    baseline=runner.score(row,{'token_ids':{'A':10,'B':11}})
    with intervene(tiny.model.layers[1],2,np.eye(24)[0],.2,0): zero=runner.score(row,{'token_ids':{'A':10,'B':11}})
    assert max(abs(zero[k]-baseline[k]) for k in zero)<2e-3
    with intervene(tiny.model.layers[1],2,np.eye(24)[0],.2,1): changed=runner.score(row,{'token_ids':{'A':10,'B':11}})
    assert any(changed[k]!=baseline[k] for k in changed)
    assert runner.resource()['mps_current_allocated_bytes']>0
