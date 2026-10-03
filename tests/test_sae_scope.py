import json
from pathlib import Path

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from quality_dimensions.sae_core import encode_decode, sparse_vector, sae_index, validate_weights
from quality_dimensions.sae_collection import _select_trace, _write_chunk, _check_chunk, _write_jsonl, _jsonl, _capture_layers, _legacy, _source_v3_run
from quality_dimensions.hooks import capture
from quality_dimensions.suite_cache import SuiteActivationCache


def toy_weights():
    return {'W_enc':torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]),
            'W_dec':torch.tensor([[1.,-1.,0.,0.],[0.,0.,1.,-1.]]),
            'b_enc':torch.zeros(4),'b_dec':torch.zeros(2)}


def test_topk_relu_and_reconstruction():
    w=toy_weights(); validate_weights(w,2,4)
    ids, values, diag=encode_decode(np.array([[2.,-3.],[0.,0.]],np.float32),w,k=2,hidden_size=2,width=4)
    assert set(ids[0])=={0,3}
    np.testing.assert_allclose(sparse_vector(ids[0],values[0],4),[2,0,0,3])
    assert diag['error_norm'][0]==0
    assert diag['relative_error'][0]==0
    assert diag['cosine'][0]==pytest.approx(1)
    assert diag['nnz'][0]==2
    assert np.isnan(diag['relative_error'][1])
    assert diag['flags'][1]&1
    negative=toy_weights();negative['W_enc']=torch.ones(4,2)
    _,zero_values,_=encode_decode(np.array([[-1.,0.]],np.float32),negative,k=2,hidden_size=2,width=4)
    np.testing.assert_array_equal(zero_values,[[0.,0.]])


def test_shape_and_hook_numbering():
    assert sae_index(1)==0 and sae_index(28)==27
    with pytest.raises(ValueError): sae_index(0)
    w=toy_weights(); w['W_dec']=w['W_dec'][:1]
    with pytest.raises(ValueError): validate_weights(w,2,4)
    blocks=torch.nn.ModuleList([torch.nn.Identity() for _ in range(28)])
    x=torch.randn(1,4,2)
    with capture([blocks[3],blocks[27]],2) as found:
        blocks[3](x+3); blocks[27](x+27)
    np.testing.assert_allclose(found[1].numpy(),(x+3)[0,2].numpy())
    np.testing.assert_allclose(found[2].numpy(),(x+27)[0,2].numpy())
    runner=SimpleNamespace(blocks=blocks)
    with _capture_layers(runner,[3],[1,3]) as selected:
        blocks[3](x+3)
    np.testing.assert_allclose(selected[3],(x+3)[0,[1,3]].numpy())


def test_chunk_resume_identity_and_sparse_round_trip(tmp_path):
    run=tmp_path; (run/'sae_repository.json').write_text(json.dumps({'k':2}))
    identity={'version':'sae_scope_v1'}; ck={'sha256':'a'*64}
    path=run/'features/main/layer3/chunk000000.npz'
    arrays={'ids':np.array([[0,3]],np.int32),'values':np.array([[2.,3.]],np.float32),
            'raw_norm':np.array([1.],np.float32),'error_norm':np.array([0.],np.float32),
            'relative_error':np.array([0.],np.float32),'cosine':np.array([1.],np.float32),
            'nnz':np.array([2],np.int16),'flags':np.array([0],np.int16)}
    _write_chunk(path,identity,3,ck,['x'],arrays)
    assert _check_chunk(path,identity,3,ck,['x'])
    with pytest.raises(ValueError): _check_chunk(path,identity,4,ck,['x'])
    with np.load(path,allow_pickle=False) as d:
        np.testing.assert_allclose(sparse_vector(d['ids'][0],d['values'][0],4),[2,0,0,3])


def test_metadata_selection_is_deterministic(tmp_path):
    cases=[{'case_id':str(i),'text':str(i),'case_type':'binding' if i%2 else 'lexical','quality_ids':['size'],
            'template_id':'plain','entity_split':'train','template_split':'train','combination_split':'train'} for i in range(20)]
    a=_select_trace(cases,8)
    assert a==_select_trace(list(reversed(cases)),8)
    assert {r['case_type'] for r in a}=={'binding','lexical'}
    assert sum(r['case_type']=='binding' for r in a)==4
    path=tmp_path/'records.jsonl';_write_jsonl(path,[{'readout_id':'r','case_id':'a'},{'readout_id':'r','case_id':'b'}])
    assert len(list(_jsonl(path)))==2


def test_copied_collection_resolves_v3_cache_and_run_from_overrides(tmp_path, monkeypatch):
    identity={'block_numbers':[4], 'hidden_size':2}
    cache_base=tmp_path/'copied_cache'
    cache=SuiteActivationCache(cache_base, identity)
    v3=tmp_path/'copied_v3'
    v3.mkdir()
    collection=tmp_path/'collection'
    collection.mkdir()
    (collection/'source_v3.json').write_text(json.dumps({
        'run':'/original_mac/runs/copied_v3',
        'activation_cache':{'root':'/original_mac/cache/quality_suite_v3/missing',
                            'identity':identity}}))
    monkeypatch.setenv('QD_V3_CACHE_BASE',str(cache_base))
    monkeypatch.setenv('QD_V3_RUN_DIR',str(v3))
    assert _legacy(collection).root == cache.root
    assert _source_v3_run(collection) == v3
    monkeypatch.setenv('QD_V3_CACHE_BASE',str(tmp_path/'wrong_cache'))
    with pytest.raises(FileNotFoundError, match='QD_V3_CACHE_BASE'):
        _legacy(collection)
