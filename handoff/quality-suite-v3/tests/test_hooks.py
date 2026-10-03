import numpy as np
import pytest
import torch
from quality_dimensions.hooks import capture,intervene,output_tensor,replace_tensor
from quality_dimensions.model import Runner
from quality_dimensions.interventions import random_directions


def test_block_capture_final_norm_and_storage(tiny):
    ids=torch.tensor([[1,2,3,4,5]])
    norm_input=[]; handle=tiny.model.norm.register_forward_pre_hook(lambda _m,args:norm_input.append(args[0].detach().clone()))
    with torch.inference_mode(),capture(tiny.model.layers,-1) as values:
        output=tiny.model(ids,use_cache=False).last_hidden_state
    handle.remove()
    assert list(values)==[1,2,3]
    for v in values.values():
        assert v.shape==(24,) and v.dtype==torch.float32
        assert v.untyped_storage().nbytes()==v.numel()*v.element_size()
    assert torch.equal(values[3],norm_input[0][0,-1])
    assert not torch.allclose(values[3],output[0,-1])
    assert all(not block._forward_hooks for block in tiny.model.layers)


def test_cleanup_on_exception(tiny):
    with pytest.raises(RuntimeError),capture(tiny.model.layers,-1): raise RuntimeError('deliberate')
    with pytest.raises(RuntimeError),intervene(tiny.model.layers[0],-1,np.ones(24),1,1): raise RuntimeError('deliberate')
    assert all(not block._forward_hooks for block in tiny.model.layers)


def test_causal_prefix(tiny):
    with torch.inference_mode(),capture(tiny.model.layers,1) as a:
        tiny.model(torch.tensor([[1,2,3,4]]),use_cache=False)
    with torch.inference_mode(),capture(tiny.model.layers,1) as b:
        tiny.model(torch.tensor([[1,2,9,10]]),use_cache=False)
    for layer in a: torch.testing.assert_close(a[layer],b[layer],atol=1e-6,rtol=0)


def test_observation_zero_and_nonzero_downstream(tiny):
    ids=torch.tensor([[1,2,3,4]]); u=random_directions(24,1,1)[0]
    before={k:v.clone() for k,v in tiny.state_dict().items()}
    with torch.inference_mode():
        baseline=tiny(ids,use_cache=False).logits
        with capture(tiny.model.layers,-1): observed=tiny(ids,use_cache=False).logits
        with intervene(tiny.model.layers[1],-1,u,.3,0): zero=tiny(ids,use_cache=False).logits
        seen=[]
        pre=tiny.model.layers[1].register_forward_hook(lambda m,a,o:seen.append(output_tensor(o).clone()))
        with intervene(tiny.model.layers[1],-1,u,.3,1) as stats:
            post=tiny.model.layers[1].register_forward_hook(lambda m,a,o:seen.append(output_tensor(o).clone()))
            changed=tiny(ids,use_cache=False).logits
            post.remove()
        pre.remove()
    torch.testing.assert_close(observed,baseline,atol=1e-6,rtol=0)
    torch.testing.assert_close(zero,baseline,atol=1e-6,rtol=0)
    assert torch.equal(seen[0][:,:-1],seen[1][:,:-1])
    torch.testing.assert_close(seen[1][0,-1]-seen[0][0,-1],torch.tensor(u*.3,dtype=torch.float32),atol=1e-7,rtol=0)
    assert not torch.allclose(changed[:,-1],baseline[:,-1],atol=1e-5,rtol=0)
    assert stats['calls']==1
    for key,value in tiny.state_dict().items(): assert torch.equal(before[key],value)


def test_structured_output_and_equal_norms(tiny):
    value=torch.randn(1,4,24); sentinel=object()
    assert replace_tensor((value,sentinel),value.clone())[1] is sentinel
    assert isinstance(replace_tensor([value,sentinel],value.clone()),list)
    norms=[]
    for u in random_directions(24,4,11):
        with torch.inference_mode(),intervene(tiny.model.layers[0],-1,u,2,-.5) as stats:
            tiny(torch.tensor([[1,2,3]]),use_cache=False)
        norms.append(stats['actual_delta_norm'])
    np.testing.assert_allclose(norms,1,atol=2e-7)


def test_single_token_score_matches_direct_logits(tiny):
    runner=Runner(tiny,dict(backend='cpu',num_hidden_layers=3),1)
    row=dict(token_ids=[1,2,3],readout_index=2)
    actual=runner.score(row,{'token_ids':{'A':10,'B':11}})
    with torch.inference_mode(): direct=tiny(torch.tensor([[1,2,3]]),use_cache=False).logits[0,-1].float().log_softmax(-1)
    assert actual['A']==pytest.approx(direct[10].item(),abs=1e-6)
    assert actual['B']==pytest.approx(direct[11].item(),abs=1e-6)
