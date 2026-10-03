"""Direct complete-block output hooks; block numbers in the public API are 1-based."""
from contextlib import contextmanager
import numpy as np
import torch


def output_tensor(output):
    if isinstance(output,torch.Tensor): return output
    if isinstance(output,(tuple,list)) and output and isinstance(output[0],torch.Tensor): return output[0]
    raise TypeError(f'Unsupported decoder output structure: {type(output)}')


def replace_tensor(output,tensor):
    if isinstance(output,torch.Tensor): return tensor
    if isinstance(output,tuple): return (tensor,)+output[1:]
    if isinstance(output,list): return [tensor]+output[1:]
    raise TypeError(type(output))


def finite(tensor,what):
    if not torch.isfinite(tensor).all().item(): raise FloatingPointError(f'Nonfinite {what}')


@contextmanager
def capture(blocks,position):
    values={}; handles=[]
    def make_hook(i):
        def hook(_module,_args,output):
            h=output_tensor(output)
            if h.ndim!=3 or h.shape[0]!=1: raise ValueError('Expected batch size 1 residual tensor')
            value=h[0,position,:].detach().to(device='cpu',dtype=torch.float32).clone()
            finite(value,'activation')
            values[i+1]=value
        return hook
    try:
        for i,block in enumerate(blocks): handles.append(block.register_forward_hook(make_hook(i)))
        yield values
    finally:
        for handle in handles: handle.remove()


@contextmanager
def intervene(block,position,direction,scale,strength):
    stats={'calls':0}
    def hook(_module,_args,output):
        stats['calls']+=1
        if stats['calls']!=1: raise RuntimeError('Intervention fired more than once')
        h=output_tensor(output); original=h[0,position,:]
        delta=torch.as_tensor(np.array(direction,copy=True),device=h.device,dtype=h.dtype)*(float(scale)*float(strength))
        changed=h.clone(); changed[0,position,:]=original+delta
        finite(changed[0,position,:],'intervened activation')
        norm=float(original.float().norm().item())
        stats.update(activation_norm=norm,requested_norm=abs(float(scale)*float(strength)),
                     actual_delta_norm=float((changed[0,position,:]-original).float().norm().item()),
                     unchanged_coordinate_fraction=float((changed[0,position,:]==original).float().mean().item()),
                     norm_ratio=abs(float(scale)*float(strength))/norm if norm else None)
        return replace_tensor(output,changed)
    handle=block.register_forward_hook(hook)
    try:
        yield stats
        if stats['calls']!=1: raise RuntimeError('Intervention did not fire')
    finally: handle.remove()


@contextmanager
def patch(block,position,donor,recipient_reference=None,direction=None,random_direction=None):
    """Full-vector, component-only, or matched-norm random donor patch.

    Reference vectors are immutable CPU snapshots of the aligned readouts.
    For random controls the norm equals the FULL/component displacement supplied
    by the same recipient/donor pair, never a separate variance scale.
    """
    stats={'calls':0}
    def hook(_module,_args,output):
        stats['calls']+=1
        if stats['calls']!=1: raise RuntimeError('Patch fired more than once')
        h=output_tensor(output);original=h[0,position,:]
        donor32=torch.as_tensor(np.array(donor,copy=True),device=h.device,dtype=torch.float32)
        delta=donor32-original.float()
        if recipient_reference is not None:
            expected=torch.as_tensor(np.array(recipient_reference,copy=True),device=h.device,dtype=h.dtype)
            if not torch.allclose(original,expected,atol=2e-3 if h.dtype==torch.float16 else 1e-5,rtol=0): raise ValueError('Recipient activation differs from saved patch reference')
        if direction is not None:
            u=torch.as_tensor(np.array(direction,copy=True),device=h.device,dtype=torch.float32)
            delta=u*torch.dot(u,delta)
        if random_direction is not None:
            random=torch.as_tensor(np.array(random_direction,copy=True),device=h.device,dtype=torch.float32)
            delta=random*delta.norm()
        changed=h.clone();changed[0,position]=(original.float()+delta).to(h.dtype)
        applied=changed[0,position].float()-original.float();finite(applied,'patch delta')
        stats.update(requested_norm=float(delta.norm()),actual_delta_norm=float(applied.norm()),activation_norm=float(original.float().norm()),unchanged_coordinate_fraction=float((changed[0,position]==original).float().mean()))
        return replace_tensor(output,changed)
    handle=block.register_forward_hook(hook)
    try:
        yield stats
        if stats['calls']!=1:raise RuntimeError('Patch did not fire')
    finally:handle.remove()


def validate_patch_alignment(recipient,donor,site,layer,num_layers):
    if recipient['concept_id']!=donor['concept_id'] or recipient['subject']!=donor['subject']:raise ValueError('Patch subject identity mismatch')
    for key in ['family','task','clause_order','query_role']:
        if recipient[key]!=donor[key]:raise ValueError(f'Patch alignment mismatch: {key}')
    index=recipient['entity_index'] if site=='entity' else recipient['readout_index']
    donor_index=donor['entity_index'] if site=='entity' else donor['readout_index']
    if site=='entity' and recipient['entity_token_id']!=donor['entity_token_id']:raise ValueError('Patch entity token mismatch')
    if site=='entity' and index<recipient['readout_index'] and layer==num_layers:raise ValueError('Final-block earlier-token patch has no remaining attention path to answer')
    return dict(recipient_index=index,donor_index=donor_index,token_length_difference=donor['token_count']-recipient['token_count'],readout=site)
