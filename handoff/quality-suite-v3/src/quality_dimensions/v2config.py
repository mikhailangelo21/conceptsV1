from pathlib import Path
import yaml


def load_config(path):
    path=Path(path).resolve();c=yaml.safe_load(path.read_text());c['config_path']=str(path)
    for k in ['source_run','run_root','benchmark_pairs']:
        p=Path(c[k]);c[k]=str(p if p.is_absolute() else path.parent.parent/p)
    if c['model']!='Qwen/Qwen3-1.7B-Base': raise ValueError('Refinement keeps the original checkpoint')
    if c['backend'] not in ['mps','cpu']: raise ValueError('Use tested MPS or explicitly separate CPU check')
    if c['max_tokens']>256: raise ValueError('Maximum 256 tokens')
    if c['strengths']!=[-1,-.5,0,.5,1]: raise ValueError('Refinement retains the original prespecified doses')
    return c
