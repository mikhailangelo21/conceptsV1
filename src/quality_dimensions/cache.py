from pathlib import Path
import numpy as np
from .util import atomic_json, atomic_npz, read_json, digest


class ActivationCache:
    """One bounded atomic shard per prompt, with self-describing fingerprint.
    A shard is the completion record; no separate counter can outrun its write.
    """
    def __init__(self,root,identity):
        self.identity=identity; self.key=digest(identity); self.root=Path(root)/self.key
        self.root.mkdir(parents=True,exist_ok=True)
        manifest=self.root/'identity.json'
        if manifest.exists() and read_json(manifest)!=identity: raise ValueError('Incompatible cache identity')
        if not manifest.exists(): atomic_json(manifest,identity)

    def row_key(self,row):
        return digest(dict(tokens=row['token_ids'],position=row['readout_index'],hook='complete_block_pre_final_norm'))

    def path(self,row): return self.root/(self.row_key(row)+'.npz')

    def get(self,row):
        path=self.path(row)
        if not path.exists(): return None
        with np.load(path,allow_pickle=False) as data:
            if str(data['fingerprint'])!=self.key or str(data['row_key'])!=self.row_key(row): raise ValueError('Incompatible activation shard')
            a=data['activation'].copy()
        expected=(self.identity['num_hidden_layers'],self.identity['hidden_size'])
        if a.shape!=expected or a.dtype!=np.float32 or not np.isfinite(a).all(): raise ValueError('Corrupt activation shard')
        return a

    def put(self,row,activation):
        a=np.asarray(activation,dtype=np.float32)
        if a.shape!=(self.identity['num_hidden_layers'],self.identity['hidden_size']) or not np.isfinite(a).all(): raise ValueError('Invalid activation shape/data')
        atomic_npz(self.path(row),activation=a,fingerprint=np.array(self.key),row_key=np.array(self.row_key(row)))
