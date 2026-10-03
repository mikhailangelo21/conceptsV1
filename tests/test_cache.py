import numpy as np
import pytest
from quality_dimensions.cache import ActivationCache
from quality_dimensions.util import atomic_json


def test_atomic_resume_and_mismatch(tmp_path):
    identity=dict(num_hidden_layers=2,hidden_size=3,revision='abc'); cache=ActivationCache(tmp_path,identity)
    rows=[dict(token_ids=[1,i+2],readout_index=1) for i in range(3)]
    cache.put(rows[0],np.ones((2,3),np.float32))
    # Uncommitted work is not a completion record.
    (cache.root/'interrupted.tmp').write_bytes(b'partial')
    resumed=ActivationCache(tmp_path,identity)
    assert resumed.get(rows[0]) is not None and resumed.get(rows[1]) is None
    for row in rows:
        if resumed.get(row) is None: resumed.put(row,np.zeros((2,3),np.float32))
    assert len(list(cache.root.glob('*.npz')))==3
    altered=ActivationCache(tmp_path,dict(identity,revision='other')); assert altered.get(rows[0]) is None
    atomic_json(cache.root/'identity.json',dict(identity,revision='corrupt'))
    with pytest.raises(ValueError,match='Incompatible'): ActivationCache(tmp_path,identity)
