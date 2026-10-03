"""Selective-layer activation cache for quality-suite v3."""
from pathlib import Path

import numpy as np

from .util import atomic_json, atomic_npz, digest, read_json


class SuiteActivationCache:
    def __init__(self, root, identity):
        self.identity = identity
        self.key = digest(identity)
        self.root = Path(root) / "quality_suite_v3" / self.key
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / "identity.json"
        if manifest.exists() and read_json(manifest) != identity:
            raise ValueError("Incompatible suite cache identity")
        if not manifest.exists():
            atomic_json(manifest, identity)

    def row_key(self, row):
        return digest({
            "tokens": row["token_ids"], "position": row["readout_index"],
            "hook": "complete_block_output_pre_final_rmsnorm",
            "blocks": self.identity["block_numbers"],
        })

    def path(self, row):
        return self.root / f"{self.row_key(row)}.npz"

    def get(self, row):
        path = self.path(row)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            if str(data["fingerprint"]) != self.key or str(data["row_key"]) != self.row_key(row):
                raise ValueError("Incompatible suite activation shard")
            value = data["activation"].copy()
        shape = (len(self.identity["block_numbers"]), self.identity["hidden_size"])
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Corrupt suite activation shard")
        return value

    def put(self, row, activation):
        value = np.asarray(activation, dtype=np.float32)
        shape = (len(self.identity["block_numbers"]), self.identity["hidden_size"])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("Invalid suite activation")
        atomic_npz(self.path(row), activation=value, fingerprint=np.array(self.key),
                   row_key=np.array(self.row_key(row)))
