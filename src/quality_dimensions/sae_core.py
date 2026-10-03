"""Qwen-Scope residual SAE encoding; feature coordinates are checkpoint/layer local."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import torch

WIDTH = 32768
HIDDEN = 2048


def sha256_file(path, block=4 * 1024 * 1024):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for piece in iter(lambda: f.read(block), b''):
            h.update(piece)
    return h.hexdigest()


def sae_index(block_number):
    if not isinstance(block_number, int) or not 1 <= block_number <= 28:
        raise ValueError('V3 block numbers must be 1..28')
    return block_number - 1


def validate_weights(weights, hidden=HIDDEN, width=WIDTH):
    required = {'W_enc': (width, hidden), 'W_dec': (hidden, width),
                'b_enc': (width,), 'b_dec': (hidden,)}
    for name, shape in required.items():
        if name not in weights or tuple(weights[name].shape) != shape:
            raise ValueError(f'Invalid SAE {name} shape; expected {shape}')
        if not torch.isfinite(weights[name]).all():
            raise ValueError(f'Nonfinite SAE {name}')
    return required


def encode_decode(hidden, weights, k=100, device='cpu', hidden_size=HIDDEN, width=WIDTH):
    """Full-demo convention: TopK(ReLU(h @ W_enc.T + b_enc), K)."""
    validate_weights(weights, hidden_size, width)
    h = torch.as_tensor(np.asarray(hidden), device=device, dtype=torch.float32)
    if h.ndim != 2 or h.shape[1] != hidden_size or not torch.isfinite(h).all():
        raise ValueError('Invalid raw activation batch')
    if not 1 <= k <= width:
        raise ValueError('Invalid K')
    w = {name: value.to(device=device, dtype=torch.float32) for name, value in weights.items() if name in {'W_enc','W_dec','b_enc','b_dec'}}
    with torch.inference_mode():
        pre = h @ w['W_enc'].T + w['b_enc']
        values, ids = torch.topk(torch.relu(pre), k, dim=1)
        reconstruction = torch.sum(w['W_dec'].T[ids] * values.unsqueeze(-1), dim=1) + w['b_dec']
        err = (h - reconstruction).norm(dim=1)
        norm = h.norm(dim=1)
        rec_norm = reconstruction.norm(dim=1)
        cosine = (h * reconstruction).sum(dim=1) / (norm * rec_norm).clamp_min(1e-30)
        relative = torch.where(norm > 0, err / norm.clamp_min(1e-30), torch.full_like(err, torch.nan))
        flags = ((norm == 0).to(torch.int16) | ((rec_norm == 0).to(torch.int16) << 1) |
                 ((~torch.isfinite(err) | ~torch.isfinite(relative) | ~torch.isfinite(cosine)).to(torch.int16) << 2))
    diagnostics = {'raw_norm': norm.cpu().numpy().astype('float32'),
                   'error_norm': err.cpu().numpy().astype('float32'),
                   'relative_error': relative.cpu().numpy().astype('float32'),
                   'cosine': cosine.cpu().numpy().astype('float32'),
                   'nnz': (values > 0).sum(dim=1).cpu().numpy().astype('int16'),
                   'flags': flags.cpu().numpy().astype('int16')}
    return ids.cpu().numpy().astype('int32'), values.cpu().numpy().astype('float32'), diagnostics


def sparse_vector(ids, values, width=WIDTH):
    ids = np.asarray(ids, dtype=np.int32); values = np.asarray(values, dtype=np.float32)
    if ids.ndim != 1 or ids.shape != values.shape or np.any(ids < 0) or np.any(ids >= width) or len(set(ids.tolist())) != len(ids):
        raise ValueError('Invalid sparse row')
    out = np.zeros(width, dtype=np.float32)
    out[ids] = values
    return out


def load_checkpoint(path, expected_sha256=None, hidden=HIDDEN, width=WIDTH):
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise ValueError('SAE checkpoint SHA-256 mismatch')
    weights = torch.load(path, map_location='cpu', weights_only=True)
    if 'state_dict' in weights: weights = weights['state_dict']
    validate_weights(weights, hidden, width)
    return weights
