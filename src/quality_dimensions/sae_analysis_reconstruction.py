"""Partitioned PCA reconstruction diagnostics using frozen train-fitted reducers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .sae_analysis import _raw_matrix, _save_rows
from .sae_analysis_core import checksum, feature_matrix, read_collection
from .util import atomic_json


def _baseline_ss(x, mean):
    squared = float(x.power(2).sum()) if sparse.issparse(x) else float(np.sum(x ** 2))
    cross = float(np.sum(x @ mean))
    return max(squared - 2 * cross + x.shape[0] * float(mean @ mean), 0.)


def reconstruction_check(run_dir):
    run = Path(run_dir).expanduser().resolve()
    config = json.loads((run / "config.json").read_text())
    source = (Path(__file__).resolve().parents[2] / config["source_run"]).resolve()
    collection = read_collection(source)
    selection = json.loads((run / "sample_ids.json").read_text())
    layers = json.loads((run / "identity.json").read_text())["selected_layers"]
    output = []
    for layer in layers:
        keys, all_f, _ = feature_matrix(collection, layer)
        lookup = {rid: i for i, rid in enumerate(keys)}
        for readout in ("entity", "final"):
            ids = [rid for split in ("train", "validation", "test", "challenge")
                   for rid in selection[readout][split] if rid in lookup]
            matrices = {"F": all_f[[lookup[rid] for rid in ids]],
                        "H": _raw_matrix(collection, layer, ids)}
            for view, matrix in matrices.items():
                path = run / "linear" / f"layer{layer}_{readout}_{view}_pca.npz"
                with np.load(path) as d:
                    if d["ids"].tolist() != ids:
                        raise ValueError("PCA coordinates and source matrices have unmatched readouts")
                    mean = d["mean"].astype(np.float64)
                    scores = d["coordinates"].astype(np.float64)
                    rank = scores.shape[1]
                id_to_row = {rid: i for i, rid in enumerate(ids)}
                for split in ("train", "validation", "test", "challenge"):
                    idx = [id_to_row[rid] for rid in selection[readout][split] if rid in id_to_row]
                    if not idx:
                        continue
                    baseline = _baseline_ss(matrix[idx], mean)
                    residual = max(baseline - float(np.sum(scores[idx] ** 2)), 0.)
                    output.append({"layer": layer, "readout": readout, "view": view,
                                   "partition": split, "readout_vectors": len(idx),
                                   "fit_vectors": len(selection[readout]["train"]),
                                   "rank": rank, "relative_rmse": float(np.sqrt(residual / max(baseline, 1e-30))),
                                   "preprocessing": "train-only centering, raw activation scale"})
        _save_rows(run / "linear" / "partitioned_reconstruction.csv", output)
        print(f"PCA reconstruction layer {layer}", flush=True)
    atomic_json(run / "linear" / "partitioned_reconstruction_provenance.json",
                {"source_sha256": checksum(Path(__file__)), "fit": "frozen main PCA files",
                 "source_collection_manifest_sha256": checksum(source / "manifest.json")})
    return run / "linear" / "partitioned_reconstruction.csv"
