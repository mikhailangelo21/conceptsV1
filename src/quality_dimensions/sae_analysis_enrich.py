"""Optional post-core diagnostics, saved separately so the original analysis run is preserved."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from .sae_analysis import _predict_one, _qualities, _raw_matrix, _save_rows, _scores
from .sae_analysis_core import checksum, centered_sparse_pca, feature_matrix, read_collection
from .sae_core import load_checkpoint
from .util import atomic_json


def _source(run):
    config = json.loads((run / "config.json").read_text())
    project = Path(__file__).resolve().parents[2]
    return read_collection((project / config["source_run"]).resolve()), config


def _rank_predictions(run, collection):
    out = []
    for layer in (1, 11, 27):
        for readout in ("entity", "final"):
            for view in ("H", "F"):
                path = run / "linear" / f"layer{layer}_{readout}_{view}_pca.npz"
                if not path.exists():
                    continue
                with np.load(path) as d:
                    ids = d["ids"].tolist()
                    coordinates = d["coordinates"].copy()
                for rank in (1, 3, 8, 32):
                    for quality in _qualities(collection):
                        _, metrics = _predict_one(coordinates[:, :rank], ids, collection, quality,
                                                  f"{view}_PCA{rank}", layer, readout, "core")
                        out.extend(metrics)
        _save_rows(run / "predictions" / "additional_ranks.csv", out)


def _stability(run):
    regex = re.compile(r"layer(\d+)_(entity|final)_(H|F)_(UMAP|t-SNE)_2d_seed(\d+)\.npz")
    groups = defaultdict(list)
    for path in (run / "embeddings").glob("*_2d_seed*.npz"):
        m = regex.fullmatch(path.name)
        if m:
            groups[m.group(1, 2, 3, 4)].append((int(m.group(5)), path))
    rows = []
    for (layer, readout, view, method), variants in groups.items():
        variants.sort()
        for a in range(len(variants)):
            for b in range(a + 1, len(variants)):
                seed_a, path_a = variants[a]
                seed_b, path_b = variants[b]
                with np.load(path_a) as d:
                    ids_a, x = d["ids"].tolist(), d["coordinates"].copy()
                with np.load(path_b) as d:
                    ids_b, y = d["ids"].tolist(), d["coordinates"].copy()
                if ids_a != ids_b:
                    raise ValueError("Stability comparison has unmatched observations")
                n = len(x)
                na = NearestNeighbors(n_neighbors=11).fit(x).kneighbors(return_distance=False)[:, :10]
                nb = NearestNeighbors(n_neighbors=11).fit(y).kneighbors(return_distance=False)[:, :10]
                overlap = float(np.mean([len(set(u) & set(v)) / 10 for u, v in zip(na, nb)]))
                rng = np.random.default_rng(1729)
                u = rng.integers(0, n, 10000)
                v = rng.integers(0, n, 10000)
                ok = u != v
                dx = np.linalg.norm(x[u[ok]] - x[v[ok]], axis=1)
                dy = np.linalg.norm(y[u[ok]] - y[v[ok]], axis=1)
                rows.append({"layer": int(layer), "readout": readout, "view": view, "method": method,
                             "seed_a": seed_a, "seed_b": seed_b, "n_matched": n,
                             "neighbour_overlap_k10": overlap,
                             "distance_rank_spearman": float(spearmanr(dx, dy).statistic),
                             "procrustes_disparity": float(procrustes(x, y)[2])})
    _save_rows(run / "embeddings" / "stability.csv", rows)


def _local_dimension(run, collection):
    output = []
    rng = np.random.default_rng(1729)
    for layer in (1, 11, 27):
        for readout in ("entity", "final"):
            path = run / "reconstruction" / f"layer{layer}_{readout}_aligned.npz"
            if not path.exists():
                continue
            with np.load(path) as d:
                ids = d["ids"].tolist()
                h = d["H"].copy()
            keys, f, _ = feature_matrix(collection, layer)
            lookup = {rid: i for i, rid in enumerate(keys)}
            matrices = {"H": h, "F": f[[lookup[rid] for rid in ids]].toarray()}
            for view, matrix in matrices.items():
                if len(matrix) < 45:
                    continue
                centres = rng.choice(len(matrix), min(40, len(matrix)), replace=False)
                for k in (20, 40):
                    nn = NearestNeighbors(n_neighbors=k).fit(matrix)
                    neighbours = nn.kneighbors(matrix[centres], return_distance=False)
                    prs, entropy, dim90 = [], [], []
                    for indices in neighbours:
                        local = matrix[indices].astype(np.float64)
                        local -= local.mean(axis=0)
                        gram = local @ local.T
                        eigen = np.linalg.eigvalsh(gram)
                        eigen = np.maximum(eigen[eigen > 1e-10], 0)
                        if len(eigen) < 2:
                            continue
                        weights = eigen / eigen.sum()
                        prs.append(float(1 / np.sum(weights ** 2)))
                        entropy.append(float(np.exp(-np.sum(weights * np.log(weights)))))
                        dim90.append(int(np.searchsorted(np.cumsum(weights[::-1]), .9) + 1))
                    output.append({"layer": layer, "readout": readout, "view": view,
                                   "neighbourhood": k, "centres": len(prs),
                                   "median_local_participation_ratio": float(np.median(prs)) if prs else np.nan,
                                   "median_local_entropy_rank": float(np.median(entropy)) if entropy else np.nan,
                                   "median_local_90pct_dimension": float(np.median(dim90)) if dim90 else np.nan,
                                   "input": "original residual H or original sparse F, bounded 512-row cohort"})
        _save_rows(run / "dimension" / "local_pca.csv", output)
    # Full-spectrum rank measures are not asserted from a truncated 64-PC fit.
    spectra = []
    for path in (run / "linear").glob("*_pca.npz"):
        with np.load(path) as d:
            ratio = d["explained_variance_ratio"]
        weight = ratio / max(float(ratio.sum()), 1e-30)
        spectra.append({"fit": path.stem, "computed_components": len(ratio),
                        "variance_fraction_captured": float(ratio.sum()),
                        "participation_ratio_within_computed_spectrum": float(1 / np.sum(weight ** 2)),
                        "entropy_rank_within_computed_spectrum": float(np.exp(-np.sum(weight * np.log(np.maximum(weight, 1e-30))))),
                        "warning": "normalised over truncated spectrum only; not full effective rank"})
    _save_rows(run / "dimension" / "truncated_spectrum.csv", spectra)


def _decoder_dictionary(run, collection):
    from huggingface_hub import try_to_load_from_cache
    layer = 11
    repo = collection["repository"]
    ck = repo["checkpoints"][str(layer)]
    path = try_to_load_from_cache(repo["repo"], ck["filename"], revision=repo["revision"])
    if not isinstance(path, str):
        _save_rows(run / "dictionary" / "status.csv",
                   [{"layer": layer, "status": "skipped_checkpoint_unavailable"}])
        return
    catalogue = pd.read_csv(run / "features" / "candidates.csv")
    chosen = catalogue[catalogue["layer"] == layer].sort_values("train_fires", ascending=False)["feature_id"].drop_duplicates().head(256).to_numpy(dtype=int)
    if len(chosen) < 3:
        _save_rows(run / "dictionary" / "status.csv",
                   [{"layer": layer, "status": "too_few_selected_directions"}])
        return
    weights = load_checkpoint(path, ck["sha256"])
    directions = weights["W_dec"][:, chosen].T.numpy().copy()
    model = PCA(n_components=min(64, len(chosen) - 1), svd_solver="randomized", random_state=1729).fit(directions)
    out = run / "dictionary"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "layer11_decoder_direction_pca.npz",
                        feature_ids=chosen.astype(np.int32), checkpoint_sha256=np.asarray(ck["sha256"]),
                        mean=model.mean_.astype(np.float32), components=model.components_.astype(np.float32),
                        coordinates=model.transform(directions).astype(np.float32),
                        explained_variance_ratio=model.explained_variance_ratio_.astype(np.float32))
    _save_rows(out / "status.csv", [{"layer": layer, "status": "complete", "selected_directions": len(chosen),
                                    "meaning": "dictionary directions as observations, not activation co-occurrence"}])


def _contrast_transfer(run, collection):
    table = pd.read_csv(run / "contrasts" / "matched.csv")
    outputs = []
    for layer in (1, 11, 27):
        keys, f, _ = feature_matrix(collection, layer)
        lookup = {rid: i for i, rid in enumerate(keys)}
        for readout in ("entity", "final"):
            subset = table[(table["layer"] == layer) & (table["readout"] == readout) &
                           (~table["quality"].isin(["multiple", "nuisance"]))]
            for quality, group in subset.groupby("quality"):
                train = group[(group["entity_split"] == "train") & (group["template_split"] == "train")]
                if len(train) < 3:
                    continue
                deltas = []
                for row in train.itertuples():
                    if row.low_readout_id in lookup and row.high_readout_id in lookup:
                        deltas.append(f[lookup[row.high_readout_id]] - f[lookup[row.low_readout_id]])
                if len(deltas) < 3:
                    continue
                direction = np.asarray(sparse.vstack(deltas).mean(axis=0)).ravel()
                direction /= max(np.linalg.norm(direction), 1e-30)
                for part, selection in (("train", train), ("validation", group[group["entity_split"] == "validation"]),
                                        ("test", group[group["entity_split"] == "test"]),
                                        ("heldout_wording", group[group["template_split"] != "train"])):
                    scores, families = [], set()
                    for row in selection.itertuples():
                        if row.low_readout_id in lookup and row.high_readout_id in lookup:
                            delta = f[lookup[row.high_readout_id]] - f[lookup[row.low_readout_id]]
                            scores.append(float(np.asarray(delta @ direction).ravel()[0]))
                            families.add(row.entity_family)
                    if scores:
                        outputs.append({"layer": layer, "readout": readout, "quality": quality,
                                        "partition": part, "train_contrasts": len(train),
                                        "contrasts": len(scores), "entity_families": len(families),
                                        "fraction_correctly_signed": float(np.mean(np.asarray(scores) > 0)),
                                        "median_signed_projection": float(np.median(scores)),
                                        "fit": "mean training low-to-high F contrast, unit-normalised"})
        _save_rows(run / "contrasts" / "direction_transfer.csv", outputs)


def _transfer_partitions(run, collection):
    path = run / "predictions" / "rows.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["view"].isin(["H", "F", "H_PCA16", "F_PCA16", "H_PCA64", "F_PCA64"])]
    metadata = {r["record_id"]: r["metadata"] for r in collection["records"] if r["kind"] == "case"}
    frames = []
    for name, source_partition, test in (
        ("test_heldout_wording", "heldout_wording", lambda m: m["entity_split"] == "test" and m["template_split"] != "train"),
        ("validation_heldout_wording", "heldout_wording", lambda m: m["entity_split"] == "validation" and m["template_split"] != "train"),
        ("test_heldout_factorial", "heldout_combination", lambda m: m["entity_split"] == "test" and m["case_type"] == "factorial"),
    ):
        selected = frame[frame["partition"] == source_partition].copy()
        selected["meta"] = selected["record_id"].map(metadata)
        selected = selected[selected["meta"].map(lambda m: isinstance(m, dict) and test(m))]
        if len(selected):
            selected["evaluation"] = name
            frames.append(selected)
    if not frames:
        _save_rows(run / "predictions" / "transfer.csv", [])
        return
    merged = pd.concat(frames, ignore_index=True)
    output = []
    for (evaluation, layer, readout, view, quality), group in merged.groupby(
            ["evaluation", "layer", "readout", "view", "quality"]):
        if quality == "hue":
            actual = np.asarray([json.loads(v) for v in group["actual"]], dtype=float)
            predicted = np.asarray([json.loads(v) for v in group["predicted"]], dtype=float)
            score = _scores(actual, predicted, True)
        else:
            score = _scores(group["actual"].to_numpy(dtype=float),
                            group["predicted"].to_numpy(dtype=float), False)
        output.append({"evaluation": evaluation, "layer": layer, "readout": readout,
                       "view": view, "quality": quality, "records": len(group),
                       "entity_families": len({m["entity_group_id"] for m in group["meta"]}),
                       **score})
    _save_rows(run / "predictions" / "transfer.csv", output)


def _scaling_sensitivity(run, collection):
    from .sae_analysis_core import pca_dimensions
    layer, readout = 11, "final"
    pca_path = run / "linear" / f"layer{layer}_{readout}_F_pca.npz"
    with np.load(pca_path) as d:
        ids = d["ids"].tolist()
        train = set(d["train_ids"].tolist())
    train_idx = np.asarray([i for i, rid in enumerate(ids) if rid in train])
    keys, f, _ = feature_matrix(collection, layer)
    lookup = {rid: i for i, rid in enumerate(keys)}
    f = f[[lookup[rid] for rid in ids]]
    h = _raw_matrix(collection, layer, ids)
    rows = []
    for view, matrix in (("H", h), ("F", f)):
        if view == "H":
            scaled = normalize(matrix, norm="l2")
            fit = PCA(n_components=32, svd_solver="randomized", random_state=1729).fit(scaled[train_idx])
            ratio = fit.explained_variance_ratio_
        else:
            scaled = normalize(matrix, norm="l2")
            fit = centered_sparse_pca(scaled[train_idx], 32, 1729)
            ratio = fit["explained_variance_ratio"]
        rows.append({"layer": layer, "readout": readout, "view": view,
                     "preprocessing": "row_l2_normalization", "training_rows": len(train_idx),
                     "variance_fraction_first32": float(np.sum(ratio)),
                     **{f"dim_{k}": v for k, v in pca_dimensions(ratio).items()}})
    train_f = f[train_idx]
    fires = np.asarray((train_f != 0).sum(axis=0)).ravel()
    mean = np.asarray(train_f.mean(axis=0)).ravel()
    sd = np.sqrt(np.maximum(np.asarray(train_f.power(2).mean(axis=0)).ravel() - mean ** 2, 0))
    eligible = fires >= 20
    reference = float(np.median(sd[eligible])) if eligible.any() else 1.
    factors = np.ones(f.shape[1], dtype=np.float32)
    factors[eligible] = np.minimum(5., reference / np.maximum(sd[eligible], 1e-8))
    scaled = f.multiply(factors).tocsr()
    fit = centered_sparse_pca(scaled[train_idx], 32, 1729)
    ratio = fit["explained_variance_ratio"]
    np.savez_compressed(run / "linear" / "layer11_final_F_bounded_scale.npz",
                        factors=factors, train_ids=np.asarray([ids[i] for i in train_idx]),
                        mean=fit["mean"], components=fit["components"],
                        explained_variance_ratio=ratio)
    rows.append({"layer": layer, "readout": readout, "view": "F",
                 "preprocessing": "train_fitted_feature_sd_scaling_min20fires_max5x",
                 "training_rows": len(train_idx), "variance_fraction_first32": float(np.sum(ratio)),
                 **{f"dim_{k}": v for k, v in pca_dimensions(ratio).items()}})
    _save_rows(run / "linear" / "sensitivity.csv", rows)


def enrich(run_dir):
    run = Path(run_dir).expanduser().resolve()
    collection, _ = _source(run)
    state_path = run / "enrichment_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "version": "sae_analysis_enrichment_v1", "source_sha256": checksum(Path(__file__)),
        "stages": {}, "status": "partial"}
    if state["source_sha256"] != checksum(Path(__file__)):
        if state["status"] == "complete":
            raise ValueError("Completed enrichment source changed; use a new analysis run")
        old = state_path.with_name("enrichment_state_previous.json")
        if not old.exists():
            atomic_json(old, state)
        state = {"version": "sae_analysis_enrichment_v1", "source_sha256": checksum(Path(__file__)),
                 "stages": {}, "status": "partial"}
    stages = [("ranks", lambda: _rank_predictions(run, collection)),
              ("stability", lambda: _stability(run)),
              ("local_dimension", lambda: _local_dimension(run, collection)),
              ("dictionary", lambda: _decoder_dictionary(run, collection)),
              ("contrast_transfer", lambda: _contrast_transfer(run, collection)),
              ("transfer_partitions", lambda: _transfer_partitions(run, collection)),
              ("scaling_sensitivity", lambda: _scaling_sensitivity(run, collection))]
    for name, function in stages:
        if state["stages"].get(name) == "complete":
            continue
        state["current_stage"] = name
        atomic_json(state_path, state)
        function()
        state["stages"][name] = "complete"
        state.pop("current_stage", None)
        atomic_json(state_path, state)
        print(f"SAE enrichment {name}: complete", flush=True)
    state["status"] = "complete"
    atomic_json(state_path, state)
    return run
