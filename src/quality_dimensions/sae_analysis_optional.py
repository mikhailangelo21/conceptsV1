"""Bounded optional manifold/topology adapters. All embeddings are descriptive unless documented."""
from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import procrustes
from scipy.stats import spearmanr

from .sae_analysis import _quality_rows, _save_rows
from .sae_analysis_core import checksum, eligible_fit, embedding_metrics, feature_matrix, read_collection
from .util import atomic_json


def diffusion_map(x, components=2, time=1):
    """Finite-sample alpha=0 Gaussian diffusion map on a bounded fixed cohort."""
    x = np.asarray(x, dtype=np.float64)
    distances = cdist(x, x, metric="sqeuclidean")
    neighbour = np.partition(distances, min(15, len(x) - 1), axis=1)[:, min(15, len(x) - 1)]
    epsilon = max(float(np.median(neighbour)), 1e-12)
    kernel = np.exp(-distances / epsilon)
    degree = kernel.sum(axis=1)
    symmetric = kernel / np.sqrt(degree[:, None] * degree[None, :])
    values, vectors = np.linalg.eigh(symmetric)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    coordinates = vectors[:, 1:components + 1] / np.sqrt(degree[:, None])
    coordinates *= values[None, 1:components + 1] ** time
    return coordinates.astype(np.float32), {"epsilon": epsilon, "diffusion_time": time, "alpha": 0}


def _optional_embedding(name, x, dimensions, seed):
    if name == "PaCMAP":
        module = importlib.import_module("pacmap")
        kwargs = {"n_components": dimensions, "n_neighbors": 10, "MN_ratio": .5, "FP_ratio": 2.}
        signature = inspect.signature(module.PaCMAP)
        if "random_state" in signature.parameters:
            kwargs["random_state"] = seed
        else:
            np.random.seed(seed)
        return module.PaCMAP(**kwargs).fit_transform(x), kwargs
    if name == "TriMap":
        module = importlib.import_module("trimap")
        kwargs = {"n_dims": dimensions, "verbose": False}
        signature = inspect.signature(module.TRIMAP)
        if "random_state" in signature.parameters:
            kwargs["random_state"] = seed
        else:
            np.random.seed(seed)
        return module.TRIMAP(**kwargs).fit_transform(x), kwargs
    if name == "PHATE":
        module = importlib.import_module("phate")
        kwargs = {"n_components": dimensions, "knn": 15, "random_state": seed, "verbose": False}
        return module.PHATE(**kwargs).fit_transform(x), kwargs
    if name == "diffusion_maps":
        return diffusion_map(x, dimensions, 1)
    raise ValueError(f"Unknown optional method: {name}")


def _topology(run, collection, layer=11, limit=240):
    try:
        from ripser import ripser
    except ImportError:
        return {"method": "persistent_homology", "status": "skipped_uninstalled", "reason": "ripser unavailable"}
    types = {r["readout_id"]: r["readout_type"] for r in collection["readouts"]}
    hue = sorted({record["readout_id"] for record in collection["records"]
                  if record["kind"] == "case" and record["metadata"]["case_type"] == "hue"
                  and types[record["readout_id"]] == "final"})[:limit]
    keys, f, _ = feature_matrix(collection, layer)
    lookup = {rid: i for i, rid in enumerate(keys)}
    hue = [rid for rid in hue if rid in lookup]
    if len(hue) < 30:
        return {"method": "persistent_homology", "status": "too_few_hue_rows", "rows": len(hue)}
    x = f[[lookup[rid] for rid in hue]]
    distances = pairwise_distances(x, metric="euclidean")
    diagram = ripser(distances, distance_matrix=True, maxdim=1)["dgms"][1]
    finite = diagram[np.isfinite(diagram[:, 1])]
    lifetimes = finite[:, 1] - finite[:, 0]
    observed = float(np.max(lifetimes)) if len(lifetimes) else 0.
    # Feature-wise independent permutation destroys code coactivation while retaining
    # each selected column's empirical marginal. It is a diagnostic null, not a test of semantics.
    dense = x.toarray()
    nonzero_columns = np.flatnonzero(np.any(dense, axis=0))
    rng = np.random.default_rng(1729)
    nulls = []
    for _ in range(3):
        shuffled = dense.copy()
        for j in nonzero_columns:
            rng.shuffle(shuffled[:, j])
        null_distances = pairwise_distances(shuffled, metric="euclidean")
        null_diagram = ripser(null_distances, distance_matrix=True, maxdim=1)["dgms"][1]
        null_finite = null_diagram[np.isfinite(null_diagram[:, 1])]
        nulls.append(float(np.max(null_finite[:, 1] - null_finite[:, 0])) if len(null_finite) else 0.)
    out = run / "extended"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "hue_h1_layer11.npz", ids=np.asarray(hue), h1_diagram=diagram,
                        null_max_lifetimes=np.asarray(nulls), distances=distances.astype(np.float32))
    return {"method": "persistent_homology", "status": "complete", "rows": len(hue),
            "observed_max_h1_lifetime": observed, "null_max_h1_lifetime_median": float(np.median(nulls)),
            "interpretation": "exploratory original-F Euclidean topology; not proof of a hue circle"}


def _nonlinear_sensitivity(run, config):
    import umap
    rows, stability = [], []
    for view in ("H", "F"):
        baseline = run / "embeddings" / f"layer11_final_{view}_t-SNE_2d_seed1729.npz"
        with np.load(baseline) as d:
            ids = d["ids"].tolist()
            x = d["input_pca64"].copy()
            train = set(d["train_ids"].tolist())
        train_idx = np.asarray([i for i, rid in enumerate(ids) if rid in train])
        random_embeddings = []
        for neighbours in (10, 30, 50):
            for minimum_distance in (.05, .3):
                seed = config["seed"]
                reducer = umap.UMAP(n_components=2, n_neighbors=neighbours,
                                    min_dist=minimum_distance, random_state=seed,
                                    transform_seed=seed)
                reducer.fit(x[train_idx])
                coords = reducer.transform(x)
                metrics = embedding_metrics(x, coords, config["neighbourhood_sizes"], seed)
                rows.append({"view": view, "method": "UMAP", "neighbours": neighbours,
                             "min_dist": minimum_distance, "seed": seed,
                             "mode": "heldout_transform", **metrics})
                np.savez_compressed(run / "extended" /
                                    f"sensitivity_UMAP_{view}_n{neighbours}_d{minimum_distance}.npz",
                                    ids=np.asarray(ids), coordinates=coords.astype(np.float32),
                                    train_ids=np.asarray([ids[i] for i in train_idx]))
        for perplexity in (5, 30, 50):
            seed = config["seed"]
            coords = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                          init="random", learning_rate="auto").fit_transform(x)
            rows.append({"view": view, "method": "t-SNE", "perplexity": perplexity,
                         "seed": seed, "initialization": "random",
                         "mode": "descriptive_all_splits",
                         **embedding_metrics(x, coords, config["neighbourhood_sizes"], seed)})
        for seed in config["stochastic_seeds"]:
            coords = TSNE(n_components=2, perplexity=30, random_state=seed,
                          init="random", learning_rate="auto").fit_transform(x)
            random_embeddings.append((seed, coords))
            np.savez_compressed(run / "extended" / f"tsne_random_{view}_seed{seed}.npz",
                                ids=np.asarray(ids), coordinates=coords.astype(np.float32), input_pca64=x)
        for i in range(len(random_embeddings)):
            for j in range(i + 1, len(random_embeddings)):
                seed_a, a = random_embeddings[i]
                seed_b, b = random_embeddings[j]
                near_a = NearestNeighbors(n_neighbors=11).fit(a).kneighbors(return_distance=False)[:, :10]
                near_b = NearestNeighbors(n_neighbors=11).fit(b).kneighbors(return_distance=False)[:, :10]
                overlap = float(np.mean([len(set(u) & set(v)) / 10 for u, v in zip(near_a, near_b)]))
                rng = np.random.default_rng(1729)
                u = rng.integers(0, len(a), 10000)
                v = rng.integers(0, len(a), 10000)
                valid = u != v
                da = np.linalg.norm(a[u[valid]] - a[v[valid]], axis=1)
                db = np.linalg.norm(b[u[valid]] - b[v[valid]], axis=1)
                stability.append({"view": view, "method": "t-SNE", "initialization": "random",
                                  "seed_a": seed_a, "seed_b": seed_b, "n_matched": len(a),
                                  "neighbour_overlap_k10": overlap,
                                  "distance_rank_spearman": float(spearmanr(da, db).statistic),
                                  "procrustes_disparity": float(procrustes(a, b)[2])})
    _save_rows(run / "extended" / "nonlinear_sensitivity.csv", rows)
    _save_rows(run / "extended" / "tsne_random_stability.csv", stability)


def optional_methods(run_dir):
    run = Path(run_dir).expanduser().resolve()
    project = Path(__file__).resolve().parents[2]
    config = json.loads((run / "config.json").read_text())
    collection = read_collection((project / config["source_run"]).resolve())
    path = run / "linear" / "layer11_final_F_pca.npz"
    with np.load(path) as d:
        ids = d["ids"].tolist()
        x = d["coordinates"][:, :32].copy()
        train_ids = set(d["train_ids"].tolist())
    all_ids, all_x = ids, x
    n = min(config["nonlinear_rows"], len(ids))
    sample = json.loads((run / "sample_ids.json").read_text())["final"]
    quotas = {"train": n // 2, "validation": n // 6, "test": n // 6,
              "challenge": n - (n // 2 + 2 * (n // 6))}
    chosen = [rid for split, quota in quotas.items() for rid in sample[split][:quota] if rid in set(ids)]
    lookup = {rid: i for i, rid in enumerate(ids)}
    x = x[[lookup[rid] for rid in chosen]]
    rows = []
    for method in ("PaCMAP", "TriMap", "diffusion_maps", "PHATE"):
        for dimensions in (2, 3):
            try:
                coords, params = _optional_embedding(method, x, dimensions, config["seed"])
                if len(coords) != len(chosen) or coords.shape[1] != dimensions:
                    raise ValueError("Adapter returned incompatible coordinates")
                metrics = embedding_metrics(x, coords, config["neighbourhood_sizes"], config["seed"])
                outfile = run / "extended" / f"layer11_final_F_{method}_{dimensions}d.npz"
                outfile.parent.mkdir(exist_ok=True)
                np.savez_compressed(outfile, ids=np.asarray(chosen), coordinates=np.asarray(coords, dtype=np.float32),
                                    input_pca32=x, parameters=np.asarray(json.dumps(params)))
                rows.append({"method": method, "dimensions": dimensions,
                             "mode": "descriptive_all_splits", "status": "complete", **metrics})
            except ImportError as exc:
                rows.append({"method": method, "dimensions": dimensions,
                             "status": "skipped_uninstalled", "reason": str(exc)})
            except Exception as exc:
                rows.append({"method": method, "dimensions": dimensions,
                             "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    rows.append(_topology(run, collection))
    # PLS is supervised and receives a separate score table; it does not enter
    # unsupervised geometry comparisons.
    train_rows = [r for r in _quality_rows(collection, all_ids, "size") if eligible_fit(r[2])]
    test_rows = [r for r in _quality_rows(collection, all_ids, "size")
                 if r[2]["entity_split"] == "test" and r[2]["template_split"] == "train"]
    if len(train_rows) >= 10 and len(test_rows) >= 3:
        pls = PLSRegression(n_components=2)
        pls.fit(all_x[[r[0] for r in train_rows]], np.asarray([r[1] for r in train_rows]))
        pred = pls.predict(all_x[[r[0] for r in test_rows]]).ravel()
        actual = np.asarray([r[1] for r in test_rows])
        _save_rows(run / "extended" / "pls_supervised.csv", [{
            "quality": "size", "layer": 11, "readout": "final", "view": "F_PCA32",
            "train_records": len(train_rows), "test_records": len(test_rows),
            "test_entity_families": len({r[2]["entity_group_id"] for r in test_rows}),
            "test_spearman": float(spearmanr(actual, pred).statistic),
            "test_mae_design_code": float(np.mean(np.abs(actual - pred))),
            "status": "complete", "meaning": "supervised comparison, not unsupervised embedding"}])
    else:
        _save_rows(run / "extended" / "pls_supervised.csv", [{
            "quality": "size", "status": "insufficient_sampled_test_or_train_rows",
            "train_records": len(train_rows), "test_records": len(test_rows)}])
    _save_rows(run / "extended" / "optional_methods.csv", rows)
    _nonlinear_sensitivity(run, config)
    versions = {}
    for package in ("pacmap", "trimap", "phate", "ripser", "umap-learn", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    atomic_json(run / "extended" / "optional_provenance.json",
                {"source_sha256": checksum(Path(__file__)), "packages": versions,
                 "input": "layer11 final F train-fitted PCA32, fixed split-balanced cohort",
                 "seed": config["seed"], "descriptive_methods": ["PaCMAP", "TriMap", "diffusion_maps", "PHATE"]})
    resolved = pd.read_csv(run / "methods.csv")
    status_by_method = {row["method"]: row["status"] for row in rows}
    status_by_method["PLS"] = pd.read_csv(run / "extended" / "pls_supervised.csv").iloc[0]["status"]
    for idx, row in resolved.iterrows():
        if row["method"] in status_by_method:
            resolved.loc[idx, "status"] = status_by_method[row["method"]]
    resolved.to_csv(run / "extended" / "methods_resolved.csv", index=False)
    return run / "extended" / "optional_methods.csv"
