"""Versioned, resumable analysis of existing Qwen-Scope SAE collection; no model forwards."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.random_projection import SparseRandomProjection

from .sae_analysis_core import (checksum, centered_sparse_pca, eligible_fit, embedding_metrics,
                                feature_matrix, intrinsic_knn, jsonl, layer_keys, linear_cka,
                                partition, pca_dimensions, read_collection, sparse_activity,
                                stable_order, transform_centered)
from .sae_collection import _legacy, _raw_for
from .sae_core import load_checkpoint
from .util import atomic_json

VERSION = "sae_analysis_v1"


def _save_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}) if rows else ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def prepare(config_path, profile="core", run_dir=None, resume=False):
    config_path = Path(config_path).expanduser().resolve()
    config = json.loads(config_path.read_text())
    if config["schema_version"] != VERSION or profile not in {"smoke", "core", "extended"}:
        raise ValueError("Unsupported analysis config/profile")
    project = config_path.parent.parent
    source = (project / config["source_run"]).resolve()
    collection = read_collection(source)
    selected = config["smoke_layers"] if profile == "smoke" else config["core_layers"]
    identity = {"version": VERSION, "profile": profile, "config_sha256": checksum(config_path),
                "source_manifest_sha256": checksum(source / "manifest.json"),
                "analysis_code_sha256": checksum(Path(__file__)),
                "core_code_sha256": checksum(Path(__file__).with_name("sae_analysis_core.py")),
                "source_run_identity": collection["manifest"]["run_identity"],
                "selected_layers": selected}
    from .util import digest
    name = f"{VERSION}_{profile}-{digest(identity)[:12]}"
    if run_dir:
        run = Path(run_dir).expanduser().resolve()
    else:
        run = project / "runs" / name
        if not run.exists():
            local = Path.home() / ".local/share/quality-dimensions-llm/runs" / name
            local.mkdir(parents=True, exist_ok=True)
            run.symlink_to(local, target_is_directory=True)
    if (run / "identity.json").exists():
        if json.loads((run / "identity.json").read_text()) != identity or not resume:
            raise ValueError("Existing analysis run; use --resume with unchanged source/config")
    else:
        if run.exists() and any(run.iterdir()):
            raise ValueError("Nonempty output path lacks analysis identity")
        run.mkdir(parents=True, exist_ok=True)
        atomic_json(run / "identity.json", identity)
        atomic_json(run / "config.json", config)
        atomic_json(run / "state.json", {"status": "prepared", "stages": {}})
        def version(name):
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                return None
        provenance = {"python": platform.python_version(), "platform": platform.platform(),
                      "source": str(source), "source_manifest_sha256": identity["source_manifest_sha256"],
                      "config_sha256": identity["config_sha256"], "seed": config["seed"],
                      "packages": {name: version(name)
                                   for name in ["numpy", "scipy", "scikit-learn", "umap-learn", "plotly", "matplotlib"]}}
        atomic_json(run / "provenance.json", provenance)
    return run, collection, config


def _stage(run, name, function):
    state = json.loads((run / "state.json").read_text())
    if state["stages"].get(name) == "complete":
        return
    state["status"] = "running"
    state["current_stage"] = name
    atomic_json(run / "state.json", state)
    try:
        function()
    except Exception as exc:
        state = json.loads((run / "state.json").read_text())
        state["status"] = "partial"
        state["stages"][name] = {"error": f"{type(exc).__name__}: {exc}"}
        atomic_json(run / "state.json", state)
        raise
    state = json.loads((run / "state.json").read_text())
    state["stages"][name] = "complete"
    state.pop("current_stage", None)
    atomic_json(run / "state.json", state)


def audit(run, collection):
    source = collection["source"]
    manifest = collection["manifest"]
    records = collection["records"]
    readout_type = {r["readout_id"]: r["readout_type"] for r in collection["readouts"]}
    counts = Counter()
    families = defaultdict(set)
    layers = collection["identity"]["layers"]
    complete = []
    for layer in layers:
        m = manifest["layers"][f"main:{layer}"]
        t = manifest["layers"][f"trace:{layer}"]
        complete.append({"layer": layer, "main_completed": m["completed"], "main_expected": m["expected"],
                         "trace_completed": t["completed"], "trace_expected": t["expected"],
                         "relative_error_p50": m["relative_error_p50"],
                         "status": "complete" if m["completed"] == m["expected"] and t["completed"] == t["expected"]
                         else "partial" if m["completed"] else "missing"})
        present = layer_keys(source, "main", layer)
        for row in records:
            meta = row["metadata"]
            if row["kind"] == "case":
                qualities = list(meta.get("targets", {})) or ["none"]
                family = meta.get("entity_group_id", "")
                wording = meta.get("template_split", "")
                case_type = meta.get("case_type", "")
                split = meta.get("entity_split", "")
            else:
                qualities = ["comparison"]
                family = meta.get("entity_group_id", "")
                wording = meta.get("template_split", "")
                case_type = "comparison"
                split = meta.get("entity_split", "")
            for quality in qualities:
                key = (layer, row["kind"], quality, case_type, split, wording,
                       readout_type[row["readout_id"]])
                counts[(key, "expected")] += 1
                families[(key, "expected")].add(family)
                if row["readout_id"] in present:
                    counts[(key, "available")] += 1
                    families[(key, "available")].add(family)
    _save_rows(run / "audit" / "layers.csv", complete)
    rows = []
    for key in sorted({k for k, mode in counts}):
        rows.append(dict(zip(("layer", "kind", "quality", "case_type", "entity_split", "wording", "readout"), key),
                         expected_records=counts[(key, "expected")], available_records=counts[(key, "available")],
                         expected_entity_families=len(families[(key, "expected")]),
                         available_entity_families=len(families[(key, "available")])))
    _save_rows(run / "audit" / "coverage.csv", rows)
    atomic_json(run / "audit" / "summary.json", {
        "source_status": manifest["status"], "readout_vectors": len(collection["readouts"]),
        "associations": len(records), "case_associations": sum(r["kind"] == "case" for r in records),
        "complete_layers": [r["layer"] for r in complete if r["status"] == "complete"],
        "partial_layers": [r["layer"] for r in complete if r["status"] == "partial"],
        "missing_layers": [r["layer"] for r in complete if r["status"] == "missing"],
        "checkpoint_hashes": {str(layer): collection["repository"]["checkpoints"][str(layer)]["sha256"] for layer in layers},
        "hook": manifest["hook"], "k": manifest["k"], "feature_width": manifest["feature_width"]})


def sample_readouts(run, collection, config, profile):
    path = run / "sample_ids.json"
    if path.exists():
        return json.loads(path.read_text())
    by_id = {r["readout_id"]: r for r in collection["readouts"]}
    strata = defaultdict(list)
    for rid, associations in collection["by_readout"].items():
        case = [a for a in associations if a["kind"] == "case"]
        if not case:
            continue
        meta = case[0]["metadata"]
        split = "train" if eligible_fit(meta) else ("challenge" if meta["entity_split"] == "train" else meta["entity_split"])
        if split not in {"train", "validation", "test", "challenge"}:
            continue
        group = (by_id[rid]["readout_type"], split, meta["case_type"],
                 meta["quality_ids"][0] if meta.get("quality_ids") else "")
        strata[group].append(rid)
    limits = ({"train": 120, "validation": 40, "test": 40, "challenge": 20} if profile == "smoke" else
              {"train": config["max_train_per_readout"], "validation": config["max_validation_per_readout"],
               "test": config["max_test_per_readout"], "challenge": config["max_challenge_per_readout"]})
    selection = {}
    for readout in ("entity", "final"):
        selection[readout] = {}
        for split, limit in limits.items():
            buckets = [stable_order(set(ids), config["seed"]) for key, ids in sorted(strata.items())
                       if key[0] == readout and key[1] == split]
            chosen = []
            if profile == "smoke" and split == "train":
                size_bucket = stable_order(set(rid for key, group in strata.items()
                                               if key[0] == readout and key[1] == split and key[3] == "size"
                                               for rid in group), config["seed"])
                chosen.extend(size_bucket[:min(55, len(size_bucket))])
                buckets = [[rid for rid in bucket if rid not in set(chosen)] for bucket in buckets]
            while len(chosen) < limit and any(buckets):
                for bucket in buckets:
                    if bucket and len(chosen) < limit:
                        chosen.append(bucket.pop(0))
            selection[readout][split] = chosen
    atomic_json(path, selection)
    return selection


def _raw_matrix(collection, layer, ids):
    source = collection["source"]
    rows = {r["readout_id"]: r for r in collection["readouts"]}
    legacy = _legacy(source)
    from .sae_core import sae_index
    layer_map = {sae_index(block): idx for idx, block in enumerate(legacy.identity["block_numbers"])}
    vectors = []
    for rid in ids:
        value = _raw_for(source, rows[rid], layer, legacy, layer_map)
        if value is None:
            raise ValueError(f"Missing raw activation for layer {layer}, readout {rid}")
        vectors.append(value)
    return np.stack(vectors).astype(np.float32)


def _quality_rows(collection, ids, key, split=None):
    result = []
    for idx, rid in enumerate(ids):
        for record in collection["by_readout"][rid]:
            if record["kind"] != "case":
                continue
            meta = record["metadata"]
            target = meta.get("targets", {})
            if key == "hue":
                if "hue_sin" not in target or "hue_cos" not in target:
                    continue
                y = [target["hue_sin"], target["hue_cos"]]
            else:
                lexical = key.startswith("lexical__")
                q = key.removeprefix("lexical__")
                if q not in target or q.endswith("__value") or (meta["case_type"] == "lexical") != lexical:
                    continue
                y = float(target[q])
            if split == "train" and not eligible_fit(meta):
                continue
            if split in {"validation", "test"} and meta["entity_split"] != split:
                continue
            result.append((idx, y, meta, record["record_id"]))
    return result


def _qualities(collection):
    keys = set()
    for record in collection["records"]:
        if record["kind"] != "case":
            continue
        meta = record["metadata"]
        if "hue_sin" in meta["targets"] and "hue_cos" in meta["targets"]:
            keys.add("hue")
        for key in meta["targets"]:
            if key in {"hue_sin", "hue_cos"} or key.endswith("__value"):
                continue
            keys.add(("lexical__" if meta["case_type"] == "lexical" else "") + key)
    return sorted(keys)


def _scores(y, pred, hue=False):
    if hue:
        angles = np.arctan2(y[:, 0], y[:, 1])
        estimates = np.arctan2(pred[:, 0], pred[:, 1])
        err = np.abs(np.angle(np.exp(1j * (angles - estimates))))
        return {"angular_mae_degrees": float(np.degrees(err).mean()),
                "angular_median_degrees": float(np.degrees(np.median(err)))}
    y = np.asarray(y).ravel()
    pred = np.asarray(pred).ravel()
    rho = spearmanr(y, pred).statistic if len(np.unique(y)) > 1 and len(np.unique(pred)) > 1 else np.nan
    rng = np.random.default_rng(1729)
    a = rng.integers(0, len(y), min(4000, max(100, len(y) * 4)))
    b = rng.integers(0, len(y), len(a))
    valid = y[a] != y[b]
    accuracy = float(np.mean(np.sign(y[a[valid]] - y[b[valid]]) == np.sign(pred[a[valid]] - pred[b[valid]]))) if valid.any() else np.nan
    return {"spearman": float(rho), "mae_design_code": float(np.mean(np.abs(y - pred))),
            "pairwise_order_accuracy": accuracy}


def _predict_one(x, ids, collection, quality, view, layer, readout, profile):
    rows = _quality_rows(collection, ids, quality)
    train = [r for r in rows if eligible_fit(r[2])]
    validation = [r for r in rows if r[2]["entity_split"] == "validation" and r[2]["template_split"] == "train"
                  and (r[2]["case_type"] != "factorial" or r[2]["combination_split"] == "train")]
    if len(train) < 10 or len({r[2]["entity_group_id"] for r in train}) < 2:
        return [], [{"layer": layer, "readout": readout, "view": view, "quality": quality,
                     "status": "insufficient_training_families", "train_records": len(train)}]
    train_indices = [r[0] for r in train]
    y = np.asarray([r[1] for r in train], dtype=np.float32)
    if sparse.issparse(x):
        xtrain = x[train_indices]
    else:
        xtrain = x[train_indices, :]
    hue = quality == "hue"
    selected = None
    for alpha in (1., 10., 100.):
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(xtrain, y)
        if validation:
            xv = x[[r[0] for r in validation]]
            vy = np.asarray([r[1] for r in validation], dtype=np.float32)
            p = model.predict(xv)
            metric = _scores(vy, p, hue)
            score = -metric["angular_mae_degrees"] if hue else (metric["spearman"] if np.isfinite(metric["spearman"]) else -2)
        else:
            score = -alpha
        if selected is None or score > selected[0]:
            selected = (score, alpha, model)
    _, alpha, model = selected
    predictions = []
    metrics = []
    for label, subset in (("train", train), ("validation_familiar", validation),
                          ("test_familiar", [r for r in rows if r[2]["entity_split"] == "test" and r[2]["template_split"] == "train"
                                             and (r[2]["case_type"] != "factorial" or r[2]["combination_split"] == "train")]),
                          ("heldout_wording", [r for r in rows if r[2]["template_split"] != "train"]),
                          ("heldout_combination", [r for r in rows if r[2]["case_type"] == "factorial" and r[2]["combination_split"] != "train"])):
        if not subset:
            continue
        idx = [r[0] for r in subset]
        yy = np.asarray([r[1] for r in subset], dtype=np.float32)
        pp = model.predict(x[idx])
        score = _scores(yy, pp, hue)
        metrics.append({"layer": layer, "readout": readout, "view": view, "quality": quality,
                        "partition": label, "alpha_selected_on_validation": alpha,
                        "records": len(subset), "entity_families": len({r[2]["entity_group_id"] for r in subset}),
                        **score, "status": "complete"})
        if label != "train":
            for row, pred in zip(subset, pp):
                predictions.append({"layer": layer, "readout": readout, "view": view, "quality": quality,
                                    "partition": label, "record_id": row[3], "readout_id": ids[row[0]],
                                    "entity_family": row[2]["entity_group_id"], "actual": row[1],
                                    "predicted": pred.tolist() if hue else float(pred)})
    return predictions, metrics


def _pca_fit(x, train_idx, rank, seed):
    fit = centered_sparse_pca(x[train_idx], rank, seed) if sparse.issparse(x) else None
    if fit is None:
        model = PCA(n_components=min(rank, len(train_idx) - 1, x.shape[1]), svd_solver="randomized", random_state=seed)
        model.fit(x[train_idx])
        fit = {"mean": model.mean_.astype(np.float32), "components": model.components_.astype(np.float32),
               "singular_values": model.singular_values_.astype(np.float32),
               "explained_variance_ratio": model.explained_variance_ratio_.astype(np.float32),
               "total_centered_ss": float(np.sum((x[train_idx] - model.mean_) ** 2))}
    fit["all_scores"] = transform_centered(x, fit)
    return fit


def _save_pca(run, layer, readout, view, ids, fit, train_idx, x):
    path = run / "linear" / f"layer{layer}_{readout}_{view}_pca.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = fit["all_scores"]
    variance = fit["explained_variance_ratio"]
    np.savez_compressed(path, ids=np.asarray(ids), train_ids=np.asarray([ids[i] for i in train_idx]),
                        mean=fit["mean"], components=fit["components"], singular_values=fit["singular_values"],
                        explained_variance_ratio=variance, coordinates=coords,
                        total_centered_ss=np.asarray(fit["total_centered_ss"]))
    # Orthonormal components make this identity exact for both sparse and dense X.
    mean = fit["mean"]
    squared = float(x.power(2).sum()) if sparse.issparse(x) else float(np.sum(x ** 2))
    cross = float(np.sum(x @ mean))
    baseline = max(squared - 2 * cross + len(ids) * float(mean @ mean), 0.)
    residual_ss = max(baseline - float(np.sum(coords ** 2)), 0.)
    denominator = max(baseline, 1e-30)
    return {"layer": layer, "readout": readout, "view": view, "n_fit": len(train_idx),
            "n_transform": len(ids), "rank": len(variance),
            "cumulative_variance_at_rank": float(np.sum(variance)),
            "reconstruction_relative_rmse_at_rank": float(np.sqrt(residual_ss / denominator)),
            **{f"dim_{key}": value for key, value in pca_dimensions(variance).items()}}


def _feature_candidates(x, ids, collection, layer, readout, min_fires):
    rows = []
    index = {rid: i for i, rid in enumerate(ids)}
    by_case = defaultdict(list)
    readout_text = {r["readout_id"]: r["text"] for r in collection["readouts"]}
    for rid in ids:
        for record in collection["by_readout"][rid]:
            if record["kind"] == "case":
                by_case[record["metadata"]["case_id"]].append((rid, record["metadata"]))
    for quality in _qualities(collection):
        qrows = _quality_rows(collection, ids, quality)
        if quality == "hue":
            continue
        train = [r for r in qrows if eligible_fit(r[2])]
        if len(train) < 20 or len({r[2]["entity_group_id"] for r in train}) < 2:
            continue
        idx = [r[0] for r in train]
        y = np.asarray([r[1] for r in train], dtype=np.float64)
        matrix = x[idx]
        fires = np.asarray((matrix != 0).sum(axis=0)).ravel()
        active = np.where(fires >= min_fires)[0]
        if not len(active):
            continue
        centered_y = y - y.mean()
        covariance = np.asarray(matrix[:, active].T @ centered_y).ravel() / len(y)
        variance = np.asarray(matrix[:, active].power(2).mean(axis=0)).ravel() - np.asarray(matrix[:, active].mean(axis=0)).ravel() ** 2
        score = np.abs(covariance) / np.sqrt(np.maximum(variance, 1e-12) * max(y.var(), 1e-12))
        shortlisted = active[np.argsort(score)[-20:][::-1]]
        for feature in shortlisted:
            values = matrix[:, feature].toarray().ravel()
            rho = spearmanr(values, y).statistic
            if not np.isfinite(rho):
                continue
            row = {"layer": layer, "readout": readout, "quality": quality, "feature_id": int(feature),
                   "train_spearman": float(rho), "train_fires": int(fires[feature]),
                   "train_records": len(train), "train_families": len({r[2]["entity_group_id"] for r in train})}
            family_signs = []
            for family in sorted({r[2]["entity_group_id"] for r in train}):
                sub = [r for r in train if r[2]["entity_group_id"] == family]
                if len(sub) >= 4:
                    fy = np.asarray([r[1] for r in sub])
                    fv = x[[r[0] for r in sub], feature].toarray().ravel()
                    if len(np.unique(fy)) > 1 and len(np.unique(fv)) > 1:
                        family_signs.append(np.sign(spearmanr(fy, fv).statistic) == np.sign(rho))
            row["training_family_sign_agreement"] = float(np.mean(family_signs)) if family_signs else np.nan
            row["training_families_evaluable"] = len(family_signs)
            contrast_changes = []
            seen_contrasts = set()
            for record in train:
                meta = record[2]
                rid = ids[record[0]]
                for contrast in meta.get("contrasts", []):
                    cid = contrast["contrast_id"]
                    if cid in seen_contrasts:
                        continue
                    paired = [item for item in by_case.get(contrast["paired_case_id"], []) if item[0] in index]
                    if not paired:
                        continue
                    paired_rid, paired_meta = paired[0]
                    target = quality.removeprefix("lexical__")
                    if target not in paired_meta.get("targets", {}) or not eligible_fit(paired_meta):
                        continue
                    delta_y = paired_meta["targets"][target] - meta["targets"][target]
                    if delta_y == 0:
                        continue
                    delta_f = float(x[index[paired_rid], feature] - x[index[rid], feature])
                    contrast_changes.append(np.sign(delta_y) * delta_f)
                    seen_contrasts.add(cid)
            row["matched_pairs"] = len(contrast_changes)
            row["matched_signed_delta_mean"] = float(np.mean(contrast_changes)) if contrast_changes else np.nan
            active_rows = x[:, feature].toarray().ravel()
            top = np.argsort(active_rows)[-3:][::-1]
            row["representative_examples"] = [{"readout_id": ids[i], "activation": float(active_rows[i]),
                                                  "text": readout_text[ids[i]][:220]} for i in top if active_rows[i] > 0]
            wording_train = np.asarray([i for i, rid in enumerate(ids) if
                                        any(a["kind"] == "case" and a["metadata"].get("template_split") == "train"
                                            for a in collection["by_readout"][rid])])
            wording_held = np.asarray([i for i, rid in enumerate(ids) if
                                       any(a["kind"] == "case" and a["metadata"].get("template_split") != "train"
                                           for a in collection["by_readout"][rid])])
            row["heldout_wording_mean_minus_familiar"] = (float(active_rows[wording_held].mean() -
                                                                active_rows[wording_train].mean())
                                                          if len(wording_train) and len(wording_held) else np.nan)
            for split in ("validation", "test"):
                subset = [r for r in qrows if r[2]["entity_split"] == split]
                if subset:
                    v = x[[r[0] for r in subset], feature].toarray().ravel()
                    yy = np.asarray([r[1] for r in subset])
                    result = spearmanr(v, yy).statistic if len(np.unique(v)) > 1 and len(np.unique(yy)) > 1 else np.nan
                    row[f"{split}_spearman"] = float(result)
                    row[f"{split}_records"] = len(subset)
                    row[f"{split}_families"] = len({r[2]["entity_group_id"] for r in subset})
            rows.append(row)
    return rows


def _coactivation(run, layer, readout, x, train_n):
    train = x[:train_n]
    binary = train.copy()
    binary.data[:] = 1
    fires = np.asarray(binary.sum(axis=0)).ravel()
    selected = np.argsort(fires)[-32:][::-1]
    b = binary[:, selected].toarray().astype(np.float64)
    v = train[:, selected].toarray().astype(np.float64)
    prevalence = b.mean(axis=0)
    joint = (b.T @ b) / max(len(b), 1)
    expected = prevalence[:, None] * prevalence[None, :]
    lift = np.divide(joint, expected, out=np.full_like(joint, np.nan), where=expected > 0)
    denom = np.sqrt(np.maximum(prevalence * (1 - prevalence), 1e-12))
    phi = (joint - expected) / (denom[:, None] * denom[None, :])
    value_correlation = np.corrcoef(v.T)
    path = run / "activity" / f"layer{layer}_{readout}_coactivation.npz"
    np.savez_compressed(path, feature_ids=selected.astype(np.int32), prevalence=prevalence,
                        binary_joint=joint, prevalence_lift=lift, binary_phi=phi,
                        activation_value_correlation=value_correlation,
                        fit_rows=np.asarray(train_n))


def _contrasts(collection, layer, readout, all_keys, full_f, sample_ids, sample_h):
    """Matched changes are keyed by original contrast IDs, never arbitrary neighbours."""
    types = {r["readout_id"]: r["readout_type"] for r in collection["readouts"]}
    by_contrast = defaultdict(dict)
    for record in collection["records"]:
        if record["kind"] != "case" or types[record["readout_id"]] != readout:
            continue
        meta = record["metadata"]
        for contrast in meta.get("contrasts", []):
            by_contrast[contrast["contrast_id"]][meta["case_id"]] = (record["readout_id"], meta, contrast["kind"])
    feature_lookup = {rid: i for i, rid in enumerate(all_keys)}
    raw_lookup = {rid: i for i, rid in enumerate(sample_ids)}
    rows = []
    for contrast_id, endpoints in by_contrast.items():
        if len(endpoints) != 2:
            continue
        pair = list(endpoints.values())
        if any(item[0] not in feature_lookup for item in pair):
            continue
        rid_a, a, kind = pair[0]
        rid_b, b, _ = pair[1]
        changed = [(quality, float(b["targets"][quality]) - float(value))
                   for quality, value in a.get("targets", {}).items()
                   if quality in b.get("targets", {}) and quality not in {"hue_sin", "hue_cos"}
                   and not quality.endswith("__value")
                   and float(b["targets"][quality]) != float(value)]
        quality = changed[0][0] if len(changed) == 1 else "multiple" if changed else "nuisance"
        sign = np.sign(changed[0][1]) if len(changed) == 1 else 1.
        if sign < 0:
            rid_a, rid_b = rid_b, rid_a
            a, b = b, a
        fa = full_f[feature_lookup[rid_a]]
        fb = full_f[feature_lookup[rid_b]]
        delta = fb - fa
        sa = set(fa.indices.tolist())
        sb = set(fb.indices.tolist())
        row = {"layer": layer, "readout": readout, "contrast_id": contrast_id, "kind": kind,
               "quality": quality, "low_readout_id": rid_a, "high_readout_id": rid_b,
               "entity_family": a.get("entity_group_id"), "entity_split": a.get("entity_split"),
               "template_split": a.get("template_split"),
               "delta_F_norm": float(np.sqrt(delta.power(2).sum())),
               "feature_support_jaccard": len(sa & sb) / len(sa | sb) if sa or sb else np.nan,
               "delta_H_norm": np.nan}
        if rid_a in raw_lookup and rid_b in raw_lookup:
            row["delta_H_norm"] = float(np.linalg.norm(sample_h[raw_lookup[rid_b]] - sample_h[raw_lookup[rid_a]]))
        rows.append(row)
    return rows


def _hue_triplets(collection, layer, readout, all_keys, full_f, sample_ids, sample_h):
    types = {r["readout_id"]: r["readout_type"] for r in collection["readouts"]}
    triplets = defaultdict(dict)
    for record in collection["records"]:
        if record["kind"] != "case" or types[record["readout_id"]] != readout:
            continue
        meta = record["metadata"]
        for triplet in meta.get("hue_triplets", []):
            triplets[triplet["triplet_id"]][triplet["role"]] = (record["readout_id"], meta)
    fidx = {rid: i for i, rid in enumerate(all_keys)}
    hidx = {rid: i for i, rid in enumerate(sample_ids)}
    rows = []
    for triplet_id, roles in triplets.items():
        role_names = ("anchor_case_id", "near_case_id", "far_case_id")
        if not all(role in roles for role in role_names):
            continue
        ids = [roles[role][0] for role in role_names]
        if not all(rid in fidx for rid in ids):
            continue
        f = full_f[[fidx[rid] for rid in ids]]
        fnear = float(np.sqrt((f[0] - f[1]).power(2).sum()))
        ffar = float(np.sqrt((f[0] - f[2]).power(2).sum()))
        row = {"layer": layer, "readout": readout, "triplet_id": triplet_id,
               "entity_family": roles[role_names[0]][1].get("entity_group_id"),
               "entity_split": roles[role_names[0]][1].get("entity_split"),
               "F_near_distance": fnear, "F_far_distance": ffar,
               "F_order_correct": int(fnear < ffar), "H_order_correct": np.nan}
        if all(rid in hidx for rid in ids):
            h = sample_h[[hidx[rid] for rid in ids]]
            hnear = float(np.linalg.norm(h[0] - h[1]))
            hfar = float(np.linalg.norm(h[0] - h[2]))
            row.update(H_near_distance=hnear, H_far_distance=hfar, H_order_correct=int(hnear < hfar))
        rows.append(row)
    return rows


def _decode_csr(f, w, bias, batch_size=32):
    """Apply the exact SAE decoder without a feature-by-feature Gram matrix."""
    if f.shape[1] != w.shape[1] or w.shape[0] != len(bias):
        raise ValueError("SAE decoder dimensions do not match sparse feature codes")
    recon = np.empty((f.shape[0], len(bias)), dtype=np.float32)
    for start in range(0, f.shape[0], batch_size):
        piece = f[start:start + batch_size].tocoo()
        block = np.broadcast_to(bias, (piece.shape[0], len(bias))).copy()
        np.add.at(block, piece.row, (w[:, piece.col].T * piece.data[:, None]))
        recon[start:start + batch_size] = block
    return recon


def _decode_sample(collection, layer, ids, h, f, limit, output_path):
    from huggingface_hub import try_to_load_from_cache
    repo = collection["repository"]
    ck = repo["checkpoints"][str(layer)]
    path = try_to_load_from_cache(repo["repo"], ck["filename"], revision=repo["revision"])
    if not isinstance(path, str):
        return {"status": "checkpoint_not_cached", "rows": 0}
    weights = load_checkpoint(path, ck["sha256"])
    w = weights["W_dec"].numpy()
    bias = weights["b_dec"].numpy()
    n = min(limit, len(ids))
    selected = stable_order(range(len(ids)), 1729)[:n]
    selected = np.asarray(selected, dtype=int)
    fs = f[selected]
    recon = _decode_csr(fs, w, bias)
    raw = h[selected]
    residual = raw - recon
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, ids=np.asarray([ids[i] for i in selected]),
                        H=raw, H_hat=recon, E=residual,
                        checkpoint_sha256=np.asarray(ck["sha256"]))
    err = np.linalg.norm(residual, axis=1) / np.maximum(np.linalg.norm(raw, axis=1), 1e-30)
    return {"status": "complete", "rows": n, "mean_relative_error": float(err.mean()),
            "median_relative_error": float(np.median(err)),
            "raw_distance_median": float(np.median(np.linalg.norm(raw[1:] - raw[:-1], axis=1))),
            "reconstruction_distance_median": float(np.median(np.linalg.norm(recon[1:] - recon[:-1], axis=1))),
            "residual_distance_median": float(np.median(np.linalg.norm(residual[1:] - residual[:-1], axis=1)))}


def linear_stage(run, collection, config, profile):
    selection = sample_readouts(run, collection, config, profile)
    layers = config["smoke_layers"] if profile == "smoke" else config["core_layers"]
    summary, predictions, metrics, candidates, dimensions, decode, activity_summary, contrasts, hue_rows = [], [], [], [], [], [], [], [], []
    previous = {}
    for layer in layers:
        if collection["manifest"]["layers"][f"main:{layer}"]["completed"] == 0:
            continue
        keys, full_f, diagnostics = feature_matrix(collection, layer)
        lookup = {key: i for i, key in enumerate(keys)}
        readout_types = {r["readout_id"]: r["readout_type"] for r in collection["readouts"]}
        for activity_readout in ("entity", "final"):
            aidx = np.asarray([i for i, key in enumerate(keys) if readout_types[key] == activity_readout])
            activity = sparse_activity(full_f[aidx], diagnostics["relative_error"][aidx])
            activity_path = run / "activity" / f"layer{layer}_{activity_readout}.npz"
            activity_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(activity_path, **activity,
                                checkpoint_sha256=collection["repository"]["checkpoints"][str(layer)]["sha256"])
            activity_summary.append({"layer": layer, "readout": activity_readout, "rows": len(aidx),
                                     "features_seen": int(np.sum(activity["fires"] > 0)),
                                     "median_relative_error": float(np.median(diagnostics["relative_error"][aidx])),
                                     "mean_active_features": float(np.mean(diagnostics["nnz"][aidx]))})
        trace_keys, trace_f, trace_diag = feature_matrix(collection, layer, "trace")
        if trace_keys:
            trace_activity = sparse_activity(trace_f, trace_diag["relative_error"])
            np.savez_compressed(run / "activity" / f"layer{layer}_trace.npz", **trace_activity)
            activity_summary.append({"layer": layer, "readout": "trace", "rows": len(trace_keys),
                                     "features_seen": int(np.sum(trace_activity["fires"] > 0)),
                                     "median_relative_error": float(np.median(trace_diag["relative_error"])),
                                     "mean_active_features": float(np.mean(trace_diag["nnz"]))})
        for readout in ("entity", "final"):
            ids = (selection[readout]["train"] + selection[readout]["validation"] +
                   selection[readout]["test"] + selection[readout]["challenge"])
            ids = [rid for rid in ids if rid in lookup]
            if len(ids) < 10:
                continue
            fit_n = sum(rid in set(selection[readout]["train"]) for rid in ids)
            train_idx = np.arange(fit_n)
            x_f = full_f[[lookup[rid] for rid in ids]]
            h = _raw_matrix(collection, layer, ids)
            contrasts.extend(_contrasts(collection, layer, readout, keys, full_f, ids, h))
            hue_rows.extend(_hue_triplets(collection, layer, readout, keys, full_f, ids, h))
            _coactivation(run, layer, readout, x_f, fit_n)
            candidates.extend(_feature_candidates(x_f, ids, collection, layer, readout, config["minimum_feature_fires"]))
            decoded = _decode_sample(collection, layer, ids, h, x_f, config["checkpoint_decode_rows"],
                                     run / "reconstruction" / f"layer{layer}_{readout}_aligned.npz")
            decode.append({"layer": layer, "readout": readout, **decoded})
            for view, x in (("H", h), ("F", x_f)):
                fit = _pca_fit(x, train_idx, config["pca_components"], config["seed"])
                summary.append(_save_pca(run, layer, readout, view, ids, fit, train_idx, x))
                if len(ids) > 40:
                    cohort = x[:min(len(train_idx), 600)]
                    if sparse.issparse(cohort):
                        cohort = cohort.toarray()
                    dimensions.append({"layer": layer, "readout": readout, "view": view,
                                       **intrinsic_knn(cohort, ks=(5, 10, 20)),
                                       "input": "original H or sparse F (densified bounded cohort)"})
                views = [(view, x), (f"{view}_PCA2", fit["all_scores"][:, :2]),
                         (f"{view}_PCA16", fit["all_scores"][:, :16]),
                         (f"{view}_PCA64", fit["all_scores"][:, :64])]
                if view == "F":
                    svd = TruncatedSVD(n_components=min(64, x.shape[0] - 1), random_state=config["seed"]).fit(x[train_idx])
                    rp = SparseRandomProjection(n_components=64, dense_output=True, random_state=config["seed"]).fit(x[train_idx])
                    views.extend([("F_SVD64", svd.transform(x)), ("F_RP64", rp.transform(x))])
                    np.savez_compressed(run / "linear" / f"layer{layer}_{readout}_svd_rp.npz",
                                        ids=np.asarray(ids), svd_components=svd.components_, svd_coordinates=svd.transform(x),
                                        rp_coordinates=rp.transform(x), svd_train_ids=np.asarray(ids[:fit_n]))
                if profile == "smoke":
                    qualities = ["size"]
                else:
                    qualities = _qualities(collection)
                for quality in qualities:
                    for view_name, vx in views:
                        p, m = _predict_one(vx, ids, collection, quality, view_name, layer, readout, profile)
                        predictions.extend(p)
                        metrics.extend(m)
            if readout == "entity" and layer in (1, 11, 27) and "entity" in previous:
                matched = [rid for rid in ids if rid in previous["entity"][0]]
                if len(matched) >= 10:
                    old_lookup = {rid: i for i, rid in enumerate(previous["entity"][0])}
                    now_lookup = {rid: i for i, rid in enumerate(ids)}
                    summary.append({"layer": layer, "readout": readout, "view": "H_cross_layer_CKA",
                                    "previous_layer": previous["entity"][2],
                                    "n_matched": len(matched),
                                    "linear_cka": linear_cka(previous["entity"][1][[old_lookup[r] for r in matched]],
                                                             h[[now_lookup[r] for r in matched]])})
            previous[readout] = (ids, h, layer)
        print(f"SAE analysis linear layer {layer}: {len(keys)} feature rows", flush=True)
        _save_rows(run / "linear" / "pca_summary.csv", summary)
        _save_rows(run / "predictions" / "metrics.csv", metrics)
        _save_rows(run / "predictions" / "rows.csv", predictions)
        _save_rows(run / "features" / "candidates.csv", candidates)
        _save_rows(run / "dimension" / "estimates.csv", dimensions)
        _save_rows(run / "reconstruction" / "decoder_checks.csv", decode)
        _save_rows(run / "activity" / "summary.csv", activity_summary)
        _save_rows(run / "contrasts" / "matched.csv", contrasts)
        _save_rows(run / "hue" / "triplets.csv", hue_rows)


def method_registry(run):
    import importlib.util
    available = {name: importlib.util.find_spec(name) is not None
                 for name in ("umap", "pacmap", "trimap", "phate", "ripser")}
    methods = [
        ("PCA", "core", "H or F; centered train-only", True, "linear"),
        ("TruncatedSVD", "core", "F uncentered", True, "linear"),
        ("SparseRandomProjection", "core", "F nonadaptive", True, "linear"),
        ("UMAP", "core", "bounded, train-fitted PCA64 input", True, "nonlinear"),
        ("t-SNE", "core", "bounded descriptive cohort", False, "nonlinear"),
        ("PaCMAP", "extended", "bounded dense cohort", False, "nonlinear"),
        ("TriMap", "extended", "bounded dense cohort", False, "nonlinear"),
        ("diffusion_maps", "extended", "bounded graph cohort", False, "nonlinear"),
        ("PHATE", "extended", "bounded dense cohort", False, "nonlinear"),
        ("Isomap", "extended", "bounded dense cohort", True, "nonlinear"),
        ("LLE", "extended", "bounded dense cohort", False, "nonlinear"),
        ("SpectralEmbedding", "extended", "bounded dense cohort", False, "nonlinear"),
        ("MDS", "extended", "bounded dense cohort", False, "nonlinear"),
        ("KernelPCA", "extended", "bounded dense cohort", True, "nonlinear"),
        ("SparsePCA", "extended", "bounded dense cohort", True, "factor"),
        ("NMF", "extended", "nonnegative F; uncentered", True, "factor"),
        ("ICA", "extended", "bounded dense cohort", True, "factor"),
        ("FactorAnalysis", "extended", "bounded dense cohort", True, "factor"),
        ("PLS", "supervised", "training labels; separate comparison", True, "supervised")]
    _save_rows(run / "methods.csv", [
        {"method": name, "tier": tier, "input": inputs, "out_of_sample": oos, "class": cls,
         "installed": available.get(name.lower().replace("-", ""), True),
         "status": "implemented" if name in {"PCA", "TruncatedSVD", "SparseRandomProjection", "UMAP", "t-SNE"} else "adapter_pending"}
        for name, tier, inputs, oos, cls in methods])


def nonlinear_stage(run, collection, config, profile):
    from sklearn.manifold import TSNE
    try:
        import umap
    except ImportError:
        umap = None
    layers = config["smoke_layers"][:1] if profile == "smoke" else config["nonlinear_layers"]
    selection = sample_readouts(run, collection, config, profile)
    metrics = []
    for layer in layers:
        for readout in ("entity", "final"):
            limit = config["nonlinear_rows"]
            quotas = {"train": limit // 2, "validation": limit // 6,
                      "test": limit // 6, "challenge": limit - (limit // 2 + 2 * (limit // 6))}
            ids = []
            for split, quota in quotas.items():
                ids.extend(selection[readout][split][:quota])
            for view in ("H", "F"):
                path = run / "linear" / f"layer{layer}_{readout}_{view}_pca.npz"
                if not path.exists():
                    continue
                with np.load(path) as d:
                    lookup = {str(rid): i for i, rid in enumerate(d["ids"].tolist())}
                    chosen = [rid for rid in ids if rid in lookup]
                    x = d["coordinates"][[lookup[rid] for rid in chosen], :64]
                    train_set = set(d["train_ids"].tolist())
                if len(chosen) < 70:
                    continue
                fit_idx = np.asarray([i for i, rid in enumerate(chosen) if rid in train_set])
                for method in ("UMAP", "t-SNE"):
                    if method == "UMAP" and umap is None:
                        metrics.append({"layer": layer, "readout": readout, "view": view, "method": method,
                                        "status": "skipped_uninstalled"})
                        continue
                    for ndim in (2, 3):
                        seeds = config["stochastic_seeds"] if ndim == 2 else config["stochastic_seeds"][:1]
                        for seed in seeds:
                            if method == "UMAP":
                                reducer = umap.UMAP(n_components=ndim, n_neighbors=30, min_dist=.1,
                                                    random_state=seed, transform_seed=seed)
                                reducer.fit(x[fit_idx])
                                coords = reducer.transform(x)
                                mode = "heldout_transform"
                            else:
                                reducer = TSNE(n_components=ndim, perplexity=min(30, (len(x) - 1) / 3),
                                               random_state=seed, init="pca", learning_rate="auto")
                                coords = reducer.fit_transform(x)
                                mode = "descriptive_all_splits"
                            result = embedding_metrics(x, coords, config["neighbourhood_sizes"], seed)
                            result.update({"layer": layer, "readout": readout, "view": view, "method": method,
                                           "dimensions": ndim, "seed": seed, "mode": mode, "status": "complete"})
                            metrics.append(result)
                            output = run / "embeddings" / f"layer{layer}_{readout}_{view}_{method}_{ndim}d_seed{seed}.npz"
                            output.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(output, ids=np.asarray(chosen), coordinates=coords.astype(np.float32),
                                                input_pca64=x, train_ids=np.asarray([chosen[i] for i in fit_idx]))
        print(f"SAE analysis nonlinear layer {layer}", flush=True)
        _save_rows(run / "embeddings" / "faithfulness.csv", metrics)


def token_stage(run, collection, config, profile):
    layers = config["smoke_layers"] if profile == "smoke" else config["core_layers"]
    traces = {t["trace_id"]: t for t in jsonl(collection["source"] / "traces.jsonl")}
    cases = {r["metadata"]["case_id"]: r["metadata"] for r in collection["records"] if r["kind"] == "case"}
    readouts = {r["text_key"]: r for r in collection["readouts"]}
    summaries = []
    for layer in layers:
        keys, features, diag = feature_matrix(collection, layer, "trace")
        if not keys:
            summaries.append({"layer": layer, "status": "no_encoded_trace_rows"})
            continue
        groups = defaultdict(list)
        for idx, key in enumerate(keys):
            tid, position = key.rsplit(":", 1)
            if tid in traces:
                groups[tid].append((int(position), idx))
        role_stats = defaultdict(lambda: {"nnz": [], "error": [], "prompts": set(), "families": set()})
        persistence = []
        for tid, positions in groups.items():
            trace = traces[tid]
            meta = cases[trace["case_id"]]
            row = readouts[trace["text_key"]]
            text = row["text"]
            offsets = row["token_offsets"]
            target_span = meta.get("readout_char_span")
            quality_words = [q.replace("_", " ") for q in meta.get("quality_ids", [])]
            quality_spans = []
            for phrase in quality_words:
                start = text.lower().find(phrase.lower())
                if start >= 0:
                    quality_spans.append((start, start + len(phrase)))
            final_position = trace["token_count"] - 1
            seen_quality = set()
            for position, idx in sorted(positions):
                start, end = offsets[position]
                overlaps = lambda span: span is not None and start < span[1] and end > span[0]
                if overlaps(target_span):
                    role = "target_entity"
                elif any(overlaps(span) for span in quality_spans):
                    role = "literal_quality_word"
                elif position == final_position:
                    role = "final_context"
                else:
                    role = "other"
                bucket = role_stats[role]
                bucket["nnz"].append(int(diag["nnz"][idx]))
                bucket["error"].append(float(diag["relative_error"][idx]))
                bucket["prompts"].add(tid)
                bucket["families"].add(meta.get("entity_group_id", ""))
                if role == "literal_quality_word":
                    seen_quality.update(features[idx].indices.tolist())
                if position == final_position and seen_quality:
                    final_support = set(features[idx].indices.tolist())
                    persistence.append(len(seen_quality & final_support) / len(seen_quality))
        for role, stats in role_stats.items():
            summaries.append({"layer": layer, "role": role, "token_positions": len(stats["nnz"]),
                              "prompts": len(stats["prompts"]), "entity_families": len(stats["families"]),
                              "mean_active_features": float(np.mean(stats["nnz"])),
                              "median_relative_error": float(np.median(stats["error"])),
                              "quality_word_to_final_support_retention_mean": float(np.mean(persistence)) if role == "final_context" and persistence else np.nan,
                              "status": "complete"})
        _save_rows(run / "tokens" / "summary.csv", summaries)


def extended_stage(run, collection, config, profile):
    """Optional bounded adapters; each unavailable method records an explicit reason."""
    if profile != "extended":
        _save_rows(run / "extended" / "status.csv", [{"method": "all", "status": "not_requested",
                                                       "reason": "Use --profile extended for bounded optional methods"}])
        return
    from sklearn.decomposition import FastICA, FactorAnalysis, KernelPCA, NMF, SparsePCA
    from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
    selection = sample_readouts(run, collection, config, profile)
    layer = config["nonlinear_layers"][1]
    readout = "final"
    ids = selection[readout]["train"][:config["nonlinear_rows"]]
    pca_path = run / "linear" / f"layer{layer}_{readout}_F_pca.npz"
    with np.load(pca_path) as d:
        lookup = {rid: i for i, rid in enumerate(d["ids"].tolist())}
        ids = [rid for rid in ids if rid in lookup]
        dense = d["coordinates"][[lookup[rid] for rid in ids], :min(32, d["coordinates"].shape[1])]
    status = []
    methods = {
        "Isomap": lambda: Isomap(n_components=2, n_neighbors=15).fit_transform(dense),
        "LLE": lambda: LocallyLinearEmbedding(n_components=2, n_neighbors=20, method="standard").fit_transform(dense),
        "SpectralEmbedding": lambda: SpectralEmbedding(n_components=2, n_neighbors=20, random_state=config["seed"]).fit_transform(dense),
        "MDS": lambda: MDS(n_components=2, random_state=config["seed"], n_init=1, max_iter=150).fit_transform(dense),
        "KernelPCA": lambda: KernelPCA(n_components=2, kernel="rbf").fit_transform(dense),
        "SparsePCA": lambda: SparsePCA(n_components=2, random_state=config["seed"], max_iter=100).fit_transform(dense),
        "ICA": lambda: FastICA(n_components=2, random_state=config["seed"], max_iter=300).fit_transform(dense),
        "FactorAnalysis": lambda: FactorAnalysis(n_components=2, random_state=config["seed"]).fit_transform(dense),
    }
    for name, fit in methods.items():
        try:
            coords = fit()
            path = run / "extended" / f"{name}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, ids=np.asarray(ids), coordinates=np.asarray(coords, dtype=np.float32),
                                input_pca32=dense)
            status.append({"method": name, "status": "complete", "rows": len(ids),
                           **embedding_metrics(dense, coords, config["neighbourhood_sizes"])})
        except Exception as exc:
            status.append({"method": name, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    for name, module in (("PaCMAP", "pacmap"), ("TriMap", "trimap"), ("PHATE", "phate"), ("persistent_homology", "ripser")):
        try:
            __import__(module)
            status.append({"method": name, "status": "available_not_executed",
                           "reason": "Optional adapter requires explicit package-specific version review"})
        except ImportError:
            status.append({"method": name, "status": "skipped_uninstalled", "reason": f"Install {module} to enable"})
    # NMF requires the original nonnegative codes, not centered PCA coordinates.
    keys, f, _ = feature_matrix(collection, layer)
    fidx = {rid: i for i, rid in enumerate(keys)}
    chosen_f = f[[fidx[rid] for rid in ids]]
    try:
        nmf = NMF(n_components=2, init="nndsvda", random_state=config["seed"], max_iter=250)
        coords = nmf.fit_transform(chosen_f)
        np.savez_compressed(run / "extended" / "NMF.npz", ids=np.asarray(ids),
                            coordinates=coords.astype(np.float32), components=nmf.components_.astype(np.float32))
        status.append({"method": "NMF", "status": "complete", "rows": len(ids),
                       "input": "nonnegative uncentered F", "reconstruction_err": float(nmf.reconstruction_err_)})
    except Exception as exc:
        status.append({"method": "NMF", "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    _save_rows(run / "extended" / "status.csv", status)


def run_analysis(config_path, profile="core", run_dir=None, resume=False, stage="all"):
    run, collection, config = prepare(config_path, profile, run_dir, resume)
    stages = [("audit", lambda: audit(run, collection)),
              ("methods", lambda: method_registry(run)),
              ("linear", lambda: linear_stage(run, collection, config, profile)),
              ("nonlinear", lambda: nonlinear_stage(run, collection, config, profile)),
              ("tokens", lambda: token_stage(run, collection, config, profile)),
              ("extended", lambda: extended_stage(run, collection, config, profile))]
    for name, function in stages:
        if stage in ("all", name):
            _stage(run, name, function)
    state = json.loads((run / "state.json").read_text())
    if all(state["stages"].get(name) == "complete" for name, _ in stages):
        state["status"] = "complete"
        atomic_json(run / "state.json", state)
    return run
