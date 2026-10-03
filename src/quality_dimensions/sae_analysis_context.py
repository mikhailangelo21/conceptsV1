"""Factorial selectivity and observational links to saved V3 behaviour scores."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

from .sae_analysis import _save_rows
from .sae_analysis_core import checksum, eligible_fit, feature_matrix, read_collection
from .util import atomic_json


def _factorial(run, collection):
    all_rows = pd.read_csv(run / "contrasts" / "matched.csv")
    metadata = {}
    for record in collection["records"]:
        if record["kind"] == "case" and record["metadata"]["case_type"] == "factorial":
            metadata[record["readout_id"]] = record["metadata"]
    details, summary = [], []
    for layer in (1, 11, 27):
        keys, f, _ = feature_matrix(collection, layer)
        lookup = {rid: i for i, rid in enumerate(keys)}
        table = all_rows[(all_rows["layer"] == layer) & (all_rows["readout"] == "final") &
                         (all_rows["kind"] == "factorial_one_axis_increase")].copy()
        table["domain"] = table["low_readout_id"].map(
            lambda rid: metadata.get(rid, {}).get("factorial_domain"))
        table = table[table["domain"].notna()]
        for domain, domain_rows in table.groupby("domain"):
            directions = {}
            train_counts = {}
            for quality, group in domain_rows.groupby("quality"):
                train = [r for r in group.itertuples()
                         if r.entity_split == "train" and r.template_split == "train"
                         and eligible_fit(metadata[r.low_readout_id])
                         and eligible_fit(metadata[r.high_readout_id])
                         and r.low_readout_id in lookup and r.high_readout_id in lookup]
                if len(train) < 3:
                    continue
                delta = sparse.vstack([f[lookup[r.high_readout_id]] - f[lookup[r.low_readout_id]]
                                       for r in train])
                direction = np.asarray(delta.mean(axis=0)).ravel()
                norm = float(np.linalg.norm(direction))
                if norm == 0:
                    continue
                directions[quality] = direction / norm
                train_counts[quality] = len(train)
            if len(directions) < 2:
                continue
            qualities = sorted(directions)
            direction_matrix = np.stack([directions[q] for q in qualities], axis=1)
            for row in domain_rows.itertuples():
                if row.low_readout_id not in lookup or row.high_readout_id not in lookup:
                    continue
                target = row.quality
                if target not in directions:
                    continue
                delta = f[lookup[row.high_readout_id]] - f[lookup[row.low_readout_id]]
                projections = np.asarray(delta @ direction_matrix).ravel()
                target_projection = float(projections[qualities.index(target)])
                off = [float(abs(projections[i])) for i, q in enumerate(qualities) if q != target]
                other_levels = {q: metadata[row.low_readout_id]["targets"].get(q)
                                for q in qualities if q != target}
                details.append({"layer": layer, "domain": domain, "target_quality": target,
                                "entity_split": row.entity_split, "entity_family": row.entity_family,
                                "template_split": row.template_split,
                                "combination_split": metadata[row.low_readout_id]["combination_split"],
                                "contrast_id": row.contrast_id, "train_pairs_for_direction": train_counts[target],
                                "target_signed_projection": target_projection,
                                "largest_off_target_abs_projection": max(off),
                                "target_larger_than_off_target": int(target_projection > max(off)),
                                "target_correctly_signed": int(target_projection > 0),
                                "other_quality_levels": other_levels})
            test_rows = [r for r in details if r["layer"] == layer and r["domain"] == domain
                         and r["entity_split"] == "test"]
            for target in qualities:
                part = [r for r in test_rows if r["target_quality"] == target]
                if part:
                    cosine = {q: float(directions[target] @ directions[q]) for q in qualities if q != target}
                    summary.append({"layer": layer, "domain": domain, "target_quality": target,
                                    "test_contrasts": len(part),
                                    "test_entity_families": len({r["entity_family"] for r in part}),
                                    "target_correct_sign_fraction": float(np.mean([r["target_correctly_signed"] for r in part])),
                                    "target_exceeds_all_off_targets_fraction": float(np.mean([r["target_larger_than_off_target"] for r in part])),
                                    "median_target_projection": float(np.median([r["target_signed_projection"] for r in part])),
                                    "median_largest_off_target_abs": float(np.median([r["largest_off_target_abs_projection"] for r in part])),
                                    "train_direction_cosine_with_other_qualities": cosine})
        print(f"Factorial selectivity layer {layer}", flush=True)
        _save_rows(run / "factorial" / "contrast_projections.csv", details)
        _save_rows(run / "factorial" / "summary.csv", summary)


def _behaviour(run, collection):
    from .sae_collection import _source_v3_run
    scores = pd.read_csv(_source_v3_run(collection["source"]) / "behavior_scores.csv")
    comparisons = {r["metadata"]["comparison_id"]: r["readout_id"]
                   for r in collection["records"] if r["kind"] == "comparison"}
    scores["readout_id"] = scores["comparison_id"].map(comparisons)
    rows = []
    for layer in (1, 11, 27):
        keys, f, diag = feature_matrix(collection, layer)
        lookup = {rid: i for i, rid in enumerate(keys)}
        for score in scores.itertuples():
            rid = score.readout_id
            if rid not in lookup:
                continue
            i = lookup[rid]
            rows.append({"layer": layer, "comparison_id": score.comparison_id,
                         "quality_id": score.quality_id, "format": score.format,
                         "entity_family": score.entity_group_id, "entity_split": score.entity_split,
                         "template_split": score.template_split,
                         "correct_margin": score.correct_margin, "correct": score.correct,
                         "feature_l2_norm": float(np.sqrt(f[i].power(2).sum())),
                         "feature_nnz": int(diag["nnz"][i]),
                         "relative_reconstruction_error": float(diag["relative_error"][i])})
    _save_rows(run / "behavior" / "comparison_links.csv", rows)
    table = pd.DataFrame(rows)
    summary = []
    for (layer, fmt), part in table.groupby(["layer", "format"]):
        for variable in ("feature_l2_norm", "feature_nnz", "relative_reconstruction_error"):
            rho = spearmanr(part[variable], part["correct_margin"]).statistic
            summary.append({"layer": layer, "format": fmt, "feature_summary": variable,
                            "comparisons": len(part), "entity_families": part["entity_family"].nunique(),
                            "spearman_with_correct_margin": float(rho),
                            "interpretation": "observational; prompt form and difficulty may confound"})
    _save_rows(run / "behavior" / "observational_summary.csv", summary)


def context_checks(run_dir):
    run = Path(run_dir).expanduser().resolve()
    config = json.loads((run / "config.json").read_text())
    source = (Path(__file__).resolve().parents[2] / config["source_run"]).resolve()
    collection = read_collection(source)
    state_path = run / "context_state.json"
    digest = checksum(Path(__file__))
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "source_sha256": digest, "stages": {}, "status": "partial"}
    if state["source_sha256"] != digest:
        raise ValueError("Context analysis source changed; use a new run")
    for name, function in (("factorial", _factorial), ("behaviour", _behaviour)):
        if state["stages"].get(name) == "complete":
            continue
        state["current_stage"] = name
        atomic_json(state_path, state)
        function(run, collection)
        state["stages"][name] = "complete"
        state.pop("current_stage", None)
        atomic_json(state_path, state)
    state["status"] = "complete"
    atomic_json(state_path, state)
    return run
