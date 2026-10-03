"""Grouped linear analyses for quality-suite v3 cached activations."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from scipy import sparse

from .util import atomic_json, atomic_npz


def _rho(y, pred):
    if len(y) < 3 or np.ptp(y) == 0 or np.ptp(pred) == 0:
        return None
    value = float(spearmanr(y, pred).statistic)
    return value if np.isfinite(value) else None


def _score(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return {"rho": _rho(y, pred), "mae": float(np.abs(y - pred).mean()),
            "r2": float(1 - np.square(y - pred).sum() / np.square(y - y.mean()).sum()) if np.ptp(y) else None}


def _circular_score(y, pred):
    true_angle = np.arctan2(y[:, 0], y[:, 1])
    pred_angle = np.arctan2(pred[:, 0], pred[:, 1])
    error = np.abs(np.angle(np.exp(1j * (pred_angle - true_angle))))
    return {"circular_mae_degrees": float(np.degrees(error).mean()),
            "mean_resultant_cosine": float(np.cos(error).mean())}


def _ridge_cv(x, y, groups, alphas, seed=0):
    """Tune all preprocessing and regularization inside grouped training folds."""
    if not sparse.issparse(x):
        x = np.asarray(x)
    y, groups = np.asarray(y), np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        alpha = float(alphas[0])
    else:
        folds = min(3, len(unique))
        splitter = GroupKFold(folds)
        scores = []
        for alpha in alphas:
            losses = []
            for train, valid in splitter.split(x, y, groups):
                model = make_pipeline(StandardScaler(with_mean=not sparse.issparse(x)), Ridge(alpha=alpha, solver="lsqr"))
                model.fit(x[train], y[train])
                losses.append(float(np.square(y[valid] - model.predict(x[valid])).mean()))
            scores.append((float(np.mean(losses)), float(alpha)))
        alpha = min(scores)[1]
    model = make_pipeline(StandardScaler(with_mean=not sparse.issparse(x)), Ridge(alpha=alpha, solver="lsqr"))
    model.fit(x, y)
    return model, alpha


def _partition(row):
    axes = []
    if row["entity_split"] != "train": axes.append("entity_" + row["entity_split"])
    if row["template_split"] != "train": axes.append("template")
    if row["case_type"] == "factorial" and row["combination_split"] != "train": axes.append("combination")
    split = "train" if not axes else "+".join(sorted(axes))
    return split + "|" + row["case_type"] + "|" + row["template_id"]


def _eligible_fit(row):
    return (row["case_type"] != "numeric" and row["entity_split"] == "train"
            and row["template_split"] == "train"
            and (row["case_type"] != "factorial" or row["combination_split"] == "train"))


def _target_rows(cases):
    result = defaultdict(list)
    for row in cases:
        if row["case_type"] == "numeric":
            continue
        if "hue_sin" in row["targets"] and "hue_cos" in row["targets"]:
            result["hue"].append(row)
        for quality, value in row["targets"].items():
            if quality not in {"hue_sin", "hue_cos"} and not quality.endswith("__value"):
                result[("lexical__" if row["case_type"] == "lexical" else "") + quality].append(row)
    return result


def _xy(rows, activation, readout, layer_index, quality):
    x = np.stack([activation[(r["case_id"], readout)][layer_index] for r in rows])
    if quality == "hue":
        y = np.array([[r["targets"]["hue_sin"], r["targets"]["hue_cos"]] for r in rows], float)
    else:
        y = np.array([r["targets"][quality.removeprefix("lexical__")] for r in rows], float)
    return x, y


def analyze_suite(run, suite, activation, static_embeddings):
    run = Path(run); config = suite["config"]; blocks = config["extraction_block_numbers"]
    targets = _target_rows(suite["cases"]); metrics = []; predictions = []; selections = {}
    alphas = config["ridge_alphas"]; search = []; fitted_models = {}
    for quality, rows in targets.items():
        train_rows = [r for r in rows if _eligible_fit(r)]
        validation_rows = [r for r in rows if r["entity_split"] == "validation" and r["template_split"] == "train"
                           and (r["case_type"] != "factorial" or r["combination_split"] == "train")]
        if len(train_rows) < 4 or len({r["entity_group_id"] for r in train_rows}) < 2:
            metrics.append({"quality": quality, "status": "insufficient_training_groups",
                            "n_rows": len(train_rows), "n_groups": len({r['entity_group_id'] for r in train_rows})})
            continue
        best = None; fitted = {}
        for readout in ("entity", "final"):
            for layer_index, block in enumerate(blocks):
                x, y = _xy(train_rows, activation, readout, layer_index, quality)
                groups = [r["entity_group_id"] for r in train_rows]
                model, alpha = _ridge_cv(x, y, groups, alphas)
                fitted[(readout, block)] = (model, alpha)
                if validation_rows:
                    vx, vy = _xy(validation_rows, activation, readout, layer_index, quality)
                    vp = model.predict(vx)
                    rho = _rho(vy, vp) if quality != "hue" else None
                    score = -_circular_score(vy, vp)["circular_mae_degrees"] if quality == "hue" else (rho if rho is not None else -2)
                else:
                    score = -float(alpha)
                candidate = (score, -block, readout, block)
                search.append(dict(quality=quality, readout=readout, block=block, alpha=alpha,
                                   validation_score=score, validation_rows=len(validation_rows)))
                if best is None or candidate > best:
                    best = candidate
        selected_readout, selected_block = best[2], best[3]
        model, alpha = fitted[(selected_readout, selected_block)]
        fitted_models[quality] = model
        selections[quality] = {"readout": selected_readout, "block": selected_block, "ridge_alpha": alpha,
                               "selection_axis": "validation entities with training templates/combinations",
                               "training_groups": sorted({r["entity_group_id"] for r in train_rows})}
        layer_index = blocks.index(selected_block)
        xtrain, ytrain = _xy(train_rows, activation, selected_readout, layer_index, quality)

        # Mean-difference is separately estimated, with no validation/test labels.
        if quality != "hue":
            lo, hi = np.quantile(ytrain, [1/3, 2/3])
            direction = xtrain[ytrain >= hi].mean(0) - xtrain[ytrain <= lo].mean(0)
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm else direction
        else:
            direction = None

        # Train-only paired contrast SVD and rank restriction.
        case_lookup = {r["case_id"]: r for r in suite["cases"]}
        deltas = []
        for contrast in suite["contrasts"]:
            if contrast["quality_id"] != quality or contrast["low_case_id"] not in case_lookup or contrast["high_case_id"] not in case_lookup:
                continue
            low, high = case_lookup[contrast["low_case_id"]], case_lookup[contrast["high_case_id"]]
            if _eligible_fit(low) and _eligible_fit(high):
                deltas.append(activation[(high["case_id"], selected_readout)][layer_index]
                              - activation[(low["case_id"], selected_readout)][layer_index])
        subspace_model = None; rank_trials = []
        if deltas and validation_rows:
            matrix = np.stack(deltas)
            _, _, vh = np.linalg.svd(matrix, full_matrices=False)
            max_rank = min(np.linalg.matrix_rank(matrix), max(config["subspace_ranks"]))
            valid_ranks = [r for r in config["subspace_ranks"] if r <= max_rank]
            vx, vy = _xy(validation_rows, activation, selected_readout, layer_index, quality)
            for rank in valid_ranks:
                basis = vh[:rank]
                # Alpha already selected by grouped folds for the full linear model;
                # basis is trained on training contrasts and rank uses outer validation only.
                sa = alpha
                sm = make_pipeline(StandardScaler(), Ridge(alpha=sa, solver='lsqr')).fit(xtrain @ basis.T, ytrain)
                loss = float(np.square(sm.predict(vx @ basis.T) - vy).mean())
                rank_trials.append(dict(rank=rank, alpha=sa, validation_loss=loss))
            if rank_trials:
                chosen = min(rank_trials, key=lambda r: (r['validation_loss'], r['rank']))
                chosen_rank = chosen['rank']; subspace = vh[:chosen_rank]
                subspace_model = make_pipeline(StandardScaler(), Ridge(alpha=chosen['alpha'], solver='lsqr')).fit(xtrain @ subspace.T, ytrain)
            else:
                chosen_rank, subspace = 0, np.empty((0, xtrain.shape[1]))
        else:
            chosen_rank, subspace = 0, np.empty((0, xtrain.shape[1]))
        selections[quality].update(contrast_svd_rank=chosen_rank, training_contrasts=len(deltas),
                                   rank_limit=min(len(deltas), xtrain.shape[1]) if deltas else 0, rank_trials=rank_trials)
        atomic_json(run / 'model_selection.json', selections)

        for partition in sorted({_partition(r) for r in rows}):
            eval_rows = [r for r in rows if _partition(r) == partition]
            if not eval_rows:
                continue
            ex, ey = _xy(eval_rows, activation, selected_readout, layer_index, quality)
            pred = model.predict(ex)
            common = {"quality": quality, "partition": partition, "readout": selected_readout,
                      "block": selected_block, "n_rows": len(eval_rows),
                      "n_groups": len({r['entity_group_id'] for r in eval_rows}), "method": "ridge"}
            common.update(_circular_score(ey, pred) if quality == "hue" else _score(ey, pred))
            metrics.append(common)
            if subspace_model is not None:
                sp = subspace_model.predict(ex @ subspace.T)
                metrics.append({**common, 'method': 'contrast_svd_ridge', **(_circular_score(ey, sp) if quality == 'hue' else _score(ey, sp))})
            if quality != "hue":
                md = (ex - xtrain.mean(0)) @ direction
                metrics.append({**{k: v for k, v in common.items() if k not in {"rho", "mae", "r2", "method"}},
                                "method": "mean_difference", "rho": _rho(ey, md)})
                for row, target, value in zip(eval_rows, ey, pred):
                    predictions.append({"case_id": row["case_id"], "quality": quality, "partition": partition,
                                        "target": float(target), "prediction": float(value), "method": "ridge"})

        # Baselines use the identical fit rows and group-aware tuning.
        if quality != "hue":
            baseline_sets = {
                "token_length": np.array([[len(r["text"]), suite['token_counts'][r['case_id']]] for r in rows], float),
                "static_embedding": np.stack([static_embeddings[r["case_id"]] for r in rows]),
            }
            train_indices = [rows.index(r) for r in train_rows]
            for name, all_x in baseline_sets.items():
                baseline, balpha = _ridge_cv(all_x[train_indices], ytrain, [r["entity_group_id"] for r in train_rows], alphas)
                for partition in sorted({_partition(r) for r in rows}):
                    indices = [i for i, r in enumerate(rows) if _partition(r) == partition]
                    by = np.array([rows[i]["targets"][quality.removeprefix('lexical__')] for i in indices])
                    bp = baseline.predict(all_x[indices])
                    metrics.append({"quality": quality, "partition": partition, "readout": "n/a", "block": 0,
                                    "n_rows": len(indices), "n_groups": len({rows[i]['entity_group_id'] for i in indices}),
                                    "method": name, "ridge_alpha": balpha, **_score(by, bp)})
            vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
            train_text = [r["text"] for r in train_rows]
            unigram_x = vectorizer.fit_transform(train_text)
            # Vocabulary and IDF are learned afresh inside every grouped fold.
            losses = []
            for a in alphas:
                fold_losses = []
                for ti, vi in GroupKFold(min(3, len(set(r['entity_group_id'] for r in train_rows)))).split(train_text, ytrain, [r['entity_group_id'] for r in train_rows]):
                    pipe = make_pipeline(TfidfVectorizer(), Ridge(alpha=a, solver='lsqr'))
                    pipe.fit([train_text[i] for i in ti], ytrain[ti])
                    fold_losses.append(np.square(pipe.predict([train_text[i] for i in vi]) - ytrain[vi]).mean())
                losses.append((float(np.mean(fold_losses)), a))
            ualpha = min(losses)[1]
            unigram = Ridge(alpha=ualpha, solver='lsqr').fit(unigram_x, ytrain)
            for partition in sorted({_partition(r) for r in rows}):
                eval_rows = [r for r in rows if _partition(r) == partition]
                uy = np.array([r["targets"][quality.removeprefix('lexical__')] for r in eval_rows])
                up = unigram.predict(vectorizer.transform([r["text"] for r in eval_rows]))
                metrics.append({"quality": quality, "partition": partition, "readout": "n/a", "block": 0,
                                "n_rows": len(eval_rows), "n_groups": len({r['entity_group_id'] for r in eval_rows}),
                                "method": "unigram_tfidf", "ridge_alpha": ualpha, **_score(uy, up)})

        atomic_npz(run / "fits" / f"{quality}.npz", block=np.array(selected_block),
                   readout=np.array(selected_readout), contrast_subspace=subspace,
                   mean_direction=np.array([]) if direction is None else direction,
                   ridge_coef=model[-1].coef_, ridge_intercept=model[-1].intercept_,
                   feature_mean=model[0].mean_, feature_scale=model[0].scale_)

    # Adjacent directions: report cosine consistency without pooling qualities/ranges.
    adjacent = []
    for quality, rows in targets.items():
        if quality == "hue" or quality not in selections:
            continue
        selection = selections[quality]; readout = selection["readout"]; li = blocks.index(selection["block"])
        fit = [r for r in rows if _eligible_fit(r)]
        target_key = quality.removeprefix('lexical__')
        levels = sorted({r["targets"][target_key] for r in fit})
        directions = []
        for low, high in zip(levels, levels[1:]):
            lo = [activation[(r["case_id"], readout)][li] for r in fit if r["targets"][target_key] == low]
            hi = [activation[(r["case_id"], readout)][li] for r in fit if r["targets"][target_key] == high]
            if lo and hi:
                d = np.mean(hi, 0) - np.mean(lo, 0); d /= np.linalg.norm(d) or 1
                directions.append((low, high, d))
        for i in range(len(directions)):
            for j in range(i + 1, len(directions)):
                adjacent.append({"quality": quality, "block": selection["block"], "range_a": f"{directions[i][0]}->{directions[i][1]}",
                                 "range_b": f"{directions[j][0]}->{directions[j][1]}",
                                 "cosine": float(directions[i][2] @ directions[j][2])})

    pd.DataFrame(metrics).to_csv(run / "linear_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(run / "linear_predictions.csv", index=False)
    pd.DataFrame(adjacent).to_csv(run / "adjacent_direction_alignment.csv", index=False)
    atomic_json(run / "model_selection.json", selections)
    atomic_json(run / 'linear_search.json', search)
    from .suite_diagnostics import diagnostics
    diagnostics(run, suite, activation, selections, fitted_models)
    from .suite_exploration import nonlinear_exploration
    nonlinear_exploration(run, suite, activation, selections, fitted_models)
    atomic_json(run / "analysis_complete.json", {"qualities_attempted": sorted(targets),
                "qualities_selected": sorted(selections), "nonlinear_enabled": bool(config["nonlinear"]["enabled"]),
                "nonlinear_status": "disabled_by_configuration" if not config["nonlinear"]["enabled"] else "complete",
                "fit_rule": "training entities and templates; factorial training combinations; numeric challenge excluded",
                "regularization": "three-fold grouped training-entity CV; preprocessing refit in every fold"})
    return metrics
