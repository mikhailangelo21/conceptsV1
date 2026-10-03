"""Same-site Euclidean/covariance comparisons; all fitting uses V3 training rows."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.model_selection import GroupKFold

from .suite_analysis import _eligible_fit, _target_rows, _partition, _score, _circular_score, _ridge_cv
from .util import atomic_json, atomic_npz, digest

SITE = "final_post_normalization_final_prompt_token"
INCREASES = {"adjacent_increase", "full_range_increase", "binding_increase", "factorial_one_axis_increase"}


def centered_ridge(x, y, alpha):
    """No coordinate-wise standardization. Objective SSE + alpha ||B||_F².

    Center X and y on training rows, fit an unpenalized intercept. Dual eigensolve
    avoids a d-by-d solve for every quality and every grouped fold.
    """
    mean, target_mean = x.mean(0), y.mean(0)
    xc, yc = x - mean, y - target_mean
    vals, vectors = eigh(xc @ xc.T, check_finite=False)
    dual = vectors @ ((vectors.T @ yc) / (np.maximum(vals, 0) + alpha)[(...,) + (None,) * (yc.ndim - 1)])
    coef = (xc.T @ dual).T
    intercept = target_mean - mean @ coef.T
    return coef, intercept


def ridge_grouped(x, y, groups, alphas):
    groups = np.asarray(groups)
    losses = defaultdict(list)
    if len(set(groups)) >= 2:
        for ti, vi in GroupKFold(min(3, len(set(groups)))).split(x, y, groups):
            mean, ym = x[ti].mean(0), y[ti].mean(0)
            xc, yc = x[ti] - mean, y[ti] - ym
            vals, q = eigh(xc @ xc.T, check_finite=False)
            rhs = q.T @ yc
            left = (x[vi] - mean) @ xc.T @ q
            for alpha in alphas:
                pred = left @ (rhs / (np.maximum(vals, 0) + alpha)[(...,) + (None,) * (yc.ndim - 1)]) + ym
                losses[alpha].append(float(np.square(pred - y[vi]).mean()))
        alpha = min(alphas, key=lambda a: (np.mean(losses[a]), a))
    else:
        alpha = alphas[0]
    coef, intercept = centered_ridge(x, y, alpha)
    return coef, intercept, float(alpha), {str(a): float(np.mean(v)) for a, v in losses.items()}


def training_context_direction(quality, cases, contrasts, x, index):
    """Equal weight per independent family, then per eligible contrast in family."""
    by_id = {r["case_id"]: r for r in cases}
    families = defaultdict(list)
    ids = []
    for c in contrasts:
        if c["quality_id"] != quality or c["kind"] not in INCREASES:
            continue
        low, high = by_id[c["low_case_id"]], by_id[c["high_case_id"]]
        if not (_eligible_fit(low) and _eligible_fit(high)):
            continue
        if quality not in low["targets"] or high["targets"][quality] <= low["targets"][quality]:
            continue
        delta = x[index[high["case_id"]]] - x[index[low["case_id"]]]
        families[low["entity_group_id"]].append(delta)
        ids.append(c["contrast_id"])
    if not families:
        return None, [], []
    return np.mean([np.mean(v, axis=0) for v in families.values()], axis=0), sorted(families), ids


def _calibrate(scores, y):
    centered = scores - scores.mean()
    denom = float(centered @ centered)
    slope = float(centered @ (y - y.mean()) / denom) if denom > 1e-20 else 0.
    return slope, float(y.mean() - slope * scores.mean())


def _save_frame(run, name, rows):
    pd.DataFrame(rows).to_csv(Path(run) / name, index=False)


def analyze_geometry(run, suite, x, geometry, output_directions, alphas):
    run = Path(run)
    cases = suite["cases"]
    index = {r["case_id"]: i for i, r in enumerate(cases)}
    targets = _target_rows(cases)
    transformed = geometry.context(x)
    metrics, predictions, selections, transports, alignment = [], [], {}, [], []
    paired_predictions = {}
    direction_arrays = {}
    for quality, rows in targets.items():
        indices = np.array([index[r["case_id"]] for r in rows])
        ti = np.array([i for i, r in enumerate(rows) if _eligible_fit(r)])
        groups = [rows[i]["entity_group_id"] for i in ti]
        if len(ti) < 4 or len(set(groups)) < 2:
            selections[quality] = dict(status="insufficient_training_groups", n_rows=len(ti), n_groups=len(set(groups)))
            continue
        key = quality.removeprefix("lexical__")
        y = np.array([[r["targets"]["hue_sin"], r["targets"]["hue_cos"]] if quality == "hue"
                      else r["targets"][key] for r in rows], float)
        raw, cov = x[indices], transformed[indices]
        selections[quality] = dict(site=SITE, train_case_ids=[rows[i]["case_id"] for i in ti],
                                   train_groups=sorted(set(groups)), train_ids_sha256=digest([rows[i]["case_id"] for i in ti]),
                                   settings_selection="grouped training-fold MSE only; fixed final site; no test selection")
        methods = {}
        for label, features in [("euclidean", raw), ("covariance", cov)]:
            coef, intercept, alpha, losses = ridge_grouped(features[ti], y[ti], groups, alphas)
            method = f"ridge_{label}"
            methods[method] = features @ coef.T + intercept
            selections[quality][method] = dict(alpha=alpha, cv_mse=losses,
                                               preprocessing="training mean only; no feature standardization",
                                               penalty="sum squared residuals + alpha * sum(coefficient²)")
            atomic_npz(run / "fits" / f"{quality}__{method}.npz", coef=coef, intercept=intercept,
                       alpha=np.array(alpha), training_mean=features[ti].mean(0))
            if label == "euclidean":
                transported = geometry.transport_coefficients(coef)
                error = float(np.max(np.abs(cov @ transported.T + intercept - methods[method])))
                if error > 1e-8:
                    raise AssertionError(f"Coefficient transport failed: {error}")
                transports.append(dict(quality=quality, source_method=method, n_rows=len(rows), max_absolute_error=error))
        # Preserve V3's standardized ridge as a third same-site control, unchanged helper.
        legacy, alpha = _ridge_cv(raw[ti], y[ti], groups, alphas)
        methods["ridge_v3_scaled_same_site"] = legacy.predict(raw)
        coef = legacy[-1].coef_ / legacy[0].scale_
        intercept = legacy[-1].intercept_ - legacy[0].mean_ @ coef.T
        error = float(np.max(np.abs(cov @ geometry.transport_coefficients(coef).T + intercept - methods["ridge_v3_scaled_same_site"])))
        if error > 1e-8:
            raise AssertionError("V3 standardized coefficient transport failed")
        transports.append(dict(quality=quality, source_method="ridge_v3_scaled_same_site", n_rows=len(rows), max_absolute_error=error))
        selections[quality]["ridge_v3_scaled_same_site"] = dict(alpha=alpha, preprocessing="V3 train-fold StandardScaler + ridge")
        atomic_npz(run / "fits" / f"{quality}__ridge_v3_scaled_same_site.npz", coef=coef, intercept=intercept)
        if quality != "hue":
            # Context contrasts are unavailable for lexical-only models; do not reuse synthetic directions as lexical fits.
            direction, train_families, contrast_ids = training_context_direction(quality, cases, suite["contrasts"], x, index)
            if direction is not None:
                selections[quality]["context_direction"] = dict(training_groups=train_families, contrast_ids=contrast_ids,
                                                               estimator="mean within family, mean across families; only positive target-changing contrasts")
                for label, space in [("euclidean", "euclidean"), ("covariance", "context")]:
                    unit = geometry.normalize(direction, space)
                    direction_arrays[f"{quality}__{label}"] = unit
                    coefficient = unit if label == "euclidean" else geometry.apply(unit, 1)
                    scores = raw @ coefficient
                    slope, offset = _calibrate(scores[ti], y[ti])
                    methods[f"context_projection_{label}"] = slope * scores + offset
                    selections[quality][f"context_projection_{label}"] = dict(calibration_slope=slope, calibration_intercept=offset,
                        preprocessing="raw metric projection, affine OLS calibration on training rows only")
                    for c in suite["contrasts"]:
                        if c["quality_id"] != quality:
                            continue
                        a, b = index[c["low_case_id"]], index[c["high_case_id"]]
                        delta = x[b] - x[a]
                        alignment.append(dict(**c, geometry=label, site=SITE, partition=_partition(cases[b]),
                                              cosine=float(geometry.cosine(delta, unit, space)),
                                              raw_projection=float(delta @ coefficient),
                                              calibrated_delta=float(slope * (delta @ coefficient)),
                                              used_for_direction=c["contrast_id"] in contrast_ids))
            for label in ["euclidean", "covariance"]:
                if (key, label) not in output_directions:
                    continue
                # Canonical context-output pairing, NOT an output-space metric product.
                gamma = output_directions[(key, label)]
                scores = raw @ gamma
                slope, offset = _calibrate(scores[ti], y[ti])
                methods[f"output_measurement_{label}"] = slope * scores + offset
                selections[quality][f"output_measurement_{label}"] = dict(calibration_slope=slope,
                    calibration_intercept=offset, pairing="lambda.T gamma_bar, raw scores retained", training_source="original train-split lexical pairs")
                for i, r in enumerate(rows):
                    predictions.append(dict(case_id=r["case_id"], quality=quality, partition=_partition(r),
                        entity_group_id=r["entity_group_id"], method=f"output_raw_score_{label}", target=y[i], prediction=scores[i]))
                for partition in sorted({_partition(r) for r in rows}):
                    ix = [i for i, r in enumerate(rows) if _partition(r) == partition]
                    metrics.append(dict(quality=quality, method=f"output_raw_score_{label}", partition=partition,
                        site=SITE, n_rows=len(ix), n_groups=len({rows[i]["entity_group_id"] for i in ix}),
                        rho=_score(y[ix], scores[ix])["rho"], mae=None,
                        units="raw logit-direction score; MAE to ordinal codes not defined"))
        for method, pred in methods.items():
            if quality != "hue":
                paired_predictions[(quality, method)] = {r["case_id"]: float(value) for r, value in zip(rows, pred)}
            for partition in sorted({_partition(r) for r in rows}):
                ix = [i for i, r in enumerate(rows) if _partition(r) == partition]
                metrics.append(dict(quality=quality, method=method, partition=partition, site=SITE,
                                    n_rows=len(ix), n_groups=len({rows[i]["entity_group_id"] for i in ix}),
                                    **(_circular_score(y[ix], pred[ix]) if quality == "hue" else _score(y[ix], pred[ix]))))
            for i, row in enumerate(rows):
                predictions.append(dict(case_id=row["case_id"], quality=quality, partition=_partition(row),
                    entity_group_id=row["entity_group_id"], method=method,
                    target=y[i].tolist(), prediction=pred[i].tolist()))
        print(f"Analyzed {quality}: {len(ti)} training rows / {len(set(groups))} groups", flush=True)
    _save_frame(run, "metrics.csv", metrics)
    _save_frame(run, "predictions.csv", predictions)
    _save_frame(run, "context_alignment.csv", alignment)
    atomic_json(run / "model_selection.json", selections)
    atomic_json(run / "coefficient_transport.json", transports)
    atomic_npz(run / "context_directions.npz", **direction_arrays)
    paired = []
    by_id = {r["case_id"]: r for r in cases}
    for c in suite["contrasts"]:
        low, high = by_id[c["low_case_id"]], by_id[c["high_case_id"]]
        for (quality, method), values in paired_predictions.items():
            if quality.startswith("lexical__") or c["low_case_id"] not in values or c["high_case_id"] not in values:
                continue
            if quality != c["quality_id"] and c["kind"] != "factorial_one_axis_increase":
                continue
            paired.append(dict(**c, measured_quality=quality, method=method, site=SITE,
                               partition=_partition(high), low_partition=_partition(low),
                               prediction_delta=values[c["high_case_id"]] - values[c["low_case_id"]],
                               target_delta=high["targets"][quality] - low["targets"][quality],
                               on_target=quality == c["quality_id"]))
    _save_frame(run, "paired_changes.csv", paired)
    if paired:
        frame = pd.DataFrame(paired)
        frame["abs_delta"] = frame.prediction_delta.abs()
        frame["positive"] = frame.prediction_delta > 0
        keys = ["quality_id", "measured_quality", "method", "kind", "entity_split", "template_id", "template_split", "combination_split", "on_target"]
        # Each independent family has equal weight in the published summaries.
        group = frame.groupby(keys + ["entity_group_id"], dropna=False).agg(
            signed_delta=("prediction_delta", "mean"), absolute_delta=("abs_delta", "mean"),
            positive_fraction=("positive", "mean"), n_pairs=("contrast_id", "size")).reset_index()
        summary = group.groupby(keys, dropna=False).agg(
            signed_delta=("signed_delta", "mean"), absolute_delta=("absolute_delta", "mean"),
            positive_fraction=("positive_fraction", "mean"), n_pairs=("n_pairs", "sum"),
            n_groups=("entity_group_id", "nunique")).reset_index()
        summary.to_csv(run / "paired_summary.csv", index=False)
    # Circular neighbourhoods stay separate; no scalar hue direction is manufactured.
    hue = []
    for t in suite["triplets"]:
        a, near, far = [index[t[k]] for k in ["anchor_case_id", "near_case_id", "far_case_id"]]
        for name, features in [("euclidean", x), ("covariance", transformed)]:
            dn, df = float(np.linalg.norm(features[a] - features[near])), float(np.linalg.norm(features[a] - features[far]))
            hue.append({**t, "stimulus_geometry": t.get("geometry"), "geometry": name,
                        "near_distance": dn, "far_distance": df, "near_closer": dn < df})
    _save_frame(run, "hue_neighborhoods.csv", hue)
    return dict(n_targets=len(selections), n_prediction_metrics=len(metrics), n_matched_changes=len(paired),
                max_coefficient_transport_error=max((r["max_absolute_error"] for r in transports), default=0))
