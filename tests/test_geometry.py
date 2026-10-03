import json
from pathlib import Path

import numpy as np
import pytest

from quality_dimensions.geometry import Geometry, covariance_chunks, steer_final_state
from quality_dimensions.geometry_analysis import centered_ridge, ridge_grouped, training_context_direction
from quality_dimensions.geometry_pairs import completion_token, candidate_pairs


@pytest.fixture
def metric():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(7, 7))
    return Geometry.from_covariance(np.arange(7) / 10, a @ a.T + .4 * np.eye(7), mode="unregularized")


def test_chunked_population_covariance():
    x = np.random.default_rng(7).normal(size=(37, 7))
    mean, cov = covariance_chunks(x, np.arange(len(x)), chunk_rows=6)
    np.testing.assert_allclose(mean, x.mean(0), atol=1e-14)
    np.testing.assert_allclose(cov, np.cov(x.T, bias=True), atol=1e-14)


def test_metrics_complementary_transforms_and_logit_differences(metric):
    rng = np.random.default_rng(8)
    x, g = rng.normal(size=(5, 7)), rng.normal(size=(4, 7))
    sigma = (metric.eigenvectors * metric.eigenvalues) @ metric.eigenvectors.T
    inverse = np.linalg.inv(sigma)
    np.testing.assert_allclose(metric.context(x) @ metric.output(g).T, x @ g.T, atol=1e-12)
    shifted = metric.context(x) @ metric.output(g, centered=True).T
    np.testing.assert_allclose(shifted[:, 1:] - shifted[:, :1], (x @ g.T)[:, 1:] - (x @ g.T)[:, :1], atol=1e-12)
    np.testing.assert_allclose(metric.inner(g[0], g[1], "output"), g[0] @ inverse @ g[1], atol=1e-12)
    np.testing.assert_allclose(metric.inner(x[0], x[1], "context"), x[0] @ sigma @ x[1], atol=1e-12)
    for values, space, transform in [(g, "output", metric.output), (x, "context", metric.context)]:
        transformed = transform(values)
        np.testing.assert_allclose(metric.inner(values[0], values[1], space), transformed[0] @ transformed[1], atol=1e-12)
        np.testing.assert_allclose(metric.projection(values, values[0], space), transformed @ (transformed[0] / np.linalg.norm(transformed[0])), atol=1e-12)
        assert metric.cosine(values[0], values[0], space) == pytest.approx(1)
    np.testing.assert_allclose(metric.intervention_to_measurement(metric.measurement_to_intervention(g)), g, atol=1e-12)
    assert not np.allclose(metric.context(x), metric.output(x))


def test_declared_regularization_is_symmetric_positive_definite():
    cov = np.diag([0., 1., 10.])
    with pytest.raises(ValueError, match="Unsafe"):
        Geometry.from_covariance(np.zeros(3), cov, mode="unregularized")
    for mode in ["auto", "ridge"]:
        g = Geometry.from_covariance(np.zeros(3), cov, mode=mode, rcond=1e-6, ridge_relative=.01)
        assert g.diagnostics["approximation"]
        actual = g.apply(np.eye(3), 1)
        np.testing.assert_allclose(actual, actual.T)
        assert np.linalg.eigvalsh(actual).min() > 0
        np.testing.assert_allclose(g.context(np.eye(3)) @ g.output(np.eye(3)).T, np.eye(3), atol=1e-12)


def test_coefficient_transport_after_undoing_standard_scaler(metric):
    rng = np.random.default_rng(19)
    x, scaled_coef = rng.normal(size=(9, 7)), rng.normal(size=(2, 7))
    mean, scale = rng.normal(size=7), rng.uniform(.1, 3, 7)
    intercept = np.array([1., 4.])
    expected = ((x - mean) / scale) @ scaled_coef.T + intercept
    coef = scaled_coef / scale
    raw_intercept = intercept - mean @ coef.T
    actual = metric.context(x) @ metric.transport_coefficients(coef).T + raw_intercept
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_ridge_objective_and_grouped_training_only():
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(16, 7)), rng.normal(size=(16, 2))
    coef, intercept = centered_ridge(x, y, 1.)
    from sklearn.linear_model import Ridge
    expected = Ridge(alpha=1.).fit(x, y)
    np.testing.assert_allclose(x @ coef.T + intercept, expected.predict(x), atol=1e-10)
    train = np.arange(12)
    first = ridge_grouped(x[train], y[train], np.repeat([1, 2, 3], 4), [.1, 1, 10])
    y[12:] += 100000
    second = ridge_grouped(x[train], y[train], np.repeat([1, 2, 3], 4), [.1, 1, 10])
    np.testing.assert_array_equal(first[0], second[0])
    assert first[2:] == second[2:]


def test_context_direction_excludes_test_wording_and_combinations():
    cases, contrasts, x = [], [], []
    for j, (entity, wording, combo) in enumerate([("train", "train", "train"), ("test", "train", "train"),
                                              ("train", "heldout", "train"), ("train", "train", "test")]):
        for level in [0, 1]:
            cases.append(dict(case_id=f"{j}_{level}", entity_group_id=str(j), entity_split=entity,
                              template_split=wording, combination_split=combo, case_type="factorial", targets={"size": level}))
            x.append(np.array([level, level * j * 100.]))
        contrasts.append(dict(quality_id="size", kind="factorial_one_axis_increase", low_case_id=f"{j}_0",
                              high_case_id=f"{j}_1", contrast_id=str(j)))
    direction, groups, ids = training_context_direction("size", cases, contrasts, np.array(x), {r['case_id']: i for i, r in enumerate(cases)})
    np.testing.assert_array_equal(direction, [1, 0])
    assert groups == ["0"] and ids == ["0"]


class ToyTokenizer:
    all_special_ids = [9]
    def encode(self, text, add_special_tokens=False):
        return {"Answer:": [1], "Answer: apple": [1, 2], "Answer: ice cube": [1, 3, 4],
                "Answer: special": [1, 9], "Answer: boundary": [6, 7]}[text]
    def decode(self, ids):
        return {2: " apple", 9: " special"}[ids[0]]


def test_token_eligibility_exclusions():
    t = ToyTokenizer()
    assert completion_token(t, "apple")["token_id"] == 2
    assert completion_token(t, "ice cube")["exclusion"] == "multi_token_completion"
    assert completion_token(t, "special")["exclusion"] == "special_token"
    assert completion_token(t, "boundary")["exclusion"] == "prefix_tokenization_changed"


def test_metric_cache_compatibility_and_integrity(metric, tmp_path):
    identity = {"model_revision": "pinned", "vocabulary": "full", "regularization": "none"}
    metric.save(tmp_path, identity)
    loaded = Geometry.load(tmp_path, identity)
    np.testing.assert_array_equal(loaded.eigenvalues, metric.eigenvalues)
    for key in identity:
        with pytest.raises(ValueError, match="Incompatible"):
            Geometry.load(tmp_path, {**identity, key: "changed"})
    with (tmp_path / "metric.npz").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        Geometry.load(tmp_path, identity)


def test_steering_disabled_unless_reviewed(metric):
    x, gamma = np.ones(7), np.arange(7)
    with pytest.raises(ValueError, match="disabled"):
        steer_final_state(x, gamma, metric, .2)
    edited = steer_final_state(x, gamma, metric, .2, measurement_reviewed=True)
    np.testing.assert_allclose(edited - x, .2 * metric.measurement_to_intervention(gamma))


def test_smoke_analysis_artifacts_and_heldout_labels_do_not_change_fits(tmp_path):
    import copy
    from quality_dimensions.suite_data import load_suite
    from quality_dimensions.suite_analysis import _eligible_fit
    from quality_dimensions.geometry_analysis import analyze_geometry
    suite = load_suite('configs/quality_suite_v3.json', 'smoke')
    rng = np.random.default_rng(65)
    x = rng.normal(size=(len(suite['cases']), 7))
    geometry = Geometry.from_covariance(np.zeros(7), np.diag(np.arange(1, 8)))
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir(); second.mkdir()
    changed = copy.deepcopy(suite)
    for row in changed['cases']:
        if not _eligible_fit(row):
            row['targets'] = {k: v + 1234 for k, v in row['targets'].items()}
    for destination, spec in [(first, suite), (second, changed)]:
        result = analyze_geometry(destination, spec, x, geometry, {}, [.1, 1])
        assert result['n_matched_changes'] > 0
        assert result['max_coefficient_transport_error'] < 1e-8
        import pandas as pd
        hue = pd.read_csv(destination / 'hue_neighborhoods.csv')
        assert set(hue.geometry) == {'euclidean', 'covariance'}
        assert 'stimulus_geometry' in hue
    assert json.loads((first / 'model_selection.json').read_text()) == json.loads((second / 'model_selection.json').read_text())
    for path in (first / 'fits').glob('*.npz'):
        with np.load(path) as a, np.load(second / 'fits' / path.name) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
