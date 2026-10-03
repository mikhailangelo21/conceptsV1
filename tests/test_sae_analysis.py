import numpy as np
import json
import pytest
from scipy import sparse
from sklearn.decomposition import PCA

from quality_dimensions.sae_analysis import _stage, _decode_csr
from quality_dimensions.sae_analysis_reconstruction import _baseline_ss
from quality_dimensions.sae_analysis_core import (
    centered_sparse_pca, eligible_fit, embedding_metrics, intrinsic_knn,
    feature_matrix, pca_dimensions, partition, transform_centered,
)
from quality_dimensions.util import digest


def test_centered_sparse_pca_matches_dense_and_keeps_feature_ids():
    rng = np.random.default_rng(9)
    dense = rng.normal(size=(80, 30))
    dense[rng.uniform(size=dense.shape) < .65] = 0
    dense[:, 7] += 8  # a large nonzero mean must be removed
    x = sparse.csr_matrix(dense)
    fit = centered_sparse_pca(x, 5, 9)
    reference = PCA(n_components=5).fit(dense)
    np.testing.assert_allclose(fit["mean"], reference.mean_, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(fit["singular_values"], reference.singular_values_, rtol=1e-4)
    np.testing.assert_allclose(np.abs(fit["components"] @ reference.components_.T).max(axis=1), 1, atol=1e-3)
    np.testing.assert_allclose(fit["scores"], transform_centered(x, fit), atol=1e-5)
    assert fit["components"].shape[1] == 30


def test_fit_restrictions_include_numeric_wording_and_factorial():
    base = {"case_type": "graded", "entity_split": "train",
            "template_split": "train", "combination_split": "train"}
    assert eligible_fit(base)
    for change in ({"case_type": "numeric"}, {"entity_split": "test"},
                   {"template_split": "heldout"},
                   {"case_type": "factorial", "combination_split": "heldout"}):
        row = {**base, **change}
        assert not eligible_fit(row)
        assert partition(row) != "fit_train"


def test_dimension_estimators_on_line_plane_circle_and_duplicates():
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(size=240))
    line = np.column_stack((t, np.zeros_like(t), np.zeros_like(t)))
    plane = rng.uniform(size=(240, 2))
    circle = np.column_stack((np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)))
    dl = intrinsic_knn(line, (5, 10, 20))
    dp = intrinsic_knn(plane, (5, 10, 20))
    dc = intrinsic_knn(circle, (5, 10, 20))
    assert .6 < dl["twonn"] < 1.6
    assert dp["twonn"] > dl["twonn"]
    assert .6 < dc["twonn"] < 1.8
    dup = intrinsic_knn(np.vstack((line, line[:20])), (5, 10, 20))
    assert dup["n_unique"] == 240
    noisy = intrinsic_knn(line + rng.normal(0, .001, line.shape), (5, 10, 20))
    assert np.isfinite(noisy["mle_k10"])


def test_embedding_metrics_and_unreached_threshold():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(90, 4))
    result = embedding_metrics(x, x[:, :2], (5, 10), 7)
    assert .7 < result["trustworthiness_k5"] <= 1
    assert result["normalized_stress"] >= 0
    assert pca_dimensions([.3, .2])["0.8"] is None


def test_sparse_chunk_loading_preserves_keys_and_rejects_duplicate_features(tmp_path):
    identity = {"version": "fixture"}
    ck = "a" * 64
    folder = tmp_path / "features" / "main" / "layer3"
    folder.mkdir(parents=True)
    path = folder / "chunk000000.npz"
    arrays = dict(identity=np.array(digest(identity)), layer=np.array(3),
                  checkpoint_sha256=np.array(ck), keys=np.array(["a", "b"]),
                  ids=np.array([[7, 9], [9, 11]], np.int32),
                  values=np.array([[2., 1.], [3., 4.]], np.float32),
                  relative_error=np.array([.2, .4]), error_norm=np.ones(2),
                  raw_norm=np.ones(2), cosine=np.ones(2), nnz=np.array([2, 2]),
                  flags=np.zeros(2))
    np.savez(path, **arrays)
    collection = {"source": tmp_path, "identity": identity,
                  "repository": {"k": 2, "checkpoints": {"3": {"sha256": ck}}}}
    keys, matrix, diagnostic = feature_matrix(collection, 3)
    assert keys == ["a", "b"]
    assert sparse.isspmatrix_csr(matrix)
    assert matrix.shape == (2, 32768)
    assert matrix[0, 7] == 2 and matrix[1, 11] == 4
    np.testing.assert_allclose(diagnostic["relative_error"], [.2, .4])
    arrays["ids"][0] = [7, 7]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="Repeated feature ID"):
        feature_matrix(collection, 3)


def test_stage_resume_skips_completed_work(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({"status": "prepared", "stages": {}}))
    calls = []
    _stage(tmp_path, "audit", lambda: calls.append(1))
    _stage(tmp_path, "audit", lambda: calls.append(2))
    assert calls == [1]
    assert json.loads((tmp_path / "state.json").read_text())["stages"]["audit"] == "complete"


def test_sparse_decoder_and_partitioned_reconstruction_match_direct_dense_calculation():
    rng = np.random.default_rng(31)
    f = sparse.csr_matrix(np.array([[2., 0., 1., 0.], [0., 3., 0., 4.],
                                    [1., 0., 0., 0.]], dtype=np.float32))
    w = rng.normal(size=(3, 4)).astype(np.float32)
    bias = rng.normal(size=3).astype(np.float32)
    np.testing.assert_allclose(_decode_csr(f, w, bias, batch_size=2),
                               f.toarray() @ w.T + bias, rtol=1e-6, atol=1e-6)
    x = sparse.csr_matrix(rng.normal(size=(40, 9)).astype(np.float32))
    fit = centered_sparse_pca(x[:25], 4, 31)
    heldout = x[25:]
    scores = transform_centered(heldout, fit)
    mean = fit["mean"]
    direct = heldout.toarray() - mean
    direct_residual = direct - scores @ fit["components"]
    np.testing.assert_allclose(_baseline_ss(heldout, mean), np.sum(direct ** 2), rtol=1e-5)
    np.testing.assert_allclose(_baseline_ss(heldout, mean) - np.sum(scores ** 2),
                               np.sum(direct_residual ** 2), rtol=1e-5, atol=1e-5)
