"""Covariance geometry, version 1. Column-vector mathematics; NumPy rows (..., d).

Output rows G transform as G @ Sigma**(-1/2); context rows X transform as
X @ Sigma**(1/2). The context-output pairing is ALWAYS X @ G.T.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .util import atomic_json, atomic_npz, digest, read_json

IMPLEMENTATION = "quality_geometry_v1.0"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_chunk(weight, indices):
    """Only this bounded slice crosses device/dtype boundaries."""
    chunk = weight[indices]
    if hasattr(chunk, "detach"):
        chunk = chunk.detach().cpu().float().numpy()
    return np.asarray(chunk, dtype=np.float64)


def output_head_identity(head, chunk_rows=2048):
    h = hashlib.sha256()
    for start in range(0, head.weight.shape[0], chunk_rows):
        h.update(row_chunk(head.weight, slice(start, start + chunk_rows)).astype("<f4").tobytes())
    bias = getattr(head, "bias", None)
    return dict(type=f"{type(head).__module__}.{type(head).__name__}",
                shape=list(head.weight.shape), dtype=str(head.weight.dtype),
                float32_rows_sha256=h.hexdigest(),
                bias_sha256=None if bias is None else hashlib.sha256(
                    bias.detach().cpu().float().numpy().astype("<f4").tobytes()).hexdigest())


def covariance_chunks(weight, row_ids, chunk_rows=2048):
    """Uniform population covariance (denominator N), two-pass float64 on CPU."""
    row_ids = np.asarray(row_ids, dtype=np.int64)
    if len(row_ids) < 2 or len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("Covariance requires at least two distinct vocabulary rows")
    d = weight.shape[1]
    mean = np.zeros(d, np.float64)
    for start in range(0, len(row_ids), chunk_rows):
        mean += row_chunk(weight, row_ids[start:start + chunk_rows]).sum(axis=0)
    mean /= len(row_ids)
    cov = np.zeros((d, d), np.float64)
    for start in range(0, len(row_ids), chunk_rows):
        centered = row_chunk(weight, row_ids[start:start + chunk_rows]) - mean
        cov += centered.T @ centered
    cov /= len(row_ids)
    return mean, (cov + cov.T) / 2


@dataclass
class Geometry:
    mean: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    raw_eigenvalues: np.ndarray
    diagnostics: dict

    @classmethod
    def from_covariance(cls, mean, cov, mode="auto", rcond=1e-10, ridge_relative=1e-4):
        cov = np.asarray(cov, np.float64)
        if not np.isfinite(cov).all() or not np.allclose(cov, cov.T, atol=1e-12):
            raise ValueError("Covariance must be finite and symmetric")
        if mode not in {"auto", "unregularized", "ridge"} or not 0 < rcond < 1 or ridge_relative <= 0:
            raise ValueError("Invalid regularization settings")
        raw, q = np.linalg.eigh(cov)  # CPU float64; never MPS eigh.
        if raw[-1] <= 0 or raw[0] < -max(raw[-1], 1.) * 1e-10:
            raise ValueError("Covariance is not positive semidefinite")
        safe = raw[0] > raw[-1] * rcond
        if mode == "unregularized" and not safe:
            raise ValueError("Unsafe inverse covariance; choose declared auto/ridge approximation")
        if mode == "ridge":
            loading = max(ridge_relative * float(np.trace(cov)) / len(raw), raw[-1] * rcond)
            values = raw + loading
            formula = "Sigma_effective = Sigma + loading * I"
        elif not safe:
            loading = float(raw[-1] * rcond)
            values = np.maximum(raw, loading)
            formula = "Sigma_effective = Q diag(max(eigenvalue, loading)) Q.T"
        else:
            loading = 0.
            values = raw.copy()
            formula = "Sigma_effective = Sigma (unregularized)"
        diagnostics = dict(mode=mode, approximation=bool(loading), formula=formula,
                           loading=loading, rcond=rcond, ridge_relative=ridge_relative,
                           raw_min=float(raw[0]), raw_max=float(raw[-1]),
                           raw_condition=float(raw[-1] / raw[0]) if raw[0] > 0 else None,
                           effective_min=float(values[0]), effective_max=float(values[-1]),
                           effective_condition=float(values[-1] / values[0]),
                           accumulation="two-pass CPU float64, population denominator N",
                           eigendecomposition="numpy.linalg.eigh CPU float64")
        return cls(np.asarray(mean, np.float64), q, values, raw, diagnostics)

    def apply(self, rows, power):
        return ((np.asarray(rows, np.float64) @ self.eigenvectors) * self.eigenvalues**power) @ self.eigenvectors.T

    def output(self, rows, centered=False):
        return self.apply(np.asarray(rows) - self.mean if centered else rows, -.5)

    def context(self, rows):
        return self.apply(rows, .5)

    def inner(self, a, b, space):
        if space not in {"output", "context", "euclidean"}:
            raise ValueError("Specify output, context or euclidean space")
        power = {"output": -1, "context": 1, "euclidean": 0}[space]
        return np.sum(np.asarray(a) * (np.asarray(b) if power == 0 else self.apply(b, power)), axis=-1)

    def norm(self, a, space):
        return np.sqrt(np.maximum(self.inner(a, a, space), 0))

    def normalize(self, a, space):
        norm = self.norm(a, space)
        if np.any(norm <= 1e-15):
            raise ValueError("Cannot normalize a zero direction")
        return np.asarray(a) / np.expand_dims(norm, -1)

    def cosine(self, a, b, space):
        denominator = self.norm(a, space) * self.norm(b, space)
        return np.divide(self.inner(a, b, space), denominator,
                         out=np.full(np.shape(denominator), np.nan), where=denominator > 1e-15)

    def projection(self, a, direction, space):
        """Signed scalar projection in the declared metric; not context-output score."""
        return self.inner(a, self.normalize(direction, space), space)

    def measurement_to_intervention(self, gamma):
        return self.apply(gamma, -1)

    def intervention_to_measurement(self, direction):
        return self.apply(direction, 1)

    def transport_coefficients(self, coefficient_rows):
        """For X' = X Sigma**.5, B' = B Sigma**-.5; intercept unchanged.

        Undo any StandardScaler first: B = coef / scale and
        intercept_raw = intercept - B @ feature_mean.
        """
        return self.output(coefficient_rows)

    def save(self, root, identity):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        atomic_npz(root / "metric.npz", mean=self.mean, eigenvectors=self.eigenvectors,
                   eigenvalues=self.eigenvalues, raw_eigenvalues=self.raw_eigenvalues)
        atomic_json(root / "identity.json", identity)
        atomic_json(root / "diagnostics.json", self.diagnostics)
        atomic_json(root / "integrity.json", {"metric_sha256": sha256_file(root / "metric.npz")})

    @classmethod
    def load(cls, root, expected_identity):
        root = Path(root)
        if read_json(root / "identity.json") != expected_identity:
            raise ValueError("Incompatible geometry cache identity")
        if sha256_file(root / "metric.npz") != read_json(root / "integrity.json")["metric_sha256"]:
            raise ValueError("Corrupt geometry cache")
        with np.load(root / "metric.npz", allow_pickle=False) as a:
            result = cls(**{key: a[key].copy() for key in ["mean", "eigenvectors", "eigenvalues", "raw_eigenvalues"]},
                         diagnostics=read_json(root / "diagnostics.json"))
        if not np.isfinite(result.eigenvectors).all() or not np.all(result.eigenvalues > 0):
            raise ValueError("Invalid geometry eigensystem")
        return result


def prepare_metric(head, tokenizer, model_identity, settings, cache_root):
    """The caller MUST pass model.get_output_embeddings(), not input embeddings."""
    vocabulary = tokenizer.get_vocab()
    n = head.weight.shape[0]
    if max(vocabulary.values()) >= n:
        raise ValueError("Tokenizer IDs exceed output-head rows")
    sample = settings.get("sample_rows")
    ids = np.arange(n) if sample is None else np.sort(np.random.default_rng(settings["seed"]).choice(
        n, min(n, sample), replace=False))
    identity = dict(implementation=IMPLEMENTATION, model=model_identity,
                    output_head=output_head_identity(head, settings["chunk_rows"]),
                    vocabulary=dict(definition="uniform over ALL output-head rows, including special and unassigned rows",
                                    output_rows=n, tokenizer_entries=len(vocabulary),
                                    tokenizer_vocab_sha256=digest(vocabulary), special_ids=sorted(tokenizer.all_special_ids),
                                    unassigned_rows=sorted(set(range(n)) - set(vocabulary.values())),
                                    selected_rows=len(ids), row_ids_sha256=digest(ids.tolist()),
                                    sampling="full" if sample is None else "seeded subsample; integration only"),
                    numerical_settings=settings)
    root = Path(cache_root) / digest(identity)
    if (root / "integrity.json").exists():
        return Geometry.load(root, identity), root, identity
    mean, covariance = covariance_chunks(head.weight, ids, settings["chunk_rows"])
    geometry = Geometry.from_covariance(mean, covariance, settings["mode"], settings["rcond"], settings["ridge_relative"])
    geometry.save(root, identity)
    return geometry, root, identity


def steer_final_state(context_rows, gamma, geometry, strength, *, measurement_reviewed=False):
    """Optional intervention immediately before the output head; disabled by default."""
    if not measurement_reviewed:
        raise ValueError("Final-state steering is disabled until measurement review")
    return np.asarray(context_rows) + float(strength) * geometry.measurement_to_intervention(gamma)
