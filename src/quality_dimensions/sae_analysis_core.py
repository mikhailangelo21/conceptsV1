"""Read-only SAE analysis primitives. Feature coordinates are local to one checkpoint."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, svds
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

from .sae_core import WIDTH


def jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_order(items, seed):
    return sorted(items, key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).hexdigest())


def eligible_fit(meta):
    return (meta.get("case_type") != "numeric"
            and meta.get("entity_split") == "train"
            and meta.get("template_split") == "train"
            and (meta.get("case_type") != "factorial" or meta.get("combination_split") == "train"))


def partition(meta):
    if eligible_fit(meta):
        return "fit_train"
    parts = []
    if meta.get("entity_split") != "train":
        parts.append("entity_" + str(meta.get("entity_split")))
    if meta.get("template_split") != "train":
        parts.append("heldout_wording")
    if meta.get("case_type") == "factorial" and meta.get("combination_split") != "train":
        parts.append("heldout_combination")
    if meta.get("case_type") == "numeric":
        parts.append("numeric_challenge")
    return "+".join(parts) or "other_train"


def read_collection(source):
    source = Path(source)
    identity = json.loads((source / "identity.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    repository = json.loads((source / "sae_repository.json").read_text())
    if manifest["run_identity"] != hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
        # The project's digest includes a stable implementation; verify it using that implementation.
        from .util import digest
        if manifest["run_identity"] != digest(identity):
            raise ValueError("Collection identity mismatch")
    if manifest["feature_width"] != WIDTH or manifest["k"] != repository["k"]:
        raise ValueError("Collection width/K mismatch")
    readouts = list(jsonl(source / "readouts.jsonl"))
    records = list(jsonl(source / "records.jsonl"))
    by_readout = defaultdict(list)
    by_case = defaultdict(list)
    for row in records:
        by_readout[row["readout_id"]].append(row)
        if row["kind"] == "case":
            by_case[row["metadata"]["case_id"]].append(row)
    if len({r["readout_id"] for r in readouts}) != len(readouts):
        raise ValueError("Duplicate computational readout")
    return {"source": source, "identity": identity, "manifest": manifest,
            "repository": repository, "readouts": readouts, "records": records,
            "by_readout": by_readout, "by_case": by_case}


def layer_keys(source, kind, layer):
    keys = set()
    for path in sorted((Path(source) / "features" / kind / f"layer{layer}").glob("chunk*.npz")):
        with np.load(path, allow_pickle=False) as d:
            keys.update(d["keys"].tolist())
    return keys


def feature_matrix(collection, layer, kind="main"):
    """Load one layer as CSR, retaining original 32,768-wide feature IDs."""
    paths = sorted((collection["source"] / "features" / kind / f"layer{layer}").glob("chunk*.npz"))
    all_keys, data, columns, row_ind = [], [], [], []
    diag = defaultdict(list)
    checkpoint = collection["repository"]["checkpoints"][str(layer)]["sha256"]
    from .util import digest
    identity_hash = digest(collection["identity"])
    for path in paths:
        with np.load(path, allow_pickle=False) as d:
            if str(d["identity"]) != identity_hash or int(d["layer"]) != layer or str(d["checkpoint_sha256"]) != checkpoint:
                raise ValueError(f"Feature identity mismatch: {path}")
            keys = d["keys"].tolist()
            ids = d["ids"]
            vals = d["values"]
            if ids.shape != vals.shape or ids.shape[1] != collection["repository"]["k"]:
                raise ValueError(f"Sparse chunk shape mismatch: {path}")
            if np.any(ids < 0) or np.any(ids >= WIDTH) or not np.isfinite(vals).all():
                raise ValueError(f"Invalid sparse chunk: {path}")
            if np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
                raise ValueError(f"Repeated feature ID within a row: {path}")
            n = len(keys)
            row_ind.append(np.repeat(np.arange(len(all_keys), len(all_keys) + n), ids.shape[1]))
            columns.append(ids.ravel())
            data.append(vals.ravel())
            all_keys.extend(keys)
            for name in ("relative_error", "error_norm", "raw_norm", "cosine", "nnz", "flags"):
                diag[name].append(d[name].copy())
    if not all_keys:
        return [], sparse.csr_matrix((0, WIDTH), dtype=np.float32), {}
    matrix = sparse.csr_matrix((np.concatenate(data), (np.concatenate(row_ind), np.concatenate(columns))),
                               shape=(len(all_keys), WIDTH), dtype=np.float32)
    matrix.eliminate_zeros()
    if len(set(all_keys)) != len(all_keys):
        raise ValueError("Duplicate feature keys")
    return all_keys, matrix, {name: np.concatenate(parts) for name, parts in diag.items()}


def centered_sparse_pca(x, rank, seed=1729):
    """True column-centered sparse PCA through a LinearOperator, without a dense covariance."""
    x = sparse.csr_matrix(x, dtype=np.float64)
    n, p = x.shape
    k = min(rank, n - 1, p - 1)
    if k < 1:
        raise ValueError("Insufficient matrix rank")
    mean = np.asarray(x.mean(axis=0)).ravel()
    def mv(v):
        return np.asarray(x @ v).ravel() - float(mean @ v)
    def rmv(v):
        return np.asarray(x.T @ v).ravel() - mean * float(np.sum(v))
    def mm(v):
        return np.asarray(x @ v) - mean[:, None] * np.sum(v, axis=0)[None, :]
    def rmm(v):
        return np.asarray(x.T @ v) - mean[:, None] * np.sum(v, axis=0)[None, :]
    operator = LinearOperator((n, p), matvec=mv, rmatvec=rmv, matmat=mm, rmatmat=rmm, dtype=np.float64)
    u, singular, vt = svds(operator, k=k, which="LM", solver="propack", random_state=seed, tol=1e-4)
    order = np.argsort(singular)[::-1]
    singular = singular[order]
    components = vt[order]
    scores = (x @ components.T) - mean @ components.T
    total = float(x.power(2).sum() - n * np.dot(mean, mean))
    ratio = singular ** 2 / max(total, 1e-30)
    return {"mean": mean.astype(np.float32), "components": components.astype(np.float32),
            "singular_values": singular.astype(np.float32), "explained_variance_ratio": ratio.astype(np.float32),
            "scores": np.asarray(scores, dtype=np.float32), "total_centered_ss": total}


def transform_centered(x, fit):
    return np.asarray(x @ fit["components"].T - fit["mean"] @ fit["components"].T, dtype=np.float32)


def pca_dimensions(ratios):
    cumulative = np.cumsum(ratios)
    return {str(t): (int(np.searchsorted(cumulative, t) + 1) if len(cumulative) and cumulative[-1] >= t else None)
            for t in (0.5, 0.8, 0.9, 0.95)}


def sparse_activity(x, relative_error):
    n = x.shape[0]
    binary = x.copy()
    binary.data[:] = 1
    fires = np.asarray(binary.sum(axis=0)).ravel().astype(np.int32)
    total = np.asarray(x.sum(axis=0)).ravel()
    sq = np.asarray(x.power(2).sum(axis=0)).ravel()
    mean = total / max(n, 1)
    variance = np.maximum(sq / max(n, 1) - mean ** 2, 0)
    er = np.asarray(binary.T @ relative_error).ravel()
    mean_error_active = np.divide(er, fires, out=np.full(WIDTH, np.nan), where=fires > 0)
    return {"fires": fires, "frequency": fires / max(n, 1), "mean": mean,
            "variance": variance, "mean_when_active": np.divide(total, fires, out=np.zeros(WIDTH), where=fires > 0),
            "mean_error_when_active": mean_error_active}


def linear_cka(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    a = float(np.sum((x.T @ y) ** 2))
    b = float(np.sum((x.T @ x) ** 2))
    c = float(np.sum((y.T @ y) ** 2))
    return a / np.sqrt(b * c) if b > 0 and c > 0 else float("nan")


def intrinsic_knn(x, ks=(5, 10, 20)):
    """Levina-Bickel local MLE and TwoNN on unique original-space observations."""
    x = np.asarray(x, dtype=np.float64)
    x = np.unique(x, axis=0)
    if len(x) < max(ks) + 2:
        return {"n_unique": len(x), "status": "too_few"}
    nn = NearestNeighbors(n_neighbors=max(ks) + 1).fit(x)
    d, _ = nn.kneighbors(x)
    d = d[:, 1:]
    positive = np.all(d > 1e-12, axis=1)
    d = d[positive]
    out = {"n_unique": len(x), "n_zero_distance_excluded": int(len(x) - len(d))}
    if len(d) < 10:
        return {**out, "status": "too_few_positive"}
    ratio = d[:, 1] / d[:, 0]
    ratio = ratio[ratio > 1]
    out["twonn"] = float(1 / np.mean(np.log(ratio))) if len(ratio) else None
    for k in ks:
        local = (k - 1) / np.sum(np.log(d[:, k - 1, None] / d[:, :k - 1]), axis=1)
        local = local[np.isfinite(local) & (local > 0)]
        out[f"mle_k{k}"] = float(np.median(local)) if len(local) else None
    out["distance_concentration_p90_p10"] = float(np.quantile(d[:, -1], .9) / max(np.quantile(d[:, 0], .1), 1e-12))
    return out


def embedding_metrics(original, embedding, ks=(5, 10, 30), seed=1729):
    original = np.asarray(original, dtype=np.float32)
    embedding = np.asarray(embedding, dtype=np.float32)
    n = len(original)
    if n != len(embedding) or n < 5:
        raise ValueError("Unmatched embedding")
    result = {"n": n}
    for k in ks:
        if k >= n // 2:
            continue
        result[f"trustworthiness_k{k}"] = float(trustworthiness(original, embedding, n_neighbors=k))
        result[f"continuity_k{k}"] = float(trustworthiness(embedding, original, n_neighbors=k))
        ni = NearestNeighbors(n_neighbors=k + 1).fit(original).kneighbors(return_distance=False)[:, :k]
        ne = NearestNeighbors(n_neighbors=k + 1).fit(embedding).kneighbors(return_distance=False)[:, :k]
        result[f"neighbour_overlap_k{k}"] = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ni, ne)]))
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n, 12000)
    b = rng.integers(0, n, 12000)
    good = a != b
    a, b = a[good], b[good]
    dx = np.linalg.norm(original[a] - original[b], axis=1)
    dy = np.linalg.norm(embedding[a] - embedding[b], axis=1)
    result["pair_distance_spearman"] = float(spearmanr(dx, dy).statistic)
    scale = float(np.dot(dx, dy) / max(np.dot(dy, dy), 1e-30))
    result["normalized_stress"] = float(np.sqrt(np.sum((dx - scale * dy) ** 2) / max(np.sum(dx ** 2), 1e-30)))
    return result
