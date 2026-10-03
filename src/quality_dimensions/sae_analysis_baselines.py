"""Train-only lexical controls on the exact readouts used for SAE probes."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .sae_analysis import _predict_one, _qualities, _save_rows
from .sae_analysis_core import checksum, read_collection
from .util import atomic_json


def lexical_baselines(run_dir):
    run = Path(run_dir).expanduser().resolve()
    config = json.loads((run / "config.json").read_text())
    source = (Path(__file__).resolve().parents[2] / config["source_run"]).resolve()
    collection = read_collection(source)
    selected = json.loads((run / "sample_ids.json").read_text())
    lookup = {row["readout_id"]: row for row in collection["readouts"]}
    output = run / "baselines"
    output.mkdir(exist_ok=True)
    metrics, predictions = [], []
    for readout in ("entity", "final"):
        ids = [rid for split in ("train", "validation", "test", "challenge")
               for rid in selected[readout][split]]
        train_ids = set(selected[readout]["train"])
        train_indices = [i for i, rid in enumerate(ids) if rid in train_ids]
        if len(ids) != len(set(ids)) or len(train_indices) != len(train_ids):
            raise ValueError("Lexical baseline readout IDs or fit IDs are inconsistent")
        texts = [lookup[rid]["text"] for rid in ids]
        length = np.asarray([lookup[rid]["token_count"] for rid in ids], dtype=np.float32)
        centre = float(length[train_indices].mean())
        scale = float(length[train_indices].std()) or 1.0
        x_length = ((length - centre) / scale).reshape(-1, 1)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                     max_features=10000, sublinear_tf=True,
                                     dtype=np.float32)
        vectorizer.fit([texts[i] for i in train_indices])
        x_text = vectorizer.transform(texts)
        joblib.dump(vectorizer, output / f"{readout}_tfidf.joblib")
        atomic_json(output / f"{readout}_fit.json",
                    {"fit_ids": [ids[i] for i in train_indices], "all_ids": ids,
                     "token_count_train_mean": centre, "token_count_train_sd": scale,
                     "text_vocabulary_size": len(vectorizer.vocabulary_),
                     "text_ngram_range": [1, 2], "text_min_df": 2,
                     "text_max_features": 10000, "text_sublinear_tf": True})
        for name, matrix in (("PROMPT_LENGTH", x_length), ("TEXT_TFIDF", x_text)):
            for quality in _qualities(collection):
                rows, summary = _predict_one(matrix, ids, collection, quality, name,
                                             -1, readout, "core")
                predictions.extend(rows)
                metrics.extend(summary)
    _save_rows(output / "metrics.csv", metrics)
    _save_rows(output / "predictions.csv", predictions)
    atomic_json(output / "provenance.json",
                {"source_sha256": checksum(Path(__file__)),
                 "collection_manifest_sha256": checksum(source / "manifest.json"),
                 "sample_ids_sha256": checksum(run / "sample_ids.json"),
                 "fit_rule": "same eligible train readouts as SAE probes; validation selects ridge alpha",
                 "purpose": "lexical controls, not independent semantic representations"})
    return output / "metrics.csv"
