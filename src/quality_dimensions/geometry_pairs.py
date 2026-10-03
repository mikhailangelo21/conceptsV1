"""Auditable candidate output pairs from V3's existing authored noun lexicon.

These are low/high lexical associations, NOT certified counterfactual pairs.
Pair membership follows the original concept split. Cartesian pairs share nouns;
both pair counts and distinct concept counts must be reported.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import row_chunk
from .util import digest


def completion_token(tokenizer, word, prefix="Answer:", leading_space=" "):
    """Validate exact append formatting; no averaging or first-subtoken fallback."""
    base = tokenizer.encode(prefix, add_special_tokens=False)
    completion = leading_space + word
    joined = tokenizer.encode(prefix + completion, add_special_tokens=False)
    continuation = joined[len(base):] if joined[:len(base)] == base else []
    reason = None
    if joined[:len(base)] != base:
        reason = "prefix_tokenization_changed"
    elif len(continuation) != 1:
        reason = "multi_token_completion" if continuation else "empty_completion"
    elif continuation[0] in tokenizer.all_special_ids:
        reason = "special_token"
    elif tokenizer.decode(continuation) != completion:
        reason = "completion_does_not_roundtrip"
    return dict(word=word, completion=completion, token_ids=continuation,
                token_id=continuation[0] if reason is None else None, exclusion=reason)


def candidate_pairs(root, tokenizer, prefix="Answer:", leading_space=" "):
    root = Path(root)
    concepts = pd.read_csv(root / "concepts.csv", keep_default_na=False)
    labels = pd.read_csv(root / "quality_labels.csv", keep_default_na=False)
    merged = labels.merge(concepts, on="concept_id", validate="many_to_one")
    rows = []
    # Five-level labels are 1..5. Fixed low {1,2}, high {4,5}; no result-based choice.
    for (quality, split), group in merged.groupby(["quality_id", "entity_split"], sort=True):
        if quality == "hue":
            continue
        low = group[group.ordinal_score <= 2].sort_values("concept_id")
        high = group[group.ordinal_score >= 4].sort_values("concept_id")
        for a in low.to_dict("records"):
            for b in high.to_dict("records"):
                lo = completion_token(tokenizer, a["lemma"], prefix, leading_space)
                hi = completion_token(tokenizer, b["lemma"], prefix, leading_space)
                reason = lo["exclusion"] or hi["exclusion"]
                if reason is None and lo["token_id"] == hi["token_id"]:
                    reason = "identical_token"
                rows.append(dict(pair_id=digest([quality, a["concept_id"], b["concept_id"]])[:20],
                                 quality=quality, split=split, low=a["lemma"], high=b["lemma"],
                                 low_concept=a["concept_id"], high_concept=b["concept_id"],
                                 low_level=a["ordinal_score"], high_level=b["ordinal_score"],
                                 low_ids=lo["token_ids"], high_ids=hi["token_ids"],
                                 low_token=lo["token_id"], high_token=hi["token_id"],
                                 eligible=reason is None, exclusion=reason,
                                 prefix=prefix, leading_space=leading_space,
                                 source="V3 concepts.csv + quality_labels.csv (authored_demo)",
                                 confounds="Different nouns change identity, category, frequency and many properties; provisional typicality labels; shared nouns make pairs dependent."))
    return rows


def analyze_output_pairs(pairs, head, geometry, random_pairs=512, seed=20261002):
    eligible = [r for r in pairs if r["eligible"]]
    token_ids = sorted({r[k] for r in eligible for k in ["low_token", "high_token"]})
    token_vectors = row_chunk(head.weight, token_ids)
    lookup = {t: v for t, v in zip(token_ids, token_vectors)}
    rng = np.random.default_rng(seed)
    random_ids = rng.integers(0, head.weight.shape[0], (random_pairs, 2))
    # Random pairs are uniform over the same FULL output-row population, independent of labels.
    random_deltas = row_chunk(head.weight, random_ids[:, 1]) - row_chunk(head.weight, random_ids[:, 0])
    records, counts, directions = [], [], {}
    for quality in sorted({r["quality"] for r in pairs}):
        candidates = [r for r in pairs if r["quality"] == quality]
        good = [r for r in eligible if r["quality"] == quality]
        train = [r for r in good if r["split"] == "train"]
        for split in ["train", "validation", "test"]:
            sub = [r for r in good if r["split"] == split]
            counts.append(dict(quality=quality, split=split, candidate_pairs=sum(r["split"] == split for r in candidates),
                               eligible_pairs=len(sub), independent_concepts=len({r[k] for r in sub for k in ["low_concept", "high_concept"]}),
                               training_pairs=len(train)))
        if not train:
            continue
        differences = np.stack([lookup[r["high_token"]] - lookup[r["low_token"]] for r in train])
        mean = differences.mean(0)
        for name, space in [("euclidean", "euclidean"), ("covariance", "output")]:
            direction = geometry.normalize(mean, space)
            directions[(quality, name)] = direction
            for row in good:
                delta = lookup[row["high_token"]] - lookup[row["low_token"]]
                records.append(dict(quality=quality, geometry=name, evaluation=row["split"], pair_id=row["pair_id"],
                                    low_concept=row["low_concept"], high_concept=row["high_concept"],
                                    cosine=float(geometry.cosine(delta, direction, space)),
                                    projection=float(geometry.projection(delta, direction, space))))
            # A separate diagnostic, never relabelled as genuinely held-out concepts.
            if len(train) > 1:
                for i, row in enumerate(train):
                    other = (differences.sum(0) - differences[i]) / (len(train) - 1)
                    if geometry.norm(other, space) > 1e-15:
                        records.append(dict(quality=quality, geometry=name, evaluation="train_pair_LOO_shared_concepts",
                                            pair_id=row["pair_id"], low_concept=row["low_concept"], high_concept=row["high_concept"],
                                            cosine=float(geometry.cosine(differences[i], other, space)),
                                            projection=float(geometry.projection(differences[i], other, space))))
            cosine = geometry.cosine(random_deltas, direction, space)
            projection = geometry.projection(random_deltas, direction, space)
            for i in range(random_pairs):
                records.append(dict(quality=quality, geometry=name, evaluation="uniform_random_output_pairs",
                                    pair_id=f"random_{i}", low_concept=int(random_ids[i, 0]), high_concept=int(random_ids[i, 1]),
                                    cosine=float(cosine[i]), projection=float(projection[i])))
    return records, counts, directions, token_ids, token_vectors
