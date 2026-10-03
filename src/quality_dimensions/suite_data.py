"""Validated, versioned loader for the quality-suite v3 fixture bank."""
from __future__ import annotations

import csv
import json
import re
import hashlib
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from .util import digest

SCHEMA_VERSION = "3.0.0"
CASE_TYPES = {"lexical", "graded", "controls", "binding", "numeric", "factorial", "hue"}
ENTITY_SPLITS = {"train", "validation", "test"}
TEMPLATE_SPLITS = {"train", "heldout"}
COMBINATION_SPLITS = {"train", "test"}
SOURCE_KINDS = {
    "demo_ordinal", "synthetic_ordinal", "synthetic_order_from_stated_context",
    "synthetic_stated_numeric_order", "synthetic_stated_numeric",
    "synthetic_factorial_specification", "synthetic_color_convention",
}


def _jsonl(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _unique(rows, key, source):
    values = [r[key] for r in rows]
    duplicates = [v for v, n in Counter(values).items() if n > 1]
    if duplicates:
        raise ValueError(f"Duplicate {key} in {source}: {duplicates[:3]}")
    return {r[key]: r for r in rows}


def load_suite(config_path="configs/quality_suite_v3.json", profile="smoke"):
    """Load and validate the suite and exact profile selection.

    Validation deliberately covers the complete bank, while returned cases and
    questions are the immutable profile selection.  The legacy size CSV is never
    read or rewritten here.
    """
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != SCHEMA_VERSION or config.get("suite") != "quality_suite_v3":
        raise ValueError("Unsupported quality-suite schema")
    if config.get("configuration_kind") != "suite_spec_not_legacy_qd_config":
        raise ValueError("Suite config must not be routed through the legacy loader")
    if config.get("batch_size") != 1 or config.get("dtype") != "float16":
        raise ValueError("V3 validates batch size 1 and FP16 device inference")
    root = (config_path.parent.parent / config["dataset_root"]).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Manifest/config schema mismatch")
    for relative, expected in manifest['generated_files'].items():
        # Config may be a separately saved experimental variant; data bank is immutable.
        if relative.startswith('data/quality_suite_v3/'):
            path = root / relative.removeprefix('data/quality_suite_v3/')
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Suite data fingerprint mismatch: {relative}')
    profile_path = root / "profiles" / f"{profile}.json"
    if not profile_path.exists():
        raise ValueError(f"Unknown suite profile: {profile}")
    selected = json.loads(profile_path.read_text())
    if selected.get("name") != profile:
        raise ValueError("Profile name does not match filename")

    all_cases = []
    for case_type in sorted(CASE_TYPES):
        rows = _jsonl(root / "cases" / f"{case_type}.jsonl")
        if any(r.get("case_type") != case_type for r in rows):
            raise ValueError(f"Mismatched case_type in {case_type}.jsonl")
        all_cases.extend(rows)
    case_by_id = _unique(all_cases, "case_id", "cases")
    comparisons = _jsonl(root / "comparisons.jsonl")
    comparison_by_id = _unique(comparisons, "comparison_id", "comparisons")
    contrasts = pd.read_csv(root / "contrasts.csv", keep_default_na=False).to_dict("records")
    contrast_by_id = _unique(contrasts, "contrast_id", "contrasts")
    triplets = pd.read_csv(root / "hue_triplets.csv", keep_default_na=False).to_dict("records")
    triplet_by_id = _unique(triplets, "triplet_id", "hue triplets")
    registry = json.loads((root / "target_registry.json").read_text())

    for row in all_cases:
        if row["entity_split"] not in ENTITY_SPLITS or row["template_split"] not in TEMPLATE_SPLITS:
            raise ValueError(f"Invalid split in {row['case_id']}")
        if row["combination_split"] not in COMBINATION_SPLITS:
            raise ValueError(f"Invalid combination split in {row['case_id']}")
        start, end = row["readout_char_span"]
        if not (0 <= start < end <= len(row["text"])) or row["text"][start:end] != row["readout_entity"]:
            raise ValueError(f"Invalid character readout span in {row['case_id']}")
        if end != len(row['text']) - 1 or not row['text'].endswith(row['readout_entity'] + '.'):
            raise ValueError('Readout must be the final contextualized entity mention')
        if row['label_kind'] not in SOURCE_KINDS or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in row['targets'].values()):
            raise ValueError('Unknown label source or nonfinite target')
        roles = {'plain':'train','numeric':'train','paraphrase':'heldout','reordered':'heldout'}
        if roles.get(row['template_id']) != row['template_split']:
            raise ValueError('Template role mismatch')
        if row['case_type']=='factorial':
            expected = 'test' if sum((i+1)*v for i,v in enumerate(row['factorial_indices'])) % 5 == 1 else 'train'
            if row['combination_split'] != expected:
                raise ValueError('Factorial held-out combination rule mismatch')
        if not set(row["targets"]) <= set(registry):
            raise ValueError(f"Unknown target in {row['case_id']}")
        known_qualities = {q.split("__", 1)[0] for q in registry} | {"hue"}
        if not set(row["quality_ids"]) <= known_qualities:
            raise ValueError(f"Unknown quality in {row['case_id']}")
        if row["case_type"] == "numeric" and not all(k.endswith("__value") for k in row["targets"]):
            raise ValueError("Numeric challenge targets must use __value keys")
        if row["case_type"] != "numeric" and any(k.endswith("__value") for k in row["targets"]):
            raise ValueError("Physical numeric targets cannot be mixed with ordinal cases")
        if row["case_type"] == "factorial" and row.get("combination_split") not in COMBINATION_SPLITS:
            raise ValueError("Factorial row lacks a held-out-combination split")
    entity_splits = {}
    for row in all_cases:
        old = entity_splits.setdefault(row["entity_group_id"], row["entity_split"])
        if old != row["entity_split"]:
            raise ValueError(f"Entity family crosses splits: {row['entity_group_id']}")
    template_splits = {}
    for row in all_cases:
        old = template_splits.setdefault(row["template_id"], row["template_split"])
        if old != row["template_split"]:
            raise ValueError(f"Template role changes across rows: {row['template_id']}")
    for row in comparisons:
        if len(row.get("candidates", [])) != 2 or row.get("correct_index") not in (0, 1):
            raise ValueError(f"Invalid behavioral candidates in {row['comparison_id']}")
        if row["entity_split"] not in ENTITY_SPLITS or row["template_split"] not in TEMPLATE_SPLITS:
            raise ValueError(f"Invalid behavioral split in {row['comparison_id']}")
        if row['candidate_meanings'][row['correct_index']] != row['expected_relation']:
            raise ValueError('Behavioral meaning/answer mapping mismatch')
        if entity_splits.get(row['entity_group_id']) != row['entity_split']:
            raise ValueError('Behavioral entity split mismatch')
    for row in contrasts:
        if row["low_case_id"] not in case_by_id or row["high_case_id"] not in case_by_id:
            raise ValueError(f"Contrast foreign key failure: {row['contrast_id']}")
        low, high = case_by_id[row["low_case_id"]], case_by_id[row["high_case_id"]]
        if low["entity_group_id"] != high["entity_group_id"]:
            raise ValueError(f"Contrast crosses entity families: {row['contrast_id']}")
        if row['kind']=='factorial_one_axis_increase':
            changed=[q for q in low['primitive_axes'] if low['specified_values'][q] != high['specified_values'][q]]
            if changed != [row['changed_axis']]: raise ValueError('Factorial contrast changes multiple axes')
        if row['kind']=='binding_increase':
            if Counter(re.findall(r'\w+',low['text'].lower())) != Counter(re.findall(r'\w+',high['text'].lower())):
                raise ValueError('Binding contrast changes word inventory')
    for row in triplets:
        for key in ("anchor_case_id", "near_case_id", "far_case_id"):
            if row[key] not in case_by_id or case_by_id[row[key]]["case_type"] != "hue":
                raise ValueError(f"Hue triplet foreign key failure: {row['triplet_id']}")

    def choose(ids, mapping, kind):
        if len(ids) != len(set(ids)): raise ValueError(f'Duplicate profile {kind} IDs')
        missing = set(ids) - set(mapping)
        if missing:
            raise ValueError(f"Profile has unknown {kind} IDs: {sorted(missing)[:3]}")
        return [mapping[value] for value in ids]

    cases = choose(selected["case_ids"], case_by_id, "case")
    questions = choose(selected["comparison_ids"], comparison_by_id, "comparison")
    selected_contrasts = choose(selected["contrast_ids"], contrast_by_id, "contrast")
    selected_triplets = choose(selected["triplet_ids"], triplet_by_id, "triplet")
    if len(cases) != selected["case_count"] or len(questions) != selected["comparison_count"]:
        raise ValueError("Profile count metadata mismatch")
    source_kinds = {r["label_kind"] for r in cases + questions}
    if not source_kinds <= SOURCE_KINDS:
        raise ValueError(f"Unknown label source kinds: {sorted(source_kinds - SOURCE_KINDS)}")
    config.update(config_path=str(config_path), dataset_root=str(root), profile=profile)
    fingerprint = digest({"config": config, "manifest": manifest, "profile": selected})
    return dict(config=config, manifest=manifest, profile=selected, cases=cases,
                questions=questions, contrasts=selected_contrasts, triplets=selected_triplets,
                registry=registry, fingerprint=fingerprint)


def tokenize_case(tokenizer, case, max_tokens=256):
    encoded = tokenizer(case["text"], add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"Invalid token length in {case['case_id']}; truncation is forbidden")
    start, end = case["readout_char_span"]
    overlapping = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    if not overlapping or case["text"][start:end] != case["readout_entity"]:
        raise ValueError(f"Tokenizer/entity span mismatch in {case['case_id']}")
    entity_index = overlapping[-1]
    if offsets[entity_index][1] > end and offsets[entity_index][0] < start:
        raise ValueError(f"Entity span lies inside an inseparable token in {case['case_id']}")
    base = dict(case_id=case["case_id"], text=case["text"], token_ids=ids,
                token_offsets=offsets, token_count=len(ids))
    return [dict(base, readout="entity", readout_index=entity_index),
            dict(base, readout="final", readout_index=len(ids) - 1)]


def tokenize_question(tokenizer, question, max_tokens=256):
    row = dict(question)
    row["text"] = row.pop("prefix")
    row["choices"] = {str(i): answer for i, answer in enumerate(row.pop("candidates"))}
    from .scoring import encode_choices
    return encode_choices(tokenizer, row, max_tokens)


def suite_plan(suite, hidden_size=2048):
    cases, questions = suite["cases"], suite["questions"]
    requests = 2 * len(cases)
    independent = {
        "activation_entity_groups": len({r["entity_group_id"] for r in cases}),
        "activation_templates": len({r["template_id"] for r in cases}),
        "behavior_entity_groups": len({r["entity_group_id"] for r in questions}),
    }
    return {
        "profile": suite["config"]["profile"], "activation_cases": len(cases),
        "readouts_per_case": 2, "activation_requests_before_deduplication": requests,
        "extraction_text_forwards_upper_bound":len({r['text'] for r in cases}),
        "behavior_questions": len(questions), "behavior_forwards_lower_bound": len(questions),
        "behavior_forwards_upper_bound": 2 * len(questions),
        "selected_blocks": suite["config"]["extraction_block_numbers"],
        "expected_activation_bytes_before_deduplication": requests * len(suite["config"]["extraction_block_numbers"]) * hidden_size * 4,
        "independent_counts": independent,
        "note": "Constructed variants are repeated scenarios, not independent human observations.",
    }


def import_human_ratings(input_path, output_path, suite_root="data/quality_suite_v3"):
    """Validate actual independent ratings; blank or invented ratings are rejected."""
    input_path, output_path = Path(input_path), Path(output_path)
    frame = pd.read_csv(input_path, keep_default_na=False)
    required = {"concept_id", "quality_id", "rater_id", "rating", "source", "license"}
    if not required <= set(frame):
        raise ValueError(f"Ratings require columns: {sorted(required)}")
    if frame.empty or (frame[["rater_id", "rating", "source", "license"]] == "").any().any():
        raise ValueError("Ratings, independent rater IDs, source, and license must be present")
    valid_concepts = set(pd.read_csv(Path(suite_root) / "concepts.csv")["concept_id"])
    valid_qualities = {r["key"] for r in json.loads((Path(suite_root) / "qualities.json").read_text())}
    if not set(frame.concept_id) <= valid_concepts or not set(frame.quality_id) <= valid_qualities:
        raise ValueError("Ratings contain unknown concept or quality IDs")
    frame["rating"] = pd.to_numeric(frame["rating"], errors="raise")
    if not frame.rating.map(lambda x: float("-inf") < x < float("inf")).all():
        raise ValueError("Ratings must be finite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists(): raise ValueError('Refusing to overwrite existing ratings')
    if Path(suite_root).resolve() in output_path.resolve().parents:
        raise ValueError('Independent ratings must be stored outside generated suite fixtures')
    if frame.duplicated(['concept_id','quality_id','rater_id']).any():
        raise ValueError('Duplicate concept/quality/rater ratings')
    frame.to_csv(output_path, index=False)
    return output_path
