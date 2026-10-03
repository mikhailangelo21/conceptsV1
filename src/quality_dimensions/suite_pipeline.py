"""Resumable smoke/laptop execution for quality-suite v3."""
from __future__ import annotations

import datetime as dt
import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import BudgetExpired, Runner, disk_preflight, environment, resolve, seed_all
from .scoring import score_choices
from .suite_analysis import analyze_suite
from .suite_cache import SuiteActivationCache
from .suite_data import load_suite, suite_plan, tokenize_case, tokenize_question
from .suite_report import suite_report
from .util import atomic_json, atomic_npz, digest, read_json


def _now(): return dt.datetime.now(dt.timezone.utc).isoformat()


def _code_hash():
    return digest({p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))})


def _model_config(config):
    # The V3 spec explicitly requests the already-pinned pilot revision.
    revisions = []
    for path in [Path('runs/pilot_m2-792030855aa5/model.json')]:
        try:
            row = read_json(path)
            if row.get("model") == config["model_id"] and row.get("model_revision"):
                revisions.append(row["model_revision"])
        except (OSError, json.JSONDecodeError):
            pass
    requested=config['model_revision']
    if requested == 'reuse_and_record_existing_pinned_revision':
        if not revisions: raise ValueError('Original pinned model revision not found; set an explicit revision in a separate suite config')
        revision=revisions[0]
    else:
        revision=requested
    return {"model": config["model_id"], "revision": revision, "backend": config["backend"], "attention": "eager"}


def prepare_suite(config_path="configs/quality_suite_v3.json", profile="smoke", run_dir=None, resume=False):
    suite = load_suite(config_path, profile)
    code_hash = _code_hash(); identity_hash = digest([suite["fingerprint"], code_hash])
    run = Path(run_dir) if run_dir else Path("runs") / f"quality_suite_v3_{profile}-{identity_hash[:12]}"
    if run_dir is None and not run.exists():
        local_run = Path.home()/'.local/share/quality-dimensions-llm/runs'/run.name
        local_run.mkdir(parents=True,exist_ok=True)
        run.parent.mkdir(parents=True,exist_ok=True)
        run.symlink_to(local_run,target_is_directory=True)
    identity = {"schema_version": "3.0.0", "profile": profile, "suite_fingerprint": suite["fingerprint"],
                "code_hash": code_hash, "run_fingerprint": identity_hash}
    if (run / "identity.json").exists():
        old = read_json(run / "identity.json")
        if old["run_fingerprint"] != identity_hash:
            raise ValueError("Run identity changed; use a new run directory")
        if not resume:
            raise ValueError(f"Run exists: {run}; use --resume")
    else:
        if run.exists() and any(run.iterdir()):
            raise ValueError("Refusing a nonempty run directory without a matching V3 identity")
        run.mkdir(parents=True, exist_ok=True)
        atomic_json(run / "identity.json", identity)
        atomic_json(run / 'source_snapshot.json', {p.name:p.read_text() for p in sorted(Path(__file__).parent.glob('*.py'))})
        atomic_json(run / "config.json", suite["config"])
        atomic_json(run / "profile.json", suite["profile"])
        atomic_json(run / "environment.json", environment())
        atomic_json(run / "plan.json", suite_plan(suite))
        atomic_json(run / "state.json", {"status": "prepared", "created": _now(), "failures": []})
        atomic_json(run / "label_provenance.json", dict(Counter(r["label_kind"] for r in suite["cases"] + suite["questions"])))
        atomic_json(run / "comparison_to_original_pilot.json", {
            "original": "size-only lexical/context pilot in separate immutable run directories",
            "v3": "15 candidate qualities; controlled binding, numeric, circular and factorial families",
            "shared_claim": "linear accessibility is evaluated; neither run by itself establishes human perceptual geometry",
            "mutation": "No original pilot input, run, or cache path is written by qd suite.",
        })
    return run, suite


def _cache_identity(metadata, config):
    keys = ["model", "model_revision", "tokenizer_revision", "backend", "dtype", "attention",
            "num_hidden_layers", "hidden_size", "versions", "hook", "implementation_hash"]
    result = {k: metadata[k] for k in keys}
    result.update(block_numbers=config["extraction_block_numbers"],
                  readout_semantics=["last_overlapping_contextual_entity_subtoken", "final_prompt_token"],
                  stored_dtype="float32")
    return result


def _static_embeddings(run, runner, tokenized_cases):
    path = run / "static_embeddings.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return {str(key): value for key, value in zip(data["case_ids"], data["values"])}
    unique = {}
    for row in tokenized_cases:
        unique.setdefault(row["case_id"], row["token_ids"])
    values = []
    embedding = runner.model.get_input_embeddings()
    with torch.inference_mode():
        for ids in unique.values():
            tensor = torch.tensor(ids, device=runner.backend)
            values.append(embedding(tensor).float().mean(0).cpu().numpy())
    atomic_npz(path, case_ids=np.array(list(unique)), values=np.asarray(values, np.float32))
    return {key: value for key, value in zip(unique, values)}


def _behavior_outputs(run):
    shards = sorted((run / "behavior_shards").glob("*.json")) if (run / "behavior_shards").exists() else []
    rows = [read_json(path) for path in shards]
    pd.DataFrame(rows).to_csv(run / "behavior_scores.csv", index=False)
    summary = []
    if rows:
        frame = pd.DataFrame(rows)
        for (quality, form, entity_split, template_split), group in frame.groupby(["quality_id", "format", "entity_split", "template_split"]):
            summary.append({"quality_id": quality, "format": form, "n": len(group),
                            "entity_split":entity_split, "template_split":template_split,
                            "n_groups":group.entity_group_id.nunique(),
                            "accuracy": float(group.correct.mean()),
                            "mean_correct_margin": float(group.correct_margin.mean()),
                            "first_choice_rate": float((group.native_margin > 0).mean()),
                            "mean_precision_disagreement": float(group.precision_disagreement.dropna().mean()) if group.precision_disagreement.notna().any() else None})
    pd.DataFrame(summary, columns=["quality_id", "format", "entity_split", "template_split", "n", "n_groups", "accuracy", "mean_correct_margin", "first_choice_rate", "mean_precision_disagreement"]).to_csv(run / "behavior_summary.csv", index=False)
    if rows:
        bias = frame.groupby(['quality_id','format','entity_split','template_split','option_order'],dropna=False).agg(n=('correct','size'),accuracy=('correct','mean'),candidate_zero_rate=('native_margin',lambda s:float((s>0).mean()))).reset_index()
        bias.to_csv(run/'answer_bias.csv',index=False)


def run_suite(config_path="configs/quality_suite_v3.json", profile="smoke", run_dir=None,
              resume=False, max_model_minutes=None, stage="all"):
    run, suite = prepare_suite(config_path, profile, run_dir, resume)
    plan = read_json(run / "plan.json")
    print(json.dumps(plan, indent=2), flush=True)
    state = read_json(run / "state.json"); runner = None; cache = None; tokenized_cases = []
    seed_all(suite['manifest']['seed'])
    try:
        if stage == 'analyze':
            # CPU-only reanalysis must remain usable after the forward budget expires.
            from transformers import AutoTokenizer
            metadata=read_json(run/'model.json')
            tokenizer=AutoTokenizer.from_pretrained(metadata['model'],revision=metadata['tokenizer_revision'],local_files_only=True)
            cm=read_json(run/'activation_cache.json')
            cache=SuiteActivationCache(Path(cm['root']).parents[1],cm['identity'])
            for case in suite['cases']: tokenized_cases.extend(tokenize_case(tokenizer,case,suite['config']['max_tokens']))
            suite['token_counts']={r['case_id']:r['token_count'] for r in tokenized_cases}
            with np.load(run/'static_embeddings.npz',allow_pickle=False) as data:
                static={str(k):v for k,v in zip(data['case_ids'],data['values'])}
            activation={(r['case_id'],r['readout']):cache.get(r) for r in tokenized_cases}
            if any(a is None for a in activation.values()): raise RuntimeError('Extraction incomplete; cached-only analysis cannot run')
            analyze_suite(run,suite,activation,static)
            complete=all((run/name).exists() for name in ['extraction_complete.json','behavior_complete.json','analysis_complete.json'])
            state.update(status='complete' if complete else 'partial',reason='CPU-only cached analysis completed; no model loaded or forward budget consumed.')
            print(f'Report: {run / "report.md"}',flush=True)
            return run
        model_config = _model_config(suite["config"])
        cfg, tokenizer, metadata = resolve(model_config, model_config["revision"])
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Model
        metadata["implementation_hash"] = digest([inspect.getsource(Qwen3DecoderLayer.forward), inspect.getsource(Qwen3Model.forward)])
        if (run/'model.json').exists():
            previous=read_json(run/'model.json')
            if _cache_identity(previous,suite['config']) != _cache_identity(metadata,suite['config']):
                raise ValueError('Model/backend/library identity changed; start a separate run rather than mixing cached scores')
        identity = read_json(run / "identity.json")
        identity.update(model_id=metadata["model"], model_revision=metadata["model_revision"], dtype=metadata["dtype"])
        atomic_json(run / "identity.json", identity); atomic_json(run / "model.json", metadata)
        for case in suite["cases"]:
            tokenized_cases.extend(tokenize_case(tokenizer, case, suite["config"]["max_tokens"]))
        questions = [tokenize_question(tokenizer, row, suite["config"]["max_tokens"]) for row in suite["questions"]]
        suite['token_counts'] = {r['case_id']: r['token_count'] for r in tokenized_cases}
        # One shard per actual tokens/position; labels remain in the run index only.
        unique = {}
        for row in tokenized_cases:
            key = digest([row["token_ids"], row["readout_index"]])
            unique.setdefault(key, row)
        plan.update(actual_activation_requests_after_deduplication=len(unique),
                    extraction_text_forwards_upper_bound=len({tuple(r['token_ids']) for r in unique.values()}),
                    token_count_range=[min(r["token_count"] for r in tokenized_cases + questions), max(r["token_count"] for r in tokenized_cases + questions)])
        atomic_json(run / "plan.json", plan)
        atomic_json(run / "activation_index.json", [
            {"case_id": r["case_id"], "readout": r["readout"],
             "request_key": digest([r["token_ids"], r["readout_index"]])}
            for r in tokenized_cases
        ])
        # Desktop may be evicted by iCloud during a run. V3 owns a separate local
        # cache and never moves or modifies any existing pilot cache.
        cache_root = Path.home()/'.local/share/quality-dimensions-llm/cache'
        cache = SuiteActivationCache(cache_root, _cache_identity(metadata, suite["config"]))
        atomic_json(run/'activation_files.json',[
            dict(case_id=r['case_id'],readout=r['readout'],readout_index=r['readout_index'],
                 token_ids=r['token_ids'],cache_key=cache.row_key(r),cache_file=str(cache.path(r)))
            for r in tokenized_cases])
        atomic_json(run / "activation_cache.json", {"root": str(cache.root), "identity": cache.identity})
        missing = sum(cache.get(row) is None for row in unique.values())
        disk_preflight(metadata, missing * len(suite["config"]["extraction_block_numbers"]) * cfg.hidden_size * 4)
        budget = max_model_minutes or suite["config"]["max_model_minutes"]
        used = sum(s['model_seconds'] for s in read_json(run/'resources.json')['sessions']) if (run/'resources.json').exists() else 0.
        if used >= budget*60: raise BudgetExpired('Cumulative model-execution budget exhausted; cached rows preserved')
        runner = Runner.load(model_config, metadata, budget-used/60)
        atomic_json(run / "model.json", metadata)
        static = _static_embeddings(run, runner, tokenized_cases)
        if stage in {"all", "extract"}:
            by_text = {}
            for row in unique.values(): by_text.setdefault(tuple(row['token_ids']), []).append(row)
            completed = 0
            for i, rows in enumerate(by_text.values(), 1):
                missing_rows = [row for row in rows if cache.get(row) is None]
                if missing_rows:
                    arrays = runner.extract_readouts(missing_rows, suite['config']['extraction_block_numbers'])
                    for row,array in zip(missing_rows,arrays): cache.put(row,array)
                completed += len(rows)
                if i % 25 == 0 or i == len(by_text):
                    print(f"Extracted {completed}/{len(unique)} activation requests ({i}/{len(by_text)} texts); model seconds {runner.elapsed:.1f}", flush=True)
                state.update(status="running", activation_requests_complete=completed, updated=_now())
                atomic_json(run / "state.json", state)
            atomic_json(run / "extraction_complete.json", {"requests": len(unique), "cache_fingerprint": cache.key})
        if stage in {"all", "behavior"}:
            shard_root = run / "behavior_shards"; shard_root.mkdir(exist_ok=True)
            for i, row in enumerate(questions, 1):
                shard = shard_root / f"{row['comparison_id']}.json"
                if not shard.exists():
                    result = score_choices(runner, row)
                    correct_margin = result["native_margin"] if row["correct_index"] == 0 else -result["native_margin"]
                    precise_margin = result["precise_margin"]
                    precise_correct = None if precise_margin is None else (precise_margin if row["correct_index"] == 0 else -precise_margin)
                    atomic_json(shard, {"comparison_id": row["comparison_id"], "quality_id": row["quality_id"],
                        "format": row["format"], "entity_group_id": row["entity_group_id"], "entity_split": row["entity_split"],
                        "template_split": row["template_split"], "option_order": row["option_order"], "correct_index": row["correct_index"],
                        "answer_mapping":row['answer_mapping'], "candidate_meanings":row['candidate_meanings'],
                        "candidate_ids":row['candidate_ids'], "candidate_texts":row['choices'],
                        "native_margin": result["native_margin"], "correct_margin": correct_margin, "correct": correct_margin > 0,
                        "precise_correct_margin": precise_correct,
                        "precision_disagreement": None if precise_correct is None else abs(correct_margin - precise_correct),
                        "single_token": result["single_token"], "log_probs": result["log_probs"], "label_kind": row["label_kind"]})
                if i % 25 == 0 or i == len(questions):
                    print(f"Scored {i}/{len(questions)} behavior questions; model seconds {runner.elapsed:.1f}", flush=True)
            _behavior_outputs(run)
            atomic_json(run / "behavior_complete.json", {"questions": len(questions), "scoring": "exact teacher-forced conditional sequence log probability"})
        if stage in {"all", "analyze"}:
            activation = {(row["case_id"], row["readout"]): cache.get(row) for row in tokenized_cases}
            if any(value is None for value in activation.values()):
                raise RuntimeError("Activation extraction is incomplete")
            analyze_suite(run, suite, activation, static)
            from .suite_causal import causal_followup
            causal_followup(run, suite, runner, tokenizer)
        del runner.model; runner.model = None
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
        complete = all((run / name).exists() for name in ("extraction_complete.json", "behavior_complete.json", "analysis_complete.json"))
        state.update(status="complete" if complete else "partial", reason="All V3 screen stages completed." if complete else f"{stage} stage completed.")
    except BudgetExpired as exc:
        state.update(status="partial", reason=str(exc))
    except (Exception, KeyboardInterrupt) as exc:
        state.setdefault("failures", []).append(f"{type(exc).__name__}: {exc}")
        state.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _behavior_outputs(run)
        state["updated"] = _now()
        if runner is not None:
            resources = read_json(run / "resources.json") if (run / "resources.json").exists() else {"sessions": []}
            resources["sessions"].append({"ended": _now(), "model_seconds": runner.elapsed, "forwards": len(runner.timings)})
            atomic_json(run / "resources.json", resources)
        atomic_json(run / "state.json", state)
        suite_report(run)
    print(f"Report: {run / 'report.md'}", flush=True)
    return run
