"""Versioned extension; V3 inputs, completed runs and shard caches are read-only."""
from __future__ import annotations

import datetime as dt
import gc
import hashlib
import inspect
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from transformers import AutoModelForCausalLM, AutoTokenizer

from .geometry import IMPLEMENTATION, Geometry, prepare_metric, sha256_file
from .geometry_analysis import SITE, analyze_geometry
from .geometry_pairs import candidate_pairs, analyze_output_pairs
from .model import Runner, environment
from .suite_cache import SuiteActivationCache
from .suite_data import load_suite, tokenize_case
from .util import atomic_json, atomic_npz, digest, read_json

LOCAL_ROOT = Path.home() / ".local/share/quality-dimensions-llm"


def load_geometry_config(path):
    path = Path(path).resolve()
    c = read_json(path)
    if c.get("schema_version") != IMPLEMENTATION:
        raise ValueError("Unsupported geometry configuration")
    root = path.parent.parent
    c["base_config"] = str((root / c["base_config"]).resolve())
    c["source_run"] = str((root / c["source_run"]).resolve())
    c["config_path"] = str(path)
    return c


def _source_audit(c, suite):
    source = Path(c["source_run"])
    ident, metadata = read_json(source / "identity.json"), read_json(source / "model.json")
    cache = read_json(source / "activation_cache.json")
    if ident["model_revision"] != c["model_revision"] or metadata["model"] != c["model_id"]:
        raise ValueError("Pinned source model mismatch")
    if metadata["backend"] != c["backend"] or metadata["dtype"] != "torch.float16":
        raise ValueError("Legacy activation conversion requires identical device/dtype")
    snapshot = read_json(source / "source_snapshot.json")
    if "before final RMSNorm" not in metadata["hook"] or 28 not in cache["identity"]["block_numbers"]:
        raise ValueError("Source is not the audited complete-block cache")
    # Failing closed prevents interpreting vectors using a different transformer implementation.
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Model
    installed_hash = digest([inspect.getsource(Qwen3DecoderLayer.forward), inspect.getsource(Qwen3Model.forward)])
    if installed_hash != cache["identity"]["implementation_hash"]:
        raise ValueError("Transformer implementation differs from cached V3 states")
    for package in ["torch", "transformers"]:
        if environment()["versions"][package] != cache["identity"]["versions"][package]:
            raise ValueError(f"{package} version differs from V3 cache")
    manifest = read_json(source / "config.json")
    if suite["manifest"]["schema_version"] != "3.0.0" or manifest["extraction_block_numbers"] != cache["identity"]["block_numbers"]:
        raise ValueError("V3 data/cache configuration mismatch")
    audit = dict(source_run=str(source), source_identity=ident,
                 snapshot_sha256=sha256_file(source / "source_snapshot.json"),
                 authoritative_hook="complete block 28 output, pre final RMSNorm",
                 primary_site=SITE, conversion="frozen model.model.norm, same device and FP16 as source",
                 snapshot_module_hashes={name: hashlib.sha256(snapshot[name].encode()).hexdigest()
                                         for name in ["model.py", "hooks.py", "suite_analysis.py", "suite_pipeline.py"]},
                 original_sites="Historical earlier-block/entity-token analyses retained separately; no paper metric is assigned to them.",
                 installed_transformer_implementation_hash=installed_hash)
    return audit, metadata, cache


def _load_model(c):
    tok = AutoTokenizer.from_pretrained(c["model_id"], revision=c["model_revision"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(c["model_id"], revision=c["model_revision"],
                    dtype=torch.float16, attn_implementation="eager", local_files_only=True)
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    if model.config.hidden_size != 2048 or len(model.model.layers) != 28:
        raise ValueError("Unexpected pinned model architecture")
    return model, tok


def _metric(c, model, tokenizer, vocabulary):
    settings = {k: v for k, v in c["metric"].items() if k != "smoke_sample_rows"}
    settings.update(seed=c["seed"], sample_rows=c["metric"]["smoke_sample_rows"] if vocabulary == "smoke" else None)
    return prepare_metric(model.get_output_embeddings(), tokenizer,
                          dict(model=c["model_id"], revision=c["model_revision"], dtype="torch.float16",
                               transformers=environment()["versions"]["transformers"]), settings,
                          LOCAL_ROOT / "cache/quality_geometry_v1/metrics")


def prepare_geometry_metric(config_path, vocabulary="full"):
    c = load_geometry_config(config_path)
    torch.set_num_threads(c["cpu_threads"])
    with threadpool_limits(c["cpu_threads"]):
        model, tokenizer = _load_model(c)
        geometry, root, identity = _metric(c, model, tokenizer, vocabulary)
    print(root, flush=True)
    print(geometry.diagnostics, flush=True)
    return root


def _read_cached_states(c, suite, tokenizer, cache_record):
    if not (Path(cache_record["root"]) / "identity.json").is_file():
        raise FileNotFoundError("Original V3 cache is absent; it will not be created or modified")
    cache = SuiteActivationCache(Path(cache_record["root"]).parents[1], cache_record["identity"])
    # Existing identity must already exist: this extension may not create an old cache.
    if str(cache.root) != cache_record["root"]:
        raise ValueError("Unexpected V3 cache path")
    rows, arrays, ledger = [], [], []
    block = cache.identity["block_numbers"].index(28)
    for case in suite["cases"]:
        row = next(r for r in tokenize_case(tokenizer, case, suite["config"]["max_tokens"]) if r["readout"] == "final")
        value = cache.get(row)
        if value is None:
            raise FileNotFoundError(f"Required original activation is absent: {cache.path(row)}; use the documented new-run extraction command")
        rows.append(row)
        arrays.append(value[block])
        ledger.append(dict(case_id=case["case_id"], path=str(cache.path(row)),
                           row_key=cache.row_key(row), sha256=sha256_file(cache.path(row))))
    return rows, np.stack(arrays), ledger


def _norm_states(model, states, backend, chunk=128):
    result = []
    with torch.inference_mode():
        for start in range(0, len(states), chunk):
            h = torch.as_tensor(states[start:start + chunk], dtype=torch.float16, device=backend)
            result.append(model.model.norm(h).float().cpu().numpy())
    return np.concatenate(result)


def _verify_site(c, suite, rows, raw, converted, model, metadata):
    runner = Runner(model, metadata, c["max_model_minutes"])
    eligible = [i for i, r in enumerate(suite["cases"]) if r["entity_split"] == "train" and r["template_split"] == "train"]
    # Spread predetermined checks across case families without selecting on observed fit.
    eligible.sort(key=lambda i: (suite["cases"][i]["case_type"], suite["cases"][i]["case_id"]))
    selected = [eligible[i] for i in np.linspace(0, len(eligible) - 1, min(len(eligible), c["site_verification"]["n_train_examples"]), dtype=int)]
    records = []
    tol = c["site_verification"]
    for i in selected:
        state, block, native = runner.final_output_readout(rows[i])
        residual_error = float(np.max(np.abs(raw[i] - block)))
        state_error = float(np.max(np.abs(converted[i] - state)))
        with torch.inference_mode():
            reproduced = model.get_output_embeddings()(torch.as_tensor(converted[i], dtype=torch.float16,
                            device=c["backend"])[None, None, :])[0, 0].float().cpu().numpy()
        logit_error = float(np.max(np.abs(native - reproduced)))
        passed = (residual_error <= tol["cached_residual_atol"] and state_error <= tol["converted_state_atol"]
                  and np.allclose(native, reproduced, atol=tol["native_logits_atol"], rtol=tol["native_logits_rtol"]))
        records.append(dict(case_id=rows[i]["case_id"], n_logits=len(native), cached_block_max_error=residual_error,
                            converted_state_max_error=state_error, native_logit_max_error=logit_error, passed=bool(passed)))
    result = dict(site=SITE, tolerances=tol, examples=records, passed=all(r["passed"] for r in records),
                  forwards=len(records), model_seconds=runner.elapsed, peak_rss_bytes=max(r["rss_bytes"] for r in runner.resources))
    if not result["passed"]:
        raise ValueError(f"Final readout conversion/logit verification failed: {result}")
    return result


def _archive_controls(source, run):
    copied = []
    for name in ["linear_metrics.csv", "linear_predictions.csv", "model_selection.json", "paired_summary.csv",
                 "factorial_selectivity.csv", "numeric_challenges.csv", "circular_neighborhoods.csv",
                 "behavior_summary.csv", "independent_counts_by_family.json", "identity.json", "analysis_complete.json"]:
        path = source / name
        if path.exists():
            target = run / "v3_historical_control" / name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            copied.append(dict(file=name, sha256=sha256_file(path), source=str(path)))
    atomic_json(run / "v3_historical_control" / "provenance.json", copied)


def verify_historical_transport(source, geometry):
    """Check actual saved V3 coefficients, including undoing their StandardScaler.

    At earlier/entity sites this is ONLY an invertible-coordinate algebra check;
    no semantic interpretation of the output covariance at those sites is made.
    """
    files = read_json(source / "activation_files.json")
    blocks = read_json(source / "activation_cache.json")["identity"]["block_numbers"]
    matrices, records = {}, []
    for path in sorted((source / "fits").glob("*.npz")):
        with np.load(path, allow_pickle=False) as fit:
            block, readout = int(fit["block"]), str(fit["readout"])
            key = (block, readout)
            if key not in matrices:
                selected = [r for r in files if r["readout"] == readout][:3]
                vectors = []
                for r in selected:
                    with np.load(r["cache_file"], allow_pickle=False) as shard:
                        vectors.append(shard["activation"][blocks.index(block)])
                matrices[key] = np.asarray(vectors, np.float64)
            x = matrices[key]
            coef = fit["ridge_coef"] / fit["feature_scale"]
            intercept = fit["ridge_intercept"] - fit["feature_mean"] @ coef.T
            original = ((x - fit["feature_mean"]) / fit["feature_scale"]) @ fit["ridge_coef"].T + fit["ridge_intercept"]
            transported = geometry.context(x) @ geometry.transport_coefficients(coef).T + intercept
            error = float(np.max(np.abs(original - transported)))
            if error > 1e-8:
                raise AssertionError(f"Historical coefficient transport failed: {path.name}: {error}")
            records.append(dict(quality=path.stem, block=block, readout=readout, n_vectors=len(x),
                                max_absolute_error=error, source_fit_sha256=sha256_file(path),
                                scope="coordinate algebra only; no paper-grounded metric claim at legacy site"))
    return records


def run_geometry(config_path, profile="smoke", run_dir=None, vocabulary=None, stage="all", resume=False):
    c = load_geometry_config(config_path)
    vocabulary = vocabulary or ("smoke" if profile == "smoke" else "full")
    suite = load_suite(c["base_config"], profile)
    source_audit, metadata, cache_record = _source_audit(c, suite)
    sources = {p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))}
    identity = dict(implementation=IMPLEMENTATION, config=c, profile=profile, vocabulary=vocabulary,
                    suite_fingerprint=suite["fingerprint"], code_hash=digest(sources),
                    source_identity=source_audit["source_identity"])
    key = digest(identity)
    run = Path(run_dir) if run_dir else Path("runs") / f"quality_geometry_v1_{profile}-{key[:12]}"
    if run_dir is None and not run.exists():
        target = LOCAL_ROOT / "runs" / run.name
        target.mkdir(parents=True, exist_ok=True)
        run.parent.mkdir(exist_ok=True)
        run.symlink_to(target, target_is_directory=True)
    if (run / "identity.json").exists():
        if read_json(run / "identity.json") != identity or not resume:
            raise ValueError("Existing run requires matching identity and --resume; use a new directory after code/config changes")
    else:
        if run.exists() and any(run.iterdir()):
            raise ValueError("Refusing to write a nonempty unrecognized run directory")
        run.mkdir(parents=True, exist_ok=True)
        atomic_json(run / "identity.json", identity)
        atomic_json(run / "source_snapshot.json", sources)
        atomic_json(run / "config.json", c)
        atomic_json(run / "source_audit.json", source_audit)
        atomic_json(run / "environment.json", environment())
        atomic_json(run / "profile.json", suite["profile"])
        _archive_controls(Path(c["source_run"]), run)
    print(f"Geometry run: {run}", flush=True)
    started = time.monotonic()
    atomic_json(run / "state.json", dict(status="running", stage=stage, updated=dt.datetime.now(dt.timezone.utc).isoformat()))
    torch.set_num_threads(c["cpu_threads"])
    try:
        with threadpool_limits(c["cpu_threads"]):
            if stage != "analyze":
                print("Loading pinned output head; preparing/reusing metric", flush=True)
                model, tokenizer = _load_model(c)
                geometry, metric_root, metric_identity = _metric(c, model, tokenizer, vocabulary)
                atomic_json(run / "metric_reference.json", dict(root=str(metric_root), identity=metric_identity,
                                                               diagnostics=geometry.diagnostics))
                print("Evaluating token eligibility and output directions", flush=True)
                pairs = candidate_pairs(suite["config"]["dataset_root"], tokenizer,
                                        c["output_pairs"]["prefix"], c["output_pairs"]["leading_space"])
                atomic_json(run / "output_pair_candidates.json", pairs)
                records, counts, directions, token_ids, token_vectors = analyze_output_pairs(
                    pairs, model.get_output_embeddings(), geometry, c["output_pairs"]["random_pairs"], c["seed"])
                pd.DataFrame(records).to_csv(run / "output_alignment.csv", index=False)
                pd.DataFrame(counts).to_csv(run / "output_pair_counts.csv", index=False)
                atomic_npz(run / "output_directions.npz", **{f"{q}__{g}": v for (q, g), v in directions.items()})
                atomic_npz(run / "output_tokens.npz", token_ids=np.array(token_ids), vectors=token_vectors)
                print("Reading original final-block shards (read-only)", flush=True)
                rows, raw, ledger = _read_cached_states(c, suite, tokenizer, cache_record)
                atomic_json(run / "source_activation_ledger.json", ledger)
                model.to(c["backend"])
                converted = _norm_states(model, raw, c["backend"])
                verification = _verify_site(c, suite, rows, raw, converted, model, metadata)
                verification["norm_source"] = inspect.getsource(type(model.model.norm).forward)
                verification["norm_weight_sha256"] = hashlib.sha256(model.model.norm.weight.detach().cpu().float().numpy().tobytes()).hexdigest()
                atomic_json(run / "site_verification.json", verification)
                atomic_npz(run / "final_states.npz", case_ids=np.array([r["case_id"] for r in suite["cases"]]), values=converted)
                atomic_json(run / "final_states_identity.json", dict(site=SITE, source_cache=cache_record,
                            ledger_sha256=sha256_file(run / "source_activation_ledger.json"),
                            norm_sha256=verification["norm_weight_sha256"],
                            states_sha256=sha256_file(run / "final_states.npz")))
                del model, tokenizer, raw
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                atomic_json(run / "preparation_complete.json", dict(status="complete", site=SITE,
                            n_cases=len(converted), source_shards=len({r['path'] for r in ledger}),
                            model_seconds=verification["model_seconds"], interventions_run=0))
            else:
                ref = read_json(run / "metric_reference.json")
                geometry = Geometry.load(ref["root"], ref["identity"])
                if sha256_file(run / "final_states.npz") != read_json(run / "final_states_identity.json")["states_sha256"]:
                    raise ValueError("Final state artifact integrity failure")
                with np.load(run / "final_states.npz", allow_pickle=False) as a:
                    if list(a["case_ids"]) != [r["case_id"] for r in suite["cases"]]:
                        raise ValueError("Case ordering mismatch")
                    converted = a["values"].copy()
                with np.load(run / "output_directions.npz", allow_pickle=False) as a:
                    directions = {tuple(k.split("__")): a[k].copy() for k in a.files}
            if stage != "prepare":
                atomic_json(run / "historical_coefficient_transport.json", verify_historical_transport(Path(c["source_run"]), geometry))
                results = analyze_geometry(run, suite, converted.astype(np.float64), geometry, directions, c["ridge_alphas"])
                atomic_json(run / "analysis_complete.json", results)
                from .geometry_report import geometry_report
                geometry_report(run)
            atomic_json(run / "state.json", dict(status="prepared" if stage == "prepare" else "complete",
                        elapsed_wall_seconds=time.monotonic() - started, interventions_run=0,
                        updated=dt.datetime.now(dt.timezone.utc).isoformat()))
    except Exception as exc:
        atomic_json(run / "state.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
        raise
    print(f"Completed: {run}", flush=True)
    return run


def run_final_steering(run_dir, case_id, quality, strength, measurement_reviewed=False):
    """Explicit opt-in output-head intervention using saved final states and head rows.

    Does not re-run the transformer. Reports algebraic logits for the saved eligible
    output vocabulary subset, without claiming changes to earlier computations.
    """
    from .geometry import steer_final_state
    if not measurement_reviewed:
        raise ValueError("Steering is disabled by default; review measurement results before --measurement-reviewed")
    if not np.isfinite(strength):
        raise ValueError("Strength must be finite")
    run = Path(run_dir)
    ref = read_json(run / "metric_reference.json")
    geometry = Geometry.load(ref["root"], ref["identity"])
    with np.load(run / "final_states.npz", allow_pickle=False) as a:
        index = list(a["case_ids"]).index(case_id)
        state = a["values"][index].astype(float)
    with np.load(run / "output_directions.npz", allow_pickle=False) as a:
        direction = a[f"{quality}__covariance"]
    with np.load(run / "output_tokens.npz", allow_pickle=False) as a:
        ids, vectors = a["token_ids"], a["vectors"]
    edited = steer_final_state(state, direction, geometry, strength, measurement_reviewed=True)
    result = dict(case_id=case_id, quality=quality, strength=strength, site=SITE,
                  intervention="lambda <- lambda + strength * Sigma_effective^-1 gamma_bar",
                  measurement_reviewed=True, model_forwards=0,
                  scope="float64 algebra on saved FP16 model state/head rows; eligible output tokens only",
                  token_ids=ids, before=state @ vectors.T, after=edited @ vectors.T,
                  delta=(edited - state) @ vectors.T)
    path = run / "optional_steering" / f"{digest([case_id, quality, strength])[:16]}.json"
    if path.exists():
        raise ValueError("This steering artifact already exists")
    atomic_json(path, result)
    return path
