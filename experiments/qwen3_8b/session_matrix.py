"""Compare paired storage/consumer code on one frozen multi-session trace.

All timings include full-logit observation. This is a synchronous development
experiment, not HTTP serving, an SLO result or a construction-method comparison.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from transformers import AutoTokenizer

from engines.qwen3_8b_kv.engine import Qwen3Engine
from engines.qwen3_8b_kv.kv_store import KVStore as BaseStore
from experiments.qwen3_8b.baseline_consumer import SessionRuntime as BaseConsumer
from experiments.qwen3_8b.identity import file_sha, model_identity


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2)+"\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_trace(tokenizer):
    lengths = [512, 1024, 2048, 3072, 2048, 1024]
    order = [0, 1, 2, 0, 3, 1, 4, 0, 5, 2, 0, 3, 1, 0, 4, 5, 0, 2, 1, 3, 0, 4, 0, 5]
    seen, result = set(), []
    for index, sid in enumerate(order):
        if sid not in seen:
            text = f"Session {sid}: analyze how a model uses stored context to answer new questions. "
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids = (ids*((lengths[sid]+len(ids)-1)//len(ids)))[:lengths[sid]]
            seen.add(sid)
        else:
            ids = tokenizer.encode(f"\nTurn {index} for session {sid}: explain the next important detail.\n", add_special_tokens=False)
        result.append({"index": index, "session_id": f"s{sid}", "input_ids": ids, "max_new_tokens": 4})
    return result


def execute(engine, consumer_class, store_class, events, gpu_budget, output, base_allocated, golden=None):
    output.mkdir(parents=True, exist_ok=False)
    store = store_class(max_bytes=4*1024**3)
    runner = consumer_class(engine, store, gpu_budget_bytes=gpu_budget)
    gc.collect()
    torch.cuda.synchronize()
    # Captured by the trusted caller before importing or constructing candidates.
    torch.cuda.reset_peak_memory_stats()
    rows, seen = [], set()
    failure = None
    start_run = time.perf_counter()
    try:
        for event in events:
            digests = []
            def observe(sid, step, logits):
                if sid != event["session_id"] or step != len(digests):
                    raise AssertionError("observer request/index mismatch")
                raw = logits.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                digests.append(hashlib.sha256(raw).hexdigest())
            seen.add(event["session_id"])
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = runner.turn(event["session_id"], event["input_ids"], event["max_new_tokens"], observer=observe)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter()-start)*1000
            extra = torch.cuda.memory_allocated()-base_allocated
            row = {"index": event["index"], "session_id": event["session_id"], "elapsed_ms": elapsed_ms,
                   "token_ids": result["token_ids"], "token_logprobs": result["token_logprobs"],
                   "logit_sha256": digests, "gpu_extra_allocated_bytes": extra,
                   "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated()-base_allocated,
                   "reported_stats": runner.stats(), "status": "pass"}
            if len(digests) != event["max_new_tokens"] or len(row["token_ids"]) != event["max_new_tokens"]:
                row.update(status="fail", error="missing actual outputs or observations")
            if golden is not None:
                reference = golden[event["index"]]
                for field in ("token_ids", "token_logprobs", "logit_sha256"):
                    if row[field] != reference[field]:
                        row.update(status="fail", error=f"reference mismatch: {field}", expected=reference[field])
                        break
            # The allowance covers tensor-allocation rounding, not another KV copy.
            if extra > gpu_budget+2*1024**2:
                row.update(status="fail", error="completed-turn GPU budget exceeded")
            if store.used_bytes > 4*1024**3:
                row.update(status="fail", error="host committed-payload budget exceeded")
            rows.append(row)
            with (output/"requests.jsonl").open("a") as file:
                file.write(json.dumps(row)+"\n")
            if row["status"] != "pass":
                raise AssertionError(row["error"])
    except Exception as error:
        failure = {"error_type": type(error).__name__, "error": str(error),
                   "event_index": event["index"], "session_id": event["session_id"],
                   "observed_logit_sha256": digests, "observed_steps": len(digests),
                   "elapsed_ms_until_failure": (time.perf_counter()-start)*1000}
        write(output/"failure.json", failure)
    finally:
        cleanup_errors = []
        for sid in seen:
            try:
                runner.close(sid)
            except Exception as error:
                cleanup_errors.append({"session_id": sid, "error": str(error)})
        gc.collect()
        torch.cuda.synchronize()
        remaining = torch.cuda.memory_allocated()-base_allocated
        if remaining > 2*1024**2 or store.used_bytes != 0 or cleanup_errors:
            failure = failure or {"error": "state not released on close"}
        cleanup = {"gpu_remaining_bytes": remaining, "host_remaining_payload_bytes": store.used_bytes,
                   "errors": cleanup_errors}
    summary = {"status": "pass" if failure is None and len(rows) == len(events) else "fail",
               "scope": "observer-instrumented sequential development trace", "gpu_session_budget_bytes": gpu_budget,
               "turns": len(rows), "wall_seconds_including_checks_and_cleanup": time.perf_counter()-start_run,
               "sum_turn_ms": sum(r["elapsed_ms"] for r in rows), "cleanup": cleanup}
    write(output/"summary.json", summary)
    print(json.dumps({"run": output.name, **summary}), flush=True)
    del runner, store
    gc.collect()
    torch.cuda.synchronize()
    if torch.cuda.memory_allocated() > base_allocated+2*1024**2:
        raise RuntimeError("candidate leaked GPU state after object deletion; stop evaluation")
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    root, candidate, out = Path.cwd(), Path(args.candidate), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    inventory = subprocess.check_output(["nvidia-smi", "-i", args.gpu_uuid,
        "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    memory, utilization = [int(v) for v in inventory.split(",")]
    write(out/"preflight.json", {"gpu_uuid": args.gpu_uuid, "observed": inventory,
                                "status": "busy" if memory >= 500 or utilization >= 5 else "idle"})
    if memory >= 500 or utilization >= 5:
        raise RuntimeError(f"selected GPU is busy: {inventory}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    torch.set_num_threads(4)
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-") == args.gpu_uuid.removeprefix("GPU-")
    task = json.loads((root/"experiments/qwen3_8b/model.json").read_text())
    identity = model_identity(args.model_path, task)
    assert file_sha(candidate/"engine.py") == file_sha(root/"engines/qwen3_8b_kv/engine.py")
    paths = [root/"experiments/qwen3_8b/session_matrix.py", root/"experiments/qwen3_8b/baseline_consumer.py",
             root/"engines/qwen3_8b_kv/engine.py", root/"engines/qwen3_8b_kv/kv_store.py",
             candidate/"consumer.py", candidate/"kv_store.py", root/"experiments/qwen3_8b/session-contract.md"]
    source = {str(p): file_sha(p) for p in paths}
    write(out/"provenance.json", {"source_sha256": source, "model_files": identity,
        "gpu_uuid": args.gpu_uuid, "preflight": inventory, "torch": torch.__version__,
        "is_independent_construction_comparison": False, "scope": "cross-combined component ablations of one joint candidate"})
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    events = make_trace(tokenizer)
    write(out/"trace.json", {"scope": "synthetic development sessions, no heldout claim", "events": events})
    engine = Qwen3Engine(args.model_path)
    with torch.inference_mode():
        warm = engine.forward([1, 2])
        del warm
        gc.collect()
        torch.cuda.synchronize()
        base_allocated = torch.cuda.memory_allocated()
        reference, golden = execute(engine, BaseConsumer, BaseStore, events, 4*1024**3,
                                    out/"resident-reference", base_allocated)
        if reference["status"] != "pass":
            raise RuntimeError("no-offload reference failed")
        new_store = load_module("joint_candidate_store", candidate/"kv_store.py").KVStore
        new_consumer = load_module("joint_candidate_consumer", candidate/"consumer.py").SessionRuntime
        if torch.cuda.memory_allocated() > base_allocated:
            raise RuntimeError("candidate import allocated hidden CUDA state outside runtime ownership")
        variants = {"S0-E0": (BaseStore, BaseConsumer), "S1-E0": (new_store, BaseConsumer),
                    "S0-E1": (BaseStore, new_consumer), "S1-E1": (new_store, new_consumer)}
        warmups = {}
        for label, (store, consumer) in variants.items():
            warmup, _ = execute(engine, consumer, store, events, 768*1024**2,
                                out/f"{label}-warmup", base_allocated, golden=golden)
            warmups[label] = warmup
        summaries = {key: [] for key in variants}
        for repeat in range(args.repeats):
            order = list(variants) if repeat % 2 == 0 else list(reversed(variants))
            for label in order:
                store, consumer = variants[label]
                summary, _ = execute(engine, consumer, store, events, 768*1024**2,
                                     out/f"{label}-r{repeat}", base_allocated, golden=golden)
                summaries[label].append(summary)
        # Same high-budget condition for the generated pair: do extra host copies hurt?
        unconstrained, _ = execute(engine, new_consumer, new_store, events, 4*1024**3,
                                   out/"resident-joint-control", base_allocated, golden=golden)
    result = {label: {"status": "pass" if all(r["status"] == "pass" for r in rows) else "fail",
                     "sum_turn_ms": [r["sum_turn_ms"] for r in rows],
                     "median_sum_turn_ms": statistics.median(r["sum_turn_ms"] for r in rows)
                         if all(r["status"] == "pass" for r in rows) else None}
              for label, rows in summaries.items()}
    assert all(file_sha(p) == digest for p, digest in source.items()), "source changed during evaluation"
    write(out/"summary.json", {"status": "measured" if all(r["status"] == "pass" for r in result.values())
                               and unconstrained["status"] == "pass"
                               and all(r["status"] == "pass" for r in warmups.values()) else "incomplete",
        "scope": "single joint construction, synthetic trace, cross-combined components; not independent search or SOTA comparison",
        "warmup_policy": "one complete trace per component combination, separately recorded and excluded from timing medians",
        "warmups": warmups, "variants": result, "resident_reference": reference, "resident_joint_control": unconstrained})


if __name__ == "__main__":
    main()
