"""Actual 8B host snapshot validation and bounded development microbenchmark.

Preservation is relative to the fixed engine, not an independent model oracle.
No HTTP service, SLO, NVMe, eviction or joint-search performance is claimed.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import weakref

import torch
from transformers import AutoTokenizer

from engines.qwen3_8b_kv.engine import Qwen3Engine, RequestState
from engines.qwen3_8b_kv.kv_store import KVStore
from experiments.qwen3_8b.identity import file_sha, model_identity

REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2)+"\n")


def byte_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.detach().cpu().contiguous().view(torch.uint8),
        b.detach().cpu().contiguous().view(torch.uint8))


def tensor_sha(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensors(state):
    return [t for pair in state.kv for t in pair]+[state.logits]


def cpu_snapshot(state):
    return [t.detach().cpu().clone() for t in tensors(state)]


def assert_snapshot(state, expected):
    actual = tensors(state)
    assert len(actual) == len(expected), "snapshot tensor count mismatch"
    for i, (a, b) in enumerate(zip(actual, expected)):
        if not byte_equal(a, b):
            raise AssertionError(f"snapshot bytes differ: request={state.request_id}, tensor={i}, "
                                 f"actual_sha256={tensor_sha(a)}, expected_sha256={tensor_sha(b)}")


def identity(engine, state):
    return {"model_revision": REVISION, "config_sha256": engine.config_sha256,
            "dtype": "bfloat16", "adapter": None, "position_start": 0,
            "prefix": state.prompt_ids+state.generated}


def make_states(engine, inputs, checkpoint, output_tokens=16):
    states = [RequestState(f"s{i}", ids, checkpoint+output_tokens, True, False)
              for i, ids in enumerate(inputs)]
    engine._forward_states(inputs, states)
    for _ in range(checkpoint):
        engine.next_tokens(states)
    return states


def correctness_case(engine, inputs, checkpoint, name, store_type=KVStore):
    output_tokens = 16
    control = make_states(engine, inputs, checkpoint, output_tokens)
    expected = [cpu_snapshot(s) for s in control]
    scores, tokens = [], []
    for _ in range(output_tokens):
        scores.append([s.logits.cpu().clone() for s in control])
        tokens.append(engine.next_tokens(control))
    del control
    gc.collect()
    torch.cuda.synchronize()
    states = make_states(engine, inputs, checkpoint, output_tokens)
    for state, snapshot in zip(states, expected):
        assert_snapshot(state, snapshot)  # engine repeatability before storage
    store = store_type(max_bytes=1024**3)
    identities = [identity(engine, s) for s in states]
    payload = sum(t.numel()*t.element_size() for s in states for t in tensors(s))
    refs = [weakref.ref(t) for s in states for t in tensors(s)]
    before_put = torch.cuda.memory_allocated()
    for state, ident in zip(states, identities):
        store.put(state.request_id, state.kv, state.logits, ident)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= before_put, "host store retained additional CUDA storage"
    assert store.used_bytes == payload
    before = torch.cuda.memory_allocated()
    for state in states:
        state.kv = state.logits = None
    gc.collect()
    torch.cuda.synchronize()
    released = before-torch.cuda.memory_allocated()
    assert all(ref() is None for ref in refs), "original CUDA tensors retained"
    assert released >= payload, f"CUDA storage still retained: {released} < {payload}"
    assert torch.cuda.memory_allocated() <= before_put-payload, "snapshot still owns CUDA storage"
    before_get = torch.cuda.memory_allocated()
    for state, ident, snapshot in zip(states, identities, expected):
        state.kv, state.logits = store.get(state.request_id, ident, "cuda")
        assert_snapshot(state, snapshot)
    # A declared corrupt-restore control must be rejected by the real byte gate.
    saved = states[0].kv[0][0].clone()
    states[0].kv[0][0][0, 0, 0, 0].add_(1)
    try:
        assert_snapshot(states[0], expected[0])
    except AssertionError:
        negative_rejected = True
    else:
        raise AssertionError("corrupt restore escaped byte comparison")
    states[0].kv[0][0].copy_(saved)
    del saved
    for step in range(output_tokens):
        for state, expected_logits in zip(states, scores[step]):
            if not byte_equal(state.logits, expected_logits):
                raise AssertionError(f"continuation logits differ: passed_steps={step}, request={state.request_id}, "
                    f"actual_sha256={tensor_sha(state.logits)}, expected_sha256={tensor_sha(expected_logits)}")
        actual_tokens = engine.next_tokens(states)
        if actual_tokens != tokens[step]:
            raise AssertionError(f"continuation tokens differ: passed_steps={step}, "
                f"requests={[s.request_id for s in states]}, actual={actual_tokens}, expected={tokens[step]}")
    for state in states:
        state.kv = state.logits = None
    gc.collect()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= before_get, "store retained a CUDA copy after get"
    for state in states:
        store.delete(state.request_id)
    assert store.used_bytes == 0
    return {"case": name, "status": "pass", "batch": len(inputs), "input_tokens": len(inputs[0]),
            "checkpoint_after_decode_tokens": checkpoint, "continuation_tokens_per_request": output_tokens,
            "full_logits_bitwise_equal": True, "snapshot_bytes_equal": True,
            "payload_bytes": payload, "cuda_allocator_bytes_released": released,
            "no_additional_cuda_storage_after_put": True,
            "no_retained_cuda_copy_after_get": True,
            "original_cuda_tensor_references_released": True, "corrupt_restore_rejected": negative_rejected}


def current_stream_case():
    store = KVStore(max_bytes=1024**2)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        k = torch.arange(4096, device="cuda", dtype=torch.float32).reshape(1, 2, 32, 64)
        v = k+3
        scores = torch.tensor([1., -2., 3.], device="cuda")
        store.put("stream", [(k, v)], scores, {"prefix": [3, 7]})
        kv, restored_scores = store.get("stream", {"prefix": [3, 7]}, "cuda")
        # The returned tensors must be usable on the caller's current stream.
        delta = (kv[0][1]-kv[0][0]).sum().item()
        assert delta == 12288
        assert byte_equal(k, kv[0][0]) and byte_equal(scores, restored_scores)
    store.delete("stream")
    return {"case": "nondefault_current_cuda_stream", "status": "pass"}


def reject_hidden_cuda_copies(engine, inputs):
    class RetainOnPut(KVStore):
        def put(self, key, kv, logits, ident):
            super().put(key, kv, logits, ident)
            self.hidden_cuda = kv[0][0].clone()

    class RetainOnGet(KVStore):
        def get(self, key, ident, device):
            kv, logits = super().get(key, ident, device)
            self.hidden_cuda = kv[0][0].clone()
            return kv, logits

    results = []
    for bad_store, signature in [(RetainOnPut, "host store retained additional CUDA storage"),
                                 (RetainOnGet, "store retained a CUDA copy after get")]:
        try:
            correctness_case(engine, inputs, 0, bad_store.__name__, store_type=bad_store)
        except AssertionError as error:
            assert str(error) == signature, f"negative control failed at another gate: {error}"
            results.append({"mutation": bad_store.__name__, "status": "rejected", "signature": signature})
        else:
            raise AssertionError(f"hidden CUDA copy escaped gate: {bad_store.__name__}")
        gc.collect()
        torch.cuda.synchronize()
    return results


def benchmark_case(engine, ids, name, journal):
    state = RequestState(name, ids, 16, True, False)
    torch.cuda.synchronize()
    start = time.perf_counter()
    engine.prefill(state)
    torch.cuda.synchronize()
    initial_prefill_ms = (time.perf_counter()-start)*1000
    store = KVStore(max_bytes=1024**3)
    ident = identity(engine, state)
    start = time.perf_counter()
    store.put(name, state.kv, state.logits, ident)
    torch.cuda.synchronize()
    save_ms = (time.perf_counter()-start)*1000
    payload = store.used_bytes
    del state
    gc.collect()
    torch.cuda.synchronize()
    without_state = torch.cuda.memory_allocated()
    observations = []
    expected_tokens = None
    # 3 ABBA blocks, 6 observations per path; fixed after correctness, no tuning.
    for repeat in range(3):
        for mode in ("recompute", "restore", "restore", "recompute"):
            state = RequestState(name, ids, 16, True, False)
            torch.cuda.synchronize()
            start = time.perf_counter()
            if mode == "recompute":
                engine.prefill(state)
            else:
                state.kv, state.logits = store.get(name, ident, "cuda")
            torch.cuda.synchronize()
            ready_ms = (time.perf_counter()-start)*1000
            for _ in range(16):
                engine.next_token(state)
            torch.cuda.synchronize()
            total_ms = (time.perf_counter()-start)*1000
            actual = list(state.generated)
            if expected_tokens is None:
                expected_tokens = actual
            row = {"case": name, "sample_index": len(observations), "block": repeat,
                   "mode": mode, "ready_ms": ready_ms, "resume_and_decode16_ms": total_ms,
                   "output_tokens": actual, "tokens_equal": actual == expected_tokens}
            if not row["tokens_equal"]:
                row["expected_tokens"] = expected_tokens
            del state
            gc.collect()
            torch.cuda.synchronize()
            row["cuda_bytes_after_request_release"] = torch.cuda.memory_allocated()
            row["cuda_bytes_without_request"] = without_state
            # Persist even rejected samples before asserting either gate.
            with journal.open("a") as file:
                file.write(json.dumps(row)+"\n")
            observations.append(row)
            assert row["tokens_equal"], "timed continuation differs after restore/recompute"
            assert row["cuda_bytes_after_request_release"] <= without_state, "timed store retained CUDA copy"
    medians = {mode: {field: statistics.median(r[field] for r in observations if r["mode"] == mode)
                     for field in ("ready_ms", "resume_and_decode16_ms")} for mode in ("recompute", "restore")}
    saving = medians["recompute"]["ready_ms"]-medians["restore"]["ready_ms"]
    store.delete(name)
    return {"case": name, "input_tokens": len(ids), "status": "measured_preserving_tokens",
            "payload_bytes": payload, "includes_next_token_fp32_logits": True,
            "initial_prefill_ms": initial_prefill_ms, "save_ms_single_observation": save_ms,
            "medians": medians, "save_amortization_reuses_estimate": save_ms/saving if saving > 0 else None,
            "samples": observations}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--correctness-only", action="store_true")
    args = p.parse_args()
    root = Path.cwd()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inventory = subprocess.check_output(["nvidia-smi", "-i", args.gpu_uuid,
        "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    memory, utilization = [int(v) for v in inventory.split(",")]
    write(out/"preflight.json", {"gpu_uuid": args.gpu_uuid, "observed": inventory,
                                "status": "busy" if memory >= 500 or utilization >= 5 else "idle"})
    if memory >= 500 or utilization >= 5:
        raise RuntimeError(f"selected GPU is occupied: {inventory}; no process was stopped")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    torch.set_num_threads(4)
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    assert actual_uuid.removeprefix("GPU-") == args.gpu_uuid.removeprefix("GPU-")
    task = json.loads((root/"experiments/qwen3_8b/model.json").read_text())
    model_files = model_identity(args.model_path, task)
    provenance = {"model_revision": REVISION, "model_files": model_files,
        "gpu_uuid": actual_uuid, "preflight": inventory, "torch": torch.__version__,
        "scope": "storage-only development preservation relative to frozen engine; no model accuracy or SOTA claim",
        "source_sha256": {str(path): file_sha(path) for path in [
            Path("experiments/qwen3_8b/host_roundtrip.py"), Path("engines/qwen3_8b_kv/engine.py"),
            Path("engines/qwen3_8b_kv/kv_store.py"), Path("experiments/qwen3_8b/host-store-contract.md")]}}
    write(out/"provenance.json", provenance)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    texts = ["Explain how an inference engine restores a session's key and value tensors. ",
             "Summarize the design of an analytical database and its column storage. "]
    def inputs(length, count=1):
        result = []
        for text in texts[:count]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            result.append((ids*((length+len(ids)-1)//len(ids)))[:length])
        return result
    trace = {str(n): inputs(n, 2) for n in (128, 512, 1024, 2048, 3072)}
    write(out/"trace.json", {"scope": "synthetic repeated-text development workload, no independent holdout", "inputs": trace})
    engine = Qwen3Engine(args.model_path)
    checks = []
    with torch.inference_mode():
        write(out/"stream-correctness.json", current_stream_case())
        write(out/"negative-controls.json", reject_hidden_cuda_copies(engine, trace["128"][:1]))
        for length, batch, checkpoint in ((128, 1, 0), (512, 1, 0), (512, 1, 7),
                (1024, 1, 0), (1024, 2, 0), (1024, 2, 7), (2048, 1, 0), (3072, 1, 0), (3072, 1, 7)):
            name = f"n{length}-b{batch}-d{checkpoint}"
            try:
                row = correctness_case(engine, trace[str(length)][:batch], checkpoint, name)
            except Exception as error:
                row = {"case": name, "status": "fail", "input_tokens": length, "batch": batch,
                       "checkpoint_after_decode_tokens": checkpoint, "error_type": type(error).__name__,
                       "error": str(error)}
                checks.append(row)
                write(out/"correctness.json", {"cases": checks, "status": "fail"})
                write(out/"summary.json", {"status": "fail", "stage": "correctness", "failure": row})
                raise
            checks.append(row)
            write(out/"correctness.json", {"cases": checks, "status": "partial"})
            print(json.dumps(row), flush=True)
        write(out/"correctness.json", {"cases": checks, "status": "pass"})
        measurements = []
        for length in (() if args.correctness_only else (128, 512, 1024, 2048, 3072)):
            row = benchmark_case(engine, trace[str(length)][0], f"n{length}", out/"samples.jsonl")
            measurements.append(row)
            write(out/"measurements.json", {"cases": measurements, "status": "partial"})
            print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
    write(out/"measurements.json", {"cases": measurements,
        "status": "not_run" if args.correctness_only else "measured_preserving_tokens"})
    assert all(file_sha(path) == sha for path, sha in provenance["source_sha256"].items())
    write(out/"summary.json", {"status": "pass", "scope": provenance["scope"],
        "correctness_cases": len(checks), "measurement_cases": len(measurements),
        "wall_seconds": time.monotonic()-started, "joint_optimization_implemented": False})


if __name__ == "__main__":
    main()
