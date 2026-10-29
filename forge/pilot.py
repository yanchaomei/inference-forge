"""Bounded first-engine run. This is not a VibeSys autonomous campaign."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import time
from urllib.request import urlopen

from .evaluate import assess_interleaving, check_numerics, evaluate_trace, write_json
from .replay import reduce_trace
from .processes import run_bounded
from .task import load_task


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_identity(root):
    files = {str(p.relative_to(root)): file_sha(p) for p in sorted((root/"forge").glob("*.py"))}
    return {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_worker(args):
    root = Path.cwd().resolve()
    task_path = Path(args.task).resolve()
    task = load_task(task_path)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    events_file = out/"stages.jsonl"
    started = time.perf_counter()

    def event(kind, **fields):
        with events_file.open("a") as target:
            target.write(json.dumps({"event": kind, "elapsed_seconds": time.perf_counter()-started,
                                     "monotonic_ns": time.perf_counter_ns(), **fields}) + "\n")

    identity = package_identity(root)
    snapshot = out/"snapshot"
    shutil.copytree(root/"forge", snapshot/"forge", ignore=shutil.ignore_patterns("__pycache__"))
    write_json(snapshot/"task.json",task)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": task["gpu_uuid"], "PYTHONPATH": str(snapshot),
                "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TORCH_HOME": str(root/"cache"/"torch")})
    gpu_inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader"], text=True)
    gpu_processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"], text=True)
    selected = next((line for line in gpu_inventory.splitlines() if line.split(",")[0].strip()==task["gpu_uuid"]), None)
    busy_processes = [line for line in gpu_processes.splitlines() if line.split(",")[0].strip()==task["gpu_uuid"]
                      and (not line.split(",")[2].strip().split()[0].isdigit() or int(line.split(",")[2].strip().split()[0])>=128)]
    if (selected is None or int(selected.split(",")[1].strip().split()[0]) >= 500
            or int(selected.split(",")[2].strip().split()[0]) >= 5 or busy_processes):
        write_json(out/"blocked.json", {"status":"blocked", "reason":"selected GPU not idle", "inventory":gpu_inventory,"compute_processes":gpu_processes})
        raise SystemExit("selected GPU not idle; no process was terminated")
    model = Path(task["model_path"])
    event("model_identity_started")
    model_files = {p.name: {"bytes": p.stat().st_size, "sha256": file_sha(p)} for p in model.iterdir()
                   if p.is_file() and (p.suffix==".safetensors" or p.name in ["config.json","tokenizer.json","tokenizer_config.json"])}
    if model_files.get("model.safetensors",{}).get("sha256") != task["model_safetensors_sha256"]:
        write_json(out/"summary.json",{"status":"blocked","stage":"model_identity","reason":"weight hash does not match frozen task"})
        raise SystemExit("model identity mismatch")
    provenance = {"task":task, "task_sha256":file_sha(task_path), "source":identity,
                  "recorded_at_utc":datetime.now(timezone.utc).isoformat(),"hostname":socket.gethostname(),
                  "base_revision":args.base_revision, "model_files":model_files,
                  "python":sys.version, "packages":{}, "gpu_inventory":gpu_inventory,"compute_processes":gpu_processes,
                  "construction_kind":"Codex session bootstrap; not a VibeSys campaign or controlled comparison",
                  "agent_token_cost":None, "agent_api_cost":None,
                  "unknown_cost_reason":"not exposed by this session; not recorded as zero"}
    for name in ["torch","transformers","safetensors","pytest"]:
        try:
            provenance["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            provenance["packages"][name] = None
    write_json(out/"provenance.json", provenance)
    event("reference_started")
    try:
        with (out/"reference.log").open("w") as log:
            subprocess.run([sys.executable,"-m","forge.reference","--task",str(snapshot/"task.json"),"--output",str(out/"goldens.json")],
                           cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=task["timeout_seconds"])
    except Exception as exc:
        event("reference_failed",error=repr(exc))
        write_json(out/"summary.json",{"status":"blocked","stage":"reference","error":repr(exc)})
        raise
    event("reference_completed")
    golden = json.loads((out/"goldens.json").read_text())
    port = free_port()
    server_command = [sys.executable,"-m","forge.server","--model",task["model_path"],
                      "--port",str(port),"--dtype",task["dtype"],"--event-log",str(out/"server-events.jsonl")]
    base_url = f"http://127.0.0.1:{port}"
    write_json(out/"commands.json", {"server":server_command,"cwd":str(snapshot),"gpu_uuid":task["gpu_uuid"]})
    event("server_starting", command=server_command)
    summary = {"status":"blocked"}
    with (out/"server.log").open("w") as server_log:
        process = subprocess.Popen(server_command, cwd=snapshot, env=env, stdout=server_log,
                                   stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+120
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}; see server.log")
                try:
                    with urlopen(base_url+"/health", timeout=2) as response:
                        health = json.load(response)
                    break
                except OSError:
                    if time.monotonic()>deadline:
                        raise TimeoutError("server readiness timeout")
                    time.sleep(.2)
            event("server_ready", health=health)
            numeric = check_numerics(base_url,golden,task)
            write_json(out/"numerics.json",numeric)
            event("numerics_completed",status=numeric["status"])
            trace = [{k:c[k] for k in ["request_id","input_ids","max_tokens"]} for c in golden["cases"]]
            write_json(out/"trace.json",trace)
            event_path = out/"server-events.jsonl"
            event_offset = event_path.stat().st_size
            streaming = evaluate_trace(base_url,trace,golden)
            write_json(out/"streaming.json",streaming)
            with event_path.open() as source:
                source.seek(event_offset)
                cohort_events = [json.loads(line) for line in source if line.strip()]
            write_json(out/"streaming-server-events.json",cohort_events)
            interleaving = assess_interleaving(cohort_events,[e["request_id"] for e in trace])
            write_json(out/"interleaving.json",interleaving)
            event("streaming_completed",status=streaming["status"])
            benchmark = None
            if task.get("benchmark_repeats", 0):
                warmup = evaluate_trace(base_url,trace,golden)
                write_json(out/"benchmark-warmup.json",warmup)
                samples = []
                for repeat in range(task["benchmark_repeats"]):
                    sample = evaluate_trace(base_url,trace,golden)
                    write_json(out/f"benchmark-{repeat:02d}.json",sample)
                    samples.append(sample)
                valid = warmup["status"] == "pass" and all(x["status"] == "pass" for x in samples)
                benchmark = {"status":"pass" if valid else "fail", "repeats":len(samples),
                             "median_tokens_per_second":statistics.median(x["output_tokens_per_second"] for x in samples) if valid else None,
                             "scope":"fixed small development cohort; not production SLO goodput"}
                write_json(out/"benchmark.json",benchmark)
            negative = evaluate_trace(base_url,trace,golden,mutation_request_id="r2")
            write_json(out/"negative-control.json",negative)

            def negative_probe(events):
                result = evaluate_trace(base_url,events,golden,mutation_request_id="r2")
                probe_index = len(list(out.glob("probe-*.json")))
                write_json(out/f"probe-{probe_index:03d}.json",result)
                return {"status":result["status"],"signature":result.get("signature")}

            reduction = reduce_trace(trace,negative_probe,"token_mismatch:r2:0",repeats=2,max_probes=24)
            write_json(out/"reduction.json",reduction)
            event("negative_control_completed",status=negative["status"],reduction=reduction["status"])
            source_unchanged = package_identity(root)==identity
            success = (numeric["status"]==streaming["status"]==interleaving["status"]=="pass" and negative["status"]=="fail"
                       and reduction["status"]=="reduced" and source_unchanged)
            if benchmark is not None:
                success = success and benchmark["status"] == "pass"
            summary = {"status":"pass" if success else "fail", "numeric_status":numeric["status"],
                       "streaming_status":streaming["status"], "streaming_requests":len(streaming.get("requests",[])),
                       "interleaving":interleaving,
                       "streaming_tokens":sum(len(r["token_ids"]) for r in streaming.get("requests",[])),
                       "negative_control_status":negative["status"], "reduction_status":reduction["status"],
                       "reduced_request_ids":[e["request_id"] for e in reduction["events"]],
                       "source_unchanged":source_unchanged,
                       "benchmark":benchmark,
                       "negative_control_scope":"declared receive-token mutation, not a naturally discovered engine/state bug",
                       "performance_scope":"bootstrap diagnostic timing; no strong-framework comparison or final SLO claim"}
        except Exception as exc:
            summary["error"] = repr(exc)
            event("execution_failed",error=repr(exc))
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            event("owned_server_stopped",returncode=process.returncode)
            summary["wall_seconds"] = time.perf_counter()-started
            write_json(out/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False))
    raise SystemExit(0 if summary["status"]=="pass" else 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-revision", default="uncommitted")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        return run_worker(args)
    task_path = Path(args.task).resolve()
    out = Path(args.output).resolve()
    if out.exists():
        raise SystemExit("output already exists; use a new run identity")
    task = load_task(task_path)
    command = [sys.executable,"-m","forge.pilot","--task",str(task_path),"--output",str(out),
               "--base-revision",args.base_revision,"--_worker"]
    result = run_bounded(command, task["timeout_seconds"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/"launcher.json", result)
    if result["timed_out"]:
        previous = json.loads((out/"summary.json").read_text()) if (out/"summary.json").exists() else None
        write_json(out/"summary.json", {"status":"blocked", "stage":"whole_run_budget",
                   "reason":"absolute pilot deadline exceeded", "launcher":result,"previous_summary":previous})
    raise SystemExit(124 if result["timed_out"] else result["returncode"])


if __name__ == "__main__":
    main()
