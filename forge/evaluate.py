"""Independent reference generation and HTTP-boundary evaluation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import time
from threading import Barrier
from urllib.request import Request, urlopen


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def post_json(url, body, timeout=120):
    request = Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def generate_reference(task, output):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    torch.manual_seed(task["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_path = task["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager").to(task["device"]).eval()
    started = time.perf_counter()
    cases = []
    with torch.inference_mode():
        for index, prompt in enumerate(task["prompts"]):
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            tensor = torch.tensor([ids], device=task["device"])
            result = model(tensor, use_cache=True)
            generated = []
            for step in range(task["max_new_tokens"]):
                token = int(result.logits[0, -1].argmax())
                generated.append(token)
                if step + 1 < task["max_new_tokens"]:
                    result = model(torch.tensor([[token]], device=task["device"]),
                                   past_key_values=result.past_key_values, use_cache=True)
            # An independent full-prefix pass is compared with candidate cache decoding.
            full = model(torch.tensor([ids + generated[:-1]], device=task["device"]), use_cache=False)
            observations = []
            for step, token in enumerate(generated):
                logits = full.logits[0, len(ids) - 1 + step].float()
                top = torch.topk(logits, 2).values
                observations.append({"index": step, "top1": int(logits.argmax()),
                                     "logprob": float(logits.log_softmax(-1)[token]),
                                     "top1_margin": float(top[0] - top[1])})
            cases.append({"request_id": f"r{index + 1}", "prompt": prompt,
                          "input_ids": ids, "max_tokens": len(generated),
                          "token_ids": generated,
                          "text": tokenizer.decode(generated, clean_up_tokenization_spaces=False),
                          "decoded_prefixes":[tokenizer.decode(generated[:n], clean_up_tokenization_spaces=False) for n in range(1,len(generated)+1)],
                          "observations": observations})
    report = {"reference_kind": "transformers-qwen2-forward-and-full-prefix",
              "transformers_version": transformers.__version__, "torch_version": torch.__version__,
              "dtype": "float32", "fixed_output_length_ignore_eos": True,
              "task_id": task["task_id"], "seconds": time.perf_counter() - started,
              "model_config_sha256": hashlib.sha256((Path(model_path)/"config.json").read_bytes()).hexdigest(),
              "cases": cases}
    write_json(output, report)
    del model
    torch.cuda.empty_cache()
    return report


class StreamFailure(RuntimeError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def assess_interleaving(rows, request_ids):
    allowed = set(request_ids)
    active, max_active, compressed, seen = set(), 0, [], set()
    repeated_after_switch = False
    for row in rows:
        request = row.get("request_id")
        if request not in allowed:
            continue
        if row.get("event") == "accepted":
            active.add(request)
            max_active = max(max_active, len(active))
        elif row.get("event") == "finished":
            active.discard(request)
        elif row.get("event") == "token" and (not compressed or compressed[-1] != request):
            if request in seen:
                repeated_after_switch = True
            compressed.append(request)
            seen.add(request)
    passed = max_active >= 2 and len(seen) >= 2 and repeated_after_switch
    return {"status":"pass" if passed else "fail", "max_inflight":max_active,
            "compressed_token_order":compressed, "token_interleaving_observed":repeated_after_switch}


def stream_request(base_url, event, barrier=None):
    if barrier is not None:
        barrier.wait(timeout=30)
    started = time.perf_counter_ns()
    body = {"input_ids": event["input_ids"], "max_tokens": event["max_tokens"],
            "request_id": event["request_id"], "stream": True}
    request = Request(base_url + "/v1/completions", json.dumps(body).encode(),
                      {"Content-Type": "application/json"})
    tokens, stamps, final, raw_lines, texts = [], [], None, [], []
    saw_done = False
    try:
        with urlopen(request, timeout=120) as response:
            for line in response:
                raw_lines.append(line.decode(errors="replace"))
                line = line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                obj = json.loads(data)
                if "error" in obj:
                    raise RuntimeError(str(obj["error"]))
                if obj.get("request_id") != event["request_id"]:
                    raise RuntimeError("response request identity mismatch")
                if "token_id" in obj:
                    if obj.get("index") != len(tokens) or type(obj["token_id"]) is not int:
                        raise RuntimeError("non-sequential or invalid token event")
                    if not isinstance(obj.get("text"),str):
                        raise RuntimeError("token text must be a string")
                    tokens.append(obj["token_id"])
                    stamps.append(time.perf_counter_ns())
                    texts.append(obj["text"])
                if "finish_reason" in obj:
                    final = obj
        ended = time.perf_counter_ns()
        if not saw_done or final is None or not tokens:
            raise RuntimeError("incomplete stream")
        if tokens != final["token_ids"] or len(tokens) != final["usage"]["completion_tokens"]:
            raise RuntimeError("stream/final token accounting mismatch")
        if len(tokens) != event["max_tokens"] or final["finish_reason"] != "length":
            raise RuntimeError("fixed output length or finish reason mismatch")
        if final["usage"]["prompt_tokens"] != len(event["input_ids"]):
            raise RuntimeError("prompt token accounting mismatch")
    except Exception as exc:
        raise StreamFailure(str(exc), {"request_id":event["request_id"], "token_ids":list(tokens),
                            "text":"".join(texts),
                            "token_timestamps_ns":list(stamps), "raw_sse_lines":list(raw_lines),
                            "started_ns":started,"ended_ns":time.perf_counter_ns(),
                            "partial_final":final,"error":repr(exc)}) from exc
    intervals = [(b-a)/1e6 for a,b in zip(stamps, stamps[1:])]
    return {"request_id": event["request_id"], "token_ids": tokens, "usage": final["usage"],
            "started_ns": started, "ended_ns": ended, "token_timestamps_ns": stamps,
            "ttft_ms": (stamps[0]-started)/1e6, "total_ms": (ended-started)/1e6,
            "mean_itl_ms": statistics.mean(intervals) if intervals else None,
            "finish_reason": final["finish_reason"], "raw_sse_lines":raw_lines,"text":"".join(texts)}


def evaluate_trace(base_url, events, golden, *, mutation_request_id=None):
    if not events:
        return {"status": "pass", "requests": [], "scope": "empty replay"}
    barrier = Barrier(len(events))
    started = time.perf_counter()
    requests, execution_errors = [], []
    with ThreadPoolExecutor(max_workers=len(events)) as pool:
        futures = [pool.submit(stream_request, base_url, e, barrier) for e in events]
        for event, future in zip(events, futures):
            try:
                requests.append(future.result())
            except StreamFailure as exc:
                execution_errors.append(exc.evidence)
            except Exception as exc:
                execution_errors.append({"request_id":event["request_id"],"error":repr(exc)})
    if execution_errors:
        return {"status":"blocked","signature":"execution_error", "requests":requests,
                "execution_errors":execution_errors, "wall_seconds":time.perf_counter()-started}
    by_id = {case["request_id"]: case for case in golden["cases"]}
    failures = []
    for event, result in zip(events, requests):
        actual = list(result["token_ids"])
        if result["request_id"] == mutation_request_id and actual:
            # A declared transport-level negative control, never a claimed natural engine bug.
            result["unmodified_token_ids"] = list(actual)
            actual[0] = (actual[0] + 1) % 151936
            result["control_mutation"] = "first_received_token_plus_one"
            result["token_ids"] = actual
        expected = by_id[event["request_id"]]["token_ids"][:event["max_tokens"]]
        if actual != expected:
            first = next((i for i,(a,b) in enumerate(zip(actual,expected)) if a!=b), min(len(actual),len(expected)))
            failures.append({"request_id": event["request_id"], "first_mismatch": first,
                             "kind":"token_mismatch",
                             "expected": expected, "actual": actual})
        elif "decoded_prefixes" in by_id[event["request_id"]]:
            expected_text = by_id[event["request_id"]]["decoded_prefixes"][event["max_tokens"]-1]
            if result["text"] != expected_text:
                failures.append({"request_id":event["request_id"],"first_mismatch":0,"kind":"text_mismatch",
                                 "expected":expected_text,"actual":result["text"]})
    elapsed = time.perf_counter() - started
    return {"status": "fail" if failures else "pass",
            "signature": f"{failures[0]['kind']}:{failures[0]['request_id']}:{failures[0]['first_mismatch']}" if failures else None,
            "failures": failures, "requests": requests, "wall_seconds": elapsed,
            "output_tokens_per_second": sum(len(r["token_ids"]) for r in requests)/elapsed,
            "control_mutation_request_id": mutation_request_id,
            "scope": "development streaming boundary, not final SLO goodput"}


def check_numerics(base_url, golden, task):
    if not golden.get("cases"):
        return {"status": "fail", "cases": [], "reason": "no independent reference cases"}
    cases = []
    for case in golden["cases"]:
        modes = {}
        for mode in ["cached", "recompute"]:
            observed = post_json(base_url + "/debug/trace", {
                "input_ids": case["input_ids"], "continuation_ids": case["token_ids"], "mode": mode})
            rows = observed["observations"]
            errors = [abs(a["logprob"]-b["logprob"]) for a,b in zip(rows,case["observations"])]
            top1_wrong = [i for i,(a,b) in enumerate(zip(rows,case["observations"])) if a["top1"]!=b["top1"]]
            count_ok = len(rows) == len(case["observations"])
            cache_ok = observed.get("cache_lengths") == list(range(len(case["input_ids"]), len(case["input_ids"])+len(rows)))
            passed = (count_ok and cache_ok and bool(errors) and not top1_wrong
                      and statistics.mean(errors) <= task["mean_logprob_error_max"]
                      and max(errors) <= task["max_logprob_error_max"])
            modes[mode] = {"status": "pass" if passed else "fail", "count_ok": count_ok, "cache_lengths_ok": cache_ok,
                           "mean_logprob_error": statistics.mean(errors) if errors else None,
                           "max_logprob_error": max(errors) if errors else None,
                           "top1_mismatch_positions": top1_wrong, "observed": observed}
        cases.append({"request_id": case["request_id"], "modes": modes})
    return {"status": "pass" if all(m["status"]=="pass" for c in cases for m in c["modes"].values()) else "fail", "cases": cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--goldens", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    task = json.loads(Path(args.task).read_text())
    golden = json.loads(Path(args.goldens).read_text())
    numeric = check_numerics(args.url, golden, task)
    events = [{k:c[k] for k in ["request_id","input_ids","max_tokens"]} for c in golden["cases"]]
    streaming = evaluate_trace(args.url, events, golden)
    report = {"status": "pass" if numeric["status"]==streaming["status"]=="pass" else "fail",
              "numeric": numeric, "streaming": streaming}
    write_json(args.output, report)
    print(json.dumps({"status":report["status"],"report":args.output}))
    raise SystemExit(0 if report["status"]=="pass" else 1)


if __name__ == "__main__":
    main()
