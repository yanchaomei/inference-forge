"""Run existing public API contracts against a generated component pair."""
import argparse
from pathlib import Path
import runpy

from experiments.qwen3_8b.session_matrix import load_module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    args = p.parse_args()
    root = Path(args.candidate)
    store_class = load_module("checked_candidate_store", root/"kv_store.py").KVStore
    consumer_class = load_module("checked_candidate_consumer", root/"consumer.py").SessionRuntime
    namespace = runpy.run_path("experiments/qwen3_8b/tests/test_host_store.py")
    for name, test in namespace.items():
        if name.startswith("test_"):
            test.__globals__["KVStore"] = store_class
            test()
            print(f"PASS generated store: {name}", flush=True)
    namespace = runpy.run_path("experiments/qwen3_8b/tests/test_session_baseline.py")
    fallback = namespace["test_pending_last_output_survives_offload_and_sessions_stay_independent"]
    fallback.__globals__["SessionRuntime"] = consumer_class
    fallback()
    print("PASS generated consumer: generic-store fallback", flush=True)
    store = store_class(max_bytes=1024**2)
    runtime = consumer_class(namespace["CausalToy"](), store, gpu_budget_bytes=0)
    assert runtime.turn("a", [1, 2], 2)["token_ids"] == [5, 3]
    assert runtime.turn("b", [2], 1)["token_ids"] == [3]
    assert runtime.turn("a", [4], 2)["token_ids"] == [4, 1]
    runtime.close("a")
    runtime.close("b")
    assert store.used_bytes == 0
    print("PASS generated pair: pending token, independent sessions, release", flush=True)


if __name__ == "__main__":
    main()
