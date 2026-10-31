"""Independent optional-protocol regression probes, preserving each failure."""
import argparse
import json
from pathlib import Path
import runpy
import torch

from experiments.qwen3_8b.session_matrix import load_module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    directory = Path(args.candidate)
    store_cls = load_module("edge_candidate_store", directory/"kv_store.py").KVStore
    consumer_cls = load_module("edge_candidate_consumer", directory/"consumer.py").SessionRuntime
    toy = runpy.run_path("experiments/qwen3_8b/tests/test_session_baseline.py")["CausalToy"]
    optional = {"put_retained", "get_retained", "extend_retained"}
    cases = []
    def record(name, call):
        try:
            call()
        except Exception as error:
            cases.append({"case": name, "status": "fail", "error_type": type(error).__name__, "error": str(error)})
        else:
            cases.append({"case": name, "status": "pass"})

    def partial(allowed):
        class Partial:
            def __init__(self):
                self.backing = store_cls(max_bytes=1024**2)
            def __getattr__(self, name):
                if name in optional and name not in allowed:
                    raise AttributeError(name)
                return getattr(self.backing, name)
        store = Partial()
        runtime = consumer_cls(toy(), store, gpu_budget_bytes=0)
        assert runtime.turn("a", [1, 2], 2)["token_ids"] == [5, 3]
        assert runtime.turn("a", [4], 2)["token_ids"] == [4, 1]
        assert runtime.turn("a", [0], 2)["token_ids"] == [4, 1]
        runtime.close("a")
        assert store.used_bytes == 0

    def changed_metadata(extra):
        if not all(callable(getattr(store_cls, name, None)) for name in optional):
            raise AssertionError("probe requires complete retained protocol")
        store = store_cls(max_bytes=1024**2)
        k = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
        base = [(k, k+1)]
        old = {"model": "frozen", "adapter": None, "prefix": [1, 2, 3]}
        lease = store.put_retained("a", base, torch.tensor([1., 2.]), old)
        new = {**old, "prefix": [1, 2, 3, 4]}
        if extra:
            new["new_semantic_field"] = "changed"
        else:
            del new["adapter"]
        extended = [(torch.cat([x, torch.zeros(1, 1, 1, 2)], dim=2),
                     torch.cat([y, torch.ones(1, 1, 1, 2)], dim=2)) for x, y in base]
        try:
            store.extend_retained("a", base, extended, torch.tensor([2., 3.]), new, lease)
        except ValueError:
            restored, _ = store.get("a", old, "cpu")
            assert torch.equal(restored[0][0], base[0][0])
            return
        raise AssertionError("changed non-prefix identity metadata was accepted")

    record("only_get_retained_falls_back", lambda: partial({"get_retained"}))
    record("put_extend_without_get_falls_back", lambda: partial({"put_retained", "extend_retained"}))
    record("extra_identity_field_rejected", lambda: changed_metadata(True))
    record("missing_null_identity_field_rejected", lambda: changed_metadata(False))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError("output exists")
    result = {"status": "pass" if all(x["status"] == "pass" for x in cases) else "fail", "cases": cases}
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
