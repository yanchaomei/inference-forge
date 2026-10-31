"""Public snapshot semantics; small CPU tensors, no model required."""
import torch
from unittest import TestCase

from engines.qwen3_8b_kv.kv_store import KVStore


def test_snapshot_is_lossless_owned_and_independent_on_each_read():
    original = torch.arange(24, dtype=torch.bfloat16).reshape(1, 2, 3, 4)
    k = original.transpose(1, 2)  # noncontiguous caller input
    v = -k
    logits = torch.tensor([1.25, -0.75, 4.0], dtype=torch.float32)
    expected_k, expected_v, expected_logits = k.clone(), v.clone(), logits.clone()
    identity = {"model": "frozen-revision", "prefix": [1, 2, 3]}
    store = KVStore(max_bytes=1000)
    store.put("session", [(k, v)], logits, identity)
    k.fill_(9)
    v.fill_(8)
    logits.zero_()
    identity["prefix"].append(4)
    restored, scores = store.get("session", {"model": "frozen-revision", "prefix": [1, 2, 3]}, "cpu")
    assert torch.equal(restored[0][0], expected_k)
    assert torch.equal(restored[0][1], expected_v)
    assert torch.equal(scores, expected_logits)
    assert restored[0][0].dtype == torch.bfloat16
    assert scores.dtype == torch.float32
    assert store.used_bytes == 108  # 2 * 24 BF16 elements + 3 FP32 elements
    restored[0][0].zero_()
    scores.fill_(7)
    again, scores_again = store.get("session", {"model": "frozen-revision", "prefix": [1, 2, 3]}, "cpu")
    assert torch.equal(again[0][0], expected_k)
    assert torch.equal(scores_again, expected_logits)


def test_identity_capacity_and_failed_replacement_preserve_old_entry():
    check = TestCase()
    store = KVStore(max_bytes=24)
    kv = [(torch.tensor([1., 2.]), torch.tensor([3., 4.]))]
    logits = torch.tensor([5., 6.])
    store.put("a", kv, logits, {"prefix": [1]})
    with check.assertRaises(ValueError):
        store.get("a", {"prefix": [2]}, "cpu")
    with check.assertRaises(MemoryError):
        store.put("b", kv, logits, {"prefix": [1]})
    with check.assertRaises(MemoryError):
        store.put("a", kv, torch.ones(3), {"prefix": [2]})
    old, scores = store.get("a", {"prefix": [1]}, "cpu")
    assert torch.equal(old[0][0], kv[0][0]) and torch.equal(scores, logits)
    assert store.used_bytes == 24
    store.put("a", [], torch.tensor([7.]), {"prefix": [2]})
    assert store.used_bytes == 4
    store.delete("a")
    assert store.used_bytes == 0
    with check.assertRaises(KeyError):
        store.get("a", {"prefix": [2]}, "cpu")
    with check.assertRaises(KeyError):
        store.delete("a")


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}", flush=True)
    print(f"{len(tests)} tests passed", flush=True)
