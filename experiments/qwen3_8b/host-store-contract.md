# Qwen3-8B lossless host KV store: construction contract v1

This is a storage-only starting point for later joint storage/consumer synthesis.
The engine seed was previously generated; its math is frozen in this experiment.
Write only `kv_store.py`. Do not edit or replace `engine.py`, read another
candidate/reference/test, use SSH/credentials/Git, or run inference/tests/builds
on this Mac. The external controller evaluates on the remote A40.

Implement `KVStore(max_bytes: int)` with this public API:

- `put(key: str, kv, logits, identity: dict) -> None`: store a lossless, owned
  CPU snapshot of `kv`, a list of `(K,V)` tensors, and next-token `logits`.
  Model tensors may originate on CPU or CUDA. Shapes/dtypes must be preserved.
  Completion means all required CUDA-to-host transfers are complete. Never
  retain a CUDA tensor or a view/reference to caller-owned tensor storage.
- `get(key: str, identity: dict, device) -> (kv, logits)`: verify the entire
  identity, then return fresh tensors on the requested device. Completion means
  returned data are ready for the caller's ordinary stream. Later mutation of a
  returned snapshot must not change the stored entry or another returned copy.
- `delete(key: str) -> None`: remove the entry and release its charged bytes.
- read-only `used_bytes` property: exact sum of stored tensor payload bytes,
  excluding metadata and allocator overhead. Never exceed `max_bytes`.

Raise `KeyError` for absent get/delete; `ValueError` for a present key with a
different identity; `MemoryError` if a put would exceed the payload budget.
Replacement is atomic with respect to these synchronous API calls: a failed
replacement leaves the old value and byte count intact. No implicit eviction.
Identity is JSON-compatible metadata, copied at put time; it includes weight
revision, config identity, precision, positions and exact input token prefix.
No approximate prefix matches, output memoization or lossy conversion.

Single controller thread; no concurrent put/delete or cancellation support is
claimed. GPU input may have been produced on the caller's current CUDA stream.
Torch tensor operations and pinned host memory are allowed. No model weights,
reference outputs, frameworks' model APIs or filesystem cache are allowed.
Do not preallocate memory beyond the declared payload budget. Prefer a small,
understandable implementation; optimize only based on measured feedback.

The first external evaluation checks tensor bytes/dtypes/shapes, isolation,
identity rejection, capacity/replacement accounting and release of original
CUDA storage. It then compares same-engine continuation before and after a
snapshot at prefill and during decode, including equal-length batches. Full
logits and output tokens must match exactly on these fixed execution paths.
These tests establish preservation relative to the engine, not general model
accuracy or equivalence to vLLM/SGLang. Profiling and final heldout tests remain
outside the generated module.
