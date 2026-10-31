# Multi-session Qwen3-8B joint storage/consumer construction, v1

Build a specialized multi-turn session runtime for the supplied frozen engine.
The editable files are `consumer.py` and `kv_store.py`. Preserve `engine.py`
bytes and real Qwen3 computation; no HF model/generate or other engine delegation.
No tests/builds/inference on this Mac, no SSH/credentials/Git, no other candidates,
references or evaluator files. The controller runs all measurements remotely.

The public consumer API is `SessionRuntime(engine, store, gpu_budget_bytes)`:

- `turn(session_id, input_ids, max_new_tokens, observer=None) -> dict`, returning
  `token_ids` and `token_logprobs`, exactly max_new_tokens, greedy/ignore-EOS.
  First input is the initial prompt. Later input is appended after ALL previous
  inputs and outputs for that session. Different sessions must stay independent.
- Call `observer(session_id, output_index, logits)` once BEFORE selecting each
  output token, with the actual unmodified full-vocabulary model logits.
  The same mathematical and scheduling path must execute with/without observer.
- `close(session_id)` releases that session's CPU and GPU cached state.
- `stats()` returns counters useful for feedback; counters are diagnostics and
  are not a replacement for externally measured memory or time.

The seed engine intentionally skips the forward pass after the last requested
output. Thus its KV covers `input + generated[:-1]`; the final generated token
is pending and MUST be consumed with the next input. The saved logits predict
that pending token. Do not silently drop or double-consume it.

One request executes at a time in this pilot. At each completed turn, live GPU
session state must fit the supplied 768 MiB budget (weights excluded). The
trusted evaluator additionally measures actual CUDA allocations and peak use.
All-resident control uses a separately labeled 4 GiB budget. No cross-turn hidden
GPU caches outside the budget, output memoization or future-trace access.

The supplied KVStore supports put/get/delete/used_bytes, with the same lossless,
identity, ownership, capacity and replacement semantics as the host-store task.
Keep these APIs intact, including arbitrary tensor values. Host payload budget
is 4 GiB. You may introduce additional operations (e.g. append, layer or batch
access) and change the consumer to use them. If an optional operation is absent,
the consumer must fall back to the standard API, preserving correctness. This
allows cross-combining old/new storage and old/new consumer for diagnosis.

Inspect the real baseline for improvement opportunities. CPU backing of a live
GPU state, incremental immutable-prefix handling, transfer granularity and
consumer-aware access are allowed design choices, not prescribed solutions.
New operations must state their guarantees and verify identities/positions.
Do not weaken the original generic put semantics just to assume unchanged data.

Evaluation: independent no-offload reference on exactly the same inputs/history;
every measured output's full-logit digest and token must match; quiescent GPU
allocation must respect the common budget; close must release GPU state. Six
sessions, initial lengths 512/1024/2048/3072, short new input and four output
tokens per turn, mixed reuse/evictions, total context below4096. No HTTP/SLO,
NVMe, online adaptation or general BF16-equivalence claim follows from this task.
