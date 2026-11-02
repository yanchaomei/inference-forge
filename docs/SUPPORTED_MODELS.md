# Supported models

The serving registry is intentionally small. Run `inference-forge models --json` to see targets available through `serve`.

| Model | Entry point | Precision | Measured hardware | Context and output |
| --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-0.5B-Instruct | `inference-forge serve MODEL_PATH` | float32 | Single NVIDIA A40 | Input + output ≤512 tokens; 1–64 output tokens |
| Qwen/Qwen3-8B | Separate KV and session experiment scripts | BF16 | Single NVIDIA A40, TP=1 | Up to 4096 total context in the documented experiments |

**Qwen3-8B is not a serving target.** Its configuration and experimental Python APIs do not imply support through the HTTP service. The `serve` CLI targets CUDA; CPU execution is available in the functional test harness. Other models, quantization formats, tensor parallelism, and additional hardware are not currently validated.

## Qwen2.5-0.5B serving

- Model revision: `7ae557604adf67be50417f59c2c2f167def9a775`.
- Model weights SHA256: `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`, recorded in the [task definition](../tasks/qwen25_05b/task.json).
- Runtime: packaged Qwen2 backend with per-request KV and batched decode; fixed-length greedy generation.
- Interface: text completions, plain-text chat, and SSE. See [API limits](API.md#request-parameters).
- Environment: Linux, Python 3.10+, CUDA-enabled PyTorch, Transformers, and safetensors. Dependency ranges are in [pyproject.toml](../pyproject.toml).

Download the complete checkpoint and point `serve` at its local directory. The API model ID is an alias selected by `--served-model-name`, not a model downloader or an architecture selector.

The new installed CLI and SDK/chat path pass [A40 functional acceptance](../results/runtime-checkpoint-a40-20261102.json) with Torch2.11.0+cu128, Transformers5.3.0 and OpenAI2.54.0. A separate [CPU check](../results/runtime-checkpoint-cpu-20261102.json) uses Torch2.10.0+cpu and Transformers5.22.1. Both use the pinned real checkpoint and compare with an independent reference.

Historical validation for v0.2.0 additionally covered 12 full-vocabulary reference comparisons on an A40. See that version's [release validation record](../results/release-v0.2.0-validation.json).

## Qwen3-8B experiments

- Model revision: `b968826d9c46dd6066d109eabc6255188de91218`.
- Separate environment: Torch `2.10.0+cu128`, Transformers `4.57.6`, safetensors `0.8.0`.
- Interfaces: host KV save/restore, versioned storage, and a session consumer with a GPU-residency budget.
- Entry point: [experiment guide](../experiments/qwen3_8b/README.md), including independent dependency installation and verification commands.
- Evidence: [host-state results](../results/qwen3-8b-host-state-20261031.json), [session results](../results/qwen3-8b-sessions-20261031.json), and [export checks](../results/qwen3-8b-export-validation-20261031.json).

Use the experiment's own environment; the 0.5B serving dependencies are not the tested 8B combination. The session API has no HTTP service, cancellation, or concurrent-thread control.

## Requesting another model

Open a [model request](https://github.com/yanchaomei/inference-forge/issues/new?template=model-request.yml) with the checkpoint and revision, intended GPU and precision, input/output lengths, and the workload you need. Include an official model/license link. A model becomes a supported serving target after a runnable path and corresponding correctness checks are available.
