# v0.3.0-rc1: install, serve, and connect

This release candidate closes the path from a Python installation to the generated inference service. The backend now lives in the installed package, and `inference-forge serve MODEL_PATH` starts the service without depending on an examples directory.

## What is included

- Unified `serve`, `models`, `doctor`, and `construct` commands.
- An OpenAI SDK-compatible subset for text and plain-text chat completions, including SSE and usage chunks.
- Compatibility with the original construction command and native streaming evaluator.
- A runnable SDK client, English/Chinese README, API and model documentation, developer guide, issue forms, and PR template.
- Wheel and source-distribution artifacts. The project has not been uploaded to PyPI.

## Validation

The remote CI recorded in [the validation file](../results/runtime-usability-20261102.json) passes 29 core/CLI checks and 13 serving/numerical CPU checks. The latter use real HTTP calls, the OpenAI Python client, a real Transformers tokenizer, and a tiny random Qwen2 checkpoint compared with Transformers. Both the wheel smoke check and HTTP tests run outside the source checkout.

The pinned Qwen2.5-0.5B checkpoint also passes functional acceptance on CPU and on an A40. The A40 check launches the installed CLI from a separate working directory, checks four reference cases in cached/recompute modes, sends four concurrent native streams, and compares normal/streaming SDK text and chat results. See the [CPU record](../results/runtime-checkpoint-cpu-20261102.json) and [A40 record](../results/runtime-checkpoint-a40-20261102.json). These are functional checks; performance measurements retain their original v0.2.0 scope.

## Use

Follow [Quickstart](QUICKSTART.md) to prepare the local model, then:

```bash
inference-forge doctor --json
inference-forge serve /path/to/Qwen2.5-0.5B-Instruct \
  --served-model-name qwen2.5-0.5b
```

The serving target remains Qwen2.5-0.5B-Instruct, float32, fixed-length greedy generation, up to 512 total tokens and 64 output tokens. See [API](API.md) for supported request fields and [Supported models](SUPPORTED_MODELS.md) for the separate 8B experiments.
