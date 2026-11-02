# Roadmap

InferenceForge aims to deliver engines that are easy to install, connect to an application, and reproduce. This roadmap describes priorities, not committed release dates.

## Current focus: a usable 0.5B runtime

The current source tree packages the generated Qwen2 backend, adds a unified CLI, and provides an OpenAI SDK-compatible text/chat subset with SSE. The documentation and contribution templates follow that user path. The package version is a 0.3.0 release candidate; release checks must cover both the installed package and the real GPU service.

The supported serving model remains Qwen2.5-0.5B-Instruct on the documented short-context, float32 path. Qwen3-8B storage and session experiments keep their separate entry points.

## Next priorities

| Priority | Intended outcome | What demonstrates completion |
| --- | --- | --- |
| Reproducible installation | A release can be installed and served outside a checkout | Wheel installation, CLI/backend checks, and a real GPU request tied to that release |
| Runtime regression coverage | Numerical and protocol changes remain compatible | Independent logits checks plus SDK, streaming, concurrent-request, and failure checks |
| Complete 8B serving | A usable Qwen3-8B engine and service | Fixed checkpoint, full forward/KV checks, concurrency checks, and a documented runnable entry |
| Meaningful performance comparisons | Results that help users select an engine | Matched vLLM/SGLang workloads, latency and throughput distributions, raw samples, and failures |
| Construction comparisons | Measure the value of execution feedback | Fixed generator, starting point, budget, and held-out tasks; VibeServe/MetaInfer comparisons |
| Better session storage | Improve complete multi-turn request cost | Longer traces, component controls, state correctness, memory budgets, and repeatable end-to-end gains |

## How scope expands

Add one working model/hardware/workload combination at a time. Expand the support matrix after correctness and measured behavior are documented. Long context, quantization, MoE, additional hardware, and broader serving features remain future work.

Contributions that remove a real installation obstacle, reproduce a failure, improve a concrete workload, or add a clear example are useful now. Open a [model request or issue](https://github.com/yanchaomei/inference-forge/issues/new/choose) with the task you need to run.

See [Specialization](SPECIALIZATION.md) for the design rationale and [Supported models](SUPPORTED_MODELS.md) for the current limits.
