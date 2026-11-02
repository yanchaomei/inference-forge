# InferenceForge

[![CPU contracts](https://github.com/yanchaomei/inference-forge/actions/workflows/check.yml/badge.svg?branch=main)](https://github.com/yanchaomei/inference-forge/actions/workflows/check.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Build and run specialized inference engines for a model, a GPU, and a workload.**

[English](README.md) · [简体中文](README.zh-CN.md) · [Quickstart](docs/QUICKSTART.md) · [API](docs/API.md) · [Supported models](docs/SUPPORTED_MODELS.md) · [Contributing](CONTRIBUTING.md)

InferenceForge uses coding agents and feedback from real execution to build inference engines. The resulting engine runs on its own: no agent or model-service account is needed to serve requests.

Start with the included Qwen2.5-0.5B engine, connect through the OpenAI Python SDK, or use the construction tools to develop a new candidate. The project is for researchers and AI infrastructure developers exploring model, hardware, and workload specialization.

## Install and serve

Use Linux with Python 3.10+ and an NVIDIA GPU. The current serving target is **Qwen2.5-0.5B-Instruct, float32, single GPU**, validated historically on an A40. It accepts up to 512 total tokens and generates 1–64 tokens per request with fixed-length greedy decoding.

Install from source; this project is not currently published on PyPI:

```bash
git clone https://github.com/yanchaomei/inference-forge.git
cd inference-forge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[inference]'
inference-forge doctor --json
```

Download the model using the [quickstart](docs/QUICKSTART.md#download-the-model), then start the service with its local directory:

```bash
inference-forge serve /path/to/Qwen2.5-0.5B-Instruct \
  --port 8000 --served-model-name qwen2.5-0.5b
```

The default address is `http://127.0.0.1:8000`. Check readiness with `curl --fail http://127.0.0.1:8000/health`.

## Use the OpenAI SDK

```bash
python -m pip install 'openai>=2.54,<3'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[{"role": "user", "content": "Explain KV caching in one sentence."}],
    max_tokens=32,
    temperature=0,
)
print(response.choices[0].message.content)
```

Text completions, plain-text chat messages, and streaming are supported. The service uses the model's chat template. See [API examples and supported parameters](docs/API.md) for streaming, usage, and request limits. Authentication and TLS belong at your gateway when exposing the service beyond loopback.

## Build an engine

The construction loop starts from a runnable seed, asks a coding agent to modify it, and evaluates candidates on a separate GPU host. Correctness checks and measured workload results feed the next iteration.

```text
Model + hardware + workload → Coding agent → Candidate engine → GPU evaluation
                                  ↑                                  │
                                  └──────── execution feedback ──────┘
```

Use `inference-forge construct --help` for the construction CLI and [Development](docs/DEVELOPMENT.md#construct-a-candidate) for setup and a full command. Construction requires a configured coding agent; serving an existing engine does not.

## Models and experiments

| Target | What you can run | Guide |
| --- | --- | --- |
| Qwen2.5-0.5B-Instruct / float32 / single A40 | Packaged engine, batched decoding, HTTP and SSE | [Quickstart](docs/QUICKSTART.md) |
| Qwen3-8B / BF16 / single A40 | Separate KV save/restore and multi-session experiments | [8B experiments](experiments/qwen3_8b/README.md) |

Qwen3-8B is not a `serve` target. Run `inference-forge models --json` to list serving targets, or consult the [support matrix](docs/SUPPORTED_MODELS.md).

On a single A40 with float32, four concurrent requests, and 16 output tokens per request, the 0.5B batched candidate measured **2.34×** the throughput of this project's initial sequential runtime. The separate 8B storage/session experiment recorded a **0.815%** reduction in median total time over its original components across three development repeats. See the [0.5B comparison and scope](docs/EXPERIMENTS.md) and [exploratory 8B results](experiments/qwen3_8b/README.md#结果与未解决的问题) for raw samples and interpretation.

## Documentation

| I want to… | Start here |
| --- | --- |
| Install, download a model, and send a request | [Quickstart](docs/QUICKSTART.md) |
| Connect an application | [API reference](docs/API.md) |
| Check models, precision, and hardware | [Supported models](docs/SUPPORTED_MODELS.md) |
| Change the code or construct an engine | [Development](docs/DEVELOPMENT.md) |
| Understand the design | [Architecture](docs/ARCHITECTURE.md) · [Specialization](docs/SPECIALIZATION.md) |
| See what's next | [Roadmap](docs/ROADMAP.md) |
| Review a release | [Releases](https://github.com/yanchaomei/inference-forge/releases) · [Release candidate notes](docs/RELEASE-v0.3.0rc1.md) · [Changelog](CHANGELOG.md) |

## Contribute

Reproducible failures, clearer examples, numerical checks, and measured performance improvements are welcome. Start with the [contribution guide](CONTRIBUTING.md), [issue forms](https://github.com/yanchaomei/inference-forge/issues/new/choose), or [security policy](SECURITY.md).

## Resources

- [Demo and video](https://yanchaomei.github.io/inference-forge/) — the recorded 0.5B construction example.
- [34-page project introduction](https://yanchaomei.github.io/inference-forge/project-introduction.pdf) and [submission materials](docs/SUBMISSION.md).
- [v0.2.0 validation record](results/release-v0.2.0-validation.json) — the historical HTTP/SSE, concurrency, and numerical checks for that release.

Code is licensed under [MIT](LICENSE). Model weights are downloaded separately under their own licenses. See [NOTICE](NOTICE.md) for dependencies and research references.
