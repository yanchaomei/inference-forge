# Quickstart

Run the included Qwen2.5-0.5B engine and send your first request. No coding-agent account is needed. [中文入口](../README.zh-CN.md) · [API](API.md) · [Model support](SUPPORTED_MODELS.md)

## Install

Use a Linux machine with Python 3.10+, an NVIDIA GPU, and a working CUDA-enabled PyTorch installation. A single A40 is the measured hardware for this engine. The runtime uses float32, a 512-token total context limit, and fixed-length greedy output of 1–64 tokens.

```bash
git clone https://github.com/yanchaomei/inference-forge.git
cd inference-forge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[inference]'
nvidia-smi
inference-forge doctor --json
inference-forge models --json
```

This installs from the checkout; the package is not currently published on PyPI. `doctor` reports dependencies and CUDA availability. It does not load a model. Add `--require-gpu` when a script should fail if CUDA is unavailable. Keep the virtual environment active in each terminal used below.

## Download the model

Choose an available GPU and a model directory. The example pins the model revision used by the existing 0.5B task:

```bash
export CUDA_VISIBLE_DEVICES=0
export FORGE_MODEL_PATH="$PWD/cache/models/Qwen2.5-0.5B-Instruct"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    local_dir=os.environ["FORGE_MODEL_PATH"],
)
PY
```

If you already have this checkpoint, set `FORGE_MODEL_PATH` to its local directory containing `config.json`, tokenizer files, and weights. Serving reads local files after download. Model weights are not included in this repository; follow the model publisher's license.

## Start the service

```bash
inference-forge serve "$FORGE_MODEL_PATH" \
  --port 8000 --served-model-name qwen2.5-0.5b \
  --event-log runs/demo-events.jsonl
```

When the service reports `"event": "ready"`, use another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models

curl --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-0.5b","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
```

The completion includes `choices[0].text`, token usage, and the original `text` and `token_ids` fields. The model name must match `--served-model-name` when supplied.

## Send a chat request

```bash
python -m pip install 'openai>=1.50,<3'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "What does a KV cache store?"},
    ],
    max_tokens=32,
    temperature=0,
)
print(response.choices[0].message.content)
print(response.usage)
```

`api_key="unused"` satisfies the SDK's client setup; the service has no built-in authentication. Chat accepts plain-text `system`, `user`, and `assistant` messages and applies the tokenizer's chat template. The template and conversation history count toward the 512-token context limit.

## Stream the output

Using the same client:

```python
stream = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[{"role": "user", "content": "用一句话介绍中国科学技术大学。"}],
    max_tokens=32,
    temperature=0,
    stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    if chunk.usage:
        print("\nUsage:", chunk.usage)
```

Output is fixed-length greedy decoding: generation finishes after `max_tokens` with `finish_reason="length"`, including when an EOS token appears earlier. It may contain special tokens. See the [API reference](API.md) for supported controls and errors.

## Use a remote GPU

Run the installation and service on the GPU machine. From your client machine, forward its loopback port:

```bash
ssh -L 8000:127.0.0.1:8000 user@gpu-host
```

Then use the same local SDK address. The service binds to `127.0.0.1` by default. Use `--host` only when you need another interface, with authentication and TLS provided by your gateway. Stop the service with `Ctrl+C`; `--event-log` records request and scheduling events.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| `serve` is not recognized | Install the current checkout into the active environment. The v0.2.0 CLI predates these subcommands. |
| CUDA is unavailable | Read `doctor --json` and `nvidia-smi`; check the selected GPU and CUDA support in PyTorch. |
| Model or tokenizer cannot be found | Point to the complete local checkpoint directory, not a single weights file. |
| Port is in use | Change `--port` and the client URL together. |
| HTTP 400 | Check `max_tokens` (1–64), total context (512), plain-text messages, and supported parameters. |
| HTTP 404 for a model | Use the name returned by `/v1/models`. |
| No coding-agent login | Serving works without it. Login is needed only for construction. |

## Next steps

- [API reference](API.md): text completions, streaming, and error handling.
- [Development](DEVELOPMENT.md): code layout, verification, and candidate construction.
- [8B experiments](../experiments/qwen3_8b/README.md): separate environment and KV/session APIs.
- [v0.2.0 quickstart](https://github.com/yanchaomei/inference-forge/blob/v0.2.0/docs/QUICKSTART.md): reproduce the historical release with its original commands.
