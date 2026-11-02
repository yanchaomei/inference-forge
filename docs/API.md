# Serving API

The service provides an OpenAI SDK-compatible subset for text completions and plain-text chat. Start it with the [quickstart](QUICKSTART.md).

```bash
inference-forge serve /path/to/Qwen2.5-0.5B-Instruct \
  --port 8000 --served-model-name qwen2.5-0.5b
```

The examples below use this model alias. If `--served-model-name` is omitted, the checkpoint directory name becomes the model ID. Query `/v1/models` to discover it. The server uses local model files and does not download a model from its ID.

## Routes

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Readiness and model identity; `ready: true` on a ready service |
| GET | `/v1/models` | Model list, including the served model ID |
| POST | `/v1/completions` | One text prompt or one token-ID sequence |
| POST | `/v1/chat/completions` | One conversation of plain-text messages |

Send JSON request bodies with `Content-Type: application/json`. The server binds to `127.0.0.1` by default and has no built-in authentication or TLS. `api_key="unused"` is an SDK placeholder, not access control. Use a gateway or SSH forwarding for remote access.

## Python client

Install the client with `python -m pip install 'openai>=2.54,<3'`.

A runnable example is in [examples/clients/chat.py](../examples/clients/chat.py):

```bash
python examples/clients/chat.py --stream --prompt "Explain KV caching briefly."
```

It discovers the served model ID through `/v1/models`; pass `--model` to select it explicitly.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")

completion = client.completions.create(
    model="qwen2.5-0.5b",
    prompt="The capital of France is",
    max_tokens=16,
    temperature=0,
)
print(completion.choices[0].text)
print(completion.usage)

chat = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "What is a KV cache?"},
    ],
    max_tokens=32,
    temperature=0,
)
print(chat.choices[0].message.content)
```

Chat messages contain only `role` and string `content`. Accepted roles are `system`, `user`, and `assistant`. The tokenizer applies its chat template with an assistant generation prompt. Send conversation history on each request; this API does not retain chat sessions.

## Request parameters

| Parameter | Support |
| --- | --- |
| `model` | The exact served model ID. Required by the SDK; optional in raw JSON for legacy clients. |
| `prompt` | Completions only: a single string; prompt batches are not supported. |
| `input_ids` | Completions extension: a nonempty list of valid token IDs. Takes precedence over `prompt` if both are supplied. |
| `messages` | Chat only: a nonempty list of plain-text messages as above. |
| `max_tokens` | Integer from 1 to 64; defaults to 16. |
| `max_completion_tokens` | Chat-only alias for `max_tokens`; do not supply both. |
| `stream` | Boolean, defaults to `false`. |
| `stream_options` | With `stream=true`: `{"include_usage": true}` adds a final usage block. |
| `temperature` | Only `0`, the default. |
| `top_p` | Only `1`, the default. |
| `n` | Only integer `1`, the default. |
| `frequency_penalty`, `presence_penalty` | Only `0`, the default. |
| `stop` | Omit or use `null`; stop sequences are not supported. |
| `request_id` | Optional extension: nonempty string up to 128 characters, unique among active requests. |

The tokenized input plus requested output must fit within **512 tokens**. Chat template tokens and all supplied history count toward this limit. Empty tokenized inputs are rejected.

Generation is greedy and fixed-length. It continues for `max_tokens`, even if an EOS token appears earlier, and ends with `finish_reason="length"`. Decoded output can contain special tokens. Tool calls, multimodal content, JSON mode, sampling controls outside the values above, logprobs, and other unlisted parameters return HTTP 400.

## Streaming

Using the client above:

```python
stream = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[{"role": "user", "content": "用一句话解释批量解码。"}],
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

For streaming text completions, use `client.completions.create(..., stream=True)` and read `chunk.choices[0].text` whenever `choices` is nonempty.

At the HTTP level, the response is `text/event-stream`:

1. Chat begins with an assistant-role chunk.
2. Content chunks carry incremental text; some token events have empty text while a Unicode character is incomplete.
3. A final choice reports `finish_reason="length"` without repeating the full output.
4. With `stream_options.include_usage=true`, a separate usage chunk has `choices: []` and prompt/completion/total token counts. SDK streams otherwise omit usage.
5. The stream ends with `data: [DONE]`.

## Responses and existing clients

Non-streaming responses contain `id`, `object`, `created`, `model`, `choices`, and `usage`. Chat uses `choices[0].message`; completions use `choices[0].text`.

The existing runtime fields remain available: `request_id`, `text`, `token_ids`, `finish_reason`, and non-streaming `usage`. Raw token stream events retain `token_id`, `index`, and `text`; the final event retains the full token sequence. Read standard `choices` for SDK text deltas.

Streaming usage has one compatibility rule for the original evaluator:

| Request | Usage location |
| --- | --- |
| Any stream with `stream_options.include_usage=true` | One separate final chunk with `choices: []` |
| A completions stream that omits `model` and does not request usage | The final completion event retains its original `usage` field |
| Chat streams or completions streams with `model`, without requested usage | No usage field |

The second case preserves `forge.evaluate.stream_request` and existing native clients. New SDK clients supply `model` and should request `include_usage` when they need token counts.

## Errors

Request validation happens before streaming starts. Errors use this shape:

```json
{
  "error": {
    "message": "Unsupported parameter: tools",
    "type": "invalid_request_error",
    "param": "tools",
    "code": "unsupported_parameter"
  }
}
```

| Status | Meaning |
| --- | --- |
| 400 | Unsupported parameter, malformed input, or token limit violation |
| 404 | Unknown route or model ID |
| 409 | A supplied `request_id` is already active |
| 500 | Engine or server failure |
| 503 | Scheduler unavailable or request capacity reached |
| 504 | Request deadline exceeded |

The Python SDK raises `BadRequestError` for 400 and `NotFoundError` for 404. If a failure occurs after SSE headers have been sent, the stream carries an `error` object and ends with `[DONE]`; its HTTP status cannot change.

## Compatibility

The `/v1` routes implement the subset described here. There are no embeddings, Responses API, audio, vision, tools, or structured-output endpoints. The separate [Qwen3-8B session experiments](../experiments/qwen3_8b/README.md) are not exposed through this service.
