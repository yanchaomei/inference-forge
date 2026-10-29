"""Numerical contracts against the cached, real Hugging Face model.

Run remotely with FORGE_MODEL_PATH set to the existing checkpoint.  Missing
weights are an error rather than a skipped correctness claim.
"""

import os
import concurrent.futures
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from forge.engine import QwenEngine


@pytest.mark.parametrize("text", [
    "中国科学技术大学：你好，世界！🧠🚀🙂𠮷👩‍🚀",
    "中文与 emoji：🦄🧑🏽‍💻，标点 . ! ?",
])
def test_incremental_decode_preserves_unicode_byte_token_boundaries(text):
    from forge.server import incremental_decode

    tokenizer = AutoTokenizer.from_pretrained(os.environ["FORGE_MODEL_PATH"],
                                             local_files_only=True)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    expected = tokenizer.decode(tokens, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
    naive = "".join(tokenizer.decode([token], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False) for token in tokens)
    print(f"unicode source={text!r} naive_per_token={naive!r} token_count={len(tokens)}", flush=True)
    assert expected == text
    assert naive != expected and "\ufffd" in naive
    emitted, chunks = "", []
    for index in range(len(tokens)):
        delta = incremental_decode(tokenizer, tokens[:index + 1], emitted,
                                   final=index == len(tokens) - 1)
        chunks.append(delta)
        emitted += delta
        assert expected.startswith(emitted)
    assert emitted == expected
    assert "" in chunks  # byte fragments still produce a token event, with empty text

    # A length-limited generation may end partway through a byte sequence.
    # Its final event must flush the tokenizer's complete (possibly replaced)
    # decoding rather than silently dropping those remaining bytes.
    for length in range(1, len(tokens)):
        decoded = tokenizer.decode(tokens[:length], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        if not decoded.endswith("\ufffd"):
            continue
        emitted = ""
        for index in range(length):
            emitted += incremental_decode(tokenizer, tokens[:index + 1], emitted,
                                          final=index == length - 1)
        assert emitted == decoded


@pytest.fixture(scope="module")
def models():
    path = os.environ["FORGE_MODEL_PATH"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    candidate = QwenEngine(path, dtype="float32", device="cuda")
    reference = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float32, local_files_only=True,
        attn_implementation="eager",
    ).to("cuda").eval()
    return candidate, reference


def assert_logits_match(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    print(
        f"logits shape={tuple(actual.shape)} max_abs={delta.max().item():.9g} "
        f"mean_abs={delta.mean().item():.9g}",
        flush=True,
    )
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))


@torch.inference_mode()
def test_full_prefill_matches_real_hugging_face_logits(models):
    candidate, reference = models
    tokens = [151644, 8948, 198, 9707, 11, 1268, 525, 264, 17847, 13, 151645]
    actual, cache = candidate.forward(tokens)
    expected = reference(torch.tensor([tokens], device="cuda")).logits
    assert actual.shape == (1, len(tokens), reference.config.vocab_size)
    assert_logits_match(actual, expected)
    assert len(cache) == reference.config.num_hidden_layers


@torch.inference_mode()
def test_cached_multi_token_chunks_match_full_recompute_and_reference(models):
    candidate, reference = models
    prefix = [151644, 872, 198, 3838, 374, 220]
    _, cache = candidate.forward(prefix)
    for chunk in ([16], [17, 18], [19], [20, 21, 22]):
        old_length = len(prefix)
        actual, cache = candidate.forward(chunk, cache)
        prefix += chunk
        full, _ = candidate.forward(prefix)
        expected = reference(torch.tensor([prefix], device="cuda")).logits
        assert_logits_match(actual, full[:, -len(chunk):])
        assert_logits_match(actual, expected[:, -len(chunk):])
        assert all(k.shape[-2] == old_length + len(chunk) and
                   v.shape[-2] == old_length + len(chunk) for k, v in cache)


@torch.inference_mode()
def test_interleaved_request_states_preserve_independent_histories(models):
    candidate, reference = models
    histories = [[151644, 872, 198, 9707], [16, 17, 18, 19, 20, 21]]
    states = [candidate.prefill(tokens) for tokens in histories]
    original_states = list(states)
    for request, token in [(0, 11), (1, 22), (0, 1268), (0, 525), (1, 23)]:
        histories[request].append(token)
        states[request] = candidate.step(token, states[request])
        expected = reference(torch.tensor([histories[request]], device="cuda")).logits
        assert_logits_match(states[request].last_logits, expected[:, -1, :])
        assert states[request].length == len(histories[request])
    # A returned state is a stable snapshot: a different request or a child
    # continuation must not append to or overwrite its cache in place.
    assert original_states[0].length == 4
    assert original_states[1].length == 6
    assert all(k.shape[-2] == 4 for k, _ in original_states[0].cache)
    assert all(k.shape[-2] == 6 for k, _ in original_states[1].cache)
    fresh = candidate.prefill(histories[0][:4])
    assert_logits_match(fresh.last_logits, original_states[0].last_logits)


def http_json(base, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base + route, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


@pytest.fixture(scope="module")
def live_server():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    artifacts = Path("results/raw/engine-tdd") / f"http-{time.time_ns()}"
    artifacts.mkdir(parents=True)
    events = artifacts / "events.jsonl"
    log_path = artifacts / "server.log"
    base = f"http://127.0.0.1:{port}"
    with log_path.open("w") as log:
        process = subprocess.Popen([
            sys.executable, "-m", "forge.server", "--model",
            os.environ["FORGE_MODEL_PATH"], "--port", str(port),
            "--dtype", "float32", "--event-log", str(events),
        ], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError(log_path.read_text())
                try:
                    if http_json(base, "/health")["ready"]:
                        break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                raise AssertionError("server did not become ready within 60 seconds")
            print(f"HTTP artifacts: {artifacts.resolve()}", flush=True)
            yield base, events
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


@torch.inference_mode()
def test_server_runtime_http_contract(live_server, models):
    base, event_path = live_server
    _, reference = models
    tokenizer = AutoTokenizer.from_pretrained(os.environ["FORGE_MODEL_PATH"],
                                             local_files_only=True)
    identity = http_json(base, "/v1/models")["data"][0]
    assert identity["engine_kind"] == "generated-qwen2"
    prefix = [151644, 872, 198, 9707]
    response = http_json(base, "/v1/completions", {
        "input_ids": prefix, "prompt": "ignored because input_ids is present",
        "max_tokens": 4, "request_id": "single",
    })
    history = list(prefix)
    expected_tokens = []
    for _ in range(4):
        token = int(reference(torch.tensor([history], device="cuda")).logits[0, -1].argmax())
        expected_tokens.append(token)
        history.append(token)
    assert response["token_ids"] == expected_tokens
    assert response["request_id"] == "single"
    assert response["usage"] == {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8}
    assert isinstance(response["text"], str)

    continuation = [11, 1268, 525]
    traces = [http_json(base, "/debug/trace", {
        "input_ids": prefix, "continuation_ids": continuation, "mode": mode,
    }) for mode in ("cached", "recompute")]
    assert traces[0]["cache_lengths"] == traces[1]["cache_lengths"] == [4, 5, 6]
    history = list(prefix)
    for index, token in enumerate(continuation):
        logits = reference(torch.tensor([history], device="cuda")).logits[0, -1]
        logprob = float(logits.log_softmax(-1)[token])
        for trace in traces:
            observed = trace["observations"][index]
            assert observed["index"] == index
            assert observed["top1"] == int(logits.argmax())
            assert observed["logprob"] == pytest.approx(logprob, abs=1e-3)
        history.append(token)

    barrier = threading.Barrier(2)

    def consume(request_id, tokens):
        barrier.wait(timeout=10)
        request = urllib.request.Request(base + "/v1/completions", data=json.dumps({
            "input_ids": tokens, "max_tokens": 8,
            "request_id": request_id, "stream": True,
        }).encode(), headers={"Content-Type": "application/json"})
        chunks = []
        done = False
        with urllib.request.urlopen(request, timeout=30) as stream:
            assert stream.headers.get_content_type() == "text/event-stream"
            for line in stream:
                line = line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                chunks.append(json.loads(data))
        assert done
        per_token, final = chunks[:-1], chunks[-1]
        assert [c["index"] for c in per_token] == list(range(8))
        assert all(c["request_id"] == request_id for c in chunks)
        assert final["finish_reason"] == "length"
        assert final["token_ids"] == [c["token_id"] for c in per_token]
        assert final["usage"]["completion_tokens"] == 8
        assert "".join(c["text"] for c in per_token) == final["text"]
        assert final["text"] == tokenizer.decode(final["token_ids"], skip_special_tokens=False,
                                                   clean_up_tokenization_spaces=False)
        return final["token_ids"]

    prefixes = [prefix, [16, 17, 18, 19, 20]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(consume, name, tokens) for name, tokens in
                   zip(("stream-a", "stream-b"), prefixes)]
        generated = [future.result(timeout=40) for future in futures]
    for tokens, actual in zip(prefixes, generated):
        expected = http_json(base, "/v1/completions", {
            "input_ids": tokens, "max_tokens": 8,
        })["token_ids"]
        assert actual == expected
    records = [json.loads(line) for line in event_path.read_text().splitlines()]
    tokens = [row for row in records if row["event"] == "token" and
              row["request_id"] in ("stream-a", "stream-b")]
    order = [row["request_id"] for row in tokens]
    assert len(order) == 16
    assert any(order[i] == order[i + 2] != order[i + 1]
               for i in range(len(order) - 2)), order
    assert all(type(row["monotonic_ns"]) is int for row in records)
    print(f"round-robin token order: {order}", flush=True)

    for invalid in ({"input_ids": [], "max_tokens": 1},
                    {"input_ids": [1], "max_tokens": 65},
                    {"input_ids": [1] * 512, "max_tokens": 1}):
        with pytest.raises(urllib.error.HTTPError) as error:
            http_json(base, "/v1/completions", invalid)
        assert error.value.code == 400
