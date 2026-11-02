"""HTTP/SSE serving of the generated batched Qwen2 implementation."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import sys
import threading
import time
import uuid

import torch
from transformers import AutoTokenizer

from .backends.qwen2.engine import EngineState, QwenEngine
from . import protocol


MAX_CONTEXT = 512
MAX_OUTPUT = 64
MAX_INFLIGHT = 32
REQUEST_TIMEOUT = 120.0


def incremental_decode(tokenizer, token_ids, emitted_text, *, final=False):
    """Decode a token prefix and emit only its new stable text suffix.

    Byte BPE tokens can end inside a Unicode character.  Hold trailing
    replacement characters until a later token completes the bytes, while
    the final token flushes exactly the tokenizer's complete decoding.
    A caller must retain and pass back the text already emitted.
    """
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
    stable = decoded if final else decoded.rstrip("\ufffd")
    if not stable.startswith(emitted_text):
        raise ValueError("decoded token prefix changed already emitted text")
    return stable[len(emitted_text):]


class RequestError(protocol.ProtocolError):
    pass


@dataclass
class Completion:
    request_id: str
    input_ids: list[int]
    max_tokens: int
    deadline: float = field(default_factory=lambda: time.monotonic() + REQUEST_TIMEOUT)
    messages: queue.Queue = field(default_factory=lambda: queue.Queue(MAX_OUTPUT + 2))
    cancelled: threading.Event = field(default_factory=threading.Event)
    generated: list[int] = field(default_factory=list)
    emitted_text: str = ""
    state: EngineState | None = None

    def usage(self):
        return {"prompt_tokens": len(self.input_ids),
                "completion_tokens": len(self.generated),
                "total_tokens": len(self.input_ids) + len(self.generated)}


class Runtime:
    def __init__(self, engine, tokenizer, event_log=None, *, served_model_name=None):
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = served_model_name or self.engine.model_path.name
        self.created = int(time.time())
        self.gpu_lock = threading.Lock()
        self.ids_lock = threading.Lock()
        self.event_lock = threading.Lock()
        self.live_ids = set()
        self.pending = queue.Queue(MAX_INFLIGHT)
        self.stop = threading.Event()
        self.log = None
        if event_log:
            path = Path(event_log)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.log = path.open("a", buffering=1)
        self.scheduler = threading.Thread(target=self._schedule, name="token-scheduler", daemon=True)
        self.scheduler.start()

    def identity(self):
        return {"id": self.model_name, "object": "model", "created": self.created,
                "owned_by": "inference-forge",
                "engine_kind": "generated-qwen2",
                "config_hash": self.engine.config_hash,
                "parameter_count": self.engine.parameter_count}

    def event(self, event, request_id, **extra):
        if self.log is None:
            return
        with self.event_lock:
            record = {"event": event, "request_id": request_id,
                      "monotonic_ns": time.monotonic_ns(), **extra}
            self.log.write(json.dumps(record, ensure_ascii=False) + "\n")

    def tokens(self, value, name="input_ids", nonempty=True):
        if not isinstance(value, list) or (nonempty and not value):
            raise RequestError(f"{name} must be a {'nonempty ' if nonempty else ''}list")
        if len(value) > MAX_CONTEXT or any(
            type(t) is not int or not 0 <= t < self.engine.vocab_size for t in value
        ):
            raise RequestError(f"{name} contains invalid tokens or exceeds {MAX_CONTEXT}")
        return value

    def submit(self, payload):
        if self.stop.is_set() or not self.scheduler.is_alive():
            raise RequestError("scheduler unavailable", 503)
        if "input_ids" in payload:
            tokens = self.tokens(payload["input_ids"])
        elif isinstance(payload.get("prompt"), str):
            tokens = self.tokens(self.tokenizer.encode(payload["prompt"], add_special_tokens=False))
        else:
            raise RequestError("provide input_ids or a prompt string")
        maximum = payload.get("max_tokens", 16)
        if type(maximum) is not int or not 1 <= maximum <= MAX_OUTPUT:
            raise RequestError(f"max_tokens must be an integer in [1, {MAX_OUTPUT}]")
        if len(tokens) + maximum > MAX_CONTEXT:
            raise RequestError(f"prompt plus output must fit within {MAX_CONTEXT} tokens")
        if type(payload.get("stream", False)) is not bool:
            raise RequestError("stream must be boolean")
        request_id = payload.get("request_id", str(uuid.uuid4()))
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise RequestError("request_id must be a nonempty string of at most 128 characters")
        item = Completion(request_id, tokens, maximum)
        with self.ids_lock:
            if request_id in self.live_ids:
                raise RequestError("request_id is already in flight", 409)
            if len(self.live_ids) >= MAX_INFLIGHT:
                raise RequestError("request queue is full", 503)
            self.live_ids.add(request_id)
            try:
                self.event("accepted", request_id, active_count=len(self.live_ids),
                           prompt_tokens=len(tokens), max_tokens=maximum)
                self.pending.put_nowait(item)
            except Exception:
                self.live_ids.remove(request_id)
                raise
        return item

    def _finish(self, item, error=None):
        # A terminal message always fits: at most MAX_OUTPUT token messages
        # precede it, and the queue has two extra slots even for a lost client.
        message = {"request_id": item.request_id,
                   "finish_reason": "error" if error else "length",
                   "token_ids": list(item.generated), "usage": item.usage(),
                   "text": self.tokenizer.decode(item.generated, skip_special_tokens=False,
                                                  clean_up_tokenization_spaces=False)}
        if error:
            message["error"] = error
        try:
            with self.ids_lock:
                self.live_ids.discard(item.request_id)
                self.event("finished", item.request_id, active_count=len(self.live_ids),
                           finish_reason=message["finish_reason"],
                           completion_tokens=len(item.generated), error=error)
        finally:
            item.state = None
            item.messages.put_nowait(message)

    @torch.inference_mode()
    def _schedule(self):
        active = deque()
        try:
            while not self.stop.is_set():
                if not active:
                    try:
                        active.append(self.pending.get(timeout=0.1))
                    except queue.Empty:
                        continue
                while True:
                    try:
                        active.append(self.pending.get_nowait())
                    except queue.Empty:
                        break
                # A round is a snapshot: every live request receives exactly one
                # token before any of them is advanced again.  Decode requests
                # in that snapshot share one batched model invocation.
                current = [active.popleft() for _ in range(len(active))]
                ready = []
                for item in current:
                    if item.cancelled.is_set():
                        self._finish(item, "client disconnected")
                        continue
                    if time.monotonic() >= item.deadline:
                        self._finish(item, "request deadline exceeded")
                        continue
                    ready.append(item)
                if not ready:
                    continue
                try:
                    with self.gpu_lock:
                        decode = []
                        for item in ready:
                            if item.state is None:
                                item.state = self.engine.prefill(item.input_ids)
                                self.event("prefill", item.request_id,
                                           active_count=len(ready),
                                           cache_length=item.state.length)
                            else:
                                decode.append(item)
                        if decode:
                            next_states = self.engine.batch_step(
                                [item.generated[-1] for item in decode],
                                [item.state for item in decode],
                            )
                            for item, state in zip(decode, next_states):
                                item.state = state
                        tokens = [int(item.state.last_logits.argmax(-1).item())
                                  for item in ready]
                except Exception as exc:
                    for item in ready:
                        self._finish(item, f"{type(exc).__name__}: {exc}")
                    continue
                for item, token in zip(ready, tokens):
                    try:
                        index = len(item.generated)
                        item.generated.append(token)
                        text = incremental_decode(
                            self.tokenizer, item.generated, item.emitted_text,
                            final=len(item.generated) == item.max_tokens,
                        )
                        item.emitted_text += text
                        self.event("token", item.request_id, active_count=len(ready),
                                   index=index, token_id=token,
                                   cache_length=item.state.length)
                        item.messages.put_nowait({"request_id": item.request_id, "index": index,
                                                  "token_id": token, "text": text})
                        if len(item.generated) == item.max_tokens:
                            self._finish(item)
                        else:
                            active.append(item)
                    except Exception as exc:
                        self._finish(item, f"{type(exc).__name__}: {exc}")
        finally:
            while True:
                try:
                    active.append(self.pending.get_nowait())
                except queue.Empty:
                    break
            for item in active:
                self._finish(item, "scheduler stopped")

    @torch.inference_mode()
    def trace(self, payload):
        prefix = list(self.tokens(payload.get("input_ids")))
        continuation = self.tokens(payload.get("continuation_ids"), "continuation_ids", False)
        mode = payload.get("mode", "cached")
        if mode not in ("cached", "recompute"):
            raise RequestError("mode must be cached or recompute")
        if len(prefix) + len(continuation) > MAX_CONTEXT:
            raise RequestError("trace exceeds the context limit")
        observations, lengths = [], []
        # Diagnostics and the scheduler share one GPU critical section.  Trace
        # work is bounded to MAX_CONTEXT and never shares request KV objects.
        with self.gpu_lock:
            state = self.engine.prefill(prefix)
            for index, token in enumerate(continuation):
                logits = state.last_logits[0].float()
                observations.append({"index": index, "top1": int(logits.argmax().item()),
                                     "logprob": float(logits.log_softmax(-1)[token].item())})
                lengths.append(state.length)
                prefix.append(token)
                if mode == "cached":
                    state = self.engine.step(token, state)
                else:
                    state = self.engine.prefill(prefix)
        return {"observations": observations, "cache_lengths": lengths}

    def close(self):
        self.stop.set()
        self.scheduler.join(timeout=10)
        if self.log is not None and not self.scheduler.is_alive():
            self.log.close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def runtime(self):
        return self.server.runtime

    def setup(self):
        super().setup()
        self.connection.settimeout(REQUEST_TIMEOUT)

    def log_message(self, format, *args):
        print(json.dumps({"event": "http", "message": format % args}), file=sys.stderr, flush=True)

    def _json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            if self.path == "/health":
                ready = self.runtime.scheduler.is_alive() and not self.runtime.stop.is_set()
                self._json({"ready": ready, **self.runtime.identity()}, 200 if ready else 503)
            elif self.path == "/v1/models":
                self._json({"object": "list", "data": [self.runtime.identity()]})
            else:
                self._json({"error": "unknown route"}, 404)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True

    def _payload(self):
        if self.headers.get("Transfer-Encoding"):
            raise RequestError("chunked request bodies are not supported")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise RequestError("invalid Content-Length") from None
        if not 0 < length <= 1024 * 1024:
            raise RequestError("request body must be between 1 byte and 1 MiB")
        try:
            payload = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeError):
            raise RequestError("request body must be JSON") from None
        if not isinstance(payload, dict):
            raise RequestError("request body must be a JSON object")
        return payload

    def _next(self, item):
        remaining = item.deadline - time.monotonic()
        if remaining <= 0:
            raise RequestError("request deadline exceeded", 504)
        try:
            return item.messages.get(timeout=remaining)
        except queue.Empty:
            raise RequestError("request deadline exceeded", 504) from None

    def _sse(self, value):
        data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)
        self.wfile.write(("data: " + data + "\n\n").encode())
        self.wfile.flush()

    def do_POST(self):
        item, streaming = None, False
        try:
            payload = self._payload()
            if self.path == "/debug/trace":
                self._json(self.runtime.trace(payload))
                return
            if self.path not in ("/v1/completions", "/v1/chat/completions"):
                self._json(protocol.error_response(RequestError("unknown route", 404), 404), 404)
                return
            chat = self.path == "/v1/chat/completions"
            normalized, include_usage = protocol.normalize_request(
                payload, chat=chat, model=self.runtime.model_name, tokenizer=self.runtime.tokenizer)
            created = int(time.time())
            item = self.runtime.submit(normalized)
            if payload.get("stream", False):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.flush()
                streaming = True
                self.close_connection = True
                if chat:
                    self._sse(protocol.role_chunk(item.request_id, self.runtime.model_name, created))
                while True:
                    message = self._next(item)
                    if "error" in message:
                        raise RequestError(message["error"], 500)
                    self._sse(protocol.response(message, self.runtime.model_name, created,
                        chat=chat, streaming=True,
                        preserve_stream_usage=not chat and 'model' not in payload and not include_usage))
                    if "finish_reason" in message:
                        if include_usage:
                            self._sse(protocol.usage_chunk(message, self.runtime.model_name, created, chat=chat))
                        self._sse("[DONE]")
                        break
            else:
                while True:
                    message = self._next(item)
                    if "finish_reason" in message:
                        if "error" in message:
                            raise RequestError(message["error"], 500)
                        message["text"] = self.runtime.tokenizer.decode(
                            message["token_ids"], skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
                        self._json(protocol.response(message, self.runtime.model_name, created, chat=chat))
                        break
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception as exc:
            status = exc.status if isinstance(exc, protocol.ProtocolError) else 500
            message = protocol.error_response(exc, status)
            try:
                if streaming:
                    self._sse({"request_id": item.request_id, "finish_reason": "error", **message})
                    self._sse("[DONE]")
                else:
                    self._json(message, status)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            self.close_connection = True
        finally:
            if item is not None:
                item.cancelled.set()


def create_server(engine, tokenizer, *, host="127.0.0.1", port=8000,
                  served_model_name=None, event_log=None):
    """Create a service around an engine; the caller owns shutdown and close."""
    runtime = Runtime(engine, tokenizer, event_log, served_model_name=served_model_name)
    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except Exception:
        runtime.close()
        raise
    server.daemon_threads = True
    server.runtime = runtime
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--served-model-name")
    parser.add_argument("--dtype", default="float32", choices=["float32"])
    parser.add_argument("--event-log")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        parser.error("serving requires a CUDA GPU; run inference-forge doctor --require-gpu")
    engine = QwenEngine(args.model, dtype=args.dtype, device="cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    server = create_server(engine, tokenizer, host=args.host, port=args.port,
                           served_model_name=args.served_model_name, event_log=args.event_log)
    runtime = server.runtime
    print(json.dumps({"event": "ready", "ready": True, "host": args.host,
                      "port": server.server_address[1], "model": engine.model_path.name,
                      "dtype": args.dtype, **runtime.identity()}), flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime.close()


if __name__ == "__main__":
    main()
