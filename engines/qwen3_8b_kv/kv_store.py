"""Lossless, capacity-bounded host storage for model KV snapshots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch


Tensor = torch.Tensor
KV = list[tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class _Entry:
    kv: KV
    logits: Tensor
    identity: str
    payload_bytes: int


def _identity_key(identity: dict) -> str:
    return json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _host_copy(tensor: Tensor, cuda_streams: dict[torch.device, Any]) -> Tensor:
    source = tensor.detach()
    host = torch.empty_like(
        source,
        device="cpu",
        pin_memory=True,
        memory_format=torch.preserve_format,
    )
    if source.is_cuda:
        stream = torch.cuda.current_stream(source.device)
        cuda_streams[source.device] = stream
        host.copy_(source, non_blocking=True)
    else:
        host.copy_(source)
    return host


class KVStore:
    def __init__(self, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self._max_bytes = max_bytes
        self._used_bytes = 0
        self._entries: dict[str, _Entry] = {}

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    def put(self, key: str, kv: KV, logits: Tensor, identity: dict) -> None:
        identity_key = _identity_key(identity)
        payload_bytes = _tensor_bytes(logits)
        for k_tensor, v_tensor in kv:
            payload_bytes += _tensor_bytes(k_tensor) + _tensor_bytes(v_tensor)

        old = self._entries.get(key)
        old_bytes = 0 if old is None else old.payload_bytes
        if self._used_bytes - old_bytes + payload_bytes > self._max_bytes:
            raise MemoryError("KV snapshot exceeds the payload budget")

        cuda_streams: dict[torch.device, Any] = {}
        host_kv = [
            (
                _host_copy(k_tensor, cuda_streams),
                _host_copy(v_tensor, cuda_streams),
            )
            for k_tensor, v_tensor in kv
        ]
        host_logits = _host_copy(logits, cuda_streams)
        for stream in cuda_streams.values():
            stream.synchronize()
        entry = _Entry(host_kv, host_logits, identity_key, payload_bytes)

        self._entries[key] = entry
        self._used_bytes = self._used_bytes - old_bytes + payload_bytes

    def get(self, key: str, identity: dict, device: Any) -> tuple[KV, Tensor]:
        entry = self._entries[key]
        if _identity_key(identity) != entry.identity:
            raise ValueError("identity does not match the stored snapshot")

        kv = [
            (
                k_tensor.to(device=device, non_blocking=True, copy=True),
                v_tensor.to(device=device, non_blocking=True, copy=True),
            )
            for k_tensor, v_tensor in entry.kv
        ]
        logits = entry.logits.to(device=device, non_blocking=True, copy=True)
        return kv, logits

    def delete(self, key: str) -> None:
        entry = self._entries.pop(key)
        self._used_bytes -= entry.payload_bytes
