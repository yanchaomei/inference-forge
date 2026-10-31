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
    # Each logical tensor is a tuple of chunks along the sequence dimension.
    # Generic put() creates one chunk; extend_retained() adds one suffix chunk.
    kv: list[tuple[tuple[Tensor, ...], tuple[Tensor, ...]]]
    logits: Tensor
    identity: str
    identity_value: dict
    payload_bytes: int
    generation: int


@dataclass(frozen=True)
class RetainedLease:
    """Proof of an entry generation and its committed sequence position.

    The retained protocol is for a trusted consumer that does not modify a
    committed KV prefix.  A lease does not prove tensor contents are unchanged;
    replacement and deletion invalidate it through the entry generation.
    """

    key: str
    generation: int
    identity: str
    sequence_length: int


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


def _sequence_length(kv: KV) -> int:
    if not kv:
        return 0
    length = int(kv[0][0].shape[2])
    for pair in kv:
        for tensor in pair:
            if tensor.ndim < 3 or int(tensor.shape[2]) != length:
                raise ValueError("KV tensors must share sequence dimension 2")
    return length


def _copy_chunks_to_device(chunks: tuple[Tensor, ...], device: Any) -> Tensor:
    if len(chunks) == 1:
        return chunks[0].to(device=device, non_blocking=True, copy=True)
    shape = list(chunks[0].shape)
    shape[2] = sum(int(chunk.shape[2]) for chunk in chunks)
    result = torch.empty(shape, dtype=chunks[0].dtype, device=device)
    offset = 0
    for chunk in chunks:
        width = int(chunk.shape[2])
        result[:, :, offset : offset + width].copy_(chunk, non_blocking=True)
        offset += width
    return result


class KVStore:
    def __init__(self, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self._max_bytes = max_bytes
        self._used_bytes = 0
        self._entries: dict[str, _Entry] = {}
        self._next_generation = 1

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
        chunked_kv = [((k_tensor,), (v_tensor,)) for k_tensor, v_tensor in host_kv]
        entry = _Entry(
            chunked_kv,
            host_logits,
            identity_key,
            json.loads(identity_key),
            payload_bytes,
            self._next_generation,
        )
        self._next_generation += 1

        self._entries[key] = entry
        self._used_bytes = self._used_bytes - old_bytes + payload_bytes

    def get(self, key: str, identity: dict, device: Any) -> tuple[KV, Tensor]:
        entry = self._entries[key]
        if _identity_key(identity) != entry.identity:
            raise ValueError("identity does not match the stored snapshot")

        kv = [
            (_copy_chunks_to_device(k_chunks, device), _copy_chunks_to_device(v_chunks, device))
            for k_chunks, v_chunks in entry.kv
        ]
        logits = entry.logits.to(device=device, non_blocking=True, copy=True)
        return kv, logits

    def put_retained(self, key: str, kv: KV, logits: Tensor, identity: dict) -> RetainedLease:
        """Store a full snapshot and return its generation/position lease."""
        self.put(key, kv, logits, identity)
        entry = self._entries[key]
        return RetainedLease(key, entry.generation, entry.identity, _sequence_length(kv))

    def get_retained(
        self, key: str, identity: dict, device: Any
    ) -> tuple[KV, Tensor, RetainedLease]:
        """Restore without consuming the host snapshot and return its lease."""
        kv, logits = self.get(key, identity, device)
        entry = self._entries[key]
        lease = RetainedLease(key, entry.generation, entry.identity, _sequence_length(kv))
        return kv, logits, lease

    def extend_retained(
        self,
        key: str,
        *args: Any,
    ) -> RetainedLease:
        """Replace logits and append a trusted immutable-prefix KV suffix.

        The lease proves the entry generation and committed position, not tensor
        contents.  The caller guarantees it has not modified the committed
        prefix.  Both ``(kv, logits, identity, lease)`` and the legacy
        ``(base_kv, kv, logits, identity, lease)`` form are accepted.
        """
        if len(args) == 4:
            kv, logits, identity, lease = args
        elif len(args) == 5:
            _base_kv, kv, logits, identity, lease = args
        else:
            raise TypeError("extend_retained expects 4 or 5 arguments after key")
        entry = self._entries[key]
        if (
            lease.key != key
            or lease.generation != entry.generation
            or lease.identity != entry.identity
        ):
            raise ValueError("retained lease does not match the stored entry")
        old_length = lease.sequence_length
        new_length = _sequence_length(kv)
        if new_length < old_length:
            raise ValueError("KV sequence does not extend the retained prefix")
        old_prefix = entry.identity_value.get("prefix")
        new_prefix = identity.get("prefix")
        if (
            not isinstance(old_prefix, list)
            or not isinstance(new_prefix, list)
            or new_prefix[: len(old_prefix)] != old_prefix
            or len(new_prefix) != new_length
            or len(old_prefix) != old_length
        ):
            raise ValueError("identity prefix or position does not match the KV extension")
        old_metadata = {k: v for k, v in entry.identity_value.items() if k != "prefix"}
        new_metadata = {k: v for k, v in identity.items() if k != "prefix"}
        if new_metadata != old_metadata:
            raise ValueError("identity metadata changed across KV extension")
        if len(entry.kv) != len(kv):
            raise ValueError("KV layer count changed across extension")

        suffix_bytes = _tensor_bytes(logits)
        old_logits_bytes = _tensor_bytes(entry.logits)
        suffixes: list[tuple[Tensor, Tensor]] = []
        for stored_pair, new_pair in zip(entry.kv, kv):
            copied = []
            for stored_chunks, new_tensor in zip(stored_pair, new_pair):
                stored_tensor = stored_chunks[0]
                if (
                    stored_tensor.dtype != new_tensor.dtype
                    or stored_tensor.shape[:2] != new_tensor.shape[:2]
                    or stored_tensor.shape[3:] != new_tensor.shape[3:]
                ):
                    raise ValueError("KV tensor metadata changed across extension")
                suffix = new_tensor[:, :, old_length:new_length]
                copied.append(suffix)
                suffix_bytes += _tensor_bytes(suffix)
            suffixes.append((copied[0], copied[1]))
        new_payload_bytes = entry.payload_bytes - old_logits_bytes + suffix_bytes
        if self._used_bytes - entry.payload_bytes + new_payload_bytes > self._max_bytes:
            raise MemoryError("KV snapshot exceeds the payload budget")

        cuda_streams: dict[torch.device, Any] = {}
        host_suffixes = [
            (_host_copy(k_tensor, cuda_streams), _host_copy(v_tensor, cuda_streams))
            for k_tensor, v_tensor in suffixes
        ]
        host_logits = _host_copy(logits, cuda_streams)
        for stream in cuda_streams.values():
            stream.synchronize()
        chunked_kv = []
        for (k_chunks, v_chunks), (k_suffix, v_suffix) in zip(entry.kv, host_suffixes):
            chunked_kv.append((k_chunks + (k_suffix,), v_chunks + (v_suffix,)))
        identity_key = _identity_key(identity)
        generation = self._next_generation
        self._next_generation += 1
        self._entries[key] = _Entry(
            chunked_kv,
            host_logits,
            identity_key,
            json.loads(identity_key),
            new_payload_bytes,
            generation,
        )
        self._used_bytes = self._used_bytes - entry.payload_bytes + new_payload_bytes
        return RetainedLease(key, generation, identity_key, new_length)

    def delete(self, key: str) -> None:
        entry = self._entries.pop(key)
        self._used_bytes -= entry.payload_bytes
