"""A small, self contained Qwen3 inference engine.

Transformers is deliberately only used for configuration and tokenization by
the server.  Model execution below consists solely of torch operations over the
original safetensors weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


Tensor = torch.Tensor
KV = Tuple[Tensor, Tensor]


@dataclass
class RequestState:
    request_id: str
    prompt_ids: List[int]
    max_tokens: int
    ignore_eos: bool
    legacy: bool
    kv: Optional[List[KV]] = None
    logits: Optional[Tensor] = None
    generated: Optional[List[int]] = None
    generated_logprobs: Optional[List[float]] = None

    def __post_init__(self) -> None:
        self.generated = [] if self.generated is None else self.generated
        self.generated_logprobs = [] if self.generated_logprobs is None else self.generated_logprobs


class Qwen3Engine:
    """Exact BF16, single-device Qwen3 causal language model execution."""

    def __init__(self, model_dir: str, dtype: str = "bfloat16") -> None:
        if dtype != "bfloat16":
            raise ValueError("this runtime supports --dtype bfloat16 only")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.model_dir = os.path.abspath(model_dir)

        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path, "rb") as fh:
            config_bytes = fh.read()
        self.config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        self.config = json.loads(config_bytes)
        self.hidden_size = int(self.config["hidden_size"])
        self.num_layers = int(self.config["num_hidden_layers"])
        self.num_heads = int(self.config["num_attention_heads"])
        self.num_kv_heads = int(self.config["num_key_value_heads"])
        self.head_dim = int(self.config.get("head_dim", self.hidden_size // self.num_heads))
        self.eps = float(self.config["rms_norm_eps"])
        self.rope_theta = float(self.config["rope_theta"])
        self.vocab_size = int(self.config["vocab_size"])
        self.eos_token_id = int(self.config["eos_token_id"])
        self.max_context = min(4096, int(self.config["max_position_embeddings"]))
        self.weights, self.parameter_count = self._load_weights()
        inv = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, device=self.device, dtype=torch.float32) / self.head_dim)
        )
        self.inv_freq = inv

    def _load_weights(self) -> Tuple[Dict[str, Tensor], int]:
        index_path = os.path.join(self.model_dir, "model.safetensors.index.json")
        with open(index_path, "r", encoding="utf-8") as fh:
            index = json.load(fh)
        filenames = sorted(set(index["weight_map"].values()))
        weights: Dict[str, Tensor] = {}
        total = 0
        for filename in filenames:
            shard = load_file(os.path.join(self.model_dir, filename), device="cuda")
            for name, value in shard.items():
                if value.is_floating_point() and value.dtype != self.dtype:
                    value = value.to(self.dtype)
                weights[name] = value
                total += value.numel()
        missing = set(index["weight_map"]) - set(weights)
        if missing:
            raise RuntimeError(f"checkpoint is missing {len(missing)} indexed tensors")
        return weights, total

    def _w(self, name: str) -> Tensor:
        return self.weights[name]

    def _rms(self, x: Tensor, weight: Tensor) -> Tensor:
        # Qwen/HF performs the variance reduction in fp32 and casts back before
        # multiplying by the learned BF16 scale.
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * weight

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-b, a), dim=-1)

    def _rope(self, q: Tensor, k: Tensor, positions: Tensor) -> Tuple[Tensor, Tensor]:
        # HF Qwen3 duplicates the half-size frequency vector rather than
        # interleaving it; rotate_half therefore splits the head in halves.
        freq = torch.outer(positions.float(), self.inv_freq)
        emb = torch.cat((freq, freq), dim=-1)
        cos = emb.cos().to(self.dtype)[None, None, :, :]
        sin = emb.sin().to(self.dtype)[None, None, :, :]
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin

    def _layer(self, x: Tensor, layer: int, positions: Tensor, past: Optional[KV]) -> Tuple[Tensor, KV]:
        p = f"model.layers.{layer}"
        residual = x
        h = self._rms(x, self._w(f"{p}.input_layernorm.weight"))
        bsz, q_len, _ = h.shape
        q = F.linear(h, self._w(f"{p}.self_attn.q_proj.weight"))
        k = F.linear(h, self._w(f"{p}.self_attn.k_proj.weight"))
        v = F.linear(h, self._w(f"{p}.self_attn.v_proj.weight"))
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self._rms(q, self._w(f"{p}.self_attn.q_norm.weight"))
        k = self._rms(k, self._w(f"{p}.self_attn.k_norm.weight"))
        q, k = self._rope(q, k, positions)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        present = (k, v)
        repeats = self.num_heads // self.num_kv_heads
        kr = k.repeat_interleave(repeats, dim=1)
        vr = v.repeat_interleave(repeats, dim=1)
        # Spell out the transformers "eager" attention path.  In particular,
        # SDPA may select a different reduction kernel for one-token decode
        # than for prefill, which makes cached and recomputed logits diverge
        # noticeably in BF16.  QK and PV remain BF16, while softmax uses the
        # model's required fp32 accumulation.
        scores = torch.matmul(q, kr.transpose(2, 3)) * (self.head_dim ** -0.5)
        key_len = int(kr.shape[2])
        if q_len > 1:
            # Query position j corresponds to absolute position past_len+j.
            past_len = key_len - q_len
            q_pos = torch.arange(q_len, device=self.device) + past_len
            k_pos = torch.arange(key_len, device=self.device)
            causal = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
            scores.masked_fill_(causal[None, None], torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = torch.matmul(probs, vr)
        attn = attn.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        x = residual + F.linear(attn, self._w(f"{p}.self_attn.o_proj.weight"))

        residual = x
        h = self._rms(x, self._w(f"{p}.post_attention_layernorm.weight"))
        gate = F.linear(h, self._w(f"{p}.mlp.gate_proj.weight"))
        up = F.linear(h, self._w(f"{p}.mlp.up_proj.weight"))
        x = residual + F.linear(F.silu(gate) * up, self._w(f"{p}.mlp.down_proj.weight"))
        return x, present

    @torch.inference_mode()
    def _forward_tensor(self, ids: Tensor, past: Optional[List[KV]] = None) -> Tuple[Tensor, List[KV]]:
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("input_ids must not be empty")
        past_len = 0 if past is None else int(past[0][0].shape[2])
        if past_len + int(ids.shape[1]) > self.max_context:
            raise ValueError(f"context exceeds {self.max_context} tokens")
        x = F.embedding(ids, self._w("model.embed_tokens.weight"))
        positions = torch.arange(past_len, past_len + int(ids.shape[1]), device=self.device)
        new_kv: List[KV] = []
        for i in range(self.num_layers):
            x, kv = self._layer(x, i, positions, None if past is None else past[i])
            new_kv.append(kv)
        x = self._rms(x, self._w("model.norm.weight"))
        logits = F.linear(x[:, -1, :], self._w("lm_head.weight")).float()
        return logits, new_kv

    @torch.inference_mode()
    def forward(self, input_ids: Sequence[int], past: Optional[List[KV]] = None) -> Tuple[Tensor, List[KV]]:
        if not input_ids:
            raise ValueError("input_ids must not be empty")
        ids = torch.tensor([list(input_ids)], dtype=torch.long, device=self.device)
        logits, kv = self._forward_tensor(ids, past)
        return logits[0], kv

    @torch.inference_mode()
    def _forward_states(self, inputs: List[List[int]], states: List[RequestState]) -> None:
        """Run one equal-shaped batch and scatter its independent KV states."""
        if not inputs:
            return
        width = len(inputs[0])
        if any(len(row) != width for row in inputs):
            raise ValueError("batched inputs must have equal lengths")
        have_past = states[0].kv is not None
        if any((state.kv is not None) != have_past for state in states):
            raise ValueError("mixed cached and uncached batch")
        past: Optional[List[KV]] = None
        if have_past:
            cache_len = int(states[0].kv[0][0].shape[2])
            if any(int(state.kv[0][0].shape[2]) != cache_len for state in states):
                raise ValueError("batched caches must have equal lengths")
            past = []
            for layer in range(self.num_layers):
                past.append((
                    torch.cat([state.kv[layer][0] for state in states], dim=0),
                    torch.cat([state.kv[layer][1] for state in states], dim=0),
                ))
        ids = torch.tensor(inputs, dtype=torch.long, device=self.device)
        logits, batched_kv = self._forward_tensor(ids, past)
        for batch_index, state in enumerate(states):
            state.logits = logits[batch_index]
            state.kv = [
                (k[batch_index : batch_index + 1], v[batch_index : batch_index + 1])
                for k, v in batched_kv
            ]

    def prefill(self, state: RequestState) -> None:
        if len(state.prompt_ids) + state.max_tokens > self.max_context:
            raise ValueError(f"prompt plus output exceeds {self.max_context} tokens")
        state.logits, state.kv = self.forward(state.prompt_ids)

    def next_token(self, state: RequestState) -> int:
        if state.logits is None or state.kv is None:
            self.prefill(state)
        token = int(torch.argmax(state.logits).item())
        logprob = float(F.log_softmax(state.logits.float(), dim=-1)[token].item())
        if not math.isfinite(logprob):
            raise RuntimeError("selected token logprob is not finite")
        state.generated.append(token)
        state.generated_logprobs.append(logprob)
        # Do not spend a forward pass after the final requested token.
        if len(state.generated) < state.max_tokens and (state.ignore_eos or token != self.eos_token_id):
            state.logits, state.kv = self.forward([token], state.kv)
        return token

    def next_tokens(self, states: List[RequestState]) -> List[int]:
        """Greedily advance independent requests, batching compatible work."""
        if not states:
            return []
        prefill_groups: Dict[int, List[RequestState]] = {}
        for state in states:
            if state.logits is None or state.kv is None:
                if len(state.prompt_ids) + state.max_tokens > self.max_context:
                    raise ValueError(f"prompt plus output exceeds {self.max_context} tokens")
                prefill_groups.setdefault(len(state.prompt_ids), []).append(state)
        for group in prefill_groups.values():
            self._forward_states([state.prompt_ids for state in group], group)

        tokens: List[int] = []
        continuing: Dict[int, List[RequestState]] = {}
        for state in states:
            token = int(torch.argmax(state.logits).item())
            logprob = float(F.log_softmax(state.logits.float(), dim=-1)[token].item())
            if not math.isfinite(logprob):
                raise RuntimeError("selected token logprob is not finite")
            state.generated.append(token)
            state.generated_logprobs.append(logprob)
            tokens.append(token)
            if len(state.generated) < state.max_tokens and (state.ignore_eos or token != self.eos_token_id):
                cache_len = int(state.kv[0][0].shape[2])
                continuing.setdefault(cache_len, []).append(state)
        for group in continuing.values():
            self._forward_states([[state.generated[-1]] for state in group], group)
        return tokens

    @torch.inference_mode()
    def trace(self, input_ids: Sequence[int], continuation_ids: Sequence[int], mode: str) -> dict:
        if mode not in {"cached", "recompute"}:
            raise ValueError("mode must be 'cached' or 'recompute'")
        if len(input_ids) + len(continuation_ids) > self.max_context:
            raise ValueError(f"trace exceeds {self.max_context} tokens")
        observations = []
        cache_lengths = []
        if mode == "cached":
            logits, kv = self.forward(input_ids)
            for i, wanted in enumerate(continuation_ids):
                logprob = F.log_softmax(logits.float(), dim=-1)[int(wanted)].item()
                observations.append({"index": i, "top1": int(logits.argmax().item()), "logprob": float(logprob)})
                cache_lengths.append(len(input_ids) + i)
                logits, kv = self.forward([int(wanted)], kv)
        else:
            prefix = list(input_ids)
            for i, wanted in enumerate(continuation_ids):
                logits, _ = self.forward(prefix)
                logprob = F.log_softmax(logits.float(), dim=-1)[int(wanted)].item()
                observations.append({"index": i, "top1": int(logits.argmax().item()), "logprob": float(logprob)})
                cache_lengths.append(len(prefix))
                prefix.append(int(wanted))
        return {"observations": observations, "cache_lengths": cache_lengths}
