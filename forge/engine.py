"""A small, explicit Qwen2 decoder runtime built from PyTorch primitives.

The engine owns weights only.  Attention state is returned to each caller and
never stored on the engine, so independently interleaved requests stay isolated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


@dataclass(frozen=True)
class EngineState:
    last_logits: torch.Tensor
    cache: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    length: int


class QwenEngine:
    def __init__(self, model_path, dtype="float32", device="cuda"):
        self.model_path = Path(model_path).resolve()
        config_bytes = (self.model_path / "config.json").read_bytes()
        self.config = json.loads(config_bytes)
        self.config_hash = hashlib.sha256(config_bytes).hexdigest()
        if self.config.get("model_type") != "qwen2":
            raise ValueError("only Qwen2 checkpoints are supported")
        if self.config.get("use_sliding_window", False):
            raise ValueError("sliding-window attention is not supported")
        if self.config.get("hidden_act", "silu") != "silu":
            raise ValueError("only SiLU gated MLPs are supported")
        if self.config.get("rope_scaling"):
            raise ValueError("scaled RoPE is not supported")
        dtypes = {"float32": torch.float32, "float16": torch.float16,
                  "bfloat16": torch.bfloat16}
        if dtype not in dtypes:
            raise ValueError(f"unsupported dtype: {dtype}")
        self.dtype = dtypes[dtype]
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.hidden_size = self.config["hidden_size"]
        self.heads = self.config["num_attention_heads"]
        self.kv_heads = self.config["num_key_value_heads"]
        self.head_dim = self.hidden_size // self.heads
        self.layers = self.config["num_hidden_layers"]
        self.vocab_size = self.config["vocab_size"]
        self.max_context = self.config["max_position_embeddings"]
        self.eps = self.config["rms_norm_eps"]
        if self.hidden_size % self.heads or self.heads % self.kv_heads:
            raise ValueError("incompatible grouped-query head dimensions")
        raw_weights = load_file(str(self.model_path / "model.safetensors"))
        self.weights = {
            name: tensor.to(device=self.device, dtype=self.dtype)
            for name, tensor in raw_weights.items()
        }
        self.parameter_count = sum(t.numel() for t in self.weights.values())
        self.embedding = self.weights["model.embed_tokens.weight"]
        self.lm_head = (
            self.embedding if self.config.get("tie_word_embeddings", False)
            else self.weights["lm_head.weight"]
        )
        self.inv_freq = 1.0 / (
            self.config.get("rope_theta", 10000.0)
            ** (torch.arange(0, self.head_dim, 2, device=self.device,
                             dtype=torch.float32) / self.head_dim)
        )

    def _norm(self, value, weight):
        fp32 = value.float()
        normalized = fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + self.eps)
        return weight * normalized.to(value.dtype)

    @staticmethod
    def _rotate_half(value):
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    @torch.inference_mode()
    def forward(self, input_ids: list[int], cache=None):
        if not input_ids or any(type(t) is not int or not 0 <= t < self.vocab_size
                                for t in input_ids):
            raise ValueError("input_ids must be a nonempty list of valid token IDs")
        length = len(input_ids)
        past_length = 0
        if cache is not None:
            if len(cache) != self.layers:
                raise ValueError("cache must contain one K/V pair per layer")
            past_length = cache[0][0].shape[-2]
            expected = (1, self.kv_heads, past_length, self.head_dim)
            if any(k.shape != expected or v.shape != expected or
                   k.device != self.device or v.device != self.device or
                   k.dtype != self.dtype or v.dtype != self.dtype for k, v in cache):
                raise ValueError("cache dimensions, device or dtype do not match the engine")
        total_length = past_length + length
        if total_length > self.max_context:
            raise ValueError("input exceeds the checkpoint context limit")
        ids = torch.tensor([input_ids], device=self.device, dtype=torch.long)
        hidden = F.embedding(ids, self.embedding)
        positions = torch.arange(past_length, total_length, device=self.device,
                                 dtype=torch.float32)
        frequencies = positions[:, None] * self.inv_freq[None, :]
        rotary = torch.cat((frequencies, frequencies), dim=-1)
        cos = rotary.cos().to(self.dtype)[None, None, :, :]
        sin = rotary.sin().to(self.dtype)[None, None, :, :]
        key_positions = torch.arange(total_length, device=self.device)
        causal = key_positions[None, :] > positions[:, None]
        next_cache = []
        for layer in range(self.layers):
            base = f"model.layers.{layer}."
            w = self.weights
            residual = hidden
            normed = self._norm(hidden, w[base + "input_layernorm.weight"])
            q = F.linear(normed, w[base + "self_attn.q_proj.weight"],
                         w[base + "self_attn.q_proj.bias"])
            k = F.linear(normed, w[base + "self_attn.k_proj.weight"],
                         w[base + "self_attn.k_proj.bias"])
            v = F.linear(normed, w[base + "self_attn.v_proj.weight"],
                         w[base + "self_attn.v_proj.bias"])
            q = q.view(1, length, self.heads, self.head_dim).transpose(1, 2)
            k = k.view(1, length, self.kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(1, length, self.kv_heads, self.head_dim).transpose(1, 2)
            q = q * cos + self._rotate_half(q) * sin
            k = k * cos + self._rotate_half(k) * sin
            if cache is not None:
                k = torch.cat((cache[layer][0], k), dim=-2)
                v = torch.cat((cache[layer][1], v), dim=-2)
            next_cache.append((k, v))
            groups = self.heads // self.kv_heads
            expanded_k = k.repeat_interleave(groups, dim=1)
            expanded_v = v.repeat_interleave(groups, dim=1)
            scores = (q @ expanded_k.transpose(-1, -2)) * (self.head_dim ** -0.5)
            scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
            attention = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attended = (attention @ expanded_v).transpose(1, 2).reshape(1, length, -1)
            hidden = residual + F.linear(attended, w[base + "self_attn.o_proj.weight"])
            residual = hidden
            normed = self._norm(hidden, w[base + "post_attention_layernorm.weight"])
            gate = F.silu(F.linear(normed, w[base + "mlp.gate_proj.weight"]))
            up = F.linear(normed, w[base + "mlp.up_proj.weight"])
            hidden = residual + F.linear(gate * up, w[base + "mlp.down_proj.weight"])
        hidden = self._norm(hidden, self.weights["model.norm.weight"])
        return F.linear(hidden, self.lm_head), tuple(next_cache)

    def prefill(self, input_ids: list[int]) -> EngineState:
        logits, cache = self.forward(input_ids)
        return EngineState(logits[:, -1, :], cache, len(input_ids))

    def step(self, token_id: int, state: EngineState) -> EngineState:
        logits, cache = self.forward([token_id], state.cache)
        return EngineState(logits[:, -1, :], cache, state.length + 1)
