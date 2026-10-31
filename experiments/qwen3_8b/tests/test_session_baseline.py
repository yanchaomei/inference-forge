"""Behavioral session boundary test using an independent small causal oracle."""
from types import SimpleNamespace
import copy
import torch

from experiments.qwen3_8b.baseline_consumer import SessionRuntime


class CausalToy:
    device = "cpu"
    config_sha256 = "toy"

    def forward(self, ids, past=None):
        history = ([] if past is None else past[0][0].flatten().tolist())+list(ids)
        scores = torch.zeros(8)
        scores[int(sum(history)+len(history)) % 8] = 10
        values = torch.tensor(history).reshape(1, 1, -1, 1)
        return scores, [(values, values.clone())]

    def next_token(self, state):
        token = int(state.logits.argmax())
        state.generated.append(token)
        state.generated_logprobs.append(float(state.logits.log_softmax(-1)[token]))
        if len(state.generated) < state.max_tokens:
            state.logits, state.kv = self.forward([token], state.kv)
        return token


class HostStore:
    def __init__(self):
        self.values = {}

    def put(self, key, kv, logits, identity):
        self.values[key] = copy.deepcopy((kv, logits, identity))

    def get(self, key, identity, device):
        kv, logits, expected = self.values[key]
        assert identity == expected
        return copy.deepcopy((kv, logits))

    def delete(self, key):
        del self.values[key]


def test_pending_last_output_survives_offload_and_sessions_stay_independent():
    store = HostStore()
    runtime = SessionRuntime(CausalToy(), store, gpu_budget_bytes=0)
    assert runtime.turn("a", [1, 2], 2)["token_ids"] == [5, 3]
    assert runtime.turn("b", [2], 1)["token_ids"] == [3]
    # History must be [1,2,5,3,4], not [1,2,5,4] or [1,2,5,3,3,4].
    observed = []
    result = runtime.turn("a", [4], 2, lambda sid, i, logits: observed.append((sid, i, int(logits.argmax()))))
    assert result["token_ids"] == [4, 1]
    assert observed == [("a", 0, 4), ("a", 1, 1)]
    runtime.close("a")
    assert "a" not in store.values and "b" in store.values
    runtime.close("b")
    assert not store.values


if __name__ == "__main__":
    test_pending_last_output_survives_offload_and_sessions_stay_independent()
    print("session pending-token and isolation contract: PASS")
