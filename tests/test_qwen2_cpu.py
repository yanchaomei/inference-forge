"""Tiny locally-created checkpoint: installed backend math, no downloads/GPU."""
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from forge.backends.qwen2.engine import QwenEngine


def test_prefill_and_variable_length_batch_match_independent_model(tmp_path):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    config = Qwen2Config(vocab_size=37, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=64, tie_word_embeddings=True,
                        attention_dropout=0.0)
    config._attn_implementation = 'eager'
    reference = Qwen2ForCausalLM(config).float().eval()
    reference.save_pretrained(tmp_path, safe_serialization=True)
    engine = QwenEngine(tmp_path, device='cpu')
    histories = [[1,2], [3,4,5,6], [7,8,9]]
    states = [engine.prefill(ids) for ids in histories]
    originals = [(state.length, [(k.clone(),v.clone()) for k,v in state.cache]) for state in states]
    initial_states = states[:]
    with torch.inference_mode():
        for ids, state in zip(histories, states):
            expected = reference(torch.tensor([ids])).logits[:, -1, :]
            torch.testing.assert_close(state.last_logits, expected, atol=2e-5, rtol=2e-5)
        for step in range(4):
            tokens = [10+step, 15+step, 20+step]
            states = engine.batch_step(tokens, states)
            for i, token in enumerate(tokens):
                histories[i].append(token)
                expected = reference(torch.tensor([histories[i]])).logits[:, -1, :]
                torch.testing.assert_close(states[i].last_logits, expected, atol=2e-5, rtol=2e-5)
                assert torch.equal(states[i].last_logits.argmax(-1), expected.argmax(-1))
    for state, (length, cache) in zip(initial_states, originals):
        assert state.length == length
        for actual, expected in zip(state.cache, cache):
            for a, e in zip(actual, expected):
                assert torch.equal(a,e)
