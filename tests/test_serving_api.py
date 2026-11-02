"""Real HTTP and OpenAI SDK contracts with a deterministic CPU test engine.

This validates transport/scheduling, not real-model numerical quality or speed.
"""
from dataclasses import dataclass
from pathlib import Path
import threading
import urllib.request
import json

import pytest
import torch
from openai import OpenAI, BadRequestError, NotFoundError

from forge.serving import create_server


@dataclass
class State:
    last_logits: torch.Tensor
    length: int


class Engine:
    model_path = Path('/test/tiny')
    config_hash = 'cpu-test-model'
    parameter_count = 0
    vocab_size = 4

    def state(self, length):
        logits = torch.zeros(1, 4)
        logits[0, 1 + length % 3] = 10
        return State(logits, length)

    def prefill(self, ids):
        return self.state(len(ids))

    def step(self, token, state):
        return self.state(state.length + 1)

    def batch_step(self, tokens, states):
        return [self.step(token, state) for token, state in zip(tokens, states)]


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [0] * len(text)

    def decode(self, ids, **kwargs):
        return ''.join(['_', '你', '好', '！'][token] for token in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        self.messages = messages
        return self.encode('|'.join(m['role'] + ':' + m['content'] for m in messages) + '|assistant:')


@pytest.fixture
def service():
    tokenizer = Tokenizer()
    server = create_server(Engine(), tokenizer, port=0, served_model_name='tiny')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    with OpenAI(base_url=base + '/v1', api_key='local-test', max_retries=0) as client:
        try:
            yield client, base, tokenizer
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()
            server.runtime.close()


def test_sdk_text_completions_and_existing_token_fields(service):
    client, base, _ = service
    response = client.completions.create(model='tiny', prompt='abc', max_tokens=3, temperature=0)
    assert response.object == 'text_completion'
    assert response.choices[0].text == '你好！'
    assert response.choices[0].finish_reason == 'length'
    assert response.usage.completion_tokens == 3
    request = urllib.request.Request(base + '/v1/completions',
        json.dumps({'prompt':'abc', 'max_tokens':3}).encode(), {'Content-Type':'application/json'})
    with urllib.request.urlopen(request) as r:
        legacy = json.load(r)
    assert legacy['token_ids'] == [1,2,3] and legacy['text'] == '你好！'
    assert legacy['request_id']


def test_sdk_chat_uses_template_and_stream_does_not_repeat_final_text(service):
    client, _, tokenizer = service
    messages = [{'role':'system','content':'简洁回答'}, {'role':'user','content':'你好'}]
    normal = client.chat.completions.create(model='tiny', messages=messages, max_tokens=4, temperature=0)
    assert tokenizer.messages == messages
    assert normal.object == 'chat.completion' and normal.choices[0].message.role == 'assistant'
    chunks = list(client.chat.completions.create(model='tiny', messages=messages, max_tokens=4,
                                               temperature=0, stream=True))
    assert ''.join(c.choices[0].delta.content or '' for c in chunks) == normal.choices[0].message.content
    assert chunks[-1].choices[0].finish_reason == 'length'
    assert all(c.object == 'chat.completion.chunk' and c.model == 'tiny' for c in chunks)


def test_streamed_completions_include_usage_on_request(service):
    client, _, _ = service
    chunks = list(client.completions.create(model='tiny', prompt='abc', max_tokens=3, stream=True,
                                           stream_options={'include_usage':True}))
    assert ''.join(c.choices[0].text for c in chunks if c.choices) == '你好！'
    assert chunks[-1].choices == [] and chunks[-1].usage.completion_tokens == 3
    assert chunks[-2].choices[0].finish_reason == 'length'


@pytest.mark.parametrize('kwargs', [{'temperature':0.7}, {'n':2}, {'stop':['!']}, {'max_tokens':65}])
def test_unsupported_generation_controls_return_sdk_errors(service, kwargs):
    client, _, _ = service
    with pytest.raises(BadRequestError) as error:
        client.completions.create(model='tiny', prompt='abc', **kwargs)
    assert error.value.body['message']


def test_unknown_model_and_unsupported_messages_are_clear_errors(service):
    client, _, _ = service
    assert client.models.list().data[0].id == 'tiny'
    with pytest.raises(NotFoundError):
        client.completions.create(model='missing', prompt='abc')
    with pytest.raises(BadRequestError):
        client.chat.completions.create(model='tiny', messages=[{'role':'user','content':[{'type':'text','text':'hi'}]}])
