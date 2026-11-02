"""Functional acceptance with real Qwen2.5 weights on a CPU test host.

Checks the installed engine/server and real SDK. This is not a GPU benchmark.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import threading

from openai import OpenAI
import torch
from transformers import AutoTokenizer

from forge.backends.qwen2.engine import QwenEngine
from forge.evaluate import generate_reference, check_numerics, evaluate_trace, write_json
from forge.serving import create_server
import forge.serving
import forge.protocol
import forge.backends.qwen2.engine


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    model = Path(args.model)
    expected_sha = 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
    assert sha(model/'model.safetensors') == expected_sha, 'checkpoint identity mismatch'
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    conversations = [
        [{'role':'user','content':'Explain KV caching briefly.'}],
        [{'role':'system','content':'简洁回答。'}, {'role':'user','content':'中国科学技术大学位于哪里？'}],
    ]
    prompts = ['The capital of France is', '中国科学技术大学位于'] + [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in conversations]
    task = {'model_path':str(model),'dtype':'float32','device':'cpu','task_id':'installed-checkpoint-cpu',
            'prompts':prompts,'max_new_tokens':8,'mean_logprob_error_max':.001,'max_logprob_error_max':.01}
    golden = generate_reference(task, out/'reference.json')
    gc.collect()
    engine = QwenEngine(model, device='cpu')
    server = create_server(engine, tokenizer, port=0, served_model_name='checkpoint-smoke')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    observations = []
    try:
        numeric = check_numerics(base, golden, task)
        write_json(out/'numerics.json', numeric)
        assert numeric['status'] == 'pass', 'numerical check failed'
        events = [{k:c[k] for k in ('request_id','input_ids','max_tokens')} for c in golden['cases']]
        native = evaluate_trace(base, events, golden)
        write_json(out/'native-streaming.json', native)
        assert native['status'] == 'pass', 'native streaming check failed'
        with OpenAI(base_url=base+'/v1', api_key='unused', max_retries=0, timeout=120) as client:
            for i, prompt in enumerate(prompts[:2]):
                reply = client.completions.create(model='checkpoint-smoke', prompt=prompt,
                                                  max_tokens=8, temperature=0)
                expected = golden['cases'][i]
                assert reply.choices[0].text == expected['text']
                assert reply.usage.completion_tokens == 8
                assert reply.model_extra['token_ids'] == expected['token_ids']
                chunks = list(client.completions.create(model='checkpoint-smoke', prompt=prompt,
                    max_tokens=8, temperature=0, stream=True, stream_options={'include_usage':True}))
                assert ''.join(c.choices[0].text for c in chunks if c.choices) == expected['text']
                assert chunks[-1].usage.completion_tokens == 8
                observations.append({'route':'completions','case':i,'token_ids':reply.model_extra['token_ids'],
                                     'text':reply.choices[0].text,'stream_matches':True})
            for i, messages in enumerate(conversations):
                reply = client.chat.completions.create(model='checkpoint-smoke', messages=messages,
                                                       max_tokens=8, temperature=0)
                expected = golden['cases'][i+2]
                assert reply.choices[0].message.content == expected['text']
                assert reply.model_extra['token_ids'] == expected['token_ids']
                chunks = list(client.chat.completions.create(model='checkpoint-smoke', messages=messages,
                    max_tokens=8, temperature=0, stream=True, stream_options={'include_usage':True}))
                assert ''.join(c.choices[0].delta.content or '' for c in chunks if c.choices) == expected['text']
                assert chunks[-1].usage.completion_tokens == 8
                observations.append({'route':'chat.completions','case':i,'token_ids':reply.model_extra['token_ids'],
                                     'text':reply.choices[0].message.content,'stream_matches':True})
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        server.runtime.close()
    report = {'status':'pass','scope':'real Qwen2.5-0.5B CPU functional acceptance; no GPU or performance claim',
              'model_revision':'7ae557604adf67be50417f59c2c2f167def9a775','model_sha256':expected_sha,
              'device':'cpu','dtype':'float32','reference_cases':len(golden['cases']),
              'numeric_modes_per_case':2,'native_concurrent_requests':len(events),
              'sdk_cases':observations,'versions':{n:importlib.metadata.version(n) for n in
                  ('torch','transformers','safetensors','openai','agentic-inference-forge')},
              'source_sha256':{m.__name__:sha(m.__file__) for m in
                  (forge.serving,forge.protocol,forge.backends.qwen2.engine)}}
    write_json(out/'summary.json', report)
    print(json.dumps({'status':'pass','device':'cpu','reference_cases':4,'sdk_cases':len(observations)},indent=2))


if __name__ == '__main__':
    main()
