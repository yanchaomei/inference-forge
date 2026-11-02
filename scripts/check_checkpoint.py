"""Functional acceptance with real Qwen2.5 weights on a test host.

CUDA checks launch the installed CLI; CPU checks use the same service factory.
This is functional acceptance, not a performance benchmark.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.request import urlopen



def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


@contextmanager
def running_service(model, tokenizer, device, out):
    if device == 'cpu':
        from forge.backends.qwen2.engine import QwenEngine
        from forge.serving import create_server
        server = create_server(QwenEngine(model, device='cpu'), tokenizer, port=0,
                               served_model_name='checkpoint-smoke')
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f'http://127.0.0.1:{server.server_address[1]}'
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()
            server.runtime.close()
        return
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port = sock.getsockname()[1]
    command = [str(Path(sys.executable).with_name('inference-forge')), 'serve', str(model),
               '--port', str(port), '--served-model-name', 'checkpoint-smoke',
               '--event-log', str(out/'server-events.jsonl')]
    base = f'http://127.0.0.1:{port}'
    with (out/'server.log').open('w') as log:
        worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        try:
            for _ in range(120):
                if worker.poll() is not None:
                    raise RuntimeError('installed CLI exited before readiness; see server.log')
                try:
                    with urlopen(base+'/health', timeout=1) as response:
                        health = json.load(response)
                    if health['ready'] and health['id'] == 'checkpoint-smoke':
                        break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('installed CLI readiness timeout')
            yield base
        finally:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGINT)
                try:
                    worker.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    p.add_argument('--gpu-uuid', help='Required for CUDA; selects an idle device before model loading')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    model = Path(args.model)
    if args.device == 'cuda':
        if not args.gpu_uuid:
            p.error('--device cuda requires --gpu-uuid')
        rows = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader'], text=True)
        if any(r.split(',')[0].strip() == args.gpu_uuid for r in rows.splitlines()):
            raise RuntimeError('selected GPU has an existing compute process')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
        (out/'preflight.json').write_text(json.dumps({'gpu_uuid':args.gpu_uuid,'processes':rows})+'\n')
    from openai import OpenAI
    import torch
    from transformers import AutoTokenizer
    from forge.evaluate import generate_reference, check_numerics, evaluate_trace, write_json
    import forge.serving
    import forge.protocol
    import forge.backends.qwen2.engine
    assert Path(forge.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), 'use the installed package outside the source checkout'
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
    task = {'model_path':str(model),'dtype':'float32','device':args.device,'task_id':'installed-checkpoint','seed':4271,
            'prompts':prompts,'max_new_tokens':8,'mean_logprob_error_max':.001,'max_logprob_error_max':.01}
    golden = generate_reference(task, out/'reference.json')
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    observations = []
    with running_service(model, tokenizer, args.device, out) as base:
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
    report = {'status':'pass','scope':f'real Qwen2.5-0.5B {args.device} functional acceptance; not a performance comparison',
              'model_revision':'7ae557604adf67be50417f59c2c2f167def9a775','model_sha256':expected_sha,
              'device':args.device,'dtype':'float32','reference_cases':len(golden['cases']),
              'installed_cli_launched':args.device=='cuda',
              'gpu_name':torch.cuda.get_device_name(0) if args.device=='cuda' else None,
              'checker_sha256':sha(__file__),
              'numeric_modes_per_case':2,'native_concurrent_requests':len(events),
              'sdk_cases':observations,'versions':{n:importlib.metadata.version(n) for n in
                  ('torch','transformers','safetensors','openai','agentic-inference-forge')},
              'source_sha256':{m.__name__:sha(m.__file__) for m in
                  (forge.serving,forge.protocol,forge.backends.qwen2.engine)}}
    write_json(out/'summary.json', report)
    print(json.dumps({'status':'pass','device':args.device,'reference_cases':4,'sdk_cases':len(observations)},indent=2))


if __name__ == '__main__':
    main()
