"""Chat with a running InferenceForge server using the OpenAI SDK."""
import argparse

from openai import OpenAI


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--model', help='Defaults to the first model returned by the service')
    p.add_argument('--prompt', default='Explain KV caching in one sentence.')
    p.add_argument('--max-tokens', type=int, default=32)
    p.add_argument('--stream', action='store_true')
    args = p.parse_args()
    with OpenAI(base_url=args.base_url, api_key='unused') as client:
        model = args.model or client.models.list().data[0].id
        request = dict(model=model, messages=[{'role':'user','content':args.prompt}],
                       max_tokens=args.max_tokens, temperature=0)
        if args.stream:
            for chunk in client.chat.completions.create(**request, stream=True,
                    stream_options={'include_usage':True}):
                if chunk.choices:
                    print(chunk.choices[0].delta.content or '', end='', flush=True)
                if chunk.usage:
                    print(f'\nTokens: {chunk.usage.completion_tokens}')
        else:
            result = client.chat.completions.create(**request)
            print(result.choices[0].message.content)


if __name__ == '__main__':
    main()
