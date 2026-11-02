"""Serving and construction commands; help works without GPU dependencies."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

MODELS = [{'id':'Qwen/Qwen2.5-0.5B-Instruct', 'serving':True, 'dtype':'float32',
           'context_tokens':512, 'max_output_tokens':64,
           'revision':'7ae557604adf67be50417f59c2c2f167def9a775'}]


def port_number(value):
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError('port must be between 1 and 65535')
    return port


def environment_report():
    packages = {}
    for name in ('agentic-inference-forge', 'torch', 'transformers', 'safetensors'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {'python':platform.python_version(), 'platform':platform.system(),
              'packages':packages, 'cuda_available':None}
    if packages['torch']:
        try:
            import torch
            report['cuda_available'] = torch.cuda.is_available()
        except (ImportError, OSError) as exc:
            report['torch_error'] = str(exc)
    return report


def parser():
    p = argparse.ArgumentParser(prog='inference-forge', description='Build and run specialized inference engines.')
    try:
        version = importlib.metadata.version('agentic-inference-forge')
    except importlib.metadata.PackageNotFoundError:
        version = 'development'
    p.add_argument('--version', action='version', version=version)
    sub = p.add_subparsers(dest='command', required=True)
    serve = sub.add_parser('serve', help='Run the packaged Qwen2 inference service')
    serve.add_argument('model', help='Local Qwen2.5-0.5B-Instruct checkpoint directory')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=port_number, default=8000)
    serve.add_argument('--served-model-name', help='Model ID exposed by the HTTP API')
    serve.add_argument('--event-log', help='Write request and scheduling events as JSONL')
    sub.add_parser('construct', help='Generate and evaluate candidates (requires source checkout)')
    doctor = sub.add_parser('doctor', help='Inspect Python, inference dependencies and CUDA')
    doctor.add_argument('--json', action='store_true')
    doctor.add_argument('--require-gpu', action='store_true')
    models = sub.add_parser('models', help='List the directly supported serving models')
    models.add_argument('--json', action='store_true')
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Existing campaign commands begin with --task/--output/etc. Keep their
    # parser and meaning; the new subcommand provides an explicit entry point.
    if argv and (argv[0] == 'construct' or (argv[0].startswith('--') and argv[0] not in ('--help','--version')):
        if argv[0] == 'construct':
            argv = argv[1:]
        from .campaign import main as construct
        return construct(argv)
    p = parser()
    args = p.parse_args(argv)
    if args.command == 'models':
        if args.json:
            print(json.dumps(MODELS, indent=2))
        else:
            for m in MODELS:
                print(f"{m['id']}  {m['dtype']}  context={m['context_tokens']}  max_output={m['max_output_tokens']}")
            print('Qwen3-8B state/session experiments: see experiments/qwen3_8b/README.md.')
        return 0
    if args.command == 'doctor':
        report = environment_report()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"Python {report['python']} on {report['platform']}")
            for name, version in report['packages'].items():
                print(f"{name}: {version or 'not installed'}")
            print(f"CUDA available: {report['cuda_available']}")
        return int(args.require_gpu and report['cuda_available'] is not True)
    model = Path(args.model).expanduser()
    if not (model/'config.json').is_file():
        p.error('model must be a local checkpoint directory containing config.json; see docs/QUICKSTART.md')
    try:
        from .serving import main as serve
    except ImportError as exc:
        p.error(f'inference dependencies unavailable: {exc}. Install the [inference] extra.')
    options = ['--model', str(model), '--host', args.host, '--port', str(args.port)]
    if args.served_model_name:
        options += ['--served-model-name', args.served_model_name]
    if args.event_log:
        options += ['--event-log', args.event_log]
    return serve(options)
