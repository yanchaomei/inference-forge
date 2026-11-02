"""Exercise the installed/public CLI without requiring model dependencies."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command(*args):
    return subprocess.run([sys.executable, '-m', 'forge', *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=20)


def test_help_exposes_serving_and_construction():
    result = command('--help')
    assert result.returncode == 0, result.stderr
    for name in ('serve', 'construct', 'doctor', 'models'):
        assert name in result.stdout
    served = command('serve', '--help')
    assert served.returncode == 0, served.stderr
    assert '--port' in served.stdout and '--served-model-name' in served.stdout


def test_supported_models_and_doctor_are_machine_readable():
    result = command('models', '--json')
    assert result.returncode == 0, result.stderr
    models = json.loads(result.stdout)
    assert any(m['id'] == 'Qwen/Qwen2.5-0.5B-Instruct' for m in models)
    assert all(m['serving'] is True for m in models)
    result = command('doctor', '--json')
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['python'] and 'torch' in report['packages']
    assert 'cuda_available' in report


def test_constructor_and_legacy_help_remain_available():
    result = command('construct', '--help')
    assert result.returncode == 0, result.stderr
    assert '--task' in result.stdout and '--remote-root' in result.stdout


def test_bad_port_is_rejected_before_loading_a_model():
    result = command('serve', '/missing/model', '--port', '70000')
    assert result.returncode == 2
    assert 'port' in result.stderr.lower()
    assert 'Traceback' not in result.stderr
