# Development

[Contributing](../CONTRIBUTING.md) · [API](API.md) · [Roadmap](ROADMAP.md)

Use a source checkout to develop the runtime or construct new candidates. Serving an installed engine only needs the inference dependencies; construction also needs the repository's tasks and source files.

## Package names

| Name | Meaning |
| --- | --- |
| InferenceForge | Project |
| `agentic-inference-forge` | Python distribution name; currently installed from source |
| `forge` | Importable Python package |
| `inference-forge` | CLI for serving, environment checks, model listing, and construction |

Use `inference-forge --version` for the installed version and `--help` for commands. `models --json` lists serving targets; `doctor --json` reports Python, dependencies, and CUDA. `doctor --json --require-gpu` returns a nonzero status when CUDA is unavailable.

## Code layout

| Path | Responsibility |
| --- | --- |
| `forge/cli.py` | `serve`, `models`, `doctor`, and construction dispatch |
| `forge/protocol.py` | Request validation and SDK response formatting |
| `forge/serving.py` | Packaged HTTP/SSE service and batch scheduling |
| `forge/backends/qwen2/engine.py` | Generated Qwen2 numerical backend used by serving |
| `forge/campaign.py` | Candidate construction, remote execution, and feedback |
| `forge/engine.py`, `forge/server.py` | Original construction seed |
| `forge/evaluate.py`, `forge/pilot.py` | Independent checks and task evaluation |
| `forge/replay.py` | Reduction of external failing request traces |
| `examples/generated/batched/` | Compatibility imports and launch path for the generated engine |
| `tasks/`, `targets/` | Model, hardware, workload, and evaluation configuration |
| `engines/`, `experiments/qwen3_8b/` | Separate 8B storage and session experiments |
| `results/` | Public measurements and validation records |

The numerical backend and the construction seed serve different roles. Change the packaged backend for the default service; change the seed only when the construction starting point should change. Keep historical results tied to their original revisions.

## Development setup

```bash
git clone https://github.com/yanchaomei/inference-forge.git
cd inference-forge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

Perform builds, tests, and model runs on a Linux test host or in CI. In the project's workflow, code editing and Git operations stay local; execution runs on the remote host. Use an isolated environment and output directory for each experiment.

## Choose the relevant checks

The commands below run from the checkout on the test host. They describe verification procedures, not results for an arbitrary revision.

**CPU task and CLI checks** need no model weights:

```bash
python -m pytest tests/test_campaign.py tests/test_task.py \
  tests/test_evaluate.py tests/test_replay.py tests/test_processes.py \
  tests/test_8b_export.py tests/test_cli.py -q
```

**HTTP and real SDK checks** use a deterministic CPU test engine. Install a CPU PyTorch wheel, then the service and client dependencies in a separate environment:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[inference,test,client]'
python -m pytest tests/test_serving_api.py -q
```

These checks exercise real HTTP requests and client parsing. They do not establish model numerical quality or GPU speed.

`make test-api` also runs `tests/test_qwen2_cpu.py`. That check creates a tiny random Qwen2 checkpoint locally, compares the packaged backend with an independent Transformers forward pass, and checks different-length cached sequences. It downloads no model and needs no GPU. The real checkpoint/GPU checks below cover a separate execution environment.

For the common CPU commands, use `make test-cpu` and `make test-api`. `make build` builds the wheel and source distribution after installing `build`.

**Real-model checks** require the pinned checkpoint and a CUDA environment. The existing seed regression suite runs as follows:

```bash
python -m pip install -e '.[inference,test,client]'
export FORGE_MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
export FORGE_GPU_UUID=GPU-your-device-uuid
export CUDA_VISIBLE_DEVICES="$FORGE_GPU_UUID"
python -m pytest tests/test_engine.py -q
```

The seed suite checks `forge/engine.py` and `forge/server.py`. Changes to the packaged backend or new serving path also need direct checks of those paths: reference logits, normal/streaming SDK requests, concurrent requests, and request limits. Use [the quickstart](QUICKSTART.md) for client requests. Record the exact commit, model revision, dependencies, hardware, commands, and results.

For packaging changes, build a wheel, install it into a fresh environment, leave the checkout, clear `PYTHONPATH`, and verify the CLI and backend imports. Include a real GPU request from that installed wheel when validating a release. An editable install alone does not check package contents.

## Construct a candidate

Run construction from the repository root on your local controller. You need an installed and logged-in Codex CLI, SSH access to a Linux GPU host with tmux, and an available generator model. Install `.[inference,test]` in a separate environment on the GPU host and prepare the model there.

```bash
export FORGE_MODEL_PATH=/path/on/gpu-server/Qwen2.5-0.5B-Instruct
export FORGE_GPU_UUID=GPU-your-device-uuid

inference-forge construct --task tasks/qwen25_05b/task.json \
  --output runs/example-campaign --host user@gpu-server --port 22 \
  --key ~/.ssh/id_ed25519 --remote-root /path/on/gpu-server/inference-forge \
  --remote-python /path/to/venv/bin/python --rounds 1 \
  --generator-model YOUR_AVAILABLE_CODEX_MODEL --clean-generator-config
```

The output directory must be new. Its final name uses 1–40 lowercase letters, digits, or hyphens. The runner evaluates the seed, invokes the generator, and evaluates candidates using the fixed task. `runs/` retains prompts, generator events, usage, candidate snapshots, remote task identities, and results. After an interruption, inspect recorded remote sessions before starting again.

`inference-forge --task ...` remains supported for existing scripts. `forge-campaign` also calls the same constructor. In construction commands, `--host` and `--port` select the SSH destination; for `serve`, they select the HTTP bind address.

To evaluate the original seed without invoking a generator, run on the GPU host:

```bash
forge-pilot --task tasks/qwen25_05b/task.json --output results/raw/local-check
```

The [8B guide](../experiments/qwen3_8b/README.md) has its own construction and evaluation commands.

## Performance changes

Fix the model revision, precision, hardware, request trace, output lengths, warmup, repetitions, and measurement method before comparing revisions. Run correctness checks first. Report raw samples and variation along with medians; include failures and the complete request cost when optimizing a component.

Construction quality and runtime speed are separate comparisons. Use the same generator and budget when studying construction feedback, and matched workloads when comparing runtime performance. Keep the final holdout outside the development feedback loop.
