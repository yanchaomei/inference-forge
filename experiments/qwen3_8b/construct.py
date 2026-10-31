"""Prepare or generate a paired 8B storage/consumer candidate locally.

This command does not run GPU inference or declare the generated code correct.
Use session_matrix on the remote GPU for independent execution feedback.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from forge.campaign import invoke_generator


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--generator-model")
    p.add_argument("--timeout-seconds", type=int, default=360)
    p.add_argument("--clean-generator-config", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    if not args.prepare_only and not args.generator_model:
        p.error("supply an actually available --generator-model")
    if args.timeout_seconds <= 0:
        p.error("timeout must be positive")
    root = Path(__file__).resolve().parents[2]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    candidate = out/"candidate"
    candidate.mkdir()
    sources = {"engine.py": root/"engines/qwen3_8b_kv/engine.py",
               "kv_store.py": root/"engines/qwen3_8b_kv/kv_store.py",
               "consumer.py": root/"experiments/qwen3_8b/baseline_consumer.py"}
    for name, source in sources.items():
        shutil.copyfile(source, candidate/name)
    seed = {name: digest(path) for name, path in sources.items()}
    contract = (root/"experiments/qwen3_8b/session-contract.md").read_text()
    manifest = {"status": "prepared_not_generated", "seed_sha256": seed,
                "contract_sha256": hashlib.sha256(contract.encode()).hexdigest(),
                "generator_model": args.generator_model, "gpu_evaluation_run": False,
                "timeout_seconds": args.timeout_seconds, "generation_round_limit": 1,
                "generation_invocation_attempted": False}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return
    command = ["codex", "exec"]
    if args.clean_generator_config:
        command.append("--ignore-user-config")
    command += ["--model", args.generator_model, "--ephemeral", "--json", "--skip-git-repo-check",
                "--sandbox", "workspace-write", "-C", str(candidate), "-"]
    prompt = contract+"\nImplementation within this contract is explicitly authorized. Edit the two permitted files now; no additional planning approval is needed. Preserve engine.py. Static source reading/editing is allowed, but no local tests/inference/builds. The external controller evaluates on the GPU.\n"
    manifest["generation_invocation_attempted"] = True
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    try:
        result = invoke_generator(command, candidate, prompt, out/"generator", args.timeout_seconds)
    except OSError as error:
        manifest.update(status="generation_failed", error_type=type(error).__name__, error=str(error))
        (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        print(json.dumps({"status": "generation_failed", "error_type": type(error).__name__}))
        raise SystemExit(1)
    missing = sorted(name for name in sources if not (candidate/name).is_file())
    observed = {name: digest(candidate/name) if (candidate/name).is_file() else None for name in sources}
    valid_call = result["returncode"] == 0 and result["turn_completed"] and not result["timed_out"]
    changed = any(observed[n] != seed[n] for n in ("consumer.py", "kv_store.py"))
    if missing:
        status = "rejected_missing_files"
    elif observed["engine.py"] != seed["engine.py"]:
        status = "rejected_engine_changed"
    elif not valid_call:
        status = "generation_failed"
    else:
        status = "generated_not_validated" if changed else "no_code_change"
    manifest.update(status=status, candidate_sha256=observed, missing_files=missing, generation=result)
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"status": status, "candidate": str(candidate), "gpu_evaluation_run": False}))
    raise SystemExit(0 if status == "generated_not_validated" else 1)


if __name__ == "__main__":
    main()
