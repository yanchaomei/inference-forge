"""CPU-only integrity of published experiment records and preparation entry."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_published_sources_and_actual_request_records_match_receipts():
    provenance = json.loads((ROOT/"experiments/qwen3_8b/provenance.json").read_text())
    for name, expected in provenance["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
    result = json.loads((ROOT/"results/qwen3-8b-sessions-20261031.json").read_text())
    runs = result["runs"]
    assert len(runs) == 18
    reference = next(r["requests"] for r in runs if r["run"] == "resident-reference")
    for run in runs:
        assert len(run["requests"]) == len(reference) == 24
        for row, expected in zip(run["requests"], reference):
            assert row["status"] == "pass"
            for field in ("index", "session_id", "token_ids", "token_logprobs", "logit_sha256"):
                assert row[field] == expected[field]
            assert len(row["logit_sha256"]) == len(row["token_ids"]) == 4


def test_preparation_preserves_seed_and_does_not_claim_generation(tmp_path):
    out = tmp_path/"prepared"
    subprocess.run([sys.executable, "-m", "experiments.qwen3_8b.construct",
                    "--output", str(out), "--prepare-only"], cwd=ROOT, check=True, capture_output=True)
    manifest = json.loads((out/"manifest.json").read_text())
    assert manifest["status"] == "prepared_not_generated"
    assert manifest["gpu_evaluation_run"] is False
    assert manifest["timeout_seconds"] == 360
    assert manifest["generation_round_limit"] == 1
    assert "generation" not in manifest
    for name, expected in manifest["seed_sha256"].items():
        assert hashlib.sha256((out/"candidate"/name).read_bytes()).hexdigest() == expected


def test_generator_deleting_required_source_is_recorded_as_rejected(tmp_path):
    stub = tmp_path/"codex"
    stub.write_text(f"#!{sys.executable}\nfrom pathlib import Path\n"
                    "Path('consumer.py').unlink()\n"
                    "print('{\"type\":\"turn.completed\",\"usage\":{}}')\n")
    stub.chmod(0o755)
    out = tmp_path/"deleted-source"
    result = subprocess.run([sys.executable, "-m", "experiments.qwen3_8b.construct",
                             "--output", str(out), "--generator-model", "controlled-test-stub"],
                            cwd=ROOT, env={**os.environ, "PATH": str(tmp_path)+os.pathsep+os.environ["PATH"]},
                            capture_output=True, text=True)
    manifest = json.loads((out/"manifest.json").read_text())
    assert result.returncode != 0
    assert manifest["status"] == "rejected_missing_files"
    assert manifest["missing_files"] == ["consumer.py"]
    assert manifest["generation"]["turn_completed"] is True
