import json
from pathlib import Path
import sys

from forge.campaign import invoke_generator, materialize


def test_external_generator_edits_candidate_and_usage_is_retained(tmp_path):
    candidate = tmp_path/"candidate"
    candidate.mkdir()
    script = "from pathlib import Path; import json; Path('engine.py').write_text('changed'); print(json.dumps({'type':'turn.completed','usage':{'input_tokens':9,'output_tokens':4}}))"
    report = invoke_generator([sys.executable,"-c",script],candidate,"instruction",tmp_path/"logs",10)
    assert (candidate/"engine.py").read_text() == "changed"
    assert report["returncode"] == 0
    assert report["usage"] == {"input_tokens":9,"output_tokens":4}
    assert json.loads((tmp_path/"logs/events.jsonl").read_text())["type"] == "turn.completed"


def test_candidate_cannot_replace_evaluator_in_materialized_bundle(tmp_path):
    trusted = tmp_path/"trusted"
    (trusted/"forge").mkdir(parents=True)
    (trusted/"forge/evaluate.py").write_text("trusted evaluator")
    (trusted/"forge/engine.py").write_text("old engine")
    (trusted/"forge/server.py").write_text("server")
    candidate = tmp_path/"candidate"
    (candidate/"forge").mkdir(parents=True)
    (candidate/"forge/engine.py").write_text("new engine")
    (candidate/"forge/server.py").write_text("new server")
    (candidate/"forge/evaluate.py").write_text("fake pass")
    bundle = tmp_path/"bundle"
    materialize(trusted,candidate,bundle)
    assert (bundle/"forge/engine.py").read_text() == "new engine"
    assert (bundle/"forge/evaluate.py").read_text() == "trusted evaluator"
