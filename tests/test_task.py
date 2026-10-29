import json
from forge.task import load_task


def test_portable_task_resolves_explicit_deployment_values(tmp_path,monkeypatch):
    path=tmp_path/"task.json"
    path.write_text(json.dumps({"model_path":"@env:FORGE_MODEL_PATH","gpu_uuid":"@env:FORGE_GPU_UUID","dtype":"float32"}))
    monkeypatch.setenv("FORGE_MODEL_PATH","/models/qwen")
    monkeypatch.setenv("FORGE_GPU_UUID","GPU-example")
    assert load_task(path)=={"model_path":"/models/qwen","gpu_uuid":"GPU-example","dtype":"float32"}
