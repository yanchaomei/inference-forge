"""Load a task, with explicit local deployment values for portable examples."""
import json
import os
from pathlib import Path


def load_task(path):
    task = json.loads(Path(path).read_text())
    for field, name in [("model_path","FORGE_MODEL_PATH"),("gpu_uuid","FORGE_GPU_UUID")]:
        if task.get(field) == "@env:"+name:
            task[field] = os.environ[name]
    return task
