"""Official model-file identity checks for the standalone 8B experiment."""
import hashlib
from pathlib import Path

def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(model, task):
    model = Path(model)
    expected = task.get("model_files_sha256") or {
        "model.safetensors": task["model_safetensors_sha256"]}
    names = set(expected)
    names.update(p.name for p in model.iterdir() if p.is_file() and
                 p.name in ("config.json", "tokenizer.json", "tokenizer_config.json"))
    actual = {name: {"bytes": (model/name).stat().st_size,
                     "sha256": file_sha(model/name)} for name in sorted(names)}
    for name, sha in expected.items():
        if actual[name]["sha256"] != sha:
            raise ValueError(f"model identity mismatch: {name}")
    return actual
