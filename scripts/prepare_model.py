"""Download a frozen target and verify every file against official HF metadata.

Run on the GPU host. The endpoint may be a mirror, but the expected identity
always comes from the checked-in official metadata, never from that mirror.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def verify_file(path, entry):
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {entry['size']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            git_blob.update(chunk)
    actual = sha256.hexdigest()
    expected = entry.get("lfs", {}).get("sha256")
    if path.stat().st_size != entry["size"]:
        raise ValueError(f"size mismatch: {path.name}")
    if (expected and actual != expected) or (not expected and git_blob.hexdigest() != entry["blobId"]):
        raise ValueError(f"official identity mismatch: {path.name}")
    return {"bytes": path.stat().st_size, "sha256": actual}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--file-base-url", help="Alternative file source; official file identities remain mandatory")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    target = json.loads(Path(args.target).read_text())["model"]
    metadata = json.loads(Path(args.metadata).read_text())
    if metadata["sha"] != target["revision"] or metadata["id"] != target["repository"]:
        raise ValueError("target and official metadata differ")
    entries = [e for e in metadata["siblings"] if e["rfilename"] != ".gitattributes"]
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination / "forge-model-receipt.json"
    # A previous success must never survive a failed re-verification.
    receipt_path.unlink(missing_ok=True)
    if not args.verify_only:
        needed = sum(e["size"] for e in entries if not (destination/e["rfilename"]).exists())
        if shutil.disk_usage(destination).free < needed + 5 * 1024**3:
            raise RuntimeError("insufficient space for weights plus 5 GiB reserve")
        if args.file_base_url:
            def download(entry):
                name = entry["rfilename"]
                final = destination/name
                if final.exists():
                    verify_file(final, entry)
                    return
                partial = destination/(name+".download")
                subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                                "--retry", "2", "--connect-timeout", "20", "--max-time", "3600",
                                "--continue-at", "-", "--output", str(partial),
                                args.file_base_url.rstrip("/")+"/"+name], check=True)
                verify_file(partial, entry)
                partial.rename(final)
                print(f"downloaded and verified {name}", flush=True)
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(download, entries))
        else:
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=target["repository"], revision=target["revision"],
                              endpoint=args.endpoint, local_dir=str(destination),
                              allow_patterns=[e["rfilename"] for e in entries], max_workers=2)
    files = {}
    for entry in entries:
        name = entry["rfilename"]
        files[name] = verify_file(destination/name, entry)
        print(f"verified {name}", flush=True)
    config = json.loads((destination/"config.json").read_text())
    if config["architectures"] != [target["architecture"]]:
        raise ValueError("architecture mismatch")
    receipt = {"status": "verified", "repository": target["repository"],
               "revision": target["revision"], "download_endpoint": args.file_base_url or args.endpoint,
               "verified_at_unix": time.time(), "files": files}
    receipt_path.write_text(json.dumps(receipt, indent=2)+"\n")
    print(f"Model verified: {receipt_path}", flush=True)


if __name__ == "__main__":
    main()
