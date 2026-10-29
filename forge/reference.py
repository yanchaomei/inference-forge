import argparse
import json
from pathlib import Path

from .evaluate import generate_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    task = json.loads(Path(args.task).read_text())
    result = generate_reference(task, args.output)
    print(json.dumps({"reference": args.output, "cases": len(result["cases"]), "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
