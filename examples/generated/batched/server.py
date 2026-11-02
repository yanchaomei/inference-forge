"""Compatibility entry point. Prefer `inference-forge serve MODEL_PATH`."""
from forge.serving import Handler, Runtime, incremental_decode, main

if __name__ == '__main__':
    main()
