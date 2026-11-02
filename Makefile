PYTHON ?= python

.PHONY: test-cpu test-api build

test-cpu:
	$(PYTHON) -m pytest tests/test_campaign.py tests/test_task.py tests/test_evaluate.py tests/test_replay.py tests/test_processes.py tests/test_8b_export.py tests/test_cli.py -q

test-api:
	$(PYTHON) -m pytest tests/test_serving_api.py tests/test_qwen2_cpu.py -q

build:
	$(PYTHON) -m build
