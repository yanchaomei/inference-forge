# Changelog

## 0.3.0rc1 — 2026-11-02

- Package the generated Qwen2 numerical backend with the library. The original example launch paths remain compatibility entry points.
- Add `inference-forge serve`, `models`, `doctor`, and `construct`, with the previous construction command still accepted.
- Add OpenAI Python SDK text/chat completions, SSE, optional usage chunks, model aliases, and structured request errors.
- Preserve the native evaluator's streaming token and usage fields.
- Add English/Chinese entry pages, an API reference, support matrix, development guide, runnable client, and contribution templates.
- Build wheels in CI, install them outside the checkout, and exercise the installed HTTP service. Add a tiny local Qwen2 checkpoint test against an independent model forward pass.

This is a release candidate. CPU installation, protocol, and tiny-model checks are recorded separately from the real-checkpoint GPU regression, which is pending for this revision. The numerical backend is byte-identical to the generated backend validated in v0.2.0; the CLI and serving layer are new.

## 0.2.0 — 2026-10-31

- Publish Qwen3-8B host-state and session experiments, construction/validation commands, and raw request records.
- Provide a quickstart, project presentation, video, and submission links.
- Record the independent 0.5B service and numerical release check.

## 0.1.0 — 2026-10-30

- Publish the initial execution-guided construction loop, Qwen2 seed, and generated batched example.
- Include fixed evaluation, request-trace reduction, and the first development measurements.
