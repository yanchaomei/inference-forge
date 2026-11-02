# Contributing to InferenceForge

Contributions in English or Chinese are welcome. Useful starting points include a clearer example, a reproducible failure, a numerical check, a model request, or a measured performance improvement.

## Report a problem

Use the [issue forms](https://github.com/yanchaomei/inference-forge/issues/new/choose). Include the commit or version, model and revision, GPU, dependency versions, launch command, and smallest request that reproduces the behavior. `inference-forge doctor --json` provides a short environment report.

For performance reports, include the workload and raw repeated measurements. For a new model, describe the target hardware, precision, context lengths, and application. Please report vulnerabilities through [SECURITY.md](SECURITY.md), and remove credentials and private prompts from public reports.

## Make a change

1. Fork the repository and work on a focused branch.
2. Follow [Development](docs/DEVELOPMENT.md) for installation, code layout, and the relevant CPU or GPU checks.
3. Keep the change small enough to review. Update examples and support limits when behavior changes.
4. Open a PR describing the problem, resulting behavior, and commands actually run. Mark checks that were not run.

Documentation and protocol work can start without a GPU. Numerical, KV, scheduling, and performance changes need the corresponding real-model checks on a test host. Maintainers review the change and its validation before merging; CI results and GPU measurements are reported separately.

## Generated code and experiments

Coding-agent contributions are welcome. State the generator/model, starting code, and generation budget where they affected the change. Review the generated code and record the checks you actually ran. For experiment contributions, retain unsuccessful trials and the source identity of measured candidates.

A performance PR needs correctness results and comparable before/after measurements with the same model, precision, hardware, input/output lengths, warmup, and repetitions. Include raw samples and variation rather than only the fastest run. See [performance guidance](docs/DEVELOPMENT.md#performance-changes).

## Files and attribution

Keep weights, credentials, and private datasets out of commits. Public results should omit internal addresses and private business data while preserving the configuration needed to interpret them. Preserve upstream license and attribution notices for any reused code; see [NOTICE.md](NOTICE.md).

Submitting a contribution makes it available under the project's [MIT license](LICENSE). There is no separate contribution agreement.
