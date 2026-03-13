# Trainer SDK Wrapper

This folder documents the trainer component entry from the top-level SDK test harness.

Current model:

- SDK test entrypoints remain in `tests/trainer/kubeflow_sdk_test.go`.
- `tests/kubeflow_sdk/Makefile` exposes top-level SDK commands.
- `tests/kubeflow_sdk/trainer/Makefile` contains trainer-specific execution logic.

This keeps existing `go test ./tests/trainer ...` workflows intact while adding a dedicated SDK-centric entry point.

## Current trainer SDK test selection

The wrapper includes tests matching:

- `TestKubeflowSdk*`
- `TestOsftTrainingHub*`
- `TestLoraTrainingHub*`
- `TestSftTrainingHub*`
- `TestRhai*`
- `TestTrainingFailureScenarios`
- `TestTorchrunTrainingFailure`

Use `SDK_TEST_EXCLUDE_REGEX` to omit specific tests by name pattern.
