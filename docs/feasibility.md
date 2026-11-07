# Initial feasibility evidence

Measured on 2026-11-07. This report covers the implemented numerical/storage foundation;
it does not certify completion of Phase 0.

## Numerical parity

Independent ONNX graphs execute through ONNX Runtime. Tests compare their results with
our PyTorch path using non-square `[2, 3]` weights and seeded batch inputs for ReLU,
Tanh, Sigmoid, Softmax and Identity (`atol=1e-6`, `rtol=1e-5`). All five pass.
The linear fixture matches hand-calculated results; the XOR fixture matches all four
truth-table classifications. This validates those execution semantics, not an ONNX importer.

## Warm CPU timing

Command: `uv run python scripts/benchmark_runtime.py`

Environment: Windows 11 build 26200, AMD64 Family 25 Model 68 Stepping 1,
Python 3.12.10, PyTorch 2.15.0+cpu, one Torch thread, deterministic algorithms enabled,
seed 42. Five warmups; 50 samples except the largest batch (10 samples).

| Model | Parameters | Batch | Median | Observed p95 |
|---|---:|---:|---:|---:|
| Linear | 1 | 1 | 0.373 ms | 0.558 ms |
| Rectified curve | 10 | 1 | 0.709 ms | 1.064 ms |
| XOR | 12 | 1 | 0.416 ms | 0.552 ms |
| 15 dense layers, width 256 | 983,040 | 1 | 14.945 ms | 19.930 ms |
| 15 dense layers, width 256 | 983,040 | 256 | 57.925 ms | 67.119 ms |

These timings include parameter/input validation, execution, output conversion and run
metadata construction. They exclude startup, artifact loading, SQLite, HTTP, activation
captures and browser rendering. Small timing samples are preliminary, not guarantees.
End-to-end latency and rendering budgets remain to be measured.

## Persistence evidence

Tests verify originals remain unchanged, restore then edit preserves both branches,
restart preserves the active head, duplicate edit requests are idempotent, conflicting
concurrent edits have exactly one winner, invalid batches create no partial revision,
and corrupted tensor files fail integrity checks.

## Verification at this checkpoint

- 21 backend tests passed in the isolated project environment on Windows.
- 3 Chromium browser tests passed: edit/run/reload/restore; coordinate selection,
  model switching and input updates; and default-workspace capture.
- A 390-pixel viewport check found no horizontal document overflow.
- TypeScript/production build, Ruff checks, and frontend formatting checks passed.
- The default workspace screenshot was visually inspected.
- Windows/Linux CI is configured; Linux execution has not been verified locally.

## Remaining decisions gated by evidence

- Normalized ONNX import/export fixtures and compatibility reports.
- Worker process timeout, memory limits, cancellation and restart.
- PixiJS detail rendering, maximum visible scene, memory and frame-time budgets.
- Disk-full and interrupted-write recovery, backup/export and schema upgrades.

Do not enable user model uploads until the import and execution containment gates pass.
