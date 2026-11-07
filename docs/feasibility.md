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

## Verification at the initial checkpoint

- 21 backend tests passed in the isolated project environment on Windows.
- 3 Chromium browser tests passed: edit/run/reload/restore; coordinate selection,
  model switching and input updates; and default-workspace capture.
- A 390-pixel viewport check found no horizontal document overflow.
- TypeScript/production build, Ruff checks, and frontend formatting checks passed.
- The default workspace screenshot was visually inspected.
- Windows/Linux CI is configured; Linux execution has not been verified locally.

## Dense visualization checkpoint

The bundled `dense16` fixture reaches 16 dense layers: 4 inputs, fifteen 64-neuron
Tanh hidden layers, and 2 linear outputs. It has 966 neurons, 57,728 connections and
58,690 parameters. Seeded parameters remain distinct from trained-model claims.
Every intermediate capture is checked against independent NumPy composition, including
input sensitivity, repeatable initialization and the effect of a final-bias edit.

Instanced neurons and batched curves replace per-object rendering for dense scenes.
Overview shows every neuron with a disclosed sample of 1,536 strings; pair focus shows
all 4,096 strings between hidden layers, and neuron focus shows complete incoming/outgoing
connections. Regression checks bound draw calls below 24, exercise native node/edge
picking, scalar editing, 16-layer replay, invalidation, model switching and narrow layout.
Selection projection caches viewport dimensions to avoid per-target forced layout.

This is one tested scale on the current Windows/Chromium environment. It does not close
the maximum-width, GPU memory, frame-time or broad hardware performance gates.

## Remaining decisions gated by evidence

- Normalized ONNX import/export fixtures and compatibility reports.
- Worker process timeout, memory limits, cancellation and restart.
- PixiJS detail rendering, maximum visible scene, memory and frame-time budgets.
- Disk-full and interrupted-write recovery, backup/export and schema upgrades.

Do not enable user model uploads until the import and execution containment gates pass.
