# neuroLabs architecture and research contract

## Approved direction

A local browser application with a Python execution service. Version 1 deeply supports
dense networks; CNNs, Transformers, and LLM orchestration are later extensions. The primary
workflow is inspect → edit → run → compare → restore → export.

The user chose a complete product, a local desktop browser, solo part-time development,
strict model compatibility, and milestone-driven delivery. Initial planning assumption:
8–12 hours per week. Quality and correctness gates take precedence over dates.

## Non-negotiable contracts

- Data-only models. Never accept Python modules, pickle, `.pt`, or `.pth` uploads.
- Stable operation/tensor identifiers. Dense weights are `[output, input]`; inputs are
  `[batch, input]`. Execution is `X @ W.T + b`.
- One authoritative PyTorch CPU execution path. React visualizes returned results.
- Each committed edit creates an immutable revision, records requested and effective
  float32 values, and references content-addressed safetensors artifacts.
- A run names its revision and effective input hash. The UI must never display a result
  as current if either has changed. Original parameters remain intact.
- Atomic edits use expected-base checks and idempotency keys. A stale edit fails explicitly.
- Task metadata, desired behavior, hypotheses, and experimental observations are distinct.
- The future agent calls the same experiment services as the user; it does not drive a browser.

## Internal graph and limits

`ModelSpec` is a versioned chain of `Dense` and activation operations, with named inputs,
outputs, and parameter references. This is an operation-level graph, not a stored visual
object for every neuron. Version 1 bounds: 16 dense layers, width 256, 1 million parameters,
batch 256, float32. ReLU, Tanh, Sigmoid, Identity, and final Softmax are supported.
Shared parameters, cycles, unused tensors, and mismatched shapes are rejected.

The importer milestone will add a native JSON/safetensors package and ONNX opsets 17/18
for constrained Gemm / MatMul + Add chains. External ONNX data, custom operations,
unresolved feature dimensions, and arbitrary broadcasting are excluded. Export emits a
normalized model, not a byte-identical recreation.

## Storage and API

SQLite owns model/revision/run metadata. Tensor artifacts are independently hashed,
verified on read, and written atomically before revision metadata commits. One transaction
commits a revision and moves the active head. Restoring a revision does not delete history.
Data defaults to `%LOCALAPPDATA%/neuroLabs` on Windows and `~/.local/share/neuroLabs`
elsewhere, outside the OneDrive-synced repository. `NEUROLABS_DATA_DIR` overrides this.

The API is versioned under `/api/v1`. Loopback binding, Host/Origin validation, and a
per-launch credential protect browser requests. This is a single-user application,
not a public upload service. Local processes can access the session endpoint intentionally.

The preview exposes sample models only. Its execution is synchronous in the backend's
request thread pool. Before accepting external models or long-running experiments, add
a bounded worker process with time/memory limits, restart/cancellation, persisted jobs,
and bounded tensor slices. A subprocess is not a hostile-code sandbox.

## Visualization architecture

The primary view is a Three.js 3D graph of individual neurons and weighted connections,
following the user's clarified product requirement. Orbit controls provide rotation, pan,
and zoom. Dense operations and their following activation share a visual neuron layer;
both preactivation and postactivation remain separately captured by the runtime.
The graph contains no text or numeric labels. Direct picking opens separate neuron or
connection views in the inspector. Neuron views expose activation function, preactivation,
output and bias; connection views expose the weight, source activation and contribution.
Thin cubic curves have fixed endpoints and gently moving interior control points to look
like floating strings. Deformation is visual only; particles follow the current curve.
Reduced-motion preferences freeze string motion without changing model behavior.
Transparent keyboard controls follow actual node/curve bounds; pointer selection raycasts
the same visible geometry. No text textures, label placement or collision system is needed.
The architecture tree and parameter controls provide additional keyboard access.
Small scenes retain individual curve components. Dense scenes use one instanced neuron
mesh, one batched LineSegments geometry and instanced replay particles. Overview samples
at most 96 connections per transition while showing every neuron; layer-pair focus draws
every edge in that pair, and neuron focus draws every incoming/outgoing edge. Filtering
never changes runtime execution or captures. Selection remains tied to real tensor
coordinates; displayed counts make filtering explicit. Camera framing follows the view.
Projected keyboard targets use cached viewport dimensions to avoid per-target layout
flushes. Dense scenes cap animation at 30 fps and omit thousands of overlapping keyboard
edge targets in overview/pair views; neuron focus and inspector/matrix controls provide
individual keyboard access. HTML matrices still render the selected bundled tensor whole.
The bundled 16-layer/64-wide study exercises this path; maximum-width and broad hardware
performance/memory gates remain open.

Run & trace performs a new backend forward pass and replays its captured values. Playback
supports pause, stepping, scrubbing, and speed changes. Nodes encode signed activation;
connections encode the sign of source activation multiplied by weight. ReLU-zero outputs
are dimmed, while their incoming nonzero contributions remain visible. All dense connections
are computed; the replay does not claim the network chose one discrete route or represent
hardware execution timing. Input/revision changes invalidate the replay and pending requests.

## Research objective

Can an LLM infer and recover a network's original mathematical behavior from partially
corrupted weights without its original task, training data, or clean parameters?

The primary condition exposes architecture and corrupted weights only. Other conditions
separately allow corrupted-model execution, corruption assumptions, or limited clean
examples. The evaluator privately owns clean checkpoints, tasks, corruption masks, and
held-out data. Agent-facing metadata, filenames, API responses, and errors must not leak them.

Measure functional recovery, inference accuracy, held-out/extrapolation performance,
failure rates, and query/edit budgets. Compare against no repair, random edits, search,
and separately labeled gradient baselines. Parameter distance alone is insufficient.
Recovery can be unidentifiable; failure and ambiguity are research results. Never score
an LLM's claimed understanding without independent evaluation.

No agent is part of the current implementation. Phase 5 creates the blinded benchmark;
Phase 6 adds the investigator. Neither depends on completing CNN/Transformer support.

## Roadmap and release gates

1. **Phase 0:** numerical fixtures and independent ONNX parity; rendering stress benchmark;
   normalized ONNX roundtrip; worker resource/cancellation tests; storage recovery tests.
2. **Phase 1:** a polished real vertical slice: sample → inspect → edit → run → restore;
   persistent history, accessible navigation, and profiling.
3. **Phase 2:** bounded native/ONNX imports, compatibility reports, tensor slices, captures,
   and normalized export; every accepted operator pattern needs numerical fixtures.
4. **Phase 3:** matrix edits, snapshots, comparisons, datasets, input/parameter sweeps,
   cancellation, and portable experiment bundles.
5. **Phase 4:** gradients, finite differences, Captum attribution, bounded FGSM experiments,
   accessible polished UI, and maximum-supported-model performance gates.
6. **Phase 5:** reproducible corruption, blinded evaluation, hypotheses, Python client,
   benchmark runner, release packaging and documentation. Version 1 ends here.
7. **Phase 6:** autonomous dense-network research; separately scoped CNN/Transformer work.

Initial planning budget: 340–530 engineering hours for Phases 0–5, plus 25% contingency.
Re-estimate after the first full vertical slice. This is not a delivery guarantee.

## References

- https://github.com/lutzroeder/netron — architecture and format inspection
- https://github.com/tensorflow/playground — direct manipulation
- https://github.com/poloclub/cnn-explainer — coordinated computation views
- https://github.com/ZhangGe6/onnx-modifier — initializer editing and export
- https://pair-code.github.io/lit/ — evaluation and interpreter separation
- https://captum.ai/docs/introduction — attribution implementations
- https://github.com/poloclub/transformer-explainer — future Transformer UI
- https://docs.pytorch.org/docs/stable/notes/randomness.html — reproducibility limits
