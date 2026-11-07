# Implementation status

This is the first foundation preview, not completion of Phase 0 or Phase 1.

## Implemented

- Versioned data-only dense-operation contracts and shape/parameter validation.
- Four hand-constructed samples (linear, rectified curve, XOR and a five-layer Tanh study),
  plus a reproducible seeded 16-dense-layer study with fifteen 64-neuron hidden layers.
- Real deterministic-setting PyTorch CPU forward passes and selected intermediate captures.
- Content-addressed safetensors, integrity checking, SQLite revisions, atomic scalar/batch
  cell edits, stale-base rejection, idempotency, and revision restore/branching.
- Local FastAPI routes, session/origin guards, persistent run results, OpenAPI docs.
- React overview, operation selection, parameter grid/editor, computation breakdown,
  live inputs, response slices, original comparison, and revision history.
- Offline bundled fonts. No cloud inference, telemetry, or external model downloads at runtime.
- 3D neuron graph with orbit controls, neuron selection, real forward-pass replay,
  pause/step/scrub/speed controls, signed contributions and ReLU suppression display.
- Direct connection and neuron picking opens separate inspector views. Values, bias editing,
  activations and weighted contributions appear in the inspector, never on the graph.
- Thin floating-string curves gently deform between fixed endpoints; reduced motion freezes
  the deformation. Changes to curve shape do not affect neural-network computation.
- Near-black observatory theme without stars, glossy highlights, text or number labels.
  Transparent, projected geometry targets preserve keyboard selection.
- Live graph values update with inputs/parameter revisions; Run & trace adds optional playback.
- Inward arrows mark input neurons and outward arrows mark outputs.
- Dense visualization: 966 neurons, 57,728 computed connections, sampled overview,
  complete adjacent-layer pairs (4,096 strings), and full neuron neighborhoods.
  Instanced neurons and batched curves bound draw calls. Layer/index navigation, exact
  displayed-connection counts and camera framing support inspection at this scale.
- Compact selection inspectors: connections show source activation, one weight editor,
  and contribution; neurons show output/function, nonlinear preactivation and one bias
  editor where present. Incoming contributions and history are expandable. Full tensor
  selection/matrices live in the Weights tab. Editor text round-trips float32 values without
  long binary-representation decimals or creating revisions for unchanged effective values.

## Remaining feasibility gates

- Bidirectional ONNX normalization/import/export. Current ONNX tests compare execution
  against independently constructed graphs; they do not establish importer coverage.
- Bounded worker process, resource caps, cancellation and crash recovery.
- Wider hardware/memory profiling, maximum-width stress benchmarks and semantic zoom.
- Simulated disk-full/interrupted-write recovery and database schema migrations.
- Typed generated frontend client; current preview types are handwritten.

## Deliberate preview limits

Only bundled samples are exposed. No upload/export, agent, training, gradient tools,
or arbitrary architecture support. Bundled parameter tables are returned whole; bounded
slice APIs are required before import. History lists at most 100 recent revisions.
Restoration works through history; a dedicated undo/redo toolbar is not implemented.
Runs are synchronous and slider requests are debounced; browser abort prevents stale display
but does not cancel backend computation. All samples are sufficiently small for this preview.

Do not expose this preview on a public interface or describe it as a hardened model sandbox.
