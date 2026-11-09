# Later intervention and training workstreams

Updated 2026-11-09. These are future steps, not implemented features. The current browser
supports bundled dense models, parameter revisions, forward-pass replay and batch diagnostics.
The Python audit bridge supports an engineer's own constrained in-memory Sequential MLP.
The architecture/research contract and its feasibility gates remain authoritative.

## Approved later work: test and validate interventions

The goal is to help an engineer evaluate a proposed repair or simplification against
task requirements. Inactivity is evidence to investigate, not proof a neuron needs fixing.
An output change may be beneficial or harmful; changed behavior alone is not a quality score.

Implement in this order:

1. **Supported real models and labeled evaluation data.** Finish the constrained data-only
   import/export and bounded worker gates. Add input/target datasets, explicit task/loss
   and metric contracts, preprocessing and feature/class order, validation splits and
   protected input groups. Retain an independent final test set for evaluation after
   candidate selection.
2. **Temporary hidden-neuron ablation.** Clamp the selected post-activation value to zero
   in a candidate experiment. Evaluate the reference and candidate on identical inputs.
   Report task metrics, affected cases and group performance, and connect them to the graph.
   Keep the original unchanged; accept/save or reject the candidate explicitly.
3. **Architecture-aware revisions.** Snapshot model structure as well as parameters before
   enabling activation changes or structural edits. Preserve immutable baselines and
   stable neuron IDs/mappings. Output comparisons remain possible across compatible
   architectures; neuron-by-neuron comparisons require matched identities, not shifted indices.
4. **Activation experiments and compact pruning.** Allow a supported layer's activation
   to change in an explicit candidate branch. Add LeakyReLU only with runtime/import/export
   and numerical parity support. Remove a hidden neuron only after intervention/evaluation
   evidence: remove its incoming weight row and bias entry plus the next dense layer's
   matching input column, validate all shapes, and preserve the old revision. Support
   exporting the candidate for fine-tuning and re-evaluating it afterward.
5. **Adding neurons, later.** Require a demonstrated capacity/training use case and a
   retraining path. Specify initialization and optimizer-state handling. New outgoing
   weights can start at zero to preserve initial outputs, but new units still need training.
   Do not present extra neurons as an automatic cure for inactive existing neurons.

Important experiment semantics:

- Ablating a neuron that is already zero for every tested input necessarily produces
  no change on that batch. This does not establish safety on other inputs or rare groups.
- Normal pointwise activation changes affect the whole layer and downstream behavior.
  Compare before/after with a declared objective and allow fine-tuning/retraining.
- Masking weights does not by itself shrink dense matrices or guarantee faster inference.
  Measure actual parameter/storage size and deployment latency for structural compression.
- Preprocessing or optimizer errors should be fixed at their source. Avoid a workflow
  that encourages manually increasing biases until every neuron appears active.
- Version the intervention, exact inputs, targets, metric definitions, seed and outcomes.
  Keep failed experiments and explicit uncertainty; do not label a candidate "fixed"
  solely because activations increased or its outputs moved closer on a small probe batch.

Acceptance examples:

- A rarely active neuron protects a rare defect class: ablation worsens that group's
  recall, so the tool exposes the tradeoff and supports rejecting removal.
- An architecture edit can be reverted, restoring exactly the prior structure and parameters.
- A compact candidate matches validation tolerances and exports valid adjacent-layer shapes.
- Candidate selection/tuning uses training and validation splits; the independent final
  test set remains untouched until the selected candidate's final evaluation.

## Proposed later work: observe learning and control training

Training visualization is feasible for the supported dense scope. Training normally changes
weights and biases while topology stays fixed. Activation displays always identify their
input/probe set; node brightness is not an input-independent property of a trained neuron.

### T1. Instrumented training and checkpoint replay

Start with a trusted Python training integration before building a general training UI.
The caller's code owns the training loop; neuroLabs receives data-only telemetry/checkpoints
through a defined interface. This is not permission for arbitrary uploaded Python execution.

Capture and expose:

- Epoch and optimizer-step counters, actual learning rate and run configuration.
- Training/validation loss and declared task metrics; no invented accuracy without targets.
- Actual measured gradient statistics where available, separate from activation statistics.
- Checkpointed weights/biases and activations on a fixed, identified evaluation probe batch.
- A timeline to select/replay saved checkpoints and compare outputs/activity at epochs A/B.
- Small 2D examples may add decision boundaries; higher-dimensional data uses declared slices.

Use a bounded capture cadence (steps/epochs), storage quota and bounded scene/tensor slices.
UI playback cadence is illustrative; it must not be described as the real duration of a
forward pass, backward pass or optimizer update. Visual sampling never removes training work.

### T2. Controlled local dense trainer

After worker, dataset/loss and training-state tests pass, add a narrow built-in trainer.
Start with application-owned supported operators and supervised tasks, not arbitrary
architectures, user-supplied optimizer code or a general replacement for training frameworks.

Controls planned for the UI:

- Epoch budget, learning rate, batch size, supported optimizer, seed and checkpoint cadence.
- Start, pause/resume, stop, and stepping by one batch or epoch where supported.
- Compare runs with the same initial checkpoint, dataset split and probe set.
- Inspect a suspicious epoch/checkpoint and branch an intervention or training experiment.

Define the control semantics before implementation:

- One epoch is a pass over the training split; a batch step is one optimizer update.
- Learning-rate changes take effect at a documented optimizer-step boundary, are logged,
  and have an explicit interaction with any learning-rate scheduler.
- An epoch-budget change affects remaining work; it never undoes completed updates.
- Batch-size, optimizer, activation or width changes create a new run/explicit branch with
  a declared restore/initialization policy; never silently reuse incompatible optimizer state.
- Pause/resume checkpoints include model/architecture, optimizer state, scheduler, RNG states,
  dataset/split/preprocessing identities, counters and configuration events.
- Stopping or canceling must reach the worker; aborting a browser request is not cancellation.

Acceptance examples:

- Resuming a deterministic supported run agrees with uninterrupted training within declared
  numerical tolerances; paused state is not merely a frozen animation.
- A learning-rate change is visible in the event history and affects subsequent updates.
- Loss/gradients/parameters shown for a step belong to the same identified run/checkpoint.
- Replaying two checkpoints on the same fixed probes is reproducible.
- Overfitting or unstable training can be investigated using measured validation loss,
  gradients and checkpoint behavior; a visually active graph is not treated as success.

## Placement and scope control

The existing roadmap remains in [architecture.md](architecture.md#roadmap-and-release-gates).
Real model/data handling and task evaluation extend Phases 2–3. Candidate interventions,
architecture-aware revisions and measured gradients extend Phases 3–4. Training observation
and control are a separately estimated later workstream, with T1 before T2 and neuron
addition after a training path exists. These extensions do not mean those phases are done.

Preserve the blinded weight-recovery research objective. A full training platform, CNNs,
Transformers and all architecture-editing controls are not prerequisites for the blinded
benchmark. Clean targets/checkpoints used by an evaluator must remain inaccessible to the
agent in research conditions that prohibit them. Re-estimate delivery scope before promoting
the proposed training track to a release requirement; the original hour range excludes it.

## Existing tools to study

- [TensorFlow Playground](https://playground.tensorflow.org/) and its
  [source](https://github.com/tensorflow/playground): direct browser manipulation of learning
  rate, activation, batch size and small network structure, with training/output visualization.
  It exposes an epoch counter and run/pause/step controls; do not describe it as having an
  arbitrary epoch-budget form without checking that feature.
- [TensorBoard with PyTorch](https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html):
  logging/visualizing losses, metrics, learning rate, graphs and distributions. Monitoring
  a run is distinct from changing its optimizer settings through a control API.
- [Weights & Biases Sweeps](https://site.wandb.ai/sweeps/): configurable hyperparameter
  experiments and comparison. Requires training-code/agent integration; it is not a generic
  live knob for any independently running model.
- [PyTorch pruning](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)
  and [TensorFlow pruning](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras):
  distinguish masks from compact structures and evaluate/fine-tune candidates.
