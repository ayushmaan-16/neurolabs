# Import and export a dense model

neuroLabs accepts a native ZIP package containing architecture JSON and float32
safetensors. Browser ONNX, `.py`, `.pt`, `.pth`, Keras and arbitrary architecture imports
are not implemented. The Python exporter accepts a caller-owned exact `torch.nn.Sequential`.

## Try it without preparing a model

1. Click **Import model** in the model collection.
2. Click **Download example package**. This contains the analytic rectified curve,
   not a trained predictor or a dataset download.
3. Select that ZIP using **Model package**.
4. Review the layers, shapes, parameter count and output-verification result.
5. Click **Import into workspace**. Preview alone does not save a model.
6. Enter a numeric input vector, click **Apply input**, inspect the graph or Weights,
   and use Diagnostics with a batch matching the input width.
7. Stage edits, name/save a revision, then click **Export saved revision** below the
   input experiment. The exported ZIP can be imported into a separate workspace model.

Imported models persist across server/browser restarts. Re-select one from the collection
after refreshing. Each upload has a server-assigned identity and never overwrites a sample
or another imported model. Retrying the same live preview commit returns the same model.

## Export your own PyTorch model

Install this repository's Python package into your model's environment (it is not currently
published as a separately maintained PyPI exporter):

```sh
python -m pip install -e /path/to/neurolabs
```

In the code where your trained model and preprocessed validation inputs already exist:

```python
from neurolabs.packages import export_sequential

model.eval().cpu()
export_sequential(
    model,
    "my-model.zip",
    name="My trained classifier",
    task="classification",
    example_inputs=validation_features[:8],  # 1–16 preprocessed rows
    feature_names=["feature_a", "feature_b", "feature_c", "feature_d"],
    output_names=["class_a", "class_b", "class_c"],
    input_note="Use the training scaler, then provide features in the listed order.",
)
```

Choose feature/output names matching your actual dimensions or omit them. `task` may be
`unknown`, `regression` or `classification`; it is a declaration, not inferred from weights.
The first example input becomes the default, unless `default_input=[...]` is supplied.
Without examples, a zero input is the default unless overridden.

The helper copies supported parameters and records outputs from the source Sequential
independently of the neuroLabs runtime. It does not train the model or modify its parameters,
training mode or device. Verification examples require the source model on CPU.

A custom `nn.Module` class is not automatically traced. For a simple known feedforward
class, construct a matching plain Sequential and copy its parameters explicitly, then
compare its outputs with the original before exporting. Unsupported behavior must never
be silently removed during this conversion.

`examples/export_iris_checkpoint.py` demonstrates a concrete conversion for the 43-parameter
[sadhaklal/mlp-iris checkpoint](https://huggingface.co/sadhaklal/mlp-iris): download its
`model.safetensors` file, then run the script with that file. This small classifier expects
four **standardized** Iris measurements. Its model card describes the training split and
normalization; that scaler is not embedded in the weights. This example exports normalized
probe inputs for numerical checks, not an accuracy evaluation on Iris flowers.

## Supported model contract

- 1–16 dense layers, each dimension 1–256, at most 1,000,000 parameters including biases.
- A dense layer optionally followed by one ReLU, Tanh, Sigmoid or Identity operation.
  Softmax is accepted only as the final operation, on the feature dimension.
- Finite float32 weights/biases. Optional dense bias is supported.
- Exact built-in Sequential/Linear/activation modules; no forward overrides, hooks,
  shared parameter storage, nested modules, dropout or batch normalization.
- No branches, skip connections, convolutions, recurrent layers or attention operators.

Input vectors contain exactly `input_width` numeric features. Diagnostics accepts 2–256
rows. The package describes input order and preprocessing in text; it does **not** execute
an uploaded scaler, tokenizer or preprocessing function. Apply the original transformations
before supplying inputs. Output names describe the returned vector in order. Classification
outputs can be logits or final Softmax values depending on the declared architecture.

## Package version 1

```text
my-model.zip
  model.json
  weights.safetensors
  examples.json          # optional
```

`model.json` contains `package_version: 1`, a validated `ModelSpec`, `default_input`, and
`metadata` (`feature_names`, `output_names`, `input_note`). Extra fields are rejected.
`weights.safetensors` must have exactly the tensor IDs, shapes and F32 dtype in that spec.
Dense matrices use `[destination, source]`; execution is `X @ W.T + b`.

`examples.json`, when present, contains `inputs` and `outputs`, with 1–16 matching rows.
The importer checks output shapes and compares every output coordinate using absolute
tolerance `1e-5` and relative tolerance `1e-4`. Failure blocks import. These tolerances
cannot be overridden by an uploaded package. Passing examples checks agreement on those
examples, not accuracy, correctness of labels or equivalence on all possible inputs.
No examples produces an explicit **not supplied** result; schema/tensor/default-run checks
still apply. Exported saved revisions include a default-input round-trip example computed
by neuroLabs, not a new independent check against the original training framework.

## Bounds and execution

- Maximum upload: 6 MiB, streamed with a hard limit even without Content-Length.
- Only the three specified root entries are accepted. Nothing is extracted to disk.
  Encrypted, duplicate, path-containing and oversized entries are rejected. ZIP members
  may use stored or deflated compression.
- Per-entry uncompressed limits: 128 KiB JSON manifest, 4,100,000 bytes of safetensors,
  and 512 KiB verification JSON. Tensor headers/shapes/dtypes are checked before allocation.
- One package-validation subprocess at a time, with a 30-second deadline and process kill
  on timeout. At most four previews are retained, each for 15 minutes. Previews expire on
  server restart. The worker accepts data only and executes application-owned operators.
- Imported snapshot parameters use a bounded little-endian float32 response (at most
  4 MB) in spec tensor order. JSON parameter snapshots over 100,000 parameters are rejected.
  A tensor slice endpoint limits responses to at most 64 × 64 cells.
- Weight tables render at most 64 × 64 cells. Block selectors and coordinate navigation
  reach every entry. The graph shows every neuron with sampled connections by default.
  All-connections rendering is limited to 100,000 connections; larger models retain full
  layer-pair and selected-neuron views. Filtering never changes runtime computation.

The size/shape limits and import deadline are implemented. This remains a local preview:
general runtime jobs still use the API thread pool, browser abort does not cancel a running
forward pass, and this is not an OS memory/security sandbox. Cross-platform hardware and
memory profiling and a general cancellable execution worker remain later gates.

## API

- `POST /api/v1/imports/preview`: raw ZIP body, returns a compatibility preview and token.
- `POST /api/v1/imports/{token}`: commit the checked package as a new model.
- `GET /api/v1/import-example`: downloadable example ZIP.
- `GET /api/v1/models/{model}/revisions/{revision}/export`: saved checkpoint ZIP.
- `GET /api/v1/models/{model}/revisions/{revision}/parameter-data`: bounded binary snapshot.
- `GET /api/v1/models/{model}/revisions/{revision}/tensors/{tensor}`: bounded JSON slice;
  parameters `row`, `rows`, `column`, `columns`, with at most 64 rows/columns.

All mutations use the same local session/origin protection as parameter revisions.
Import metadata records the package SHA-256, input metadata and original verification
result. Subsequent parameter revisions do not inherit a claim that their modified weights
still agree with the original source examples.
