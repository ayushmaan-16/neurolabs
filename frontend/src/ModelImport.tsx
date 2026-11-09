import { useEffect, useRef, useState } from "react";
import { Upload, X } from "lucide-react";
import {
  api,
  parameterText,
  uploadPackage,
  type ImportPreview,
  type Model,
} from "./api";

export function ModelImport({
  onImported,
  disabled,
}: {
  onImported: (model: Model) => void;
  disabled: boolean;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const controller = useRef<AbortController | null>(null);
  const [preview, setPreview] = useState<ImportPreview | null>(null);
  const [phase, setPhase] = useState<"" | "checking" | "importing">("");
  const pending = phase !== "";
  const [dragging, setDragging] = useState(false);
  const [name, setName] = useState("");
  const [task, setTask] = useState("unknown");
  const [error, setError] = useState("");
  const [filename, setFilename] = useState("");
  const outputLayer = preview?.manifest.spec.operations
    .filter((op) => op.kind === "dense")
    .at(-1);
  const outputWidth = preview?.manifest.spec.tensors.find(
    (tensor) => tensor.id === outputLayer?.weight,
  )?.shape[0];
  useEffect(() => () => controller.current?.abort(), []);
  function close() {
    if (phase === "importing") return;
    controller.current?.abort();
    setPhase("");
    setDragging(false);
    dialog.current?.close();
  }
  async function check(file: File | undefined) {
    if (!file) return;
    controller.current?.abort();
    setFilename(file.name);
    setPreview(null);
    setError("");
    if (!/\.(zip|onnx)$/i.test(file.name) || file.size > 6 * 1024 * 1024) {
      setError(
        "Choose an .onnx model or a neuroLabs .zip package, up to 6 MiB.",
      );
      return;
    }
    const abort = new AbortController();
    controller.current = abort;
    setPhase("checking");
    try {
      const result = await uploadPackage(file, abort.signal);
      if (!abort.signal.aborted) {
        setPreview(result);
        setName(result.manifest.spec.name);
        setTask(result.manifest.spec.task);
      }
    } catch (error) {
      if (!abort.signal.aborted) setError((error as Error).message);
    } finally {
      if (!abort.signal.aborted) setPhase("");
    }
  }
  async function commit() {
    if (!preview || !name.trim()) return;
    setPhase("importing");
    setError("");
    try {
      const model = await api<Model>(`/imports/${preview.token}`, {
        name: name.trim(),
        task,
      });
      onImported(model);
      dialog.current?.close();
      setPreview(null);
    } catch (error) {
      setError((error as Error).message);
    } finally {
      setPhase("");
    }
  }
  return (
    <>
      <button
        className="import-trigger"
        disabled={disabled}
        onClick={() => {
          setError("");
          setPreview(null);
          setFilename("");
          dialog.current?.showModal();
        }}
      >
        <Upload size={14} /> Import model
      </button>
      <dialog
        ref={dialog}
        className="import-dialog"
        aria-labelledby="import-title"
        onCancel={(event) => {
          if (phase === "importing") event.preventDefault();
          else close();
        }}
      >
        <div className="import-title">
          <h2 id="import-title">Import a neural network</h2>
          <button
            aria-label="Close import"
            disabled={phase === "importing"}
            onClick={close}
          >
            <X size={18} />
          </button>
        </div>
        <p>
          Upload an ONNX model or a neuroLabs ZIP package. We detect the
          supported architecture and weights, then check compatibility. Your
          model runs locally.
        </p>
        <p className="import-limits">
          Sequential dense networks · up to 16 dense layers · widths up to 256 ·
          1 million parameters. ReLU, Tanh, Sigmoid, Identity and final Softmax.
        </p>
        <div
          role="region"
          aria-label="Model file drop zone"
          aria-disabled={pending}
          className={`import-dropzone${dragging ? " dragging" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            if (!pending) {
              event.dataTransfer.dropEffect = "copy";
              setDragging(true);
            }
          }}
          onDragLeave={(event) => {
            if (
              !(event.relatedTarget instanceof Node) ||
              !event.currentTarget.contains(event.relatedTarget)
            )
              setDragging(false);
          }}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            if (pending) return;
            if (event.dataTransfer.files.length !== 1) {
              setPreview(null);
              setError("Drop one .onnx model or .zip package at a time.");
              return;
            }
            void check(event.dataTransfer.files[0]);
          }}
        >
          <strong>Drop your model here</strong>
          <span>.onnx or .zip · up to 6 MiB · or browse below</span>
          <label className="import-file">
            Model file (.onnx or .zip)
            <input
              aria-label="Model package"
              type="file"
              accept=".onnx,.zip,application/zip"
              disabled={pending}
              onChange={(event) => {
                void check(event.target.files?.[0]);
                event.target.value = "";
              }}
            />
          </label>
        </div>
        {filename && <p>Selected package: {filename}</p>}
        <div className="import-help">
          <a href="/api/v1/import-example.onnx" download>
            Download ONNX example
          </a>
          <a href="/api/v1/import-example" download>
            Download example package
          </a>
          <a
            href="https://github.com/ayushmaan-16/neurolabs/blob/main/docs/model-import.md"
            target="_blank"
            rel="noreferrer"
          >
            Python exporter guide ↗
          </a>
        </div>
        <p className="import-example-note">
          Both examples use our analytic rectified curve. ONNX is checked
          against ONNX Runtime on generated probes; ZIP packages may supply
          their own examples.
        </p>
        {pending && (
          <p role="status">
            {phase === "checking" ? "Checking model…" : "Importing model…"}
          </p>
        )}
        {error && (
          <p className="import-error" role="alert">
            {error}
          </p>
        )}
        {preview && (
          <section
            className="import-preview"
            aria-label="Model compatibility preview"
          >
            <h3>{preview.manifest.spec.name}</h3>
            <p>
              {preview.source_format === "onnx"
                ? "ONNX model · architecture detected automatically"
                : "neuroLabs ZIP package"}
            </p>
            <div className="import-options">
              <label>
                Model name
                <input
                  value={name}
                  maxLength={120}
                  onChange={(event) => setName(event.target.value)}
                  disabled={pending}
                />
              </label>
              <label>
                Model task
                <select
                  aria-label="Model task"
                  value={task}
                  onChange={(event) => setTask(event.target.value)}
                  disabled={pending}
                >
                  <option value="unknown">Not specified</option>
                  <option value="classification">Classification</option>
                  <option value="regression">Regression</option>
                </select>
              </label>
            </div>
            {preview.source_format === "onnx" && (
              <p>
                Choose a task only if you know it from the model's
                documentation. Supply inputs with the original preprocessing and
                feature order.
              </p>
            )}
            <dl>
              <div>
                <dt>Parameters</dt>
                <dd>{preview.parameter_count.toLocaleString()}</dd>
              </div>
              <div>
                <dt>Inputs</dt>
                <dd>{preview.manifest.spec.input_width}</dd>
              </div>
              <div>
                <dt>Outputs</dt>
                <dd>{outputWidth}</dd>
              </div>
              <div>
                <dt>Dense layers</dt>
                <dd>
                  {
                    preview.manifest.spec.operations.filter(
                      (op) => op.kind === "dense",
                    ).length
                  }
                </dd>
              </div>
            </dl>
            <div className="import-layers">
              <table>
                <thead>
                  <tr>
                    <th>Operation</th>
                    <th>Function</th>
                    <th>Shape</th>
                  </tr>
                </thead>
                <tbody>
                  {preview.manifest.spec.operations.map((op) => (
                    <tr key={op.id}>
                      <td>{op.id}</td>
                      <td>{op.kind}</td>
                      <td>
                        {op.weight
                          ? preview.manifest.spec.tensors
                              .find((t) => t.id === op.weight)
                              ?.shape.join(" × ")
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p>
              {preview.verification.status === "passed"
                ? preview.verification.method === "onnxruntime_probes"
                  ? `ONNX Runtime verification passed on ${preview.verification.sample_count} generated probes. Maximum absolute error: ${preview.verification.max_abs_error}. This checks the conversion, not prediction accuracy or agreement with the original training code.`
                  : `Output verification passed on ${preview.verification.sample_count} supplied examples. Maximum absolute error: ${preview.verification.max_abs_error}.`
                : "No original-model example outputs supplied. Shapes and a default forward pass were checked; source-model numerical equivalence has not been verified."}
            </p>
            <p>{preview.manifest.metadata.input_note}</p>
            <details className="import-provenance">
              <summary>File fingerprints</summary>
              <small>Package SHA-256: {preview.sha256}</small>
              {preview.source_sha256 && (
                <small>Source ONNX SHA-256: {preview.source_sha256}</small>
              )}
            </details>
            <button
              className="import-confirm"
              disabled={pending || !name.trim()}
              onClick={() => void commit()}
            >
              Import into workspace
            </button>
          </section>
        )}
      </dialog>
    </>
  );
}

export function ImportedInputs({
  model,
  values,
  onChange,
}: {
  model: Model;
  values: number[];
  onChange: (values: number[]) => void;
}) {
  const [text, setText] = useState(values.map(parameterText).join(", "));
  const [error, setError] = useState("");
  useEffect(() => {
    setText(values.map(parameterText).join(", "));
    setError("");
  }, [JSON.stringify(values), model.spec.id]);
  return (
    <form
      className="imported-inputs"
      onSubmit={(event) => {
        event.preventDefault();
        const parts = text.split(/[,\s]+/).filter(Boolean);
        const next = parts.map(Number);
        if (
          next.length !== model.spec.input_width ||
          next.some((value) => !Number.isFinite(Math.fround(value)))
        ) {
          setError(
            `Enter ${model.spec.input_width} finite numbers in feature order.`,
          );
          return;
        }
        setError("");
        onChange(next);
      }}
    >
      <label htmlFor="imported-vector">
        Input vector · {model.spec.input_width} values
      </label>
      <textarea
        id="imported-vector"
        value={text}
        onChange={(event) => setText(event.target.value)}
        spellCheck={false}
      />
      {model.import_info?.metadata.feature_names.length ? (
        <small>
          Order: {model.import_info.metadata.feature_names.join(", ")}
        </small>
      ) : null}
      <button type="submit">Apply input</button>
      {error && <p role="alert">{error}</p>}
    </form>
  );
}
