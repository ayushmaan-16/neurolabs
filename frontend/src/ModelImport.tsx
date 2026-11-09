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
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [filename, setFilename] = useState("");
  useEffect(() => () => controller.current?.abort(), []);
  function close() {
    if (!pending) dialog.current?.close();
  }
  async function check(file: File | undefined) {
    if (!file) return;
    setFilename(file.name);
    setPreview(null);
    setError("");
    if (
      !file.name.toLowerCase().endsWith(".zip") ||
      file.size > 6 * 1024 * 1024
    ) {
      setError(
        "Choose a neuroLabs .zip package, up to 6 MiB. Use the Python exporter for your model.",
      );
      return;
    }
    const abort = new AbortController();
    controller.current?.abort();
    controller.current = abort;
    setPending(true);
    try {
      setPreview(await uploadPackage(file, abort.signal));
    } catch (error) {
      if (!abort.signal.aborted) setError((error as Error).message);
    } finally {
      if (!abort.signal.aborted) setPending(false);
    }
  }
  async function commit() {
    if (!preview) return;
    setPending(true);
    setError("");
    try {
      const model = await api<Model>(`/imports/${preview.token}`, {});
      onImported(model);
      dialog.current?.close();
      setPreview(null);
    } catch (error) {
      setError((error as Error).message);
    } finally {
      setPending(false);
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
          if (pending) event.preventDefault();
        }}
      >
        <div className="import-title">
          <h2 id="import-title">Import a neural network</h2>
          <button aria-label="Close import" disabled={pending} onClick={close}>
            <X size={18} />
          </button>
        </div>
        <p>
          Upload a neuroLabs package containing the architecture and float32
          weights. Your model runs locally.
        </p>
        <p className="import-limits">
          Sequential dense networks · up to 16 dense layers · widths up to 256 ·
          1 million parameters. ReLU, Tanh, Sigmoid, Identity and final Softmax.
        </p>
        <label className="import-file">
          Model package (.zip)
          <input
            aria-label="Model package"
            type="file"
            accept=".zip,application/zip"
            disabled={pending}
            onChange={(event) => {
              void check(event.target.files?.[0]);
              event.target.value = "";
            }}
          />
        </label>
        {filename && <p>Selected package: {filename}</p>}
        <div className="import-help">
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
          The example contains our analytic rectified curve and verification
          inputs. It demonstrates the upload workflow.
        </p>
        {pending && <p role="status">Checking model…</p>}
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
                <dt>Task</dt>
                <dd>{preview.manifest.spec.task}</dd>
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
                ? `Output verification passed on ${preview.verification.sample_count} supplied examples. Maximum absolute error: ${preview.verification.max_abs_error}.`
                : "No original-model example outputs supplied. Shapes and a default forward pass were checked; source-model numerical equivalence has not been verified."}
            </p>
            <p>{preview.manifest.metadata.input_note}</p>
            <small>Package SHA-256: {preview.sha256}</small>
            <button
              className="import-confirm"
              disabled={pending}
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
