import { useEffect, useRef, useState } from "react";
import { api, type Model, type Revision } from "./api";
import { type Audit, metric, parseBatch, probeBatch } from "./audit-data";
import type { Change } from "./draft-state";

export function Diagnostics({
  model,
  history,
  onNeuron,
  onCase,
  changes,
  blocked = false,
}: {
  model: Model;
  history: Revision[];
  onNeuron: (layer: string, index: number) => void;
  onCase: (inputs: number[]) => void;
  changes: Change[];
  blocked?: boolean;
}) {
  const [draft, setDraft] = useState(() => probeBatch(model));
  const [source, setSource] = useState("Generated probe set");
  const [baseline, setBaseline] = useState(model.original);
  const [report, setReport] = useState<Audit | null>(null);
  const [analyzedKey, setAnalyzedKey] = useState("");
  const changesKey = JSON.stringify(changes);
  const requestKey = JSON.stringify([
    model.head,
    baseline,
    draft,
    changesKey,
    blocked,
  ]);
  const [layerId, setLayerId] = useState("");
  const [filter, setFilter] = useState("all");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const controller = useRef<AbortController | null>(null);
  const fileVersion = useRef(0);
  useEffect(() => {
    controller.current?.abort();
    setPending(false);
    setReport(null);
    setError("");
  }, [model.head, baseline, draft, changesKey, blocked]);
  useEffect(
    () => () => {
      controller.current?.abort();
      fileVersion.current++;
    },
    [],
  );
  const current = report && analyzedKey === requestKey ? report : null;
  const layer =
    current?.layers.find((item) => item.id === layerId) ?? current?.layers[0];
  const neurons =
    layer?.neurons.filter((neuron) =>
      filter === "inactive"
        ? neuron.inactive_on_batch
        : filter === "saturated"
          ? (neuron.saturation_fraction ?? 0) >= 0.9
          : true,
    ) ?? [];
  const baselines = [
    model.original,
    baseline,
    ...history.map((revision) => revision.id),
  ].filter((value, index, all) => all.indexOf(value) === index);

  async function analyze() {
    if (blocked) return;
    const abort = new AbortController();
    controller.current?.abort();
    controller.current = abort;
    setError("");
    setReport(null);
    try {
      const inputs = parseBatch(draft, model.spec.input_width);
      setPending(true);
      const result = await api<Audit>(
        `/models/${model.spec.id}/diagnostics`,
        {
          revision_id: model.head,
          baseline_revision_id: baseline,
          inputs,
          changes,
        },
        abort.signal,
      );
      if (!abort.signal.aborted) {
        setReport(result);
        setAnalyzedKey(requestKey);
        setFilter("all");
      }
    } catch (error) {
      if (!abort.signal.aborted) setError((error as Error).message);
    } finally {
      if (!abort.signal.aborted) setPending(false);
    }
  }
  async function load(file: File | undefined) {
    if (!file) return;
    const version = ++fileVersion.current;
    if (file.size > 256 * 1024) {
      setError("Input CSV must be at most 256 KiB.");
      return;
    }
    try {
      const text = await file.text();
      if (version !== fileVersion.current) return;
      parseBatch(text, model.spec.input_width);
      setDraft(text);
      setSource(file.name);
      setError("");
    } catch (error) {
      if (version === fileVersion.current) setError((error as Error).message);
    }
  }
  function download() {
    if (!current) return;
    const url = URL.createObjectURL(
      new Blob(
        [JSON.stringify({ ...current, input_source: source }, null, 2)],
        { type: "application/json" },
      ),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `neurolabs-${model.spec.id}-${model.head.slice(0, 8)}${changes.length ? "-draft" : ""}-audit.json`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <section className="diagnostics" aria-label="Batch diagnostics">
      <div className="audit-heading">
        <div>
          <h3>Test behavior across inputs</h3>
          <p>
            Audit activations and compare saved revisions on the same batch.
          </p>
        </div>
      </div>
      <div className="audit-inputs">
        <div>
          <label htmlFor="audit-inputs">
            Input batch · {model.spec.input_width}{" "}
            {model.spec.input_width === 1 ? "value" : "values"} per row
          </label>
          <textarea
            id="audit-inputs"
            value={draft}
            spellCheck={false}
            onChange={(event) => {
              fileVersion.current++;
              setDraft(event.target.value);
              setSource("Custom inputs");
            }}
          />
          <p>
            CSV numbers only, without headers or targets. 2–256 rows. {source}.
          </p>
          <div className="audit-actions">
            <label className="audit-file">
              Load CSV
              <input
                aria-label="Load input CSV"
                type="file"
                accept=".csv,text/csv"
                onChange={(event) => {
                  void load(event.target.files?.[0]);
                  event.target.value = "";
                }}
              />
            </label>
            <button
              onClick={() => {
                fileVersion.current++;
                setDraft(probeBatch(model));
                setSource("Generated probe set");
              }}
            >
              Use probe set
            </button>
          </div>
        </div>
        <div className="audit-settings">
          <label htmlFor="audit-baseline">
            {changes.length
              ? "Compare unsaved draft with"
              : "Compare current revision with"}
          </label>
          <select
            id="audit-baseline"
            value={baseline}
            onChange={(event) => setBaseline(event.target.value)}
          >
            {baselines.map((id) => (
              <option key={id} value={id}>
                {id === model.original ? "Original" : "Revision"} ·{" "}
                {id.slice(0, 8)}
                {id === model.head ? " · current" : ""}
              </option>
            ))}
          </select>
          <p>
            {source === "Generated probe set"
              ? "Generated probes exercise the model; they are not a representative dataset. Load validation inputs for an engineering audit."
              : "Use inputs with the same preprocessing and feature order as the model. Reports describe this supplied batch."}
          </p>
          <button
            className="audit-run"
            onClick={() => void analyze()}
            disabled={pending || blocked}
          >
            {pending ? "Analyzing…" : "Analyze batch"}
          </button>
          <small>
            Changes to staged parameters or inputs invalidate the report. Re-run
            to refresh it.
          </small>
        </div>
      </div>
      {error && (
        <p role="alert" className="audit-error">
          {error}
        </p>
      )}
      {current && (
        <div className="audit-results">
          <div className="audit-result-heading">
            <strong>{current.sample_count} inputs analyzed</strong>
            <button onClick={download}>Download report</button>
          </div>
          <div className="audit-metrics">
            <div>
              <span>ReLU neurons zero on every input</span>
              <strong data-testid="audit-inactive">
                {current.summary.inactive_relu_neurons}
              </strong>
            </div>
            <div>
              <span>Mean absolute output change</span>
              <strong data-testid="audit-output-mean">
                {metric(current.summary.mean_abs_output_change)}
              </strong>
            </div>
            <div>
              <span>Largest absolute output change</span>
              <strong data-testid="audit-output-max">
                {metric(current.summary.max_abs_output_change)}
              </strong>
            </div>
            {current.summary.argmax_flips !== null && (
              <div>
                <span>Changed argmax predictions</span>
                <strong>
                  {current.summary.argmax_flips} / {current.sample_count}
                </strong>
              </div>
            )}
          </div>
          <p className="audit-scope">
            These observations apply to this batch. Zero ReLU outputs do not
            prove a neuron is globally dead. Output changes and argmax flips do
            not establish accuracy or explain a root cause.
          </p>
          <h4>Where does behavior change?</h4>
          <p>
            RMS measures activation magnitude. Change RMS compares matching
            neurons between the two revisions.
          </p>
          <div className="audit-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Layer</th>
                  <th>Function</th>
                  <th>Baseline RMS</th>
                  <th>Current RMS</th>
                  <th>Change RMS</th>
                  <th>Inactive ReLU</th>
                </tr>
              </thead>
              <tbody>
                {current.layers.map((item) => (
                  <tr
                    key={item.id}
                    className={layer?.id === item.id ? "audit-selected" : ""}
                  >
                    <td>
                      <button
                        aria-label={`Audit layer ${item.id}`}
                        onClick={() => {
                          setLayerId(item.id);
                          setFilter("all");
                        }}
                      >
                        {item.id}
                      </button>
                    </td>
                    <td>{item.activation}</td>
                    <td>{metric(item.baseline_rms)}</td>
                    <td>{metric(item.rms)}</td>
                    <td>{metric(item.delta_rms)}</td>
                    <td>
                      {item.activation === "relu"
                        ? `${item.inactive_neurons} / ${item.width}`
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {layer && (
            <>
              <div className="audit-neuron-heading">
                <h4>{layer.id} · neuron audit</h4>
                <label>
                  Show
                  <select
                    aria-label="Neuron audit filter"
                    value={filter}
                    onChange={(event) => setFilter(event.target.value)}
                  >
                    <option value="all">All neurons</option>
                    <option value="inactive">
                      Inactive ReLU on this batch
                    </option>
                    <option value="saturated">
                      Saturated on ≥90% of inputs
                    </option>
                  </select>
                </label>
              </div>
              <p>
                Saturation: |Tanh| ≥ 0.99; Sigmoid ≤ 0.01 or ≥ 0.99. This is
                activation saturation, not a measurement of training gradients.
              </p>
              <div className="audit-table-wrap audit-neurons">
                <table>
                  <thead>
                    <tr>
                      <th>Neuron</th>
                      <th>Range</th>
                      <th>Mean ± std</th>
                      <th>Zero %</th>
                      <th>Saturated %</th>
                      <th>Change RMS</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {neurons.map((item) => (
                      <tr
                        key={item.index}
                        className={
                          item.inactive_on_batch ? "audit-flagged" : ""
                        }
                      >
                        <td>{item.index}</td>
                        <td>
                          {metric(item.min)} … {metric(item.max)}
                        </td>
                        <td>
                          {metric(item.mean)} ± {metric(item.std)}
                        </td>
                        <td>{metric(item.zero_fraction * 100)}</td>
                        <td>
                          {item.saturation_fraction === null
                            ? "—"
                            : metric(item.saturation_fraction * 100)}
                        </td>
                        <td>{metric(item.delta_rms)}</td>
                        <td>
                          <button
                            aria-label={`Inspect audited ${layer.id} neuron ${item.index}`}
                            onClick={() => onNeuron(layer.id, item.index)}
                          >
                            Inspect in 3D
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {neurons.length === 0 && (
                  <p>No neurons match this filter on the supplied batch.</p>
                )}
              </div>
            </>
          )}
          <h4>Inputs with the largest output changes</h4>
          <p>
            Showing up to 10 cases. Inspect a case to run its exact inputs in
            the graph.
          </p>
          <div className="audit-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Input row</th>
                  <th>Baseline output</th>
                  <th>Current output</th>
                  <th>Largest change</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {[...current.cases]
                  .sort((a, b) => b.max_abs_change - a.max_abs_change)
                  .slice(0, 10)
                  .map((item) => (
                    <tr key={item.index}>
                      <td>{item.index + 1}</td>
                      <td>{item.baseline_output.map(metric).join(", ")}</td>
                      <td>{item.current_output.map(metric).join(", ")}</td>
                      <td>{metric(item.max_abs_change)}</td>
                      <td>
                        <button
                          aria-label={`Inspect input row ${item.index + 1}`}
                          onClick={() => onCase(item.inputs)}
                        >
                          Inspect in 3D
                        </button>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <p className="audit-provenance">
            {changes.length
              ? `Unsaved draft of ${model.head.slice(0, 8)}`
              : `Current ${current.revision_id.slice(0, 8)}`}{" "}
            · baseline {current.baseline_revision_id.slice(0, 8)} · input hash{" "}
            {current.input_hash.slice(0, 12)}. Export includes the full batch,
            per-neuron statistics, revision IDs and runtime provenance.
          </p>
        </div>
      )}
    </section>
  );
}
