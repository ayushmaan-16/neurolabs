import { useEffect, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  Box,
  Check,
  ChevronRight,
  CircleDot,
  FlaskConical,
  GitBranch,
  Grid2X2,
  Layers3,
  LoaderCircle,
  Network as NetworkIcon,
  RotateCcw,
  Terminal,
  X,
} from "lucide-react";
import {
  api,
  number,
  parameterText,
  type Model,
  type Parameters,
  type Revision,
  type Run,
} from "./api";
import { Network3D } from "./Network3D";
import { Plot } from "./Plot";
import { Inspector } from "./Inspector";

export function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [activeId, setActiveId] = useState("curve");
  const [selected, setSelected] = useState("hidden");
  const [selectionKind, setSelectionKind] = useState<"neuron" | "edge">(
    "neuron",
  );
  const [inputNeuron, setInputNeuron] = useState(0);
  const [tab, setTab] = useState<"architecture" | "weights">("architecture");
  const [parameters, setParameters] = useState<Parameters | null>(null);
  const [baseline, setBaseline] = useState<Parameters | null>(null);
  const [history, setHistory] = useState<Revision[]>([]);
  const [inputs, setInputs] = useState<number[]>([]);
  const [run, setRun] = useState<Run | null>(null);
  const [sweep, setSweep] = useState<Run | null>(null);
  const [originalSweep, setOriginalSweep] = useState<Run | null>(null);
  const [tensorId, setTensorId] = useState("hidden.weight");
  const [coordinate, setCoordinate] = useState<number[]>([0, 0]);
  const [editValue, setEditValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const model = models.find((item) => item.spec.id === activeId);
  const currentParameters =
    parameters?.revision_id === model?.head ? parameters : null;
  const currentRun =
    run?.revision_id === model?.head &&
    run?.inputs[0].length === inputs.length &&
    run?.inputs[0].every((value, index) => value === Math.fround(inputs[index]))
      ? run
      : null;
  const currentSweep =
    sweep?.revision_id === model?.head &&
    sweep?.inputs[0]
      .slice(1)
      .every((value, index) => value === Math.fround(inputs[index + 1]))
      ? sweep
      : null;
  const operation = model?.spec.operations.find((op) => op.id === selected);
  const parameter = currentParameters?.tensors[tensorId];
  const originalParameter =
    baseline && baseline.revision_id === model?.original
      ? baseline.tensors[tensorId]
      : undefined;
  const matrix = parameter
    ? parameter.shape.length === 1
      ? [parameter.values as number[]]
      : (parameter.values as number[][])
    : [];
  const originalMatrix = originalParameter
    ? originalParameter.shape.length === 1
      ? [originalParameter.values as number[]]
      : (originalParameter.values as number[][])
    : [];
  const totalParameters =
    model?.spec.tensors.reduce(
      (sum, t) => sum + t.shape.reduce((a, b) => a * b, 1),
      0,
    ) ?? 0;

  const refresh = async () => setModels(await api<Model[]>("/models"));
  useEffect(() => {
    void refresh().catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!model) return;
    setInputs(model.default_input);
    setSelected(model.spec.operations[0].id);
    setSelectionKind("neuron");
    const first = model.spec.operations.find((op) => op.kind === "dense")!;
    const initial = first.bias ?? first.weight!;
    setTensorId(initial);
    setCoordinate(
      model.spec.tensors.find((t) => t.id === initial)!.shape.map(() => 0),
    );
    setRun(null);
    setSweep(null);
    setOriginalSweep(null);
    setHistory([]);
  }, [model?.spec.id]);

  useEffect(() => {
    if (!model) return;
    const controller = new AbortController();
    const prefix = `/models/${model.spec.id}`;
    Promise.all([
      api<Parameters>(
        `${prefix}/revisions/${model.head}/parameters`,
        undefined,
        controller.signal,
      ),
      api<Parameters>(
        `${prefix}/revisions/${model.original}/parameters`,
        undefined,
        controller.signal,
      ),
      api<Revision[]>(`${prefix}/revisions`, undefined, controller.signal),
    ])
      .then(([values, original, revisions]) => {
        setParameters(values);
        setBaseline(original);
        setHistory(revisions);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      });
    return () => controller.abort();
  }, [model?.spec.id, model?.head]);

  useEffect(() => {
    if (!model || inputs.length !== model.spec.input_width) return;
    const controller = new AbortController();
    setRunning(true);
    const timer = setTimeout(() => {
      const rows = Array.from({ length: 65 }, (_, i) => [
        -2 + i / 16,
        ...inputs.slice(1),
      ]);
      const route = `/models/${model.spec.id}/runs`;
      Promise.all([
        api<Run>(
          route,
          {
            revision_id: model.head,
            inputs: [inputs],
            captures: [
              model.spec.input_name,
              ...model.spec.operations.map((op) => op.output),
            ],
          },
          controller.signal,
        ),
        api<Run>(
          route,
          { revision_id: model.head, inputs: rows },
          controller.signal,
        ),
        api<Run>(
          route,
          { revision_id: model.original, inputs: rows },
          controller.signal,
        ),
      ])
        .then(([result, curve, original]) => {
          if (!controller.signal.aborted) {
            setRun(result);
            setSweep(curve);
            setOriginalSweep(original);
          }
        })
        .catch((e) => {
          if (!controller.signal.aborted) setError(e.message);
        })
        .finally(() => {
          if (!controller.signal.aborted) setRunning(false);
        });
    }, 150);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [model?.spec.id, model?.head, JSON.stringify(inputs)]);

  const selectedValue = parameter
    ? coordinate.length === 1
      ? (parameter.values as number[])[coordinate[0]]
      : (parameter.values as number[][])[coordinate[0]]?.[coordinate[1]]
    : undefined;
  useEffect(() => {
    setEditValue(parameterText(selectedValue));
  }, [selectedValue, tensorId, coordinate.join(",")]);

  function selectNode(id: string, index = 0) {
    setSelected(id);
    setSelectionKind("neuron");
    if (id === "input") setInputNeuron(index);
    const op = model?.spec.operations.find((item) => item.id === id);
    if (op?.bias) {
      setTensorId(op.bias);
      setCoordinate([index]);
    } else if (op?.weight) {
      setTensorId(op.weight);
      setCoordinate([index, 0]);
    }
  }
  function selectParameter(id: string, index: number[]) {
    setSelectionKind(index.length === 2 ? "edge" : "neuron");
    const owner = model?.spec.operations.find(
      (op) => op.weight === id || op.bias === id,
    );
    if (owner) setSelected(owner.id);
    setTensorId(id);
    setCoordinate(index);
  }
  async function applyEdit() {
    if (
      !model ||
      !Number.isFinite(Number(editValue)) ||
      editValue.trim() === ""
    )
      return;
    setBusy(true);
    setError("");
    try {
      await api(`/models/${model.spec.id}/revisions`, {
        base_revision: model.head,
        idempotency_key: crypto.randomUUID(),
        changes: [
          { tensor_id: tensorId, index: coordinate, value: Number(editValue) },
        ],
      });
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function restore(revision: string) {
    if (!model) return;
    setBusy(true);
    setError("");
    try {
      await api(`/models/${model.spec.id}/checkout`, {
        expected_head: model.head,
        revision_id: revision,
      });
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const dense = operation?.kind === "dense" ? operation : undefined;
  const neuron =
    tensorId === dense?.weight || tensorId === dense?.bias ? coordinate[0] : 0;

  return (
    <div className="app-shell">
      <header className="app-header">
        <a className="brand" href="/">
          <div className="brand-icon">
            <NetworkIcon size={22} />
          </div>
          <span>
            neuro<span className="brand-light">Labs</span>
          </span>
          <span className="alpha">OBSERVATORY</span>
        </a>
        <div className="header-center">
          <span className="status-dot" />
          Local laboratory<span className="header-separator">/</span>
          <span>Dense networks</span>
        </div>
        <a
          className="api-link"
          href="http://127.0.0.1:8000/docs"
          target="_blank"
          rel="noreferrer"
        >
          <Terminal size={14} /> API reference <ArrowUpRight size={13} />
        </a>
      </header>
      <div className="workspace">
        <aside className="sidebar">
          <div className="sidebar-heading">
            <span>WORKSPACE</span>
            <FlaskConical size={15} />
          </div>
          <h1>
            Network
            <br />
            observatory.
          </h1>
          <p className="sidebar-description">
            Inspect a connection.
            <br />
            Trace the result.
          </p>
          <div className="section-label">
            SAMPLE COLLECTION{" "}
            <span>{String(models.length).padStart(2, "0")}</span>
          </div>
          <nav aria-label="Sample models" className="sample-list">
            {models.map((item, i) => (
              <button
                key={item.spec.id}
                disabled={busy}
                className={`sample-card ${item.spec.id === activeId ? "active" : ""}`}
                onClick={() => {
                  setActiveId(item.spec.id);
                  setError("");
                }}
              >
                <div className="sample-icon">
                  {i === 0 ? (
                    <ArrowUpRight size={18} />
                  ) : i === 1 ? (
                    <Activity size={18} />
                  ) : (
                    <Grid2X2 size={18} />
                  )}
                </div>
                <span>
                  <strong>{item.spec.name}</strong>
                  <small>
                    {item.spec.input_width} input
                    {item.spec.input_width > 1 ? "s" : ""} ·{" "}
                    {
                      item.spec.operations.filter((op) => op.kind === "dense")
                        .length
                    }{" "}
                    dense layer
                    {item.spec.operations.filter((op) => op.kind === "dense")
                      .length > 1
                      ? "s"
                      : ""}
                  </small>
                </span>
                <ChevronRight size={13} />
              </button>
            ))}
          </nav>
          <div className="section-label tree-heading">
            ARCHITECTURE <Layers3 size={13} />
          </div>
          <nav className="layer-list" aria-label="Architecture tree">
            {model && (
              <>
                <button
                  onClick={() => selectNode("input")}
                  className={selected === "input" ? "chosen" : ""}
                >
                  <CircleDot size={14} />
                  <span>Input</span>
                  <code>{model.spec.input_width}</code>
                </button>
                {model.spec.operations.map((op) => (
                  <button
                    key={op.id}
                    onClick={() => selectNode(op.id)}
                    className={selected === op.id ? "chosen" : ""}
                  >
                    {op.kind === "dense" ? (
                      <Box size={14} />
                    ) : (
                      <Activity size={14} />
                    )}
                    <span>{op.id}</span>
                    <code>{op.kind}</code>
                  </button>
                ))}
              </>
            )}
          </nav>
          <div className="sidebar-note">
            <strong>Local execution</strong>
            <p>
              Parameters stay on this machine. Every edit is saved as a separate
              revision.
            </p>
          </div>
          <div className="sidebar-footer">
            <span className="status-dot" />
            PyTorch · float32 · CPU
          </div>
        </aside>

        <main className="main-panel">
          {error && (
            <div className="error-banner" role="alert">
              <span>{error}</span>
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          {!model ? (
            <div className="empty-state">
              <LoaderCircle className="spin" />
              <h2>Connecting to your laboratory</h2>
              <p>Start the Python backend on port 8000.</p>
              <button
                onClick={() => void refresh().catch((e) => setError(e.message))}
              >
                Retry connection
              </button>
            </div>
          ) : (
            <>
              <section className="model-heading">
                <div>
                  <div className="breadcrumb">
                    Collection <ChevronRight size={12} /> {model.spec.task}
                  </div>
                  <h2>{model.spec.name}</h2>
                  <p>{model.spec.description}</p>
                </div>
                <button
                  className="secondary-button"
                  disabled={busy || model.head === model.original}
                  onClick={() => void restore(model.original)}
                >
                  <RotateCcw size={14} /> Restore original
                </button>
              </section>
              <div className="model-meta">
                <span>
                  <Box size={13} />
                  {totalParameters.toLocaleString()} parameters
                </span>
                <span>
                  <Layers3 size={13} />
                  {
                    model.spec.operations.filter((op) => op.kind === "dense")
                      .length
                  }{" "}
                  dense layers
                </span>
                <span>
                  <GitBranch size={13} />
                  {model.head.slice(0, 8)}
                </span>
                <span
                  className={model.head === model.original ? "" : "modified"}
                >
                  {model.head === model.original
                    ? "Original checkpoint"
                    : "Modified checkpoint"}
                </span>
              </div>
              <section className="canvas-card">
                <div className="canvas-toolbar">
                  <div className="view-tabs">
                    <button
                      className={tab === "architecture" ? "active" : ""}
                      onClick={() => setTab("architecture")}
                    >
                      <NetworkIcon size={14} />
                      Architecture
                    </button>
                    <button
                      className={tab === "weights" ? "active" : ""}
                      onClick={() => {
                        setTab("weights");
                        const weight =
                          operation?.weight ??
                          model.spec.tensors.find((t) => t.shape.length === 2)!
                            .id;
                        selectParameter(weight, [0, 0]);
                      }}
                    >
                      <Grid2X2 size={14} />
                      Weights
                    </button>
                  </div>
                  <span className="view-hint">
                    {tab === "architecture"
                      ? "3D neuron graph"
                      : "Select a cell to edit"}
                  </span>
                </div>
                <div className="canvas-content">
                  {tab === "architecture" ? (
                    <Network3D
                      key={model.spec.id}
                      model={model}
                      parameters={currentParameters}
                      selection={{ tensorId, index: coordinate }}
                      selectionKind={selectionKind}
                      onParameterSelect={selectParameter}
                      run={currentRun}
                      selected={selected}
                      neuron={selected === "input" ? inputNeuron : neuron}
                      onSelect={(id, index) => {
                        selectNode(id, index);
                      }}
                    />
                  ) : (
                    <div className="matrix-view">
                      <div className="section-label">
                        PARAMETER MATRIX{" "}
                        <code>[{parameter?.shape.join(" × ")}]</code>
                      </div>
                      <label className="sr-only" htmlFor="tensor-select">
                        Tensor
                      </label>
                      <select
                        id="tensor-select"
                        value={tensorId}
                        onChange={(event) =>
                          selectParameter(
                            event.target.value,
                            model.spec.tensors
                              .find((t) => t.id === event.target.value)!
                              .shape.map(() => 0),
                          )
                        }
                      >
                        {model.spec.tensors.map((t) => (
                          <option key={t.id} value={t.id}>
                            {t.id}
                          </option>
                        ))}
                      </select>
                      <div className="large-matrix">
                        {matrix.map((row, r) => (
                          <div className="matrix-row" key={r}>
                            {row.map((value, c) => {
                              const index =
                                parameter!.shape.length === 1 ? [c] : [r, c];
                              return (
                                <button
                                  key={c}
                                  className={`${value < 0 ? "negative" : "positive"} ${coordinate.join() === index.join() ? "selected" : ""} ${originalMatrix[r]?.[c] !== value ? "changed" : ""}`}
                                  onClick={() =>
                                    selectParameter(tensorId, index)
                                  }
                                  aria-label={`${tensorId} [${index.join(", ")}] = ${value}`}
                                >
                                  <small>[{index.join(", ")}]</small>
                                  <strong>{number(value)}</strong>
                                </button>
                              );
                            })}
                          </div>
                        ))}
                      </div>
                      <p>
                        Rows = output features · columns = input features
                        <br />
                        Bias tensors use one feature index.
                      </p>
                    </div>
                  )}
                </div>
                <div className="canvas-footer">
                  <span>
                    <span className="legend-dot mint" />
                    Positive / zero <span className="legend-dot peach" />
                    Negative
                  </span>
                  <span>
                    Nodes: activations · edges: signed weighted contributions
                  </span>
                </div>
              </section>
              <section className="experiment-panel">
                <div className="panel-title">
                  <div>
                    <span className="section-label">LIVE EXPERIMENT</span>
                    <h3>Input → response</h3>
                  </div>
                  <span className="run-status" aria-live="polite">
                    {running ? (
                      <>
                        <LoaderCircle size={12} className="spin" />
                        Computing
                      </>
                    ) : (
                      <>
                        <Check size={13} />
                        {number(currentRun?.elapsed_ms)} ms
                      </>
                    )}
                  </span>
                </div>
                <div className="experiment-grid">
                  <div className="inputs">
                    <div className="section-label">INPUT VECTOR</div>
                    {inputs.map((value, index) => (
                      <label className="input-control" key={index}>
                        <span>
                          <span>
                            Input{inputs.length > 1 ? ` ${index + 1}` : ""}
                          </span>
                          <code>{number(value)}</code>
                        </span>
                        <input
                          aria-label={`Input ${index}`}
                          type="range"
                          min="-2"
                          max="2"
                          step="0.01"
                          value={value}
                          onChange={(event) =>
                            setInputs(
                              inputs.map((v, i) =>
                                i === index ? Number(event.target.value) : v,
                              ),
                            )
                          }
                        />
                        <div className="range-labels">
                          <span>−2</span>
                          <span>2</span>
                        </div>
                      </label>
                    ))}
                    <div className="output-values">
                      <span className="section-label">
                        {model.spec.task === "classification"
                          ? "OUTPUT LOGITS"
                          : "OUTPUT VECTOR"}
                      </span>
                      <strong>
                        {currentRun?.outputs[0].map(number).join(" , ") ?? "—"}
                      </strong>
                    </div>
                  </div>
                  <div className="plot-area">
                    <div className="plot-caption">
                      <span>
                        Response across input
                        {inputs.length > 1
                          ? ` · other inputs held at ${inputs.slice(1).map(number).join(", ")}`
                          : ""}
                      </span>
                      <div>
                        <span className="legend-line" />
                        Current <span className="legend-line dashed" />
                        Original
                      </div>
                    </div>
                    <Plot current={currentSweep} baseline={originalSweep} />
                  </div>
                </div>
              </section>
              <footer className="main-footer">
                <span>Inspectable computation. Reproducible change.</span>
                <span>neuroLabs · foundation preview</span>
              </footer>
            </>
          )}
        </main>

        <Inspector
          model={model}
          operation={operation}
          kind={selectionKind}
          neuron={neuron}
          inputNeuron={inputNeuron}
          parameters={currentParameters}
          run={currentRun}
          coordinate={coordinate}
          draft={editValue}
          busy={busy}
          history={history}
          onDraft={setEditValue}
          onApply={applyEdit}
          onSelectParameter={selectParameter}
          onRestore={restore}
        />
      </div>
    </div>
  );
}
