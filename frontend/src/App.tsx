import { useEffect, useMemo, useState } from "react";
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
import { Diagnostics } from "./Diagnostics";
import { Weights } from "./Weights";
import {
  type Draft,
  type Change,
  cellKey,
  draftStorageKey,
  overlay,
  readDrafts,
} from "./draft-state";

const noChanges: Change[] = [];

export function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [activeId, setActiveId] = useState("curve");
  const [selected, setSelected] = useState("hidden");
  const [selectionKind, setSelectionKind] = useState<"neuron" | "edge">(
    "neuron",
  );
  const [inputNeuron, setInputNeuron] = useState(0);
  const [tab, setTab] = useState<"architecture" | "weights" | "diagnostics">(
    "architecture",
  );
  const [focusAuditNeuron, setFocusAuditNeuron] = useState(false);
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
  const [drafts, setDrafts] = useState<Record<string, Draft>>(readDrafts);
  const [storageError, setStorageError] = useState("");
  const [computedKey, setComputedKey] = useState("");
  const model = models.find((item) => item.spec.id === activeId);
  const draft = drafts[activeId];
  const staleDraft = !!draft && draft.base_revision !== model?.head;
  const changes = draft?.changes ?? noChanges;
  const draftKey = JSON.stringify([
    model?.spec.id,
    model?.head,
    draft?.base_revision,
    changes,
  ]);
  const savedParameters =
    parameters?.revision_id === model?.head ? parameters : null;
  const currentParameters = useMemo(
    () => overlay(savedParameters, staleDraft ? noChanges : changes),
    [savedParameters, staleDraft, changes],
  );
  const currentRun =
    computedKey === draftKey &&
    !staleDraft &&
    run?.inputs[0].length === inputs.length &&
    run?.inputs[0].every((value, index) => value === Math.fround(inputs[index]))
      ? run
      : null;
  const currentSweep =
    computedKey === draftKey &&
    !staleDraft &&
    sweep?.inputs[0]
      .slice(1)
      .every((value, index) => value === Math.fround(inputs[index + 1]))
      ? sweep
      : null;
  const operation = model?.spec.operations.find((op) => op.id === selected);
  const parameter = currentParameters?.tensors[tensorId];
  const totalParameters =
    model?.spec.tensors.reduce(
      (sum, t) => sum + t.shape.reduce((a, b) => a * b, 1),
      0,
    ) ?? 0;

  const refresh = async () => setModels(await api<Model[]>("/models"));
  useEffect(() => {
    try {
      sessionStorage.setItem(draftStorageKey, JSON.stringify(drafts));
      setStorageError("");
    } catch {
      setStorageError(
        "Draft recovery storage is unavailable. Keep this tab open until you save your revision.",
      );
    }
  }, [drafts]);
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
    setFocusAuditNeuron(false);
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
    if (!model || inputs.length !== model.spec.input_width || staleDraft) {
      setRunning(false);
      return;
    }
    const controller = new AbortController();
    setRunning(true);
    const timer = setTimeout(() => {
      const rows = Array.from({ length: 65 }, (_, i) => [
        -2 + i / 16,
        ...inputs.slice(1),
      ]);
      const route = `/models/${model.spec.id}/runs`;
      const main = changes.length
        ? api<{ run: Run; sweep: Run }>(
            `/models/${model.spec.id}/draft-preview`,
            {
              base_revision: draft!.base_revision,
              changes,
              inputs,
            },
            controller.signal,
          ).then((result) => [result.run, result.sweep])
        : Promise.all([
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
          ]);
      Promise.all([
        main,
        api<Run>(
          route,
          { revision_id: model.original, inputs: rows },
          controller.signal,
        ),
      ])
        .then(([[result, curve], original]) => {
          if (!controller.signal.aborted) {
            setRun(result);
            setSweep(curve);
            setOriginalSweep(original);
            setComputedKey(draftKey);
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
  }, [draftKey, JSON.stringify(inputs), staleDraft]);

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
      staleDraft ||
      !savedParameters ||
      !Number.isFinite(Math.fround(Number(editValue))) ||
      editValue.trim() === ""
    )
      return;
    setError("");
    const change: Change = {
      tensor_id: tensorId,
      index: [...coordinate],
      value: Math.fround(Number(editValue)),
    };
    const tensor = savedParameters!.tensors[tensorId];
    const original =
      coordinate.length === 1
        ? (tensor.values as number[])[coordinate[0]]
        : (tensor.values as number[][])[coordinate[0]][coordinate[1]];
    const next = changes.filter((item) => cellKey(item) !== cellKey(change));
    if (change.value !== original) next.push(change);
    if (next.length > 4096) {
      setError("A draft can contain at most 4,096 parameter changes.");
      return;
    }
    next.sort((a, b) => cellKey(a).localeCompare(cellKey(b)));
    setDrafts((all) => {
      const updated = { ...all };
      if (next.length)
        updated[activeId] = {
          base_revision: draft?.base_revision ?? model.head,
          idempotency_key: crypto.randomUUID(),
          changes: next,
          name: draft?.name ?? "",
        };
      else delete updated[activeId];
      return updated;
    });
  }
  function discardDraft() {
    setDrafts((all) => {
      const updated = { ...all };
      delete updated[activeId];
      return updated;
    });
    setEditValue(
      parameterText(
        savedParameters
          ? coordinate.length === 1
            ? (savedParameters.tensors[tensorId].values as number[])[
                coordinate[0]
              ]
            : (savedParameters.tensors[tensorId].values as number[][])[
                coordinate[0]
              ][coordinate[1]]
          : undefined,
      ),
    );
    setError("");
  }
  async function saveDraft() {
    if (!model || !draft || staleDraft) return;
    setBusy(true);
    setError("");
    try {
      await api(`/models/${model.spec.id}/revisions`, {
        base_revision: draft.base_revision,
        idempotency_key: draft.idempotency_key,
        changes,
        name: draft.name?.trim() || undefined,
      });
      await refresh();
      setDrafts((all) => {
        const updated = { ...all };
        delete updated[activeId];
        return updated;
      });
    } catch (e) {
      setError((e as Error).message);
      await refresh().catch(() => {});
    } finally {
      setBusy(false);
    }
  }
  async function restore(revision: string) {
    if (!model || draft) return;
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
  const unstaged =
    selectedValue !== undefined &&
    (editValue.trim() === "" ||
      !Number.isFinite(Math.fround(Number(editValue))) ||
      Math.fround(Number(editValue)) !== selectedValue);
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
              Stage related edits in a draft, preview their effect, then save
              them together as one revision.
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
                  disabled={busy || !!draft || model.head === model.original}
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
                  <span
                    className="revision-heading-name"
                    title={model.head_name}
                  >
                    {model.head_name}
                  </span>
                  {model.head.slice(0, 8)}
                </span>
                <span
                  className={model.head === model.original ? "" : "modified"}
                >
                  {draft
                    ? "Unsaved draft"
                    : model.head === model.original
                      ? "Original checkpoint"
                      : "Modified checkpoint"}
                </span>
              </div>
              {draft && (
                <section className="draft-bar" aria-label="Parameter draft">
                  <div>
                    <strong>
                      {changes.length} staged parameter change
                      {changes.length === 1 ? "" : "s"}
                    </strong>
                    <p>
                      {staleDraft
                        ? "The saved model changed. This draft has not been rebased. Discard it to use the latest revision."
                        : "Previewing unsaved changes. Save the whole draft as one revision."}
                    </p>
                    <small>
                      Based on {draft.base_revision.slice(0, 8)} · recovered on
                      refresh in this browser tab.
                    </small>
                  </div>
                  <div className="draft-actions">
                    <label className="revision-name-field">
                      Revision name
                      <input
                        aria-label="Revision name"
                        type="text"
                        maxLength={80}
                        value={draft.name ?? ""}
                        placeholder="e.g. Fix hidden bias"
                        disabled={busy}
                        onChange={(event) => {
                          const name = event.target.value;
                          setDrafts((all) => ({
                            ...all,
                            [activeId]: {
                              ...all[activeId],
                              name,
                              idempotency_key: crypto.randomUUID(),
                            },
                          }));
                        }}
                      />
                      <small>Optional · up to 80 characters</small>
                    </label>
                    <button
                      className="draft-save"
                      disabled={busy || staleDraft || unstaged}
                      onClick={() => void saveDraft()}
                    >
                      Save revision
                    </button>
                    <button disabled={busy} onClick={discardDraft}>
                      Discard draft
                    </button>
                  </div>
                  {unstaged && (
                    <p>
                      Stage or reset the current editor value before saving.
                    </p>
                  )}
                  {storageError && <p role="alert">{storageError}</p>}
                  <details>
                    <summary>Review staged changes</summary>
                    <ul>
                      {changes.map((change) => (
                        <li key={cellKey(change)}>
                          <code>
                            {change.tensor_id}[{change.index.join(", ")}] →{" "}
                            {parameterText(change.value)}
                          </code>
                          <button
                            disabled={busy}
                            aria-label={`Unstage ${change.tensor_id} [${change.index.join(", ")}]`}
                            onClick={() =>
                              setDrafts((all) => {
                                const updated = { ...all };
                                const remaining = all[activeId].changes.filter(
                                  (item) => cellKey(item) !== cellKey(change),
                                );
                                if (remaining.length)
                                  updated[activeId] = {
                                    ...all[activeId],
                                    changes: remaining,
                                    idempotency_key: crypto.randomUUID(),
                                  };
                                else delete updated[activeId];
                                return updated;
                              })
                            }
                          >
                            Undo
                          </button>
                        </li>
                      ))}
                    </ul>
                  </details>
                </section>
              )}
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
                    <button
                      className={tab === "diagnostics" ? "active" : ""}
                      onClick={() => setTab("diagnostics")}
                    >
                      <Activity size={14} /> Diagnostics
                    </button>
                  </div>
                  <span className="view-hint">
                    {tab === "architecture"
                      ? "3D neuron graph"
                      : tab === "weights"
                        ? "Select a cell to edit"
                        : "Batch audit & revision comparison"}
                  </span>
                </div>
                <div className="canvas-content">
                  {tab === "architecture" ? (
                    <Network3D
                      key={model.spec.id}
                      model={model}
                      initialDetail={focusAuditNeuron ? "neuron" : "overview"}
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
                  ) : tab === "weights" ? (
                    <Weights
                      model={model}
                      parameters={currentParameters}
                      baseline={baseline}
                      tensorId={tensorId}
                      coordinate={coordinate}
                      changes={changes}
                      onSelect={selectParameter}
                    />
                  ) : null}
                  <div
                    className="diagnostic-view"
                    hidden={tab !== "diagnostics"}
                  >
                    <Diagnostics
                      key={model.spec.id}
                      model={model}
                      history={history}
                      changes={changes}
                      draftName={draft?.name ?? ""}
                      blocked={staleDraft}
                      onNeuron={(id, index) => {
                        selectNode(id, index);
                        setFocusAuditNeuron(true);
                        setTab("architecture");
                      }}
                      onCase={(values) => {
                        setInputs(values);
                        setFocusAuditNeuron(false);
                        setTab("architecture");
                      }}
                    />
                  </div>
                </div>
                <div className="canvas-footer">
                  <span>
                    <span className="legend-dot mint" />
                    Positive / zero <span className="legend-dot peach" />
                    Negative
                  </span>
                  <span>
                    {tab === "architecture"
                      ? "Nodes: activations · edges: signed weighted contributions"
                      : tab === "weights"
                        ? "Cells: parameter values · colors: sign"
                        : "Batch statistics and revision comparison"}
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
                          min={Math.min(-2, value)}
                          max={Math.max(2, value)}
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
                          <span>{number(Math.min(-2, value))}</span>
                          <span>{number(Math.max(2, value))}</span>
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
          hasDraft={!!draft}
          editBlocked={staleDraft}
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
