import { useEffect, useMemo, useRef, useState } from "react";
import {
  createGraphScene,
  type CameraPose,
  type GraphStage,
} from "./graph-scene";
import { api, type Model, type Parameters, type Run } from "./api";
import { createDenseGraphScene } from "./dense-graph-scene";
import { graphConnections, type GraphDetail } from "./graph-detail";

export function Network3D({
  model,
  parameters,
  run,
  selected,
  neuron,
  onSelect,
  selection,
  selectionKind,
  onParameterSelect,
  initialDetail = "overview",
}: {
  model: Model;
  parameters: Parameters | null;
  run: Run | null;
  selected: string;
  neuron: number;
  onSelect: (id: string, neuron: number) => void;
  selection: { tensorId: string; index: number[] };
  selectionKind: "neuron" | "edge";
  onParameterSelect: (tensorId: string, index: number[]) => void;
  initialDetail?: GraphDetail;
}) {
  const host = useRef<HTMLDivElement>(null);
  const hitTargets = useRef(new Map<string, HTMLElement>());
  const [trace, setTrace] = useState<Run | null>(null);
  const [progress, setProgress] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [webglError, setWebglError] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const resetCamera = useRef<() => void>(() => {});
  const cameraState = useRef<CameraPose | null>(null);
  const cameraView = useRef("");
  const [detail, setDetail] = useState<GraphDetail>(initialDetail);
  const [focusLayer, setFocusLayer] = useState(1);
  const stages = useMemo(() => {
    const result: GraphStage[] = [
      {
        id: "input",
        title: "Input",
        output: model.spec.input_name,
        width: model.spec.input_width,
        relu: false,
      },
    ];
    for (let i = 0; i < model.spec.operations.length; i++) {
      const op = model.spec.operations[i];
      const next = model.spec.operations[i + 1];
      const activation =
        op.kind === "dense" && next && next.kind !== "dense" ? next : undefined;
      const width = op.weight
        ? model.spec.tensors.find((t) => t.id === op.weight)!.shape[0]
        : result.at(-1)!.width;
      result.push({
        id: op.id,
        title:
          op.kind === "dense"
            ? `${op.id} · ${activation?.kind ?? "linear"}`
            : op.kind,
        output: activation?.output ?? op.output,
        preactivation: op.output,
        width,
        weight: op.weight,
        bias: op.bias ?? undefined,
        relu: activation?.kind === "relu" || op.kind === "relu",
      });
      if (activation) i++;
    }
    return result;
  }, [model.spec]);
  const totalNodes = stages.reduce((sum, stage) => sum + stage.width, 0);
  const totalConnections = stages.reduce(
    (sum, stage, layer) =>
      sum + (layer ? stage.width * stages[layer - 1].width : 0),
    0,
  );
  const dense = totalNodes > 128 || totalConnections > 2048;
  const selectedLayer = Math.max(
    0,
    stages.findIndex((stage) => stage.id === selected),
  );
  // Small scenes only recolor their existing objects when selection changes.
  const detailSelection =
    dense && detail !== "all"
      ? JSON.stringify([selectedLayer, neuron, selectionKind, selection])
      : "";
  const connections = useMemo(
    () =>
      parameters
        ? graphConnections(
            stages,
            parameters,
            dense,
            detail,
            focusLayer,
            selectedLayer,
            neuron,
            selectionKind,
            selection,
          )
        : [],
    [stages, parameters, dense, detail, focusLayer, detailSelection],
  );
  const keyboardConnections =
    connections.length <= 512
      ? connections
      : connections.filter(
          (edge) =>
            selectionKind === "edge" &&
            edge.tensorId === selection.tensorId &&
            edge.row === selection.index[0] &&
            edge.col === selection.index[1],
        );
  const viewKey = `${detail}:${focusLayer}:${detail === "neuron" ? selectedLayer : ""}`;
  const live = useRef({
    trace,
    run,
    progress,
    selected,
    neuron,
    selection,
    selectionKind,
    onSelect,
    onParameterSelect,
  });
  live.current = {
    trace,
    run,
    progress,
    selected,
    neuron,
    selection,
    selectionKind,
    onSelect,
    onParameterSelect,
  };

  // Changing input or parameters invalidates the replay, including requests still in flight.
  useEffect(() => {
    controller.current?.abort();
    setTrace(null);
    setProgress(0);
    setPlaying(false);
    setPending(false);
  }, [run?.revision_id, run?.input_hash, model.spec.id]);
  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(
      () =>
        setProgress((value) => {
          const next = Math.min(stages.length - 1, value + 0.035 * speed);
          if (next >= stages.length - 1) setPlaying(false);
          return next;
        }),
      40,
    );
    return () => window.clearInterval(timer);
  }, [playing, speed, stages.length]);

  async function start() {
    if (!run) return;
    const abort = new AbortController();
    controller.current?.abort();
    controller.current = abort;
    setPending(true);
    setPlaying(false);
    setError("");
    setTrace(null);
    setProgress(0);
    try {
      const result = await api<Run>(
        `/models/${model.spec.id}/runs`,
        {
          revision_id: model.head,
          inputs: run.inputs,
          captures: [
            model.spec.input_name,
            ...model.spec.operations.map((op) => op.output),
          ],
        },
        abort.signal,
      );
      if (!abort.signal.aborted) {
        setTrace(result);
        setPlaying(
          !window.matchMedia("(prefers-reduced-motion: reduce)").matches,
        );
      }
    } catch (e) {
      if (!abort.signal.aborted) setError((e as Error).message);
    } finally {
      if (!abort.signal.aborted) setPending(false);
    }
  }

  useEffect(() => {
    if (!host.current || !parameters) return;
    try {
      const common = {
        mount: host.current,
        stages,
        state: () => live.current,
        targets: hitTargets.current,
        pose: cameraView.current === viewKey ? cameraState.current : null,
        savePose: (pose: CameraPose) => {
          cameraState.current = pose;
        },
      };
      const graph = dense
        ? createDenseGraphScene({
            ...common,
            connections,
            detail,
            focusLayer,
            selectedLayer,
          })
        : createGraphScene({ ...common, parameters });
      cameraView.current = viewKey;
      setWebglError(false);
      resetCamera.current = graph.reset;
      return () => {
        graph.dispose();
        resetCamera.current = () => {};
      };
    } catch {
      setWebglError(true);
    }
  }, [stages, parameters, dense, connections, viewKey]);

  const finished = !!trace && progress >= stages.length - 1;
  const stageIndex = Math.min(stages.length - 1, Math.floor(progress));
  return (
    <div className={`network-3d ${dense ? "dense-network" : ""}`}>
      <div className="trace-controls">
        <button
          className="trace-start"
          onClick={() => void start()}
          disabled={!run || pending || !parameters}
        >
          {pending ? "Computing…" : "Run & trace"}
        </button>
        <button
          disabled={!trace || finished}
          onClick={() => setPlaying(!playing)}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <button
          disabled={!trace || finished}
          onClick={() => {
            setPlaying(false);
            setProgress((value) =>
              Math.min(stages.length - 1, Math.floor(value) + 1),
            );
          }}
        >
          Step layer
        </button>
        <label>
          Speed{" "}
          <select
            aria-label="Playback speed"
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
          >
            <option value="0.5">0.5×</option>
            <option value="1">1×</option>
            <option value="2">2×</option>
          </select>
        </label>
        <button onClick={() => resetCamera.current()}>Reset camera</button>
      </div>
      {dense && (
        <div className="graph-detail-controls">
          <div className="graph-density-summary">
            <strong>{totalNodes.toLocaleString()} neurons</strong>
            <span>
              {totalConnections.toLocaleString()} connections computed
            </span>
          </div>
          <div className="graph-detail-selects">
            <label>
              Connections
              <select
                aria-label="Connection detail"
                value={detail}
                onChange={(event) =>
                  setDetail(event.target.value as GraphDetail)
                }
              >
                <option value="overview">Overview · sampled strings</option>
                <option value="all">All connections</option>
                <option value="layer">Layer pair · all strings</option>
                <option value="neuron">Selected neuron · all strings</option>
              </select>
            </label>
            {detail === "layer" && (
              <label>
                Layer pair
                <select
                  aria-label="Layer pair"
                  value={focusLayer}
                  onChange={(event) =>
                    setFocusLayer(Number(event.target.value))
                  }
                >
                  {stages.slice(1).map((stage, index) => (
                    <option key={stage.id} value={index + 1}>
                      {index === 0 ? "Input" : `Layer ${index}`} → Layer{" "}
                      {index + 1} · {stages[index].width} × {stage.width}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {detail === "neuron" && (
              <>
                <label>
                  Neuron layer
                  <select
                    aria-label="Neuron layer"
                    value={selectedLayer}
                    onChange={(event) =>
                      onSelect(stages[Number(event.target.value)].id, 0)
                    }
                  >
                    {stages.map((stage, index) => (
                      <option key={stage.id} value={index}>
                        {index === 0 ? "Input" : `Layer ${index}`} ·{" "}
                        {stage.width} neurons
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Neuron index
                  <select
                    aria-label="Neuron index"
                    value={neuron}
                    onChange={(event) =>
                      onSelect(
                        stages[selectedLayer].id,
                        Number(event.target.value),
                      )
                    }
                  >
                    {Array.from(
                      { length: stages[selectedLayer].width },
                      (_, index) => (
                        <option key={index} value={index}>
                          {index}
                        </option>
                      ),
                    )}
                  </select>
                </label>
              </>
            )}
          </div>
          <p data-testid="connection-summary">
            Showing {connections.length.toLocaleString()} of{" "}
            {totalConnections.toLocaleString()} connections.
            {detail === "all"
              ? " Every neuron and connection is shown."
              : detail === "overview"
                ? " Every neuron is shown. Focus a layer pair or neuron for all its strings."
                : detail === "neuron"
                  ? " Click a neuron to follow every incoming and outgoing connection."
                  : " Every connection in this pair is shown. Select a neuron for a clearer view."}
          </p>
        </div>
      )}
      <div className="three-viewport" ref={host}>
        <div className="graph-hit-targets">
          {stages.map((stage, layer) =>
            Array.from({ length: stage.width }, (_, index) => {
              const reached = trace ? progress >= layer : !!run;
              const value = (trace ?? run)?.activations[stage.output]?.[0]?.[
                index
              ];
              return (
                <button
                  key={`${stage.id}:${index}`}
                  className="neuron-target scene-proxy"
                  ref={(el) => {
                    if (el) hitTargets.current.set(`${layer}:${index}`, el);
                    else hitTargets.current.delete(`${layer}:${index}`);
                  }}
                  aria-label={`Select ${stage.id} neuron ${index}`}
                  data-activation={reached ? value : "pending"}
                  data-zero-output={
                    reached && stage.relu && value === 0 ? "true" : "false"
                  }
                  onClick={() => onSelect(stage.id, index)}
                />
              );
            }),
          )}
          {keyboardConnections.map((edge) => (
            <button
              key={edge.key}
              className="edge-target scene-proxy"
              ref={(el) => {
                if (el) hitTargets.current.set(edge.key, el);
                else hitTargets.current.delete(edge.key);
              }}
              aria-label={`Select connection ${edge.tensorId} [${edge.row}, ${edge.col}]`}
              onClick={() =>
                onParameterSelect(edge.tensorId, [edge.row, edge.col])
              }
            />
          ))}
        </div>
        {webglError && (
          <div className="three-error" role="alert">
            3D rendering needs WebGL. Enable hardware acceleration in your
            browser. The architecture tree and weight editor remain available.
          </div>
        )}
      </div>
      <div className="trace-timeline">
        <input
          aria-label="Propagation progress"
          type="range"
          min="0"
          max={stages.length - 1}
          step="0.01"
          value={progress}
          disabled={!trace}
          onChange={(e) => {
            setPlaying(false);
            setProgress(Number(e.target.value));
          }}
        />
        <span role="status">
          {error ||
            (!trace
              ? "Choose your input, then Run & trace."
              : finished
                ? "Forward pass complete"
                : `${stages[stageIndex].title} → ${stages[stageIndex + 1]?.title ?? "output"}`)}
        </span>
        <small>
          Replay of captured values · animation speed is illustrative, not
          runtime timing.
        </small>
      </div>
    </div>
  );
}
