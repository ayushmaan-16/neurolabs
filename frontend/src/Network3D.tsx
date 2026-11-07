import { useEffect, useMemo, useRef, useState } from "react";
import {
  createGraphScene,
  type CameraPose,
  type GraphStage,
} from "./graph-scene";
import { api, type Model, type Parameters, type Run } from "./api";

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
  const connections = useMemo(
    () =>
      stages.flatMap((stage, layer) => {
        if (!stage.weight || !parameters) return [];
        const values = parameters.tensors[stage.weight].values as number[][];
        return values.flatMap((row, target) =>
          row.map((weight, source) => ({
            layer,
            row: target,
            col: source,
            weight,
            tensorId: stage.weight!,
            key: `edge:${layer}:${target}:${source}`,
          })),
        );
      }),
    [stages, parameters],
  );
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
      const graph = createGraphScene({
        mount: host.current,
        stages,
        parameters,
        state: () => live.current,
        targets: hitTargets.current,
        pose: cameraState.current,
        savePose: (pose) => {
          cameraState.current = pose;
        },
      });
      setWebglError(false);
      resetCamera.current = graph.reset;
      return () => {
        graph.dispose();
        resetCamera.current = () => {};
      };
    } catch {
      setWebglError(true);
    }
  }, [stages, parameters]);

  const finished = !!trace && progress >= stages.length - 1;
  const stageIndex = Math.min(stages.length - 1, Math.floor(progress));
  return (
    <div className="network-3d">
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
          {connections.map((edge) => (
            <button
              key={edge.key}
              className="edge-target scene-proxy"
              ref={(el) => {
                if (el) hitTargets.current.set(edge.key, el);
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
