import { Check, CircleDot, LoaderCircle, RotateCcw } from "lucide-react";
import {
  number,
  type Model,
  type Operation,
  type Parameters,
  type Revision,
  type Run,
} from "./api";

function ParameterEditor({
  label,
  value,
  draft,
  busy,
  onDraft,
  onApply,
}: {
  label: "Weight" | "Bias";
  value: number | undefined;
  draft: string;
  busy: boolean;
  onDraft: (value: string) => void;
  onApply: () => Promise<void>;
}) {
  const valid =
    draft.trim() !== "" && Number.isFinite(Math.fround(Number(draft)));
  const changed =
    valid && value !== undefined && Math.fround(Number(draft)) !== value;
  const disabled = busy || !changed;
  return (
    <div className="inspector-editor">
      <label htmlFor="parameter-value">{label}</label>
      <div className="inspector-edit-row">
        <input
          id="parameter-value"
          data-testid={label === "Bias" ? "neuron-bias" : "edge-weight"}
          type="number"
          step="any"
          value={draft}
          disabled={busy || value === undefined}
          onChange={(event) => onDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !disabled) void onApply();
          }}
        />
        <button
          aria-label="Stage parameter edit"
          disabled={disabled}
          onClick={() => void onApply()}
        >
          {busy ? <LoaderCircle className="spin" size={13} /> : "Stage"}
        </button>
      </div>
      {changed && <span className="inspector-draft">Unstaged change</span>}
    </div>
  );
}

export function Inspector({
  model,
  operation,
  kind,
  neuron,
  inputNeuron,
  parameters,
  run,
  coordinate,
  draft,
  busy,
  history,
  onDraft,
  onApply,
  onSelectParameter,
  onRestore,
  hasDraft = false,
  editBlocked = false,
}: {
  model: Model | undefined;
  operation: Operation | undefined;
  kind: "neuron" | "edge";
  neuron: number;
  inputNeuron: number;
  parameters: Parameters | null;
  run: Run | null;
  coordinate: number[];
  draft: string;
  busy: boolean;
  history: Revision[];
  onDraft: (value: string) => void;
  onApply: () => Promise<void>;
  onSelectParameter: (id: string, index: number[]) => void;
  onRestore: (revision: string) => Promise<void>;
  hasDraft?: boolean;
  editBlocked?: boolean;
}) {
  const dense = operation?.kind === "dense" ? operation : undefined;
  const input = !operation;
  const edge = kind === "edge" && !!dense;
  const following =
    dense && model
      ? model.spec.operations[model.spec.operations.indexOf(dense) + 1]
      : undefined;
  const activation =
    following && following.kind !== "dense" ? following : undefined;
  const activationName = activation?.kind ?? "linear";
  const names: Record<string, string> = {
    relu: "ReLU",
    tanh: "Tanh",
    sigmoid: "Sigmoid",
    softmax: "Softmax",
    identity: "Identity",
    linear: "Linear",
  };
  const outputName = activation?.output ?? dense?.output;
  const inputs = dense ? run?.activations[dense.input]?.[0] : undefined;
  const weights = dense?.weight
    ? (parameters?.tensors[dense.weight]?.values as number[][] | undefined)
    : undefined;
  const bias = dense?.bias
    ? (parameters?.tensors[dense.bias]?.values as number[] | undefined)?.[
        neuron
      ]
    : undefined;
  const z = dense ? run?.activations[dense.output]?.[0]?.[neuron] : undefined;
  const output = outputName
    ? run?.activations[outputName]?.[0]?.[neuron]
    : undefined;
  const weight = edge ? weights?.[coordinate[0]]?.[coordinate[1]] : undefined;
  const source = edge ? inputs?.[coordinate[1]] : undefined;
  const contribution =
    source !== undefined && weight !== undefined ? source * weight : undefined;
  let producer =
    dense && model
      ? model.spec.operations.find((op) => op.output === dense.input)
      : undefined;
  if (producer && producer.kind !== "dense")
    producer = model?.spec.operations.find(
      (op) => op.output === producer!.input,
    );
  const sourceName = producer?.id ?? "Input";
  const identity = edge
    ? `${sourceName}[${coordinate[1]}] → ${dense!.id}[${coordinate[0]}]`
    : input
      ? `Input · neuron ${inputNeuron}`
      : dense
        ? `${dense.id} · neuron ${neuron}`
        : operation!.id;
  const editor = (
    <ParameterEditor
      label={edge ? "Weight" : "Bias"}
      value={edge ? weight : bias}
      draft={draft}
      busy={busy || editBlocked}
      onDraft={onDraft}
      onApply={onApply}
    />
  );
  return (
    <aside
      className="inspector compact-inspector"
      data-selection-kind={edge ? "edge" : input ? "input" : "neuron"}
    >
      <div className="inspector-title">
        <span>
          <CircleDot size={15} />
          Inspector
        </span>
      </div>
      {!model ? (
        <div className="inspector-body">Select a neuron or connection.</div>
      ) : (
        <>
          <header className="inspector-selection">
            <span>
              {edge
                ? "CONNECTION"
                : input
                  ? "INPUT NEURON"
                  : dense
                    ? "NEURON"
                    : "OPERATION"}
            </span>
            <h3>{identity}</h3>
          </header>
          <div className="inspector-body">
            {edge ? (
              <>
                <dl className="inspector-values">
                  <div>
                    <dt>Source activation</dt>
                    <dd data-testid="edge-input">{number(source)}</dd>
                  </div>
                </dl>
                {editor}
                <div className="inspector-result">
                  <span>
                    Current contribution <small>input × weight</small>
                  </span>
                  <strong data-testid="edge-contribution">
                    {number(contribution)}
                  </strong>
                </div>
              </>
            ) : input ? (
              <div className="inspector-output">
                <span>Input value</span>
                <strong data-testid="neuron-input">
                  {number(run?.inputs[0]?.[inputNeuron])}
                </strong>
                <small>Adjust with the input controls below the graph.</small>
              </div>
            ) : dense ? (
              <>
                <div className="inspector-output">
                  <span>
                    Output <small>{names[activationName]}</small>
                  </span>
                  <strong data-testid="neuron-activation">
                    {number(output)}
                  </strong>
                </div>
                {activationName !== "linear" &&
                  activationName !== "identity" && (
                    <dl className="inspector-values">
                      <div>
                        <dt>Pre-activation</dt>
                        <dd data-testid="neuron-preactivation">{number(z)}</dd>
                      </div>
                    </dl>
                  )}
                {dense.bias ? (
                  editor
                ) : (
                  <dl className="inspector-values">
                    <div>
                      <dt>Bias</dt>
                      <dd>Not present</dd>
                    </div>
                  </dl>
                )}
                {weights && (
                  <details className="inspector-breakdown">
                    <summary>
                      Incoming contributions{" "}
                      <span>{weights[neuron].length}</span>
                    </summary>
                    <div className="incoming-head">
                      <span>From</span>
                      <span>Input</span>
                      <span>Weight</span>
                      <span>Product</span>
                    </div>
                    {weights[neuron].map((value, index) => (
                      <button
                        className="incoming-row"
                        key={index}
                        aria-label={`Inspect incoming connection ${index}`}
                        onClick={() =>
                          onSelectParameter(dense.weight!, [neuron, index])
                        }
                      >
                        <span>{index}</span>
                        <code>{number(inputs?.[index])}</code>
                        <code>{number(value)}</code>
                        <strong>
                          {number(
                            inputs?.[index] === undefined
                              ? undefined
                              : inputs[index] * value,
                          )}
                        </strong>
                      </button>
                    ))}
                  </details>
                )}
              </>
            ) : (
              <dl className="inspector-values">
                <div>
                  <dt>Function</dt>
                  <dd>{names[operation!.kind] ?? operation!.kind}</dd>
                </div>
                <div>
                  <dt>Values</dt>
                  <dd>
                    {run?.activations[operation!.output]?.[0]
                      .map(number)
                      .join(", ") ?? "—"}
                  </dd>
                </div>
              </dl>
            )}
          </div>
          <details className="inspector-history">
            <summary>
              Revision history <span>{history.length}</span>
            </summary>
            <div className="revision-list">
              {history.map((revision) => (
                <button
                  key={revision.id}
                  disabled={busy || hasDraft || model.head === revision.id}
                  className={model.head === revision.id ? "current" : ""}
                  onClick={() => void onRestore(revision.id)}
                >
                  <span className="revision-mark" />
                  <span>
                    <strong>
                      {revision.parent
                        ? `${revision.changes.length} parameter edit${revision.changes.length > 1 ? "s" : ""}`
                        : "Original model"}
                    </strong>
                    <code>{revision.id.slice(0, 8)}</code>
                  </span>
                  {model.head === revision.id ? (
                    <Check size={13} />
                  ) : (
                    <RotateCcw size={12} />
                  )}
                </button>
              ))}
            </div>
          </details>
        </>
      )}
    </aside>
  );
}
