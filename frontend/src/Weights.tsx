import { useEffect, useRef, useState } from "react";
import { parameterText, type Model, type Parameters } from "./api";
import { cellKey, type Change } from "./draft-state";

function displayValue(value: number): string {
  if (value === 0) return "0";
  if (Math.abs(value) < 0.0001 || Math.abs(value) >= 10_000)
    return value.toExponential(3);
  return String(Number(value.toPrecision(5)));
}

export function Weights({
  model,
  parameters,
  baseline,
  tensorId,
  coordinate,
  changes,
  onSelect,
}: {
  model: Model;
  parameters: Parameters | null;
  baseline: Parameters | null;
  tensorId: string;
  coordinate: number[];
  changes: Change[];
  onSelect: (id: string, index: number[]) => void;
}) {
  const tensor = parameters?.tensors[tensorId];
  const original =
    baseline?.revision_id === model.original
      ? baseline.tensors[tensorId]
      : undefined;
  const vector = tensor?.shape.length === 1;
  const rows = tensor?.shape[0] ?? 0;
  const columns = vector ? 1 : (tensor?.shape[1] ?? 0);
  const rowStart = Math.floor(coordinate[0] / 64) * 64;
  const columnStart = vector ? 0 : Math.floor((coordinate[1] ?? 0) / 64) * 64;
  const visibleRows = Math.min(64, rows - rowStart);
  const visibleColumns = Math.min(64, columns - columnStart);
  const cells = useRef(new Map<string, HTMLButtonElement>());
  const scrollArea = useRef<HTMLDivElement>(null);
  const [row, setRow] = useState("0"),
    [column, setColumn] = useState("0");
  const [exact, setExact] = useState(false);
  useEffect(() => {
    setRow(String(coordinate[0]));
    setColumn(String(coordinate[1] ?? 0));
    const reveal = () =>
      cells.current
        .get(coordinate.join(","))
        ?.scrollIntoView({ block: "nearest", inline: "nearest" });
    reveal();
    const observer = new ResizeObserver(reveal);
    if (scrollArea.current) observer.observe(scrollArea.current);
    return () => observer.disconnect();
  }, [tensorId, coordinate.join(","), !!tensor, exact]);
  const staged = new Set(changes.map(cellKey));
  const selected = tensor
    ? vector
      ? (tensor.values as number[])[coordinate[0]]
      : (tensor.values as number[][])[coordinate[0]]?.[coordinate[1]]
    : undefined;
  const valid =
    row.trim() !== "" &&
    Number.isInteger(Number(row)) &&
    Number(row) >= 0 &&
    Number(row) < rows &&
    (vector ||
      (column.trim() !== "" &&
        Number.isInteger(Number(column)) &&
        Number(column) >= 0 &&
        Number(column) < columns));
  function jump() {
    if (valid)
      onSelect(
        tensorId,
        vector ? [Number(row)] : [Number(row), Number(column)],
      );
  }
  return (
    <div className="matrix-view tensor-view">
      <div className="tensor-heading">
        <div>
          <span className="section-label">
            {vector ? "BIAS VECTOR" : "WEIGHT MATRIX"}
          </span>
          <h3>{tensorId}</h3>
        </div>
        <code>[{tensor?.shape.join(" × ") ?? "…"}] · float32</code>
      </div>
      <label className="sr-only" htmlFor="tensor-select">
        Tensor
      </label>
      <select
        id="tensor-select"
        value={tensorId}
        onChange={(event) => {
          const item = model.spec.tensors.find(
            (item) => item.id === event.target.value,
          )!;
          onSelect(
            item.id,
            item.shape.map(() => 0),
          );
        }}
      >
        {model.spec.tensors.map((item) => (
          <option key={item.id} value={item.id}>
            {item.id} · [{item.shape.join(" × ")}]
          </option>
        ))}
      </select>
      <p>
        {vector
          ? "One bias per destination neuron, added after its weighted input sum."
          : "Rows = destination neurons. Columns = source neurons in the preceding layer. W[row, column] is the weight on that connection."}
        {changes.length
          ? " Showing the unsaved draft values."
          : ` Showing ${model.head_name}.`}
      </p>
      <div className="tensor-controls">
        <label>
          Destination row
          <input
            aria-label="Destination row"
            type="number"
            min="0"
            max={rows - 1}
            value={row}
            onChange={(event) => setRow(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") jump();
            }}
          />
        </label>
        {!vector && (
          <label>
            Source column
            <input
              aria-label="Source column"
              type="number"
              min="0"
              max={columns - 1}
              value={column}
              onChange={(event) => setColumn(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") jump();
              }}
            />
          </label>
        )}
        <button disabled={!valid} onClick={jump}>
          Select coordinate
        </button>
        <label className="tensor-precision">
          <input
            type="checkbox"
            checked={exact}
            onChange={(event) => setExact(event.target.checked)}
          />{" "}
          Full precision
        </label>
      </div>
      <div className="tensor-selection" aria-live="polite">
        Selected{" "}
        <code>
          {tensorId}[{coordinate.join(", ")}] = {parameterText(selected) || "…"}
        </code>
        <span>
          Select a cell, then edit and stage its value in the Inspector.
        </span>
      </div>
      {(rows > 64 || columns > 64) && (
        <div className="tensor-pages">
          <label>
            Rows
            <select
              aria-label="Matrix row block"
              value={rowStart}
              onChange={(event) =>
                onSelect(
                  tensorId,
                  vector
                    ? [Number(event.target.value)]
                    : [Number(event.target.value), coordinate[1]],
                )
              }
            >
              {Array.from({ length: Math.ceil(rows / 64) }, (_, page) => (
                <option value={page * 64} key={page}>
                  {page * 64}–{Math.min(rows - 1, page * 64 + 63)}
                </option>
              ))}
            </select>
          </label>
          {!vector && (
            <label>
              Columns
              <select
                aria-label="Matrix column block"
                value={columnStart}
                onChange={(event) =>
                  onSelect(tensorId, [
                    coordinate[0],
                    Number(event.target.value),
                  ])
                }
              >
                {Array.from({ length: Math.ceil(columns / 64) }, (_, page) => (
                  <option value={page * 64} key={page}>
                    {page * 64}–{Math.min(columns - 1, page * 64 + 63)}
                  </option>
                ))}
              </select>
            </label>
          )}
          <span>
            Showing rows {rowStart}–{rowStart + visibleRows - 1}
            {!vector &&
              ` · columns ${columnStart}–${columnStart + visibleColumns - 1}`}
            . Select a block or jump to a coordinate.
          </span>
        </div>
      )}
      {!tensor ? (
        <p>Loading parameters…</p>
      ) : (
        <div
          className={`tensor-scroll ${exact ? "tensor-exact" : ""}`}
          ref={scrollArea}
          role="region"
          aria-label="Parameter matrix scroll area"
          tabIndex={0}
        >
          <table className="tensor-table">
            <caption className="sr-only">
              {tensorId}: {rows} destination rows and {columns}{" "}
              {vector ? "bias column" : "source columns"}. Indices begin at
              zero.
            </caption>
            <thead>
              <tr>
                <th className="tensor-corner">
                  To ↓{!vector && <span>From →</span>}
                </th>
                {Array.from(
                  { length: visibleColumns },
                  (_, i) => i + columnStart,
                ).map((c) => (
                  <th key={c} scope="col">
                    {vector ? "Bias" : `From ${c}`}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: visibleRows }, (_, i) => i + rowStart).map(
                (r) => (
                  <tr key={r}>
                    <th scope="row">To {r}</th>
                    {Array.from(
                      { length: visibleColumns },
                      (_, i) => i + columnStart,
                    ).map((c) => {
                      const index = vector ? [r] : [r, c];
                      const value = vector
                        ? (tensor.values as number[])[r]
                        : (tensor.values as number[][])[r][c];
                      const originalValue = original
                        ? vector
                          ? (original.values as number[])[r]
                          : (original.values as number[][])[r]?.[c]
                        : undefined;
                      const picked = coordinate.join(",") === index.join(",");
                      const pending = staged.has(
                        cellKey({ tensor_id: tensorId, index }),
                      );
                      const differs =
                        originalValue !== undefined && originalValue !== value;
                      return (
                        <td key={c}>
                          <button
                            ref={(element) => {
                              if (element)
                                cells.current.set(index.join(","), element);
                              else cells.current.delete(index.join(","));
                            }}
                            className={`tensor-cell ${value < 0 ? "negative" : "positive"} ${picked ? "selected" : ""} ${differs ? "changed" : ""} ${pending ? "staged" : ""}`}
                            aria-pressed={picked}
                            aria-label={`${tensorId} [${index.join(", ")}] = ${value}`}
                            title={`${tensorId}[${index.join(", ")}] = ${parameterText(value)}${differs ? ` · ${model.original_name}: ${parameterText(originalValue)}` : ""}${pending ? " · staged" : ""}`}
                            onClick={() => onSelect(tensorId, index)}
                          >
                            {exact ? parameterText(value) : displayValue(value)}
                          </button>
                        </td>
                      );
                    })}
                  </tr>
                ),
              )}
            </tbody>
          </table>
        </div>
      )}
      <div className="tensor-legend">
        <span>
          <i className="tensor-changed-dot" /> Different from{" "}
          {model.original_name}
        </span>
        <span>Dashed border: staged edit</span>
        <span>Scroll horizontally and vertically · indices begin at 0</span>
      </div>
    </div>
  );
}
