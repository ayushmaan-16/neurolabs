import type { Model } from "./api";

export type AuditNeuron = {
  index: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  zero_fraction: number;
  baseline_zero_fraction: number;
  inactive_on_batch: boolean;
  saturation_fraction: number | null;
  delta_rms: number;
};
export type Audit = {
  format_version: number;
  model_id: string;
  revision_id: string;
  baseline_revision_id: string;
  input_hash: string;
  sample_count: number;
  scope: string;
  summary: {
    inactive_relu_neurons: number;
    mean_abs_output_change: number;
    max_abs_output_change: number;
    argmax_flips: number | null;
  };
  layers: {
    id: string;
    activation: string;
    width: number;
    rms: number;
    baseline_rms: number;
    delta_rms: number;
    inactive_neurons: number;
    neurons: AuditNeuron[];
  }[];
  cases: {
    index: number;
    inputs: number[];
    current_output: number[];
    baseline_output: number[];
    max_abs_change: number;
    argmax_changed: boolean | null;
  }[];
};

export function probeBatch(model: Model): string {
  if (model.spec.id === "xor") return "0,0\n0,1\n1,0\n1,1";
  return Array.from({ length: 33 }, (_, row) =>
    Array.from({ length: model.spec.input_width }, (_, col) =>
      col === 0
        ? -2 + row / 8
        : Number((2 * Math.sin((row + 1) * (col + 1) * 1.37)).toFixed(5)),
    ).join(","),
  ).join("\n");
}

export function parseBatch(text: string, width: number): number[][] {
  const lines = text
    .replace(/^\uFEFF/, "")
    .trim()
    .split(/\r?\n/);
  if (lines.length < 2 || lines.length > 256)
    throw new Error("Supply 2–256 input rows.");
  return lines.map((line, row) => {
    const cells = line.split(",").map((cell) => cell.trim());
    if (cells.length !== width)
      throw new Error(`Row ${row + 1}: expected ${width} input values.`);
    return cells.map((cell) => {
      if (
        !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(cell) ||
        !Number.isFinite(Math.fround(Number(cell)))
      )
        throw new Error(
          `Row ${row + 1}: every cell must be a finite float32 number.`,
        );
      return Number(cell);
    });
  });
}

export function metric(value: number): string {
  if (value === 0) return "0";
  return Math.abs(value) < 0.0001 || Math.abs(value) >= 100_000
    ? value.toExponential(3)
    : new Intl.NumberFormat("en-US", { maximumFractionDigits: 5 }).format(
        value,
      );
}
