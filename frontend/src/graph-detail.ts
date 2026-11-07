import type { Parameters } from "./api";
import type { GraphStage } from "./graph-scene";

export type GraphDetail = "overview" | "all" | "layer" | "neuron";
export type GraphConnection = {
  layer: number;
  row: number;
  col: number;
  weight: number;
  tensorId: string;
  key: string;
};

// Rendering is independent of execution: only the view is filtered.
export function graphConnections(
  stages: GraphStage[],
  parameters: Parameters,
  dense: boolean,
  detail: GraphDetail,
  focusLayer: number,
  selectedLayer: number,
  neuron: number,
  selectionKind: "neuron" | "edge",
  selection: { tensorId: string; index: number[] },
): GraphConnection[] {
  const result: GraphConnection[] = [];
  for (let layer = 1; layer < stages.length; layer++) {
    const tensorId = stages[layer].weight;
    if (!tensorId) continue;
    const matrix = parameters.tensors[tensorId].values as number[][];
    const width = matrix[0].length;
    const size = matrix.length * width;
    const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : a);
    let stride = 17;
    while (gcd(stride, width) !== 1) stride++;
    // At most 96 evenly distributed connections per transition in overview.
    const overview = new Set(
      Array.from(
        { length: Math.min(96, size) },
        (_, i) =>
          Math.floor((i * matrix.length) / Math.min(96, size)) * width +
          ((i * stride + layer * 7) % width),
      ),
    );
    matrix.forEach((row, target) =>
      row.forEach((weight, source) => {
        const picked =
          selectionKind === "edge" &&
          selection.tensorId === tensorId &&
          selection.index[0] === target &&
          selection.index[1] === source;
        const show =
          !dense ||
          detail === "all" ||
          (detail === "overview" && overview.has(target * width + source)) ||
          (detail === "layer" && layer === focusLayer) ||
          (detail === "neuron" &&
            ((layer === selectedLayer && target === neuron) ||
              (layer === selectedLayer + 1 && source === neuron))) ||
          (picked && detail !== "layer");
        if (show)
          result.push({
            layer,
            row: target,
            col: source,
            weight,
            tensorId,
            key: `edge:${layer}:${target}:${source}`,
          });
      }),
    );
  }
  return result;
}
