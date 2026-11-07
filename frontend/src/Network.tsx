import { memo, useMemo } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  Position,
  ReactFlow,
  type NodeProps,
  type Node,
} from "@xyflow/react";
import { type Model, type Run, number } from "./api";

type LayerData = {
  title: string;
  subtitle: string;
  values: number[];
  selected: boolean;
  kind: string;
};
const Layer = memo(({ data }: NodeProps<Node<LayerData>>) => (
  <div className={`layer-node ${data.selected ? "selected" : ""}`}>
    {data.kind !== "input" && <Handle type="target" position={Position.Left} />}
    <div className="node-kicker">
      {data.kind === "dense" ? "AFFINE TRANSFORM" : data.kind.toUpperCase()}
    </div>
    <strong>{data.title}</strong>
    <span className="node-subtitle">{data.subtitle}</span>
    <div className="neuron-row">
      {(data.values.length ? data.values : [undefined]).map((value, index) => (
        <div key={index} className="neuron-value">
          <span
            className={`neuron ${value !== undefined && value < 0 ? "negative" : ""}`}
          />
          <code>{number(value)}</code>
        </div>
      ))}
    </div>
    <Handle type="source" position={Position.Right} />
  </div>
));
const nodeTypes = { layer: Layer };

export function Network({
  model,
  run,
  selected,
  onSelect,
}: {
  model: Model;
  run: Run | null;
  selected: string;
  onSelect: (id: string) => void;
}) {
  const nodes = useMemo(() => {
    let width = model.spec.input_width;
    const items: Node<LayerData>[] = [
      {
        id: "input",
        type: "layer",
        position: { x: 0, y: 0 },
        data: {
          title: "Input",
          subtitle: `[batch, ${width}]`,
          values: run?.inputs[0] ?? [],
          selected: selected === "input",
          kind: "input",
        },
      },
    ];
    for (const [index, op] of model.spec.operations.entries()) {
      const previousWidth = width;
      if (op.weight)
        width = model.spec.tensors.find((t) => t.id === op.weight)!.shape[0];
      items.push({
        id: op.id,
        type: "layer",
        position: { x: (index + 1) * 242, y: 0 },
        data: {
          title:
            op.kind === "dense"
              ? op.id === "readout"
                ? "Readout"
                : "Dense layer"
              : op.kind.toUpperCase(),
          subtitle:
            op.kind === "dense"
              ? `${previousWidth} → ${width} features`
              : `[batch, ${width}]`,
          values: run?.activations[op.output]?.[0] ?? [],
          selected: selected === op.id,
          kind: op.kind,
        },
      });
    }
    return items;
  }, [model, run, selected]);
  const edges = nodes.slice(1).map((node, index) => ({
    id: `edge-${index}`,
    source: nodes[index].id,
    target: node.id,
    type: "smoothstep",
    style: { stroke: "#52786e", strokeWidth: 1.5 },
  }));
  return (
    <ReactFlow
      key={model.spec.id}
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodeClick={(_, node) => onSelect(node.id)}
      fitView
      fitViewOptions={{ padding: 0.18, maxZoom: 1 }}
      minZoom={0.3}
      maxZoom={1.8}
      nodesDraggable={false}
      nodesConnectable={false}
      deleteKeyCode={null}
      colorMode="dark"
      proOptions={{ hideAttribution: false }}
    >
      <Background
        variant={BackgroundVariant.Dots}
        gap={22}
        size={1}
        color="#28353d"
      />
      <Controls showInteractive={false} />
    </ReactFlow>
  );
}
