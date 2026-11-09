import { number, type Run } from "./api";

export function Plot({
  current,
  baseline,
  pending = false,
}: {
  current: Run | null;
  baseline: Run | null;
  pending?: boolean;
}) {
  if (!current || !baseline)
    return (
      <div className="plot-placeholder">
        {pending
          ? "Computing a response slice…"
          : "Response slice unavailable. The fixed −2 to 2 sweep could not complete; inspect the selected input above."}
      </div>
    );
  const all = [...current.outputs, ...baseline.outputs].flat();
  const lo = Math.min(...all) - 0.2,
    hi = Math.max(...all) + 0.2;
  const x = (i: number) =>
    45 + (i * 610) / Math.max(current.outputs.length - 1, 1);
  const y = (v: number) => 132 - ((v - lo) / (hi - lo)) * 112;
  const path = (run: Run, output: number) =>
    run.outputs
      .map((row, i) => `${i ? "L" : "M"}${x(i)},${y(row[output])}`)
      .join(" ");
  return (
    <svg
      className="response-chart"
      viewBox="0 0 685 158"
      role="img"
      aria-label="Output values across an input sweep from minus two to two; dashed lines show the original model"
    >
      {[0, 0.5, 1].map((t) => (
        <g key={t}>
          <line
            x1="45"
            x2="655"
            y1={20 + 112 * t}
            y2={20 + 112 * t}
            stroke="#25313b"
          />
          <text x="35" y={24 + 112 * t} textAnchor="end">
            {number(hi - (hi - lo) * t)}
          </text>
        </g>
      ))}
      {[-2, -1, 0, 1, 2].map((value, i) => (
        <text key={value} x={45 + i * 152.5} y="151" textAnchor="middle">
          {value}
        </text>
      ))}
      {current.outputs[0].map((_, index) => (
        <g key={index}>
          <path
            d={path(baseline, index)}
            fill="none"
            stroke="#8f9ba9"
            strokeWidth="1.5"
            strokeDasharray="5 5"
          />
          <path
            d={path(current, index)}
            fill="none"
            stroke={index ? "#c0a99b" : "#b4c5e9"}
            strokeWidth="2.5"
          />
        </g>
      ))}
    </svg>
  );
}
