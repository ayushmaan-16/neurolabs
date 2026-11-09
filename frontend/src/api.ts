export type Operation = {
  kind: "dense" | "relu" | "tanh" | "sigmoid" | "softmax" | "identity";
  id: string;
  input: string;
  output: string;
  weight?: string;
  bias?: string | null;
};
export type Model = {
  spec: {
    id: string;
    name: string;
    description: string;
    input_name: string;
    input_width: number;
    output_name: string;
    task: string;
    tensors: { id: string; shape: number[] }[];
    operations: Operation[];
  };
  head: string;
  original: string;
  default_input: number[];
  head_name: string;
  original_name: string;
};
export type Parameters = {
  revision_id: string;
  tensors: Record<string, { shape: number[]; values: number[] | number[][] }>;
};
export type Revision = {
  id: string;
  name: string;
  parent: string | null;
  created_at: string;
  changes: { tensor_id: string; index: number[]; old: number; value: number }[];
};

export function revisionName(
  id: string,
  model: Model,
  history: Revision[],
): string {
  return (
    (id === model.original
      ? model.original_name
      : id === model.head
        ? model.head_name
        : history.find((revision) => revision.id === id)?.name) ||
    `Revision ${id.slice(0, 8)}`
  );
}
export type Run = {
  id: string;
  revision_id: string;
  input_hash: string;
  inputs: number[][];
  outputs: number[][];
  activations: Record<string, number[][]>;
  elapsed_ms: number;
  is_draft?: boolean;
  base_revision?: string;
};

let session: Promise<string> | undefined;
export async function api<T>(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const headers: Record<string, string> = {};
  if (body !== undefined) {
    session ??= fetch("/api/v1/session")
      .then(async (response) => {
        if (!response.ok)
          throw new Error(
            "Cannot establish a local session. Is the backend running?",
          );
        return (await response.json()).token as string;
      })
      .catch((error) => {
        session = undefined;
        throw error;
      });
    headers["Content-Type"] = "application/json";
    headers["X-NeuroLabs-Session"] = await session;
  }
  const response = await fetch(`/api/v1${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  const payload = await response.json();
  if (!response.ok) {
    if (response.status === 403) session = undefined;
    throw new Error(
      payload.error?.message ??
        payload.detail?.[0]?.msg ??
        `Request failed (${response.status})`,
    );
  }
  return payload;
}

export const number = (value: number | undefined) =>
  value === undefined
    ? "—"
    : new Intl.NumberFormat("en-US", { maximumFractionDigits: 5 }).format(
        value,
      );

/** Shortest decimal that round-trips to this stored float32 parameter. */
export function parameterText(value: number | undefined): string {
  if (value === undefined) return "";
  for (let digits = 1; digits <= 9; digits++) {
    const text = String(Number(value.toPrecision(digits)));
    if (Math.fround(Number(text)) === value) return text;
  }
  return String(value);
}
