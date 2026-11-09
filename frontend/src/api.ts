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
  import_info?: {
    package_sha256: string;
    metadata: {
      feature_names: string[];
      output_names: string[];
      input_note: string;
    };
    verification: {
      status: string;
      sample_count: number;
      max_abs_error: number | null;
    };
  } | null;
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
async function sessionToken(): Promise<string> {
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
  return session;
}

export async function uploadPackage(
  file: File,
  signal: AbortSignal,
): Promise<ImportPreview> {
  const response = await fetch("/api/v1/imports/preview", {
    method: "POST",
    body: file,
    signal,
    headers: {
      "Content-Type": "application/zip",
      "X-NeuroLabs-Session": await sessionToken(),
    },
  });
  const payload = await response.json();
  if (!response.ok) {
    if (response.status === 403) session = undefined;
    throw new Error(
      payload.error?.message ?? "Could not check this model package",
    );
  }
  return payload;
}

export type ImportPreview = {
  token: string;
  manifest: {
    spec: Model["spec"];
    default_input: number[];
    metadata: NonNullable<Model["import_info"]>["metadata"];
  };
  parameter_count: number;
  sha256: string;
  verification: NonNullable<Model["import_info"]>["verification"];
};

export async function downloadPackage(
  path: string,
  filename: string,
): Promise<void> {
  const response = await fetch(`/api/v1${path}`);
  if (!response.ok) {
    const payload = await response.json();
    throw new Error(
      payload.error?.message ?? "Could not export this saved revision",
    );
  }
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function loadParameters(
  model: Model,
  revision: string,
  signal: AbortSignal,
): Promise<Parameters> {
  const prefix = `/models/${model.spec.id}/revisions/${revision}`;
  if (!model.import_info)
    return api<Parameters>(`${prefix}/parameters`, undefined, signal);
  const response = await fetch(`/api/v1${prefix}/parameter-data`, { signal });
  if (!response.ok) throw new Error("Could not load model parameters");
  if (response.headers.get("X-Revision-ID") !== revision)
    throw new Error("Parameter revision mismatch");
  const data = new DataView(await response.arrayBuffer());
  const count = model.spec.tensors.reduce(
    (sum, tensor) => sum + tensor.shape.reduce((a, b) => a * b, 1),
    0,
  );
  if (count > 1_000_000 || data.byteLength !== count * 4)
    throw new Error("Invalid parameter data size");
  let offset = 0;
  const tensors: Parameters["tensors"] = {};
  for (const tensor of model.spec.tensors) {
    const readRow = (width: number) =>
      Array.from({ length: width }, () => {
        const value = data.getFloat32(offset, true);
        offset += 4;
        return value;
      });
    tensors[tensor.id] = {
      shape: tensor.shape,
      values:
        tensor.shape.length === 1
          ? readRow(tensor.shape[0])
          : Array.from({ length: tensor.shape[0] }, () =>
              readRow(tensor.shape[1]),
            ),
    };
  }
  return { revision_id: revision, tensors };
}
export async function api<T>(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const headers: Record<string, string> = {};
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    headers["X-NeuroLabs-Session"] = await sessionToken();
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
