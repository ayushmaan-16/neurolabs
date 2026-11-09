import type { Parameters } from "./api";

export type Change = { tensor_id: string; index: number[]; value: number };
export type Draft = {
  base_revision: string;
  idempotency_key: string;
  changes: Change[];
};
export const draftStorageKey = "neurolabs.parameter-drafts.v1";
export const cellKey = (change: Pick<Change, "tensor_id" | "index">) =>
  `${change.tensor_id}|${change.index.join(",")}`;

export function readDrafts(): Record<string, Draft> {
  try {
    const parsed = JSON.parse(sessionStorage.getItem(draftStorageKey) ?? "{}");
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
      return {};
    return Object.fromEntries(
      Object.entries(parsed)
        .filter(([id, item]) => {
          const draft = item as Draft;
          return (
            /^[A-Za-z][A-Za-z0-9_.-]{0,63}$/.test(id) &&
            draft &&
            typeof draft.base_revision === "string" &&
            typeof draft.idempotency_key === "string" &&
            Array.isArray(draft.changes) &&
            draft.changes.length > 0 &&
            draft.changes.length <= 4096 &&
            draft.changes.every(
              (change) =>
                change &&
                typeof change.tensor_id === "string" &&
                Array.isArray(change.index) &&
                change.index.length >= 1 &&
                change.index.length <= 2 &&
                change.index.every(
                  (index) =>
                    Number.isInteger(index) && index >= 0 && index < 256,
                ) &&
                typeof change.value === "number" &&
                Number.isFinite(Math.fround(change.value)),
            )
          );
        })
        .slice(0, 20),
    ) as Record<string, Draft>;
  } catch {
    return {};
  }
}

export function overlay(
  saved: Parameters | null,
  changes: Change[],
): Parameters | null {
  if (!saved || !changes.length) return saved;
  const tensors = { ...saved.tensors };
  const copied = new Set<string>();
  for (const change of changes) {
    const tensor = tensors[change.tensor_id];
    if (
      !tensor ||
      tensor.shape.length !== change.index.length ||
      change.index.some((index, axis) => index >= tensor.shape[axis])
    )
      continue;
    if (!copied.has(change.tensor_id)) {
      tensors[change.tensor_id] = {
        ...tensor,
        values: structuredClone(tensor.values),
      };
      copied.add(change.tensor_id);
    }
    const values = tensors[change.tensor_id].values;
    if (change.index.length === 1)
      (values as number[])[change.index[0]] = Math.fround(change.value);
    else
      (values as number[][])[change.index[0]][change.index[1]] = Math.fround(
        change.value,
      );
  }
  return { revision_id: `${saved.revision_id}:draft`, tensors };
}
