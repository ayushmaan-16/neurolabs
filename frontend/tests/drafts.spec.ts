import { test, expect, type Page } from "@playwright/test";

async function history(page: Page) {
  return (await page.request.get("/api/v1/models/curve/revisions")).json();
}
async function stageBias(page: Page, value: string) {
  await page.getByLabel("Bias", { exact: true }).fill(value);
  await page
    .getByRole("button", { name: "Stage parameter edit", exact: true })
    .click();
}

test("multiple edits preview, survive refresh and save as exactly one revision", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await expect(page.locator(".output-values strong")).toHaveText("0.3");
  const before = await history(page);
  await stageBias(page, "2");
  await expect(page.locator(".output-values strong")).toHaveText("1.1");
  await page
    .getByRole("navigation", { name: "Architecture tree" })
    .getByRole("button", { name: "readout dense", exact: true })
    .click();
  await page.locator(".inspector-breakdown summary").click();
  await page
    .getByRole("button", { name: "Inspect incoming connection 0", exact: true })
    .click();
  await page.getByLabel("Weight", { exact: true }).fill("1.8");
  await page
    .getByRole("button", { name: "Stage parameter edit", exact: true })
    .click();
  await expect(page.locator(".output-values strong")).toHaveText("3.6");
  await expect(
    page.getByText("2 staged parameter changes", { exact: true }),
  ).toBeVisible();
  await page.screenshot({ path: ".local/revision-draft.png", fullPage: true });
  expect((await history(page)).length).toBe(before.length);
  await expect(
    page.getByRole("button", { name: "Restore original", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "Run & trace", exact: true }).click();
  await page.getByLabel("Propagation progress").fill("2");
  await expect(
    page.getByRole("status").filter({ hasText: "Forward pass complete" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Diagnostics", exact: true }).click();
  await page.getByLabel("Input batch", { exact: false }).fill("0.5\n1\n2");
  await page
    .getByRole("button", { name: "Analyze batch", exact: true })
    .click();
  await expect(
    page.getByText("3 inputs analyzed", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".audit-provenance")).toContainText(
    "Unsaved draft of",
  );
  expect((await history(page)).length).toBe(before.length);
  await page.reload();
  await expect(
    page.getByText("2 staged parameter changes", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".output-values strong")).toHaveText("3.6");
  await page
    .getByRole("button", { name: "Save revision", exact: true })
    .click();
  await expect(
    page.getByRole("region", { name: "Parameter draft" }),
  ).toHaveCount(0);
  await expect(page.locator(".output-values strong")).toHaveText("3.6");
  const after = await history(page);
  expect(after.length).toBe(before.length + 1);
  expect(after[0].changes).toHaveLength(2);
  await page
    .getByRole("button", { name: "Restore original", exact: true })
    .click();
  await expect(page.locator(".output-values strong")).toHaveText("0.3");
});

test("repeated edits collapse, restoring a value unstages it and discard never saves", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator(".output-values strong")).toHaveText("0.3");
  const before = await history(page);
  await stageBias(page, "2");
  await stageBias(page, "3");
  await expect(
    page.getByText("1 staged parameter change", { exact: true }),
  ).toBeVisible();
  await stageBias(page, "1");
  await expect(
    page.getByRole("region", { name: "Parameter draft" }),
  ).toHaveCount(0);
  await expect(page.locator(".output-values strong")).toHaveText("0.3");
  await stageBias(page, "2");
  await page.getByRole("button", { name: "The linear map" }).click();
  await expect(page.locator(".output-values strong")).toHaveText("0.5");
  await page
    .getByRole("button", { name: "A rectified curve", exact: false })
    .click();
  await expect(page.locator(".output-values strong")).toHaveText("1.1");
  await page.getByLabel("Bias", { exact: true }).fill("4");
  await expect(
    page.getByRole("button", { name: "Save revision", exact: true }),
  ).toBeDisabled();
  await page
    .getByRole("button", { name: "Discard draft", exact: true })
    .click();
  await expect(page.getByLabel("Bias", { exact: true })).toHaveValue("1");
  await expect(page.locator(".output-values strong")).toHaveText("0.3");
  expect((await history(page)).length).toBe(before.length);
  await stageBias(page, "2");
  await page.getByText("Review staged changes", { exact: true }).click();
  await page
    .getByRole("button", { name: "Unstage hidden.bias [0]", exact: true })
    .click();
  await expect(
    page.getByRole("region", { name: "Parameter draft" }),
  ).toHaveCount(0);
  expect((await history(page)).length).toBe(before.length);
});

test("a conflicting save preserves the draft and does not overwrite another revision", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator(".output-values strong")).toHaveText("0.3");
  const model = await (await page.request.get("/api/v1/models/curve")).json();
  await stageBias(page, "2");
  const token = (await (await page.request.get("/api/v1/session")).json())
    .token;
  const other = await page.request.post("/api/v1/models/curve/revisions", {
    headers: { "x-neurolabs-session": token },
    data: {
      base_revision: model.head,
      idempotency_key: "other-tab",
      changes: [{ tensor_id: "readout.bias", index: [0], value: 0.2 }],
    },
  });
  expect(other.status()).toBe(201);
  await page
    .getByRole("button", { name: "Save revision", exact: true })
    .click();
  await expect(page.getByText(/This draft has not been rebased/)).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Save revision", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByText("1 staged parameter change", { exact: true }),
  ).toBeVisible();
  expect(
    (await (await page.request.get("/api/v1/models/curve")).json()).head,
  ).toBe((await other.json()).id);
  await page
    .getByRole("button", { name: "Discard draft", exact: true })
    .click();
  await expect(page.locator(".output-values strong")).toHaveText("0.8");
  await page
    .getByRole("button", { name: "Restore original", exact: true })
    .click();
  await expect(page.locator(".output-values strong")).toHaveText("0.3");
});
