import { test, expect } from "@playwright/test";
import { execFileSync } from "node:child_process";

test.describe.configure({ timeout: 90_000 });
test.beforeAll(() => {
  const python =
    process.platform === "win32"
      ? ".venv/Scripts/python.exe"
      : ".venv/bin/python";
  execFileSync(python, ["scripts/create-import-test-fixtures.py"], {
    timeout: 60_000,
  });
});

test("upload preview, independent model, live inputs, named edits, diagnostics and export roundtrip", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  const before = await (await page.request.get("/api/v1/models")).json();
  await page.getByRole("button", { name: "Import model", exact: true }).click();
  await page
    .getByLabel("Model package", { exact: true })
    .setInputFiles(".local/import-small.zip");
  const preview = page.getByRole("region", {
    name: "Model compatibility preview",
  });
  await expect(preview).toContainText(
    "Output verification passed on 2 supplied examples",
    { timeout: 40_000 },
  );
  await expect(preview).toContainText("43");
  expect((await (await page.request.get("/api/v1/models")).json()).length).toBe(
    before.length,
  );
  await page.screenshot({ path: ".local/import-preview.png", fullPage: true });
  await page
    .getByRole("button", { name: "Import into workspace", exact: true })
    .click();
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Upload test MLP", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".output-values strong")).not.toHaveText("—");
  await expect(
    page.getByLabel("Input vector · 4 values", { exact: true }),
  ).toHaveValue("0.2, -0.7, 1.1, 0.4");
  await expect(page.locator(".neuron-target")).toHaveCount(12);
  await expect(page.locator(".output-values strong")).not.toHaveText("—");
  await page
    .getByLabel("Input vector · 4 values", { exact: true })
    .fill("0.4, 0.1, -0.7, 1");
  await page.getByRole("button", { name: "Apply input", exact: true }).click();
  await page.getByRole("button", { name: "Weights", exact: true }).click();
  await page.getByLabel("Tensor", { exact: true }).selectOption("op01.weight");
  await page.getByLabel("Destination row", { exact: true }).fill("1");
  await page.getByLabel("Source column", { exact: true }).fill("3");
  await page
    .getByRole("button", { name: "Select coordinate", exact: true })
    .click();
  await page.getByLabel("Weight", { exact: true }).fill("0.9");
  await page
    .getByRole("button", { name: "Stage parameter edit", exact: true })
    .click();
  await page
    .getByLabel("Revision name", { exact: true })
    .fill("Imported repair test");
  await page
    .getByRole("button", { name: "Save revision", exact: true })
    .click();
  await expect(page.locator(".revision-heading-name")).toHaveText(
    "Imported repair test",
  );
  await page.getByRole("button", { name: "Diagnostics", exact: true }).click();
  await page
    .getByLabel("Input batch", { exact: false })
    .fill("0.2,-0.7,1.1,0.4\n0.4,0.1,-0.7,1");
  await page
    .getByRole("button", { name: "Analyze batch", exact: true })
    .click();
  await expect(
    page.getByText("2 inputs analyzed", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".audit-provenance")).toContainText(
    "Imported repair test",
  );
  const after = await (await page.request.get("/api/v1/models")).json();
  const imported = after.find(
    (item: any) => item.spec.name === "Upload test MLP",
  );
  await page.reload();
  await page.getByRole("button", { name: /^Upload test MLP/ }).click();
  await expect(page.locator(".revision-heading-name")).toHaveText(
    "Imported repair test",
  );
  const download = page.waitForEvent("download");
  await page.getByRole("link", { name: /^Export saved revision/ }).click();
  const exportedPath = (await (await download).path())!;
  await page.getByRole("button", { name: "Import model", exact: true }).click();
  await page.getByLabel("Model package", { exact: true }).setInputFiles({
    name: "exported.zip",
    mimeType: "application/zip",
    buffer: await (await import("node:fs/promises")).readFile(exportedPath),
  });
  await expect(preview).toContainText(
    "Output verification passed on 1 supplied examples",
    { timeout: 40_000 },
  );
  await page
    .getByRole("button", { name: "Import into workspace", exact: true })
    .click();
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Upload test MLP", exact: true }),
  ).toBeVisible();
  const models = await (await page.request.get("/api/v1/models")).json();
  const copied = models
    .filter((item: any) => item.spec.name === "Upload test MLP")
    .at(-1);
  expect(copied.spec.id).not.toBe(imported.spec.id);
  const parameters = await (
    await page.request.get(
      `/api/v1/models/${copied.spec.id}/revisions/${copied.head}/parameters`,
    )
  ).json();
  expect(parameters.tensors["op01.weight"].values[1][3]).toBeCloseTo(0.9);
  await page.screenshot({ path: ".local/imported-model.png", fullPage: true });
});

test("maximum-width package uses sampled geometry and bounded matrix blocks", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await page.getByRole("button", { name: "Import model", exact: true }).click();
  await page
    .getByLabel("Model package", { exact: true })
    .setInputFiles(".local/import-large.zip");
  await expect(
    page.getByRole("region", { name: "Model compatibility preview" }),
  ).toContainText("987,137", { timeout: 40_000 });
  await page
    .getByRole("button", { name: "Import into workspace", exact: true })
    .click();
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Maximum import fixture", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".output-values strong")).not.toHaveText("—");
  await expect(page.locator(".neuron-target")).toHaveCount(4097);
  expect(
    Number(
      await page
        .locator(".three-viewport")
        .getAttribute("data-rendered-connections"),
    ),
  ).toBeLessThan(1600);
  await expect(
    page.getByLabel("Connection detail").locator('option[value="all"]'),
  ).toBeDisabled();
  await page.getByRole("button", { name: "Weights", exact: true }).click();
  await page.getByLabel("Tensor", { exact: true }).selectOption("op01.weight");
  await expect(page.locator(".tensor-cell")).toHaveCount(4096);
  await page.getByLabel("Destination row", { exact: true }).fill("255");
  await page.getByLabel("Source column", { exact: true }).fill("255");
  await page
    .getByRole("button", { name: "Select coordinate", exact: true })
    .click();
  await expect(
    page.getByLabel("Matrix row block", { exact: true }),
  ).toHaveValue("192");
  await expect(
    page.getByLabel("Matrix column block", { exact: true }),
  ).toHaveValue("192");
  await expect(page.locator(".tensor-cell.selected")).toHaveAttribute(
    "aria-label",
    /^op01.weight \[255, 255\]/,
  );
  await expect(page.getByLabel("Weight", { exact: true })).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator(".tensor-cell.selected")).toBeInViewport();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: ".local/import-wide-mobile.png",
    fullPage: true,
  });
});

test("invalid uploads report the reason and leave the collection unchanged", async ({
  page,
}) => {
  await page.goto("/");
  const before = await (await page.request.get("/api/v1/models")).json();
  await page.getByRole("button", { name: "Import model", exact: true }).click();
  await page.getByLabel("Model package", { exact: true }).setInputFiles({
    name: "model.py",
    mimeType: "text/plain",
    buffer: Buffer.from("not a package"),
  });
  await expect(page.getByRole("alert")).toContainText(
    "Choose a neuroLabs .zip package",
  );
  await page.getByLabel("Model package", { exact: true }).setInputFiles({
    name: "broken.zip",
    mimeType: "application/zip",
    buffer: Buffer.from("not a package"),
  });
  await expect(page.getByRole("alert")).toContainText("Invalid model package", {
    timeout: 40_000,
  });
  await expect(
    page.getByRole("button", { name: "Import into workspace", exact: true }),
  ).toHaveCount(0);
  expect((await (await page.request.get("/api/v1/models")).json()).length).toBe(
    before.length,
  );
  await page.getByRole("button", { name: "Close import", exact: true }).click();
  await expect(page.getByRole("dialog")).not.toBeVisible();
});
