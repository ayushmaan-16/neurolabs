import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";

test("batch inactivity, report export, 3D inspection and revision impact form one workflow", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.locator(".output-values strong")).toHaveText("0.3");
  await page.getByRole("button", { name: "Diagnostics", exact: true }).click();
  await page.getByLabel("Input batch", { exact: false }).fill("0.5\n1\n2");
  await page
    .getByRole("button", { name: "Analyze batch", exact: true })
    .click();
  await expect(page.getByTestId("audit-inactive")).toHaveText("1");
  await expect(page.getByTestId("audit-output-max")).toHaveText("0");
  await page.getByLabel("Neuron audit filter").selectOption("inactive");
  await expect(page.locator(".audit-neurons tbody tr")).toHaveCount(1);

  const downloadPromise = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Download report", exact: true })
    .click();
  const download = await downloadPromise;
  const report = JSON.parse(await readFile((await download.path())!, "utf8"));
  expect(report.sample_count).toBe(3);
  expect(report.layers[0].neurons[2].inactive_on_batch).toBe(true);
  expect(report.parameter_hash).toBe(report.baseline_parameter_hash);
  expect(report.input_hash).toHaveLength(64);
  expect(report.provenance.current.dtype).toBe("float32");
  expect(report.cases[0].inputs).toEqual([0.5]);
  await page
    .getByRole("button", {
      name: "Inspect audited hidden neuron 2",
      exact: true,
    })
    .click();
  await expect(page.locator(".three-viewport canvas")).toBeVisible();
  await expect(page.getByTestId("neuron-preactivation")).toHaveText("-0.5");
  await page.getByLabel("Bias", { exact: true }).fill("2");
  await page.getByRole("button", { name: "Stage parameter edit" }).click();
  await expect(page.locator(".output-values strong")).toHaveText("0.9");
  await page.getByRole("button", { name: "Diagnostics", exact: true }).click();
  await expect(page.getByLabel("Input batch", { exact: false })).toHaveValue(
    "0.5\n1\n2",
  );
  await expect(
    page.getByRole("button", { name: "Download report", exact: true }),
  ).toHaveCount(0);
  await page
    .getByRole("button", { name: "Analyze batch", exact: true })
    .click();
  await expect(page.getByTestId("audit-inactive")).toHaveText("0");
  await expect(page.getByTestId("audit-output-max")).toHaveText("0.6");
  await page.screenshot({
    path: ".local/batch-diagnostics.png",
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "Inspect input row 1", exact: true })
    .click();
  await expect(page.getByLabel("Input 0", { exact: true })).toHaveValue("0.5");
  await page
    .getByRole("button", { name: "Save revision", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Restore original", exact: true })
    .click();
  await expect(page.locator(".output-values strong")).toHaveText("0.3");
  expect(errors).toEqual([]);
});

test("custom CSV, dense neuron inspection, out-of-range cases and narrow audit layout", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await page.getByRole("button", { name: "Diagnostics", exact: true }).click();
  await page.getByLabel("Load input CSV").setInputFiles({
    name: "invalid.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("input\n1\n2"),
  });
  await expect(page.getByRole("alert")).toContainText("finite float32");
  await page.getByRole("button", { name: "Dense sixteen" }).click();
  await page
    .getByRole("button", { name: "Analyze batch", exact: true })
    .click();
  await expect(
    page.getByText("33 inputs analyzed", { exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: /^Audit layer / })).toHaveCount(
    16,
  );
  await page
    .getByRole("button", { name: "Audit layer layer08", exact: true })
    .click();
  await expect(page.locator(".audit-neurons tbody tr")).toHaveCount(64);
  await page
    .getByRole("button", {
      name: "Inspect audited layer08 neuron 63",
      exact: true,
    })
    .click();
  await expect(page.getByLabel("Connection detail")).toHaveValue("neuron");
  await expect(page.getByTestId("connection-summary")).toContainText(
    "Showing 128 of 57,728",
  );
  await page.getByRole("button", { name: "Diagnostics", exact: true }).click();
  await expect(
    page.getByText("33 inputs analyzed", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("Load input CSV").setInputFiles({
    name: "validation.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("0,0,0,0\n10,0,0,0"),
  });
  await expect(
    page.getByRole("button", { name: "Download report", exact: true }),
  ).toHaveCount(0);
  await page
    .getByRole("button", { name: "Analyze batch", exact: true })
    .click();
  await expect(
    page.getByText("2 inputs analyzed", { exact: true }),
  ).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page
    .getByRole("button", { name: "Inspect input row 2", exact: true })
    .click();
  await expect(page.getByLabel("Input 0", { exact: true })).toHaveValue("10");
  await expect(page.getByLabel("Input 0", { exact: true })).toHaveAttribute(
    "max",
    "10",
  );
});

test("changing a batch discards an in-flight report", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Diagnostics", exact: true }).click();
  let release!: () => void;
  let requested!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const requestStarted = new Promise<void>((resolve) => {
    requested = resolve;
  });
  await page.route("**/models/curve/diagnostics", async (route) => {
    const response = await route.fetch();
    requested();
    await gate;
    await route.fulfill({ response }).catch(() => {});
  });
  await page
    .getByRole("button", { name: "Analyze batch", exact: true })
    .click();
  await requestStarted;
  await page.getByLabel("Input batch", { exact: false }).fill("-1\n1");
  release();
  await expect(
    page.getByRole("button", { name: "Analyze batch", exact: true }),
  ).toBeEnabled();
  await expect(
    page.getByRole("button", { name: "Download report", exact: true }),
  ).toHaveCount(0);
});
