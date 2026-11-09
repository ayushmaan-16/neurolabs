import { test, expect, type Page } from "@playwright/test";

// Click the actual node or string using its matching keyboard target bounds.
// Native pointer events reach the canvas and exercise the scene's raycasting.
async function clickGraphTarget(page: Page, name: string) {
  const target = page.getByRole("button", { name, exact: true });
  await expect
    .poll(() => target.evaluate((element) => element.style.width))
    .not.toBe("");
  await target.scrollIntoViewIfNeeded();
  const bounds = await target.boundingBox();
  if (!bounds) throw new Error(`No projected bounds for ${name}`);
  await page.mouse.click(
    bounds.x + bounds.width / 2,
    bounds.y + bounds.height / 2,
  );
}

test("edit, rerun, reload, and restore an exact linear model", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "A rectified curve", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "The linear map" }).click();
  await expect(page.locator(".output-values strong")).toHaveText("0.5");
  await clickGraphTarget(page, "Select connection linear.weight [0, 0]");
  await page.getByLabel("Weight", { exact: true }).fill("1.5");
  await page.getByRole("button", { name: "Stage parameter edit" }).click();
  await expect(page.locator(".output-values strong")).toHaveText("1.5");
  await page
    .getByRole("button", { name: "Save revision", exact: true })
    .click();
  await expect(
    page.getByText("Modified checkpoint", { exact: true }),
  ).toBeVisible();
  await page.reload();
  await page.getByRole("button", { name: "The linear map" }).click();
  await expect(page.locator(".output-values strong")).toHaveText("1.5");
  await page
    .getByRole("button", { name: "Restore original", exact: true })
    .click();
  await expect(page.locator(".output-values strong")).toHaveText("0.5");
  await expect(
    page.getByRole("button", { name: "Restore original", exact: true }),
  ).toBeDisabled();
  expect(errors).toEqual([]);
});

test("matrix coordinates, model switching, inputs and narrow layout", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator(".output-values strong")).not.toHaveText("—");
  await page.getByRole("button", { name: "Weights", exact: true }).click();
  await page
    .getByRole("button", { name: "hidden.weight [2, 0] = -1", exact: true })
    .click();
  await expect(page.locator("#parameter-value")).toHaveValue("-1");
  await page.getByRole("button", { name: "The XOR classifier" }).click();
  await expect(page.locator(".output-values strong")).toHaveText("-1 , 1");
  await page.getByLabel("Input 0", { exact: true }).fill("1");
  await expect(page.locator(".output-values strong")).toHaveText("1 , -1");
  await page.getByRole("button", { name: "Architecture", exact: true }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(
    page.getByRole("heading", { name: "The XOR classifier", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
});

test("capture the default workspace for visual review", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator(".output-values strong")).toHaveText("0.3");
  await page.screenshot({
    path: ".local/workspace-preview.png",
    fullPage: true,
  });
});

test("3D playback exposes actual ReLU suppression and invalidates when inputs change", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.locator(".output-values strong")).toHaveText("0.3");
  await expect(page.locator(".three-viewport canvas")).toBeVisible();
  await page.getByRole("button", { name: "Run & trace", exact: true }).click();
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await page.getByRole("button", { name: "Step layer", exact: true }).click();
  const blocked = page.getByRole("button", {
    name: "Select hidden neuron 2",
    exact: true,
  });
  await expect(blocked).toHaveAttribute("data-activation", "0");
  await expect(blocked).toHaveAttribute("data-zero-output", "true");
  await clickGraphTarget(page, "Select hidden neuron 2");
  await expect(page.getByTestId("neuron-preactivation")).toHaveText("-0.5");
  await expect(page.getByTestId("neuron-activation")).toHaveText("0");
  await page.getByRole("button", { name: "Step layer", exact: true }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "Forward pass complete" }),
  ).toBeVisible();
  await page.screenshot({
    path: ".local/neurons-3d-preview.png",
    fullPage: true,
  });
  await page.getByLabel("Input 0", { exact: true }).fill("-1");
  await expect(blocked).toHaveAttribute("data-activation", "1");
  await expect(
    page.getByRole("button", { name: "Step layer", exact: true }),
  ).toBeDisabled();
  expect(errors).toEqual([]);
});

test("edges and neurons have distinct inspectors and edits recompute outputs", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.locator(".output-values strong")).toHaveText("0.3");
  await expect(page.locator(".edge-target")).toHaveCount(6);
  await expect(page.locator(".neuron-target")).toHaveCount(5);
  expect((await page.locator(".graph-hit-targets").textContent())?.trim()).toBe(
    "",
  );
  await expect(
    page.locator(".neuron-label, .neuron-bias, .edge-weight"),
  ).toHaveCount(0);
  await clickGraphTarget(page, "Select connection readout.weight [0, 0]");
  await expect(page.locator(".inspector")).toHaveAttribute(
    "data-selection-kind",
    "edge",
  );
  await expect(page.locator(".inspector #tensor-select")).toHaveCount(0);
  await expect(page.getByTestId("edge-input")).toHaveText("1.5");
  await expect(page.getByTestId("edge-contribution")).toHaveText("1.2");
  await page.locator("#parameter-value").fill("1.8");
  await page.getByRole("button", { name: "Stage parameter edit" }).click();
  await expect(page.locator(".output-values strong")).toHaveText("1.8");
  await clickGraphTarget(page, "Select readout neuron 0");
  await expect(page.locator(".inspector")).toHaveAttribute(
    "data-selection-kind",
    "neuron",
  );
  await expect(page.getByLabel("Bias", { exact: true })).toBeVisible();
  await expect(page.getByTestId("neuron-bias")).toHaveValue("-0.3");
  await page.locator("#parameter-value").fill("0.2");
  await page.getByRole("button", { name: "Stage parameter edit" }).click();
  await expect(page.locator(".output-values strong")).toHaveText("2.3");
  await page
    .getByRole("button", { name: "Save revision", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Restore original", exact: true })
    .click();
  await expect(page.locator(".output-values strong")).toHaveText("0.3");
  await page.screenshot({ path: ".local/space-workspace.png", fullPage: true });
  expect(errors).toEqual([]);
});

test("deeper example supports selection of all twenty strings and ten neurons", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Deep signal study" }).click();
  await expect(page.locator(".output-values strong")).not.toHaveText("—");
  await expect(page.locator(".edge-target")).toHaveCount(20);
  await expect(page.locator(".neuron-target")).toHaveCount(10);
  await expect(page.locator(".stage-label")).toHaveCount(0);
  // Keyboard users can select the same scene component without precision pointing.
  await page
    .getByRole("button", {
      name: "Select connection stage2.weight [1, 2]",
      exact: true,
    })
    .focus();
  await page.keyboard.press("Enter");
  await expect(page.locator(".inspector-selection h3")).toHaveText(
    "stage1[2] → stage2[1]",
  );
  await expect(page.locator("#parameter-value")).toHaveValue("0.5");
  await page.screenshot({
    path: ".local/deep-space-workspace.png",
    fullPage: true,
  });
});

test("nodes and strings remain selectable after zoom and orbit", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.locator(".output-values strong")).toHaveText("0.3");
  const weight = page.getByRole("button", {
    name: "Select connection readout.weight [0, 0]",
    exact: true,
  });
  const bias = page.getByRole("button", {
    name: "Select readout neuron 0",
    exact: true,
  });
  await expect
    .poll(() =>
      weight.evaluate((element) => parseFloat(element.style.width) || 0),
    )
    .toBeGreaterThan(1);
  const beforeWeight = (await weight.boundingBox())!.width;
  const beforeBias = (await bias.boundingBox())!.width;
  const viewport = await page.locator(".three-viewport canvas").boundingBox();
  await page.mouse.move(
    viewport!.x + viewport!.width / 2,
    viewport!.y + viewport!.height / 2,
  );
  await page.mouse.wheel(0, -400);
  await expect
    .poll(async () => (await weight.boundingBox())!.width)
    .toBeGreaterThan(beforeWeight * 1.15);
  await expect
    .poll(async () => (await bias.boundingBox())!.width)
    .toBeGreaterThan(beforeBias * 1.15);
  await clickGraphTarget(page, "Select connection readout.weight [0, 0]");
  await expect(page.locator(".inspector")).toHaveAttribute(
    "data-selection-kind",
    "edge",
  );
  await expect(page.locator(".inspector #tensor-select")).toHaveCount(0);
  await page.getByRole("button", { name: "Reset camera", exact: true }).click();
  await page.mouse.move(viewport!.x + 100, viewport!.y + 70);
  await page.mouse.down();
  await page.mouse.move(viewport!.x + 150, viewport!.y + 85, { steps: 12 });
  await page.mouse.up();
  await clickGraphTarget(page, "Select readout neuron 0");
  await expect(page.locator(".inspector")).toHaveAttribute(
    "data-selection-kind",
    "neuron",
  );
  await expect(page.getByLabel("Bias", { exact: true })).toBeVisible();
  await expect(page.locator(".edge-label-leaders")).toHaveCount(0);
  await page.screenshot({
    path: ".local/integrated-labels.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("input neurons have their own inspector and strings respect reduced motion", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator(".output-values strong")).toHaveText("0.3");
  await clickGraphTarget(page, "Select input neuron 0");
  await expect(page.getByTestId("neuron-input")).toHaveText("0.5");
  await expect(page.locator("#tensor-select")).toBeHidden();
  const edge = page.getByRole("button", {
    name: "Select connection readout.weight [0, 0]",
    exact: true,
  });
  const before = (await edge.boundingBox())!;
  await expect
    .poll(async () => {
      const after = (await edge.boundingBox())!;
      return Math.hypot(after.x - before.x, after.y - before.y);
    })
    .toBeGreaterThan(1);
  await expect(page.locator(".output-values strong")).toHaveText("0.3");
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.evaluate(
    () =>
      new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
      ),
  );
  const frozen = (await edge.boundingBox())!;
  await page.evaluate(
    () =>
      new Promise<void>((resolve) => {
        let frames = 0;
        const next = () => {
          if (++frames === 20) resolve();
          else requestAnimationFrame(next);
        };
        requestAnimationFrame(next);
      }),
  );
  const later = (await edge.boundingBox())!;
  expect(Math.hypot(later.x - frozen.x, later.y - frozen.y)).toBeLessThan(0.1);
  await page.screenshot({ path: ".local/string-graph.png", fullPage: true });
});

test("inspectors show one editor, collapse details, and retain restoration", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator(".output-values strong")).toHaveText("0.3");
  const panel = page.locator(".compact-inspector");
  await clickGraphTarget(page, "Select hidden neuron 2");
  await expect(panel.locator('input[type="number"]')).toHaveCount(1);
  await expect(page.getByLabel("Bias", { exact: true })).toHaveValue("0");
  await expect(page.getByTestId("neuron-preactivation")).toHaveText("-0.5");
  await expect(page.getByTestId("neuron-activation")).toHaveText("0");
  await expect(panel.locator(".inspector-breakdown")).not.toHaveAttribute(
    "open",
    "",
  );
  await expect(panel.locator(".inspector-history")).not.toHaveAttribute(
    "open",
    "",
  );
  await expect(
    panel.locator(
      ".operation-formula, .mini-matrix, #tensor-select, .inspector-footer",
    ),
  ).toHaveCount(0);
  await panel.locator(".inspector-breakdown summary").click();
  await page
    .getByRole("button", { name: "Inspect incoming connection 0", exact: true })
    .click();
  await expect(panel).toHaveAttribute("data-selection-kind", "edge");
  await expect(page.getByLabel("Weight", { exact: true })).toHaveValue("-1");
  await expect(page.getByTestId("edge-input")).toHaveText("0.5");
  await expect(page.getByTestId("edge-contribution")).toHaveText("-0.5");
  await expect(panel.locator(".inspector-breakdown")).toHaveCount(0);
  await clickGraphTarget(page, "Select readout neuron 0");
  await expect(page.getByLabel("Bias", { exact: true })).toHaveValue("-0.3");
  await expect(
    page.getByRole("button", { name: "Stage parameter edit" }),
  ).toBeDisabled();
  await page.getByLabel("Bias", { exact: true }).fill("-0.3");
  await expect(
    page.getByRole("button", { name: "Stage parameter edit" }),
  ).toBeDisabled();
  await page.getByLabel("Bias", { exact: true }).fill("0.2");
  await expect(
    panel.getByText("Unstaged change", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Stage parameter edit" }).click();
  await expect(page.locator(".output-values strong")).toHaveText("0.8");
  await page
    .getByRole("button", { name: "Save revision", exact: true })
    .click();
  await panel.locator(".inspector-history summary").click();
  await panel.getByRole("button", { name: /Original model/ }).click();
  await expect(page.locator(".output-values strong")).toHaveText("0.3");
  await panel.locator(".inspector-history summary").click();
  await page.screenshot({
    path: ".local/inspector-neuron.png",
    fullPage: true,
  });
  await clickGraphTarget(page, "Select connection readout.weight [0, 0]");
  await page.screenshot({
    path: ".local/inspector-connection.png",
    fullPage: true,
  });
});

test("a linear neuron omits duplicated preactivation and absent bias editors", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "The linear map" }).click();
  await expect(page.locator(".output-values strong")).toHaveText("0.5");
  await clickGraphTarget(page, "Select linear neuron 0");
  const panel = page.locator(".inspector");
  await expect(page.getByTestId("neuron-activation")).toHaveText("0.5");
  await expect(panel.getByText("Not present", { exact: true })).toBeVisible();
  await expect(
    panel.locator("input[type=number], [data-testid=neuron-preactivation]"),
  ).toHaveCount(0);
});
