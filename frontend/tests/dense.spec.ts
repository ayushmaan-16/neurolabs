import { test, expect, type Page } from "@playwright/test";

test.describe.configure({ timeout: 60_000 });

async function openDense(page: Page) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await page.getByRole("button", { name: "Dense sixteen" }).click();
  await expect(page.locator(".output-values strong")).not.toHaveText("—");
  await expect(page.locator(".three-viewport canvas")).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);
}

async function clickNode(page: Page, name: string) {
  const target = page.getByRole("button", { name, exact: true });
  await expect
    .poll(() => target.evaluate((el) => parseFloat(el.style.width) || 0))
    .toBeGreaterThan(2);
  await target.scrollIntoViewIfNeeded();
  const bounds = (await target.boundingBox())!;
  await page.mouse.click(
    bounds.x + bounds.width / 2,
    bounds.y + bounds.height / 2,
  );
}

test("dense overview, full layer pairs and neuron focus remain bounded and selectable", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openDense(page);
  await expect(page.locator(".neuron-target")).toHaveCount(966);
  await expect(page.getByTestId("connection-summary")).toContainText(
    "Showing 1,536 of 57,728",
  );
  await expect
    .poll(async () =>
      Number(
        await page.locator(".three-viewport").getAttribute("data-draw-calls"),
      ),
    )
    .toBeGreaterThan(0);
  expect(
    Number(
      await page.locator(".three-viewport").getAttribute("data-draw-calls"),
    ),
  ).toBeLessThan(24);
  await page.screenshot({ path: ".local/dense-overview.png", fullPage: true });

  await page.getByLabel("Connection detail").selectOption("layer");
  await page.getByLabel("Layer pair", { exact: true }).selectOption("8");
  await expect(page.getByTestId("connection-summary")).toContainText(
    "Showing 4,096 of 57,728",
  );
  await expect
    .poll(() =>
      page.locator(".three-viewport").getAttribute("data-rendered-connections"),
    )
    .toBe("4096");
  expect(
    Number(
      await page.locator(".three-viewport").getAttribute("data-draw-calls"),
    ),
  ).toBeLessThan(24);
  await page.screenshot({
    path: ".local/dense-layer-pair.png",
    fullPage: true,
  });

  await page.getByLabel("Connection detail").selectOption("neuron");
  await page.getByLabel("Neuron layer", { exact: true }).selectOption("8");
  await page.getByLabel("Neuron index", { exact: true }).selectOption("62");
  await expect(page.getByTestId("connection-summary")).toContainText(
    "Showing 128 of 57,728",
  );
  await clickNode(page, "Select layer08 neuron 63");
  await expect(page.locator(".inspector-selection h3")).toContainText(
    "layer08 · neuron 63",
  );
  await page
    .getByRole("button", {
      name: "Select connection layer08.weight [63, 17]",
      exact: true,
    })
    .focus();
  await page.keyboard.press("Enter");
  await expect(page.locator(".inspector")).toHaveAttribute(
    "data-selection-kind",
    "edge",
  );
  const source = Number(await page.getByTestId("edge-input").textContent());
  const weight = Number(
    await page.getByLabel("Weight", { exact: true }).inputValue(),
  );
  const contribution = Number(
    await page.getByTestId("edge-contribution").textContent(),
  );
  expect(Math.abs(source * weight - contribution)).toBeLessThan(0.001);
  await page.screenshot({
    path: ".local/dense-neuron-focus.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Run & trace", exact: true }).click();
  await expect(page.getByLabel("Propagation progress")).toBeEnabled();
  await page.getByLabel("Propagation progress").fill("16");
  await expect(
    page.getByRole("status").filter({ hasText: "Forward pass complete" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", {
      name: "Select layer16 neuron 0",
      exact: true,
      includeHidden: true,
    }),
  ).not.toHaveAttribute("data-activation", "pending");

  // Returning to a small model must release the dense scene and retain native selection.
  await page
    .getByRole("button", { name: "A rectified curve", exact: false })
    .click();
  await expect(page.locator(".three-viewport canvas")).toHaveCount(1);
  await expect(page.locator(".neuron-target")).toHaveCount(5);
  await expect(page.getByLabel("Connection detail")).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("dense edge picking, bias edits, input invalidation and narrow controls", async ({
  page,
}) => {
  await openDense(page);
  await page.getByLabel("Connection detail").selectOption("neuron");
  await page.getByLabel("Neuron layer", { exact: true }).selectOption("0");
  await clickNode(page, "Select connection layer01.weight [63, 0]");
  await expect(page.locator(".inspector")).toHaveAttribute(
    "data-selection-kind",
    "edge",
  );
  await expect(page.locator(".inspector-selection h3")).toHaveText(
    "Input[0] → layer01[63]",
  );
  await page.getByLabel("Neuron layer", { exact: true }).selectOption("16");
  const before = (await page.locator(".output-values strong").innerText())
    .split(",")
    .map(Number);
  const bias = page.getByLabel("Bias", { exact: true });
  await bias.fill(String(Number(await bias.inputValue()) + 1));
  await page.getByRole("button", { name: "Apply parameter edit" }).click();
  await expect
    .poll(
      async () =>
        Number(
          (await page.locator(".output-values strong").innerText()).split(
            ",",
          )[0],
        ) - before[0],
    )
    .toBeCloseTo(1, 3);
  await page.getByRole("button", { name: "Run & trace", exact: true }).click();
  await expect(page.getByLabel("Propagation progress")).toBeEnabled();
  await page.getByLabel("Input 0", { exact: true }).fill("-1");
  await expect(page.getByLabel("Propagation progress")).toBeDisabled();
  await page
    .getByRole("button", { name: "Restore original", exact: true })
    .click();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByLabel("Connection detail")).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
});
