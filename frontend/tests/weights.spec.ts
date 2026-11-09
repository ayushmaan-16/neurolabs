import { test, expect } from "@playwright/test";

test("dense matrices stay readable, scroll to coordinates and select the inspector", async ({
  page,
}) => {
  test.setTimeout(60_000);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await page.getByRole("button", { name: "Dense sixteen" }).click();
  await expect(page.locator(".output-values strong")).not.toHaveText("—");
  await page.getByRole("button", { name: "Weights", exact: true }).click();
  await page
    .getByLabel("Tensor", { exact: true })
    .selectOption("layer08.weight");
  const matrix = page.getByRole("region", {
    name: "Parameter matrix scroll area",
  });
  await expect(matrix.locator(".tensor-cell")).toHaveCount(4096);
  const initial = matrix.locator(".tensor-cell").first();
  expect((await initial.boundingBox())!.width).toBeGreaterThanOrEqual(96);
  expect(await initial.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(
    true,
  );
  const dimensions = await matrix.evaluate((el) => ({
    width: el.clientWidth,
    fullWidth: el.scrollWidth,
    height: el.clientHeight,
    fullHeight: el.scrollHeight,
  }));
  expect(dimensions.fullWidth).toBeGreaterThan(dimensions.width);
  expect(dimensions.fullHeight).toBeGreaterThan(dimensions.height);
  await page.screenshot({ path: ".local/weights-matrix.png", fullPage: true });
  await page.getByLabel("Destination row", { exact: true }).fill("63");
  await page.getByLabel("Source column", { exact: true }).fill("63");
  await page
    .getByRole("button", { name: "Select coordinate", exact: true })
    .click();
  await expect(matrix.locator(".tensor-cell.selected")).toHaveAttribute(
    "aria-label",
    /^layer08.weight \[63, 63\]/,
  );
  await expect(page.getByLabel("Weight", { exact: true })).toBeVisible();
  expect(await matrix.evaluate((el) => el.scrollLeft)).toBeGreaterThan(0);
  expect(await matrix.evaluate((el) => el.scrollTop)).toBeGreaterThan(0);
  await page.getByLabel("Full precision", { exact: true }).check();
  expect(
    (await matrix.locator(".tensor-cell.selected").boundingBox())!.width,
  ).toBeGreaterThan(170);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  expect((await matrix.boundingBox())!.width).toBeLessThan(390);
  await page.getByLabel("Tensor", { exact: true }).selectOption("layer08.bias");
  await expect(matrix.locator(".tensor-cell")).toHaveCount(64);
  await expect(
    matrix.getByRole("columnheader", { name: "Bias", exact: true }),
  ).toBeVisible();
  await page.getByLabel("Destination row", { exact: true }).fill("63");
  await page
    .getByRole("button", { name: "Select coordinate", exact: true })
    .click();
  await expect(matrix.locator(".tensor-cell.selected")).toHaveAttribute(
    "aria-label",
    /^layer08.bias \[63\]/,
  );
  await expect(page.getByLabel("Bias", { exact: true })).toBeVisible();
  await page.screenshot({ path: ".local/weights-mobile.png", fullPage: true });
});
