import { expect, test } from "@playwright/test";

test("landing renders the headline result and the key sections", async ({ page }) => {
  await page.goto("index.html");

  // hero headline + the honest best reproducible number (9-model, not a fold)
  await expect(page.locator("h1")).toContainText("out-hunts");
  await expect(page.locator(".score.big .num")).toContainText("75.34");

  // the major sections all exist
  for (const id of ["result", "datasets", "explore", "separates", "separates-cars", "eli5"]) {
    await expect(page.locator(`#${id}`)).toBeAttached();
  }

  // benchmark table shows the SFORA rows; no leftover test-fitted "100%" fold claim
  await expect(page.getByText("SFORA — 9-model ensemble")).toBeVisible();
  await expect(page.getByText("retrieval-aware")).toHaveCount(0);
});

test("separation viz: clicking a legend species isolates it in every panel", async ({ page }) => {
  await page.goto("index.html");
  const chip = page.locator("#separates .leg-chip").first();
  await expect(chip).toHaveAttribute("aria-pressed", "false");
  await chip.click();
  await expect(chip).toHaveAttribute("aria-pressed", "true");
  // the grid gets an isolate class so non-matching dots dim
  await expect(page.locator("#separates .proj-grid.iso-0")).toBeAttached();
});

test("report page renders the method, references, and GPA as the best fold", async ({ page }) => {
  await page.goto("report/index.html");
  await expect(page.locator("h1")).toContainText("HERD");
  await expect(page.locator("#fold")).toBeAttached();
  await expect(page.getByText("GPA-aligned mean").first()).toBeVisible();
  // the honest position: no test-set-fitted projection
  await expect(page.getByText("retrieval-aware")).toHaveCount(0);
  await expect(page.locator("#refs")).toBeAttached();
});
