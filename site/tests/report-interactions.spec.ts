import { expect, test } from "@playwright/test";

test("research report controls filter, sort, and switch panels", async ({ page }) => {
  await page.goto("/index.html");

  await expect(page.getByRole("heading", { name: "Interactive method anatomy" })).toBeVisible();
  await expect(page.locator(".research-answer article")).toHaveCount(4);
  await expect(page.locator(".protocol-flow article")).toHaveCount(4);
  await expect(page.locator(".result-overview-card")).toHaveCount(3);
  await expect(page.getByRole("heading", { name: "What changes from SupCon to Group SupCon?" })).toBeVisible();
  await expect(page.locator(".contrast-unit-grid article")).toHaveCount(3);
  await expect(page.locator('[data-anatomy-panel="full"]')).toBeVisible();
  await expect(page.locator('[data-anatomy-panel="supcon"]')).toBeHidden();
  await expect(page.getByRole("button", { name: "Group SupCon + XBM + Radius" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );

  await page.getByRole("button", { name: "Supervised Contrastive (SupCon)" }).click();
  await expect(page.getByRole("button", { name: "Supervised Contrastive (SupCon)" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.locator('[data-anatomy-panel="supcon"]')).toBeVisible();
  await expect(page.locator('[data-anatomy-panel="full"]')).toBeHidden();
  await expect(
    page.locator('[data-anatomy-panel="supcon"] [data-step="radius"]'),
  ).toHaveAttribute("data-enabled", "false");

  await page.getByRole("button", { name: "Group SupCon + XBM + Radius" }).click();
  await expect(page.getByRole("button", { name: "Group SupCon + XBM + Radius" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(
    page.locator('[data-anatomy-panel="full"] [data-step="radius"]'),
  ).toHaveAttribute("data-enabled", "true");

  await expect(page.locator("[data-visible-count]")).toHaveText("144");
  await expect(page.locator('[data-filter="method"] option[value="Group SupCon"]')).toHaveCount(1);
  expect(await page.locator('[data-result-row][data-method="Group SupCon"]').count()).toBeGreaterThan(0);
  await expect(page.locator(".row-badge.best").first()).toBeVisible();
  await expect(page.locator(".row-badge.ours").first()).toBeVisible();
  await expect(page.locator('tr[data-kind="worst"] .row-badge.worst').first()).toBeVisible();
  expect(await page.locator('[data-result-row][data-sign="negative"]').count()).toBeGreaterThan(0);
  await page.selectOption('[data-filter="dataset"]', "CUB");
  await expect(page.locator("[data-visible-count]")).toHaveText("48");

  const visibleDatasets = await page
    .locator('tr[data-result-row][data-view="table"]:visible td:first-child')
    .evaluateAll((nodes) => [...new Set(nodes.map((node) => node.textContent?.trim()))]);
  expect(visibleDatasets).toEqual(["CUB"]);

  await page.selectOption('[data-filter="method"]', "Group SupCon");
  await expect(page.locator("[data-visible-count]")).toHaveText("3");
  await expect(page.locator('[data-result-row][data-method="Group SupCon"]:visible .row-badge.ours').first()).toBeVisible();
  await page.selectOption('[data-filter="method"]', "all");

  await page.locator('th button[data-sort="name"]').click();
  await expect(page.locator('th button[data-sort="name"]')).toHaveAttribute("aria-pressed", "true");
  await expect(
    page.locator('tr[data-result-row][data-view="table"]:visible td:nth-child(2)').first(),
  ).toContainText("ArcFace");

  await page.locator("[data-chart-toggle]").click();
  await expect(page.locator("[data-chart-toggle]")).toHaveText("Show fewer chart rows");
  await expect(page.locator("[data-chart-toggle]")).toHaveAttribute("aria-expanded", "true");

  await page.locator('button[data-lift-tab="Cars196"]').click();
  await expect(page.locator('[data-lift-panel="Cars196"]')).toBeVisible();

  await expect(
    page.locator('[data-comparison-panel="CUB"] .comparison-ladder strong').nth(0),
  ).toHaveText("0.5287");
  await expect(
    page.locator('[data-comparison-panel="CUB"] .comparison-ladder strong').nth(2),
  ).toHaveText("0.5324");
  await expect(page.locator('[data-comparison-panel="CUB"]')).toContainText(
    "vs SupCon: +0.0037 MAP@R",
  );
  await page.locator('button[data-comparison-tab="SOP"]').click();
  await expect(page.locator('[data-comparison-panel="SOP"]')).toContainText(
    "vs SupCon: +0.0112 MAP@R",
  );

  await expect(page.locator(".limits-grid article")).toHaveCount(3);
  await expect(page.getByRole("heading", { name: "How to read the evidence" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Our validated claim, plus paper-reported reference results" })).toBeVisible();
  await expect(page.locator("#published .published-best-grid > article")).toHaveCount(3);
  await expect(page.getByRole("heading", { name: "Latest field references for paper-protocol comparison" })).toBeVisible();
  await expect(page.locator("#published .published-results").nth(1).locator("tbody tr")).toHaveCount(12);
  await expect(page.getByRole("heading", { name: "Our paper-protocol end-to-end run status" })).toBeVisible();
  await expect(page.locator("#published .published-results").nth(0).locator("tbody tr")).toHaveCount(3);
  await expect(page.locator("#published .published-results").nth(0).locator("tbody tr").first()).toContainText(
    "pending Group SupCon + XBM + Radius",
  );
  await expect(page.getByRole("heading", { name: "Observed same-architecture ResNet rows" })).toBeVisible();
  await expect(page.locator("#published [data-paper-observed] tbody tr")).toHaveCount(2);
  await expect(page.locator("#published [data-paper-observed] tbody tr").nth(1)).toContainText(
    "Group SupCon",
  );
  await expect(page.locator("#published [data-paper-observed] tbody tr").nth(1)).toContainText(
    "11.67%",
  );
  await expect(page.getByRole("heading", { name: "Historical 2022 MAP@R, P@1, and R-Precision references" })).toBeVisible();
  await expect(page.locator("#published .published-results").nth(2).locator("tbody tr")).toHaveCount(20);
  await expect(page.locator("#published .row-badge.best").first()).toBeVisible();
  await expect(page.getByRole("link", { name: /HPL WACV 2022 Table 1/ }).first()).toBeVisible();
  await expect(page.getByRole("link", { name: /Proxy Anchor CVPR 2020 Table 2/ }).first()).toBeVisible();

  await page.getByRole("button", { name: "Group SupCon equation" }).click();
  await expect(page.locator('[data-equation-panel="group"]')).toBeVisible();
});

test("research report stays readable on mobile", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 900 });
  await page.goto("/index.html");

  const viewportWidth = await page.evaluate(() => document.documentElement.clientWidth);
  const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth);

  await page.locator("#results").scrollIntoViewIfNeeded();
  await expect(page.locator("#results .result-matrix thead")).toBeHidden();
  await expect(page.locator('.result-matrix td[data-label="Result gain"]').first()).toBeVisible();
});
