import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  expect: {
    timeout: 5_000,
  },
  use: {
    baseURL: "http://127.0.0.1:8790",
    trace: "on-first-retry",
  },
  webServer: {
    command: "python3 -m http.server 8790 --bind 127.0.0.1 --directory ../reports/site",
    url: "http://127.0.0.1:8790/index.html",
    reuseExistingServer: !process.env.CI,
    timeout: 10_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
