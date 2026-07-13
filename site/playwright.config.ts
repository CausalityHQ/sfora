import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  expect: {
    timeout: 5_000,
  },
  use: {
    // The built site uses the "/sfora" base path, so serve it under that prefix
    // (via a symlink) or its JS/asset URLs 404 and interactivity never loads.
    baseURL: "http://127.0.0.1:8790/sfora/",
    trace: "on-first-retry",
  },
  webServer: {
    command:
      "rm -rf .e2e-root && mkdir -p .e2e-root && ln -sfn \"$(cd ../reports/site && pwd)\" .e2e-root/sfora && python3 -m http.server 8790 --bind 127.0.0.1 --directory .e2e-root",
    url: "http://127.0.0.1:8790/sfora/index.html",
    reuseExistingServer: !process.env.CI,
    timeout: 15_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
