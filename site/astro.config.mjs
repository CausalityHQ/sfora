import { defineConfig } from "astro/config";

export default defineConfig({
  // Served under https://causality.pl/sfora/ (GitHub Pages project subpath).
  site: "https://causality.pl",
  base: "/sfora",
  output: "static",
  outDir: "../reports/site",
});
