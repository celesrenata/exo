import { defineConfig } from "vitest/config";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import path from "path";

export default defineConfig({
  plugins: [svelte()],
  resolve: {
    alias: {
      $lib: path.resolve(__dirname, "src/lib"),
      $components: path.resolve(__dirname, "src/lib/components"),
      "$app/environment": path.resolve(__dirname, "src/lib/utils/tests/__mocks__/app-environment.ts"),
    },
  },
  test: {
    include: ["src/**/*.test.ts"],
  },
});
