import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";

// JSX uses the automatic runtime by default under Vite 8's oxc transform.
// Component tests opt into jsdom with a `// @vitest-environment jsdom` docblock.
export default defineConfig({
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  test: {
    include: ["src/**/*.test.{ts,tsx}"],
    environment: "node",
    restoreMocks: true,
    unstubEnvs: true,
    unstubGlobals: true,
  },
});
