import { defineConfig } from "vite";
import react, { reactCompilerPreset } from "@vitejs/plugin-react";
import babel from "@rolldown/plugin-babel";
import tailwindcss from "@tailwindcss/vite";
import { readFile } from "node:fs/promises";
import type { Plugin } from "vite";

/** Dev server only: serves the gitignored real-data snapshot at /dev/dataset.json, never part of a build. */
function devSnapshot(): Plugin {
  return {
    name: "strixscope-dev-snapshot",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/dev/dataset.json", async (_req, res) => {
        try {
          const body = await readFile(new URL("./dev-data/dataset.json", import.meta.url));
          res.setHeader("Content-Type", "application/json");
          res.end(body);
        } catch {
          res.statusCode = 404;
          res.end("no dev-data/dataset.json");
        }
      });
    },
  };
}

export default defineConfig({
  // React Compiler (Babel 7 AST) memoizes components and hooks automatically.
  plugins: [react(), babel({ presets: [reactCompilerPreset()] }), tailwindcss(), devSnapshot()],
  clearScreen: false,
  server: {
    port: 5174,
    strictPort: true,
    host: "127.0.0.1",
    watch: {
      ignored: ["**/src-tauri/**"],
    },
  },
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    // Default target (Chrome 111 / Safari 16.4) — the floor Tailwind 4 needs too.
    sourcemap: true,
  },
});
