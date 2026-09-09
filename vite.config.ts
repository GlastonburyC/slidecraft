import fs from "node:fs";
import path from "node:path";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// openslide-wasm requires SharedArrayBuffer, which requires cross-origin isolation.
// These headers are mandatory in dev AND in production hosting.
const crossOriginIsolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "cross-origin",
};

/**
 * Serve ONNX Runtime's assets verbatim.
 *
 * ORT loads its wasm glue with a runtime `import(url)`. Vite's dev server
 * appends `?import` to that request and then tries to transform the .mjs,
 * which 500s for files in `public/`. This middleware answers /ort/* straight
 * from disk, ignoring the query, so dev matches the production build where
 * public/ is copied as-is.
 */
function serveOrtAssets(): Plugin {
  const root = path.resolve(process.cwd(), "public/ort");
  return {
    name: "serve-ort-assets",
    configureServer(server) {
      // Registered in the body so it runs before Vite's transform middleware.
      server.middlewares.use("/ort", (req, res, next) => {
        const rel = decodeURIComponent((req.url ?? "").split("?")[0]).replace(/^\/+/, "");
        const file = path.resolve(root, rel);
        if (!file.startsWith(root) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
          return next();
        }
        res.setHeader("Content-Type", file.endsWith(".wasm") ? "application/wasm" : "text/javascript");
        for (const [k, v] of Object.entries(crossOriginIsolation)) res.setHeader(k, v);
        fs.createReadStream(file).pipe(res);
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), serveOrtAssets()],
  server: {
    headers: crossOriginIsolation,
    // DEV ONLY: lets the test harness fetch real slides off disk via /@fs/.
    fs: { allow: [process.cwd(), process.env.HOME ?? "/"] },
  },
  preview: { headers: crossOriginIsolation },
  worker: { format: "es" },
  // Never pre-bundle: esbuild rewrites `new URL('./worker.js', import.meta.url)`
  // and breaks the wasm/worker asset resolution.
  optimizeDeps: { exclude: ["@conflux-xyz/openslide-wasm"] },
  build: { target: "es2022" },
});
