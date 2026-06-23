import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react-swc";

// During `vite dev`, proxy the mesh data-plane (helexa-router, OpenAI-
// compatible) and the account control-plane (helexa-upstream) so the SPA
// talks to them same-origin without CORS. Targets are overridable via env.
//   VITE_ROUTER_BASE_URL  — helexa-router (default http://localhost:8088)
//   VITE_ACCOUNT_BASE_URL — helexa-upstream (default http://localhost:8090)
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  const router = env.VITE_ROUTER_BASE_URL || "http://localhost:8088";
  const account = env.VITE_ACCOUNT_BASE_URL || "http://localhost:8090";
  return {
    plugins: [react()],
    server: {
      proxy: {
        "/v1": { target: router, changeOrigin: true },
        "/health": { target: router, changeOrigin: true },
        // The frontend calls /api/*; helexa-upstream serves /web/v1/*.
        "/api": {
          target: account,
          changeOrigin: true,
          rewrite: (p) => p.replace(/^\/api/, "/web/v1"),
        },
      },
    },
    build: { outDir: "dist" },
  };
});
