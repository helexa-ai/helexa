import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

// Dev server proxies /api to the bench API on bob so `fetch('/api/...')`
// works without CORS/mixed-origin fuss during local development.
// For a production build hosted elsewhere, set VITE_API_BASE to the bob
// API origin (e.g. http://bob.hanzalova.internal:13132) instead.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://bob.hanzalova.internal:13132",
        changeOrigin: true,
      },
    },
  },
});
