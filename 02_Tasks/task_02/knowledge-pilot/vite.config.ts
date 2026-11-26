import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import path from "path";

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      "@app":            path.resolve(__dirname, "./src/app"),
      "@modules":        path.resolve(__dirname, "./src/modules"),
      "@shared":         path.resolve(__dirname, "./src/shared"),
      "@infrastructure": path.resolve(__dirname, "./src/infrastructure"),
      "@assets":         path.resolve(__dirname, "./src/assets"),
      "@styles":         path.resolve(__dirname, "./src/styles"),
    },
  },
});
