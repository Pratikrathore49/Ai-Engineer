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
      "@pages": path.resolve(__dirname, "./src/pages"),
      "@lib":path.resolve(__dirname,"./src/lib"),
      "@types":path.resolve(__dirname,"./src/types"),
      "@components":path.resolve(__dirname,"./src/components")
    },
  },
});