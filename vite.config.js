import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  publicDir: false,
  build: {
    rollupOptions: {
      input: resolve(import.meta.dirname, "ui/index.html"),
    },
  },
});
