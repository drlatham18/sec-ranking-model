import { resolve } from "node:path";
import { defineConfig } from "vite";
import { sites } from "@openai/sites-vite-plugin";

export default defineConfig({
  plugins: [sites()],
  publicDir: false,
  build: {
    rollupOptions: {
      input: resolve(import.meta.dirname, "ui/index.html"),
    },
  },
});
