import { copyFile, mkdir, rename, rm, writeFile } from "node:fs/promises";

await rename("dist/ui/index.html", "dist/index.html");
await rm("dist/ui", { recursive: true, force: true });
await copyFile("ui/og.png", "dist/og.png");
await mkdir("dist/server", { recursive: true });

const worker = `export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/") url.pathname = "/index.html";
    return env.ASSETS.fetch(new Request(url, request));
  }
};\n`;

await writeFile("dist/server/index.js", worker);
