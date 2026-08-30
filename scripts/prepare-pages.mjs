import { copyFile, rename, rm } from "node:fs/promises";

await rename("dist/ui/index.html", "dist/index.html");
await rm("dist/ui", { recursive: true, force: true });
await copyFile("ui/og.png", "dist/og.png");
