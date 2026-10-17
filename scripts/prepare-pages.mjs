import { copyFile, rename, rm, mkdir } from "node:fs/promises";

await rename("dist/ui/index.html", "dist/index.html");
await rm("dist/ui", { recursive: true, force: true });
await copyFile("ui/og.png", "dist/og.png");
// Include standalone conference comparison in the published Pages artifact.
await mkdir("dist/sec-vs-big-ten", { recursive: true });
await copyFile("sec-vs-big-ten/index.html", "dist/sec-vs-big-ten/index.html");
