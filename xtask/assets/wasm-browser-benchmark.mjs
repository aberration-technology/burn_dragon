#!/usr/bin/env node

import process from "node:process";
import { pathToFileURL } from "node:url";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

function argument(name) {
  const index = process.argv.indexOf(name);
  if (index < 0 || index + 1 >= process.argv.length) {
    throw new Error(`missing ${name}`);
  }
  return process.argv[index + 1];
}

async function loadPlaywright() {
  try {
    return await import("playwright");
  } catch {
    const npxRoot = path.join(os.homedir(), ".npm", "_npx");
    if (!fs.existsSync(npxRoot)) {
      throw new Error(
        "playwright is not installed; run `cargo run -p xtask -- install-playwright-chromium`",
      );
    }
    const candidates = fs
      .readdirSync(npxRoot)
      .map((entry) => path.join(npxRoot, entry, "node_modules", "playwright", "index.mjs"))
      .filter((candidate) => fs.existsSync(candidate))
      .sort((left, right) => fs.statSync(right).mtimeMs - fs.statSync(left).mtimeMs);
    if (candidates.length === 0) {
      throw new Error("playwright is not installed; run `cargo run -p xtask -- install-playwright-chromium`");
    }
    return import(pathToFileURL(candidates[0]).href);
  }
}

const url = argument("--url");
const chrome = argument("--chrome");
const screenshot = argument("--screenshot");
const resultPath = argument("--result");
const batchSize = Number.parseInt(argument("--batch-size"), 10);
const trainBatches = Number.parseInt(argument("--train-batches"), 10);
const tbpttChunkSize = Number.parseInt(argument("--tbptt-chunk-size"), 10);
if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > 8) {
  throw new Error(`invalid --batch-size ${batchSize}; expected 1..8`);
}
if (!Number.isInteger(trainBatches) || trainBatches < 1 || trainBatches > 64) {
  throw new Error(`invalid --train-batches ${trainBatches}; expected 1..64`);
}
if (
  !Number.isInteger(tbpttChunkSize) ||
  tbpttChunkSize < 1 ||
  tbpttChunkSize > 256 ||
  256 % tbpttChunkSize !== 0
) {
  throw new Error(`invalid --tbptt-chunk-size ${tbpttChunkSize}; expected a divisor of 256`);
}
const playwright = await loadPlaywright();
const browser = await playwright.chromium.launch({
  headless: false,
  executablePath: chrome,
  args: [
    "--enable-unsafe-webgpu",
    "--use-angle=vulkan",
    "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan,UseSkiaRenderer,WebGPU",
    "--disable-vulkan-surface",
    "--ignore-gpu-blocklist",
  ],
});

try {
  const page = await browser.newPage();
  const consoleMessages = [];
  page.on("console", (message) => {
    const text = message.text();
    consoleMessages.push(text);
    process.stdout.write(`${text}\n`);
  });
  for (let attempt = 0; attempt < 60; attempt += 1) {
    try {
      const response = await page.request.get(url);
      if (response.ok()) {
        break;
      }
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  const benchmarkUrl = new URL(url);
  benchmarkUrl.searchParams.set("batch_size", String(batchSize));
  benchmarkUrl.searchParams.set("train_batches", String(trainBatches));
  benchmarkUrl.searchParams.set("tbptt_chunk_size", String(tbpttChunkSize));
  await page.goto(benchmarkUrl.toString(), { waitUntil: "domcontentloaded" });
  const adapter = await page.evaluate(async () => {
    const selected = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (!selected) {
      return null;
    }
    const info = selected.info ?? {};
    return {
      vendor: info.vendor ?? null,
      architecture: info.architecture ?? null,
      description: info.description ?? null,
      is_fallback_adapter: info.isFallbackAdapter ?? null,
    };
  });
  process.stdout.write(`BROWSER_ADAPTER ${JSON.stringify(adapter)}\n`);
  if (!adapter || adapter.is_fallback_adapter || adapter.vendor === "google") {
    throw new Error(`hardware WebGPU benchmark selected a fallback adapter: ${JSON.stringify(adapter)}`);
  }

  await page.waitForFunction(
    () => document.body.innerText.includes("test result:"),
    null,
    { timeout: 180_000 },
  );
  const output = await page.locator("body").innerText();
  process.stdout.write(`${output}\n`);
  await page.screenshot({ path: screenshot, fullPage: true });
  if (!output.includes("test result: ok.")) {
    throw new Error("WASM hardware benchmark did not pass");
  }
  const resultMessage = consoleMessages
    .filter((message) => message.startsWith("BROWSER_TRAINING_BENCHMARK "))
    .at(-1);
  if (!resultMessage) {
    throw new Error("WASM hardware benchmark did not emit a structured result");
  }
  const jsonStart = resultMessage.indexOf("{");
  if (jsonStart < 0) {
    throw new Error(`malformed benchmark result: ${resultMessage}`);
  }
  const metadata = resultMessage.match(
    /^BROWSER_TRAINING_BENCHMARK\s+(\S+)\s+started_at_ms=(\d+)\s+ended_at_ms=(\d+)\s+/,
  );
  if (!metadata) {
    throw new Error(`malformed benchmark metadata: ${resultMessage}`);
  }
  const benchmark = JSON.parse(resultMessage.slice(jsonStart));
  fs.writeFileSync(
    resultPath,
    `${JSON.stringify(
      {
        adapter,
        batch_size: batchSize,
        train_batches: trainBatches,
        tbptt_chunk_size: tbpttChunkSize,
        condition: metadata[1],
        started_at_ms: Number(metadata[2]),
        ended_at_ms: Number(metadata[3]),
        benchmark,
      },
      null,
      2,
    )}\n`,
  );
} finally {
  await browser.close();
}
