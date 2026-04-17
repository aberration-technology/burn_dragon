#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

const SITE_BASE_URL = requiredEnv("BURN_DRAGON_BROWSER_CANARY_SITE_BASE_URL");
const EDGE_BASE_URL = requiredEnv("BURN_DRAGON_BROWSER_CANARY_EDGE_BASE_URL");
const PRINCIPAL_ID = requiredEnv("BURN_DRAGON_BROWSER_CANARY_PRINCIPAL_ID");
const CALLBACK_TOKEN = requiredEnv("BURN_DRAGON_BROWSER_CANARY_CALLBACK_TOKEN");
const EXPERIMENT_ID =
  process.env.BURN_DRAGON_BROWSER_CANARY_EXPERIMENT_ID ?? "nca-prepretraining";
const QUIET_WINDOW_MS = parseIntegerEnv("BURN_DRAGON_BROWSER_CANARY_QUIET_WINDOW_MS", 8_000);
const CONNECT_TIMEOUT_MS = parseIntegerEnv(
  "BURN_DRAGON_BROWSER_CANARY_CONNECT_TIMEOUT_MS",
  90_000,
);
const TRAIN_TIMEOUT_MS = parseIntegerEnv(
  "BURN_DRAGON_BROWSER_CANARY_TRAIN_TIMEOUT_MS",
  300_000,
);
const ARTIFACT_DIR =
  process.env.BURN_DRAGON_BROWSER_CANARY_ARTIFACT_DIR ??
  path.join(os.tmpdir(), `burn-dragon-browser-canary-${Date.now()}`);
const OUTPUT_JSON =
  process.env.BURN_DRAGON_BROWSER_CANARY_OUTPUT_JSON ??
  path.join(ARTIFACT_DIR, "canary-summary.json");
const HEADLESS = process.env.BURN_DRAGON_BROWSER_CANARY_HEADED === "1" ? false : true;
const SITE_OVERRIDE_DIR = process.env.BURN_DRAGON_BROWSER_CANARY_SITE_OVERRIDE_DIR?.trim() || null;

const WATCHED_CONTROL_PATHS = [
  "/portal/snapshot",
  "/directory",
  "/directory/signed",
  "/heads",
  "/leaderboard",
  "/leaderboard/signed",
  "/metrics/catchup/",
];
const EDGE_ARTIFACT_PATH_PREFIXES = ["/artifacts/heads/", "/artifacts/tickets/"];

function requiredEnv(name) {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(`missing required environment variable ${name}`);
  }
  return value;
}

function parseIntegerEnv(name, fallback) {
  const raw = process.env[name];
  if (!raw) {
    return fallback;
  }
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`invalid integer environment variable ${name}=${raw}`);
  }
  return parsed;
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

async function loadPlaywright() {
  try {
    return await import("playwright");
  } catch {
    const npxRoot = path.join(os.homedir(), ".npm", "_npx");
    if (!fs.existsSync(npxRoot)) {
      throw new Error(
        "playwright package not found; run `npx --yes playwright --version` first",
      );
    }
    const candidates = fs
      .readdirSync(npxRoot)
      .map((entry) => path.join(npxRoot, entry, "node_modules", "playwright", "index.mjs"))
      .filter((candidate) => fs.existsSync(candidate))
      .sort((left, right) => fs.statSync(right).mtimeMs - fs.statSync(left).mtimeMs);
    if (candidates.length === 0) {
      throw new Error(
        "playwright package not found in npx cache; run `npx --yes playwright --version` first",
      );
    }
    return await import(pathToFileURL(candidates[0]).href);
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: {
      "content-type": "application/json",
      ...(options.headers ?? {}),
    },
  });
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} for ${url}: ${trimPreview(text)}`);
  }
  return text ? JSON.parse(text) : null;
}

function trimPreview(text) {
  const normalized = String(text ?? "").trim();
  if (normalized.length <= 240) {
    return normalized;
  }
  return `${normalized.slice(0, 240)}...`;
}

function endpoint(baseUrl, relativePath) {
  return new URL(relativePath, baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`).toString();
}

function requestedScopes(experimentId) {
  return ["Connect", "Discover", { Train: { experiment_id: experimentId } }];
}

function firstApprovedTargetArtifactHash(snapshot) {
  const allowed = snapshot.allowed_target_artifact_hashes ?? [];
  if (allowed.length > 0) {
    return allowed[0];
  }
  const trustAllowed = snapshot.trust_bundle?.allowed_target_artifact_hashes ?? [];
  if (trustAllowed.length > 0) {
    return trustAllowed[0];
  }
  throw new Error("edge snapshot did not expose any approved browser target artifact hashes");
}

function requiredReleaseTrainHash(snapshot) {
  return (
    snapshot.required_release_train_hash ??
    snapshot.trust_bundle?.required_release_train_hash ??
    fail("edge snapshot is missing a required release train hash")
  );
}

function browserStorageSnapshot(networkId, sessionState) {
  return {
    metadata_version: 3,
    session: sessionState,
    cached_chunk_artifacts: [],
    cached_head_artifact_heads: [],
    last_head_artifact_transport: null,
    cached_microshards: [],
    stored_receipts: [],
    pending_receipts: {
      backend: "Snapshot",
      receipts: [],
    },
    submitted_receipts: [],
    last_directory_sync_at: null,
    last_signed_directory_snapshot: null,
    last_signed_leaderboard_snapshot: null,
    metrics_catchup_bundles: [],
    last_metrics_live_event: null,
    last_metrics_sync_at: null,
    last_head_id: null,
    artifact_replay_checkpoint: null,
    stored_certificate_peer_id:
      sessionState?.certificate?.claims?.peer_id ??
      sessionState?.certificate?.claims?.claims?.peer_id ??
      null,
    active_assignment: null,
    active_training_lease: null,
    updated_at: new Date().toISOString(),
  };
}

function emptyReceiptOutbox() {
  return {
    backend: "Snapshot",
    receipts: [],
  };
}

function classifyEdgeRequest(urlString, edgeHost) {
  const url = new URL(urlString);
  if (url.host !== edgeHost) {
    return {
      watchedControlPlane: false,
      artifactFallback: false,
    };
  }
  const watchedControlPlane = WATCHED_CONTROL_PATHS.some((prefix) =>
    url.pathname.startsWith(prefix),
  );
  const artifactFallback = EDGE_ARTIFACT_PATH_PREFIXES.some((prefix) =>
    url.pathname.startsWith(prefix),
  );
  return { watchedControlPlane, artifactFallback };
}

function fail(message) {
  throw new Error(message);
}

function statTileValue(tiles, label) {
  const normalizedLabel = label.trim().toLowerCase();
  for (const tile of tiles ?? []) {
    if (!Array.isArray(tile) || tile.length === 0) {
      continue;
    }
    if ((tile[0] ?? "").trim().toLowerCase() !== normalizedLabel) {
      continue;
    }
    return tile
      .slice(1)
      .map((part) => part ?? "")
      .join(" | ")
      .trim();
  }
  return null;
}

function isDialableWebRtcSeed(seed) {
  return seed.includes("/webrtc-direct") && seed.includes("/certhash/");
}

function isDialableWebTransportSeed(seed) {
  return seed.includes("/quic-v1/webtransport") && seed.includes("/certhash/");
}

function contentTypeForPath(filePath) {
  if (filePath.endsWith(".html")) return "text/html; charset=utf-8";
  if (filePath.endsWith(".js")) return "text/javascript; charset=utf-8";
  if (filePath.endsWith(".css")) return "text/css; charset=utf-8";
  if (filePath.endsWith(".json")) return "application/json; charset=utf-8";
  if (filePath.endsWith(".wasm")) return "application/wasm";
  if (filePath.endsWith(".map")) return "application/json; charset=utf-8";
  return "application/octet-stream";
}

function resolveOverrideAssetPath(overrideDir, requestUrl, siteBaseUrl) {
  const request = new URL(requestUrl);
  const siteBase = new URL(siteBaseUrl);
  if (request.origin !== siteBase.origin) {
    return null;
  }
  let pathname = request.pathname;
  const siteBasePath = siteBase.pathname === "/" ? "" : siteBase.pathname.replace(/\/$/, "");
  if (siteBasePath && pathname.startsWith(siteBasePath)) {
    pathname = pathname.slice(siteBasePath.length) || "/";
  }
  const relativePath = pathname === "/" ? "index.html" : pathname.replace(/^\/+/, "");
  const decoded = decodeURIComponent(relativePath);
  const normalized = path.normalize(decoded);
  if (normalized.startsWith("..")) {
    return null;
  }
  return path.join(overrideDir, normalized);
}

async function waitForVisible(locator, timeoutMs) {
  await locator.waitFor({ state: "visible", timeout: timeoutMs });
}

async function isVisible(locator) {
  try {
    return (await locator.count()) > 0 && (await locator.first().isVisible());
  } catch {
    return false;
  }
}

async function maybeClick(locator) {
  if ((await locator.count()) > 0 && (await locator.first().isVisible())) {
    await locator.first().click();
    return true;
  }
  return false;
}

async function enrollBrowserCanary(snapshot) {
  const provider = (snapshot.login_providers ?? []).find(
    (candidate) => candidate?.login_path && candidate.login_path.trim().length > 0,
  );
  if (!provider) {
    fail("edge snapshot did not expose a usable login provider");
  }
  const callbackPath = provider.callback_path ?? snapshot.paths?.callback_path;
  if (!callbackPath) {
    fail("edge snapshot did not expose a callback path");
  }
  const scopes = requestedScopes(EXPERIMENT_ID);
  const login = await fetchJson(endpoint(EDGE_BASE_URL, provider.login_path), {
    method: "POST",
    body: JSON.stringify({
      network_id: snapshot.network_id,
      principal_hint: PRINCIPAL_ID,
      requested_scopes: scopes,
    }),
  });
  const session = await fetchJson(endpoint(EDGE_BASE_URL, callbackPath), {
    method: "POST",
    headers: {
      "x-burn-p2p-canary-token": CALLBACK_TOKEN,
    },
    body: JSON.stringify({
      login_id: login.login_id,
      state: login.state,
    }),
  });
  const trustBundle = await fetchJson(
    endpoint(EDGE_BASE_URL, snapshot.paths.trust_bundle_path),
    { method: "GET", headers: {} },
  );
  const peerLabel = `browser-canary-${crypto.randomUUID()}`;
  const certificate = await fetchJson(endpoint(EDGE_BASE_URL, snapshot.paths.enroll_path), {
    method: "POST",
    body: JSON.stringify({
      session_id: session.session_id,
      release_train_hash: requiredReleaseTrainHash(snapshot),
      target_artifact_hash: firstApprovedTargetArtifactHash(snapshot),
      peer_id: peerLabel,
      peer_public_key_hex: crypto.randomBytes(32).toString("hex"),
      requested_scopes: scopes,
      client_policy_hash: `browser-canary-policy-${Date.now()}`,
      serial: 1,
      ttl_seconds: 900,
    }),
  });
  return {
    login,
    session,
    trustBundle,
    certificate,
    requestedScopes: scopes,
  };
}

async function runCanary() {
  ensureDir(ARTIFACT_DIR);
  const snapshot = await fetchJson(endpoint(EDGE_BASE_URL, "/portal/snapshot"), {
    method: "GET",
    headers: {},
  });
  const signedSeedsEnvelope = await fetchJson(
    endpoint(
      EDGE_BASE_URL,
      snapshot.paths?.browser_seed_advertisement_path ?? "/browser/seeds/signed",
    ),
    { method: "GET", headers: {} },
  );
  const browserConfig = await fetchJson(endpoint(SITE_BASE_URL, "/browser-app-config.json"), {
    method: "GET",
    headers: {},
  });

  if (snapshot.transports?.webtransport_gateway) {
    fail("live edge is advertising webtransport_gateway without validated native runtime support");
  }

  const signedSeeds = signedSeedsEnvelope?.payload?.payload?.seeds?.flatMap(
    (record) => record.multiaddrs ?? [],
  ) ?? [];
  const browserConfigSeeds = browserConfig.seed_node_urls ?? [];
  const signedHasWebRtcDirect = signedSeeds.some(isDialableWebRtcSeed);
  const signedHasWebTransport = signedSeeds.some(isDialableWebTransportSeed);
  const signedHasWss = signedSeeds.some((value) => value.includes("/wss"));
  const browserCapableSeedCount = Number(signedHasWebRtcDirect) + Number(signedHasWebTransport) + Number(signedHasWss);
  if (!signedSeeds.length || browserCapableSeedCount === 0) {
    fail(`signed browser seeds are missing browser-capable addresses: ${JSON.stringify(signedSeeds)}`);
  }
  for (const seed of signedSeeds) {
    if (seed.includes("/webrtc-direct") && !isDialableWebRtcSeed(seed)) {
      fail(`signed browser seed advertises malformed webrtc-direct multiaddr: ${seed}`);
    }
    if (seed.includes("/webtransport") && !isDialableWebTransportSeed(seed)) {
      fail(`signed browser seed advertises malformed webtransport multiaddr: ${seed}`);
    }
  }
  if (signedSeeds.some((value) => value.includes("/quic-v1") || value.includes("/tcp/4001"))) {
    fail(`signed browser seeds still contain native-only addresses: ${JSON.stringify(signedSeeds)}`);
  }
  if (Boolean(snapshot.transports?.webrtc_direct) !== signedHasWebRtcDirect) {
    fail(
      `signed browser seeds disagree with snapshot webrtc_direct=${snapshot.transports?.webrtc_direct}: ${JSON.stringify(signedSeeds)}`,
    );
  }
  if (Boolean(snapshot.transports?.webtransport_gateway) !== signedHasWebTransport) {
    fail(
      `signed browser seeds disagree with snapshot webtransport_gateway=${snapshot.transports?.webtransport_gateway}: ${JSON.stringify(signedSeeds)}`,
    );
  }
  if (Boolean(snapshot.transports?.wss_fallback) !== signedHasWss) {
    fail(
      `signed browser seeds disagree with snapshot wss_fallback=${snapshot.transports?.wss_fallback}: ${JSON.stringify(signedSeeds)}`,
    );
  }
  if (JSON.stringify(browserConfigSeeds) !== JSON.stringify(signedSeeds)) {
    fail(
      `browser config seeds drifted from signed browser seeds: config=${JSON.stringify(browserConfigSeeds)} signed=${JSON.stringify(signedSeeds)}`,
    );
  }

  const enrollment = await enrollBrowserCanary(snapshot);
  const sessionState = {
    session: enrollment.session,
    certificate: enrollment.certificate,
    trust_bundle: enrollment.trustBundle,
    enrolled_at: new Date().toISOString(),
    reenrollment_required: Boolean(enrollment.trustBundle?.reenrollment),
  };
  const storageSnapshot = browserStorageSnapshot(snapshot.network_id, sessionState);

  const { chromium } = await loadPlaywright();
  const browser = await chromium.launch({
    headless: HEADLESS,
    args: [
      "--enable-unsafe-webgpu",
      "--use-angle=swiftshader",
      "--enable-features=Vulkan,UseSkiaRenderer,WebGPU",
    ],
  });

  const requests = [];
  const consoleMessages = [];
  const pageErrors = [];
  let tracePath = null;
  let screenshotPath = path.join(ARTIFACT_DIR, "canary.png");
  const report = {
    site_base_url: SITE_BASE_URL,
    edge_base_url: EDGE_BASE_URL,
    principal_id: PRINCIPAL_ID,
    experiment_id: EXPERIMENT_ID,
    network_id: snapshot.network_id,
    transports: snapshot.transports,
    browser_config_seed_node_urls: browserConfigSeeds,
    signed_seed_multiaddrs: signedSeeds,
    signed_seed_transport_preference:
      signedSeedsEnvelope?.payload?.payload?.transport_policy?.preferred ?? [],
    connect_clicked: false,
    training_button_visible: false,
    connect_button_visible: false,
    get_started_button_visible: false,
    live_status_label: null,
    live_stat_tiles: [],
    live_panel_detail: null,
    live_notice_label: null,
    live_notice_detail: null,
    transport_summary: null,
    quiet_window_ms: QUIET_WINDOW_MS,
    train_timeout_ms: TRAIN_TIMEOUT_MS,
    quiet_window_control_plane_requests: [],
    artifact_http_fallback_requests: [],
    receipt_submission: null,
    console_errors: [],
    page_errors: [],
    success: false,
  };

  const context = await browser.newContext();
  try {
    if (ARTIFACT_DIR) {
      await context.tracing.start({ screenshots: true, snapshots: true });
      tracePath = path.join(ARTIFACT_DIR, "trace.zip");
    }

    if (SITE_OVERRIDE_DIR) {
      await context.route("**/*", async (route) => {
        const overridePath = resolveOverrideAssetPath(
          SITE_OVERRIDE_DIR,
          route.request().url(),
          SITE_BASE_URL,
        );
        if (!overridePath) {
          await route.continue();
          return;
        }
        if (!fs.existsSync(overridePath) || fs.statSync(overridePath).isDirectory()) {
          await route.fallback();
          return;
        }
        await route.fulfill({
          status: 200,
          body: fs.readFileSync(overridePath),
          headers: {
            "content-type": contentTypeForPath(overridePath),
            "cache-control": "no-store",
          },
        });
      });
    }

    await context.addInitScript(
      ({ networkId, storageJson, receiptJson }) => {
        const storageKey = `burn-p2p.browser.storage.${networkId}`;
        const receiptKey = `burn-p2p.browser.receipt-outbox.${networkId}`;
        try {
          window.localStorage.setItem(storageKey, storageJson);
          window.localStorage.setItem(receiptKey, receiptJson);
        } catch (error) {
          console.error("burn-dragon-canary-init-storage-failed", String(error));
        }
      },
      {
        networkId: snapshot.network_id,
        storageJson: JSON.stringify(storageSnapshot),
        receiptJson: JSON.stringify(emptyReceiptOutbox()),
      },
    );

    const page = await context.newPage();
    page.setDefaultTimeout(CONNECT_TIMEOUT_MS);
    page.on("console", (message) => {
      const entry = {
        type: message.type(),
        text: message.text(),
      };
      consoleMessages.push(entry);
      if (entry.type === "error") {
        report.console_errors.push(entry.text);
      }
    });
    page.on("pageerror", (error) => {
      const text =
        error && typeof error === "object" && "stack" in error && error.stack
          ? String(error.stack)
          : String(error);
      pageErrors.push(text);
      report.page_errors.push(text);
    });
    page.on("request", (request) => {
      const url = request.url();
      const { watchedControlPlane, artifactFallback } = classifyEdgeRequest(
        url,
        new URL(EDGE_BASE_URL).host,
      );
      requests.push({
        ts: Date.now(),
        method: request.method(),
        url,
        watchedControlPlane,
        artifactFallback,
      });
    });

    await page.goto(SITE_BASE_URL, { waitUntil: "domcontentloaded" });

    const connectButton = page.locator('button:has-text("connect")').first();
    const trainButton = page.locator('button:has-text("run browser training")').first();
    const getStartedButton = page.locator('button:has-text("get started")').first();
    let quietWindowStartedAt = null;
    const captureLiveStatus = async () => {
      report.connect_button_visible = await isVisible(connectButton);
      report.training_button_visible = await isVisible(trainButton);
      report.get_started_button_visible = await isVisible(getStartedButton);
      report.live_status_label =
        (await page
          .locator(".dragon-live-status-pill")
          .first()
          .textContent()
          .catch(() => null)) ?? null;
      report.live_panel_detail =
        (await page
          .locator(".dragon-live-shell .section-detail")
          .first()
          .textContent()
          .catch(() => null)) ?? null;
      report.live_notice_label =
        (await page
          .locator(".activity-notice-label")
          .first()
          .textContent()
          .catch(() => null)) ?? null;
      report.live_notice_detail =
        (await page
          .locator(".activity-notice-detail")
          .first()
          .textContent()
          .catch(() => null)) ?? null;
      report.live_stat_tiles = await page
        .locator(".dragon-live-stats .stat-tile")
        .evaluateAll((nodes) =>
          nodes.map((node) =>
            Array.from(node.children).map((child) => child.textContent?.trim() ?? ""),
          ),
        )
        .catch(() => []);
      report.transport_summary = statTileValue(report.live_stat_tiles, "transport");
    };

    const connectDeadline = Date.now() + CONNECT_TIMEOUT_MS;
    const sessionResumeGraceDeadline = Date.now() + 5_000;
    while (Date.now() < connectDeadline) {
      await captureLiveStatus();
      if (report.training_button_visible) {
        break;
      }
      if (report.connect_button_visible) {
        await connectButton.click();
        report.connect_clicked = true;
      }
      if (
        Date.now() >= sessionResumeGraceDeadline &&
        report.get_started_button_visible &&
        !report.connect_button_visible
      ) {
        fail("browser canary session did not resume; page returned to get started");
      }
      await page.waitForTimeout(500);
    }

    await captureLiveStatus();
    if (!report.training_button_visible) {
      report.artifact_http_fallback_requests = requests.filter((entry) => entry.artifactFallback);
      fail(
        `browser canary did not become training-ready: status=${report.live_status_label ?? "missing"} transport=${report.transport_summary ?? "missing"} notice=${report.live_notice_detail ?? report.live_panel_detail ?? "none"}`,
      );
    }

    quietWindowStartedAt = Date.now();
    await page.waitForTimeout(QUIET_WINDOW_MS);
    await captureLiveStatus();
    report.quiet_window_control_plane_requests = requests.filter(
      (entry) => entry.ts >= quietWindowStartedAt && entry.watchedControlPlane,
    );
    report.artifact_http_fallback_requests = requests.filter((entry) => entry.artifactFallback);
    if (report.quiet_window_control_plane_requests.length > 0) {
      fail(
        `browser canary observed steady-state edge polling after direct connect: ${JSON.stringify(report.quiet_window_control_plane_requests)}`,
      );
    }
    const liveTransportTile = statTileValue(report.live_stat_tiles, "transport");
    if (
      signedSeedsEnvelope?.payload?.payload?.transport_policy?.preferred?.[0] === "WebRtcDirect" &&
      snapshot.transports?.webrtc_direct &&
      !(liveTransportTile ?? "").startsWith("webrtc-direct")
    ) {
      fail(
        `browser canary did not settle on webrtc-direct despite advertised preference: ${liveTransportTile ?? "missing transport tile"}`,
      );
    }

    const receiptResponsePromise = page.waitForResponse(
      (response) =>
        response.request().method() === "POST" &&
        new URL(response.url()).host === new URL(EDGE_BASE_URL).host &&
        new URL(response.url()).pathname === "/receipts/browser" &&
        response.status() >= 200 &&
        response.status() < 300,
      { timeout: TRAIN_TIMEOUT_MS },
    );
    await trainButton.click();
    const receiptResponse = await receiptResponsePromise;
    report.receipt_submission = {
      url: receiptResponse.url(),
      status: receiptResponse.status(),
      body_preview: trimPreview(await receiptResponse.text()),
    };
    report.success = true;
    await page.screenshot({ path: screenshotPath, fullPage: true });
  } catch (error) {
    report.success = false;
    report.error = String(error);
    try {
      const failurePage = context.pages()[0];
      if (failurePage) {
        await failurePage.screenshot({ path: screenshotPath, fullPage: true });
      }
    } catch {}
    throw error;
  } finally {
    if (tracePath) {
      await context.tracing.stop({ path: tracePath }).catch(() => {});
    }
    report.artifact_http_fallback_requests = requests.filter((entry) => entry.artifactFallback);
    report.console_errors = report.console_errors.concat(
      consoleMessages.filter((entry) => entry.type === "error").map((entry) => entry.text),
    );
    report.page_errors = pageErrors;
    fs.writeFileSync(path.join(ARTIFACT_DIR, "requests.json"), JSON.stringify(requests, null, 2));
    fs.writeFileSync(
      path.join(ARTIFACT_DIR, "console.json"),
      JSON.stringify(consoleMessages, null, 2),
    );
    fs.writeFileSync(OUTPUT_JSON, JSON.stringify(report, null, 2));
    await browser.close().catch(() => {});
  }

  return report;
}

try {
  const report = await runCanary();
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
} catch (error) {
  const summaryPath = OUTPUT_JSON;
  if (fs.existsSync(summaryPath)) {
    process.stderr.write(`${fs.readFileSync(summaryPath, "utf8")}\n`);
  }
  process.stderr.write(`browser canary failed: ${error}\n`);
  process.exitCode = 1;
}
