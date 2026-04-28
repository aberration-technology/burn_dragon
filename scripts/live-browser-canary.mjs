#!/usr/bin/env node

import fs from "node:fs";
import { isIP } from "node:net";
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
const DURABLE_RECEIPT_TIMEOUT_MS = parseIntegerEnv(
  "BURN_DRAGON_BROWSER_CANARY_DURABLE_RECEIPT_TIMEOUT_MS",
  30_000,
);
const BROWSER_NAME = (
  process.env.BURN_DRAGON_BROWSER_CANARY_BROWSER ?? "chromium"
).trim().toLowerCase();
const TRANSPORT_MODE = (
  process.env.BURN_DRAGON_BROWSER_CANARY_TRANSPORT_MODE ?? "auto"
).trim().toLowerCase();
const EXPECT_TRAINING = parseBooleanEnv("BURN_DRAGON_BROWSER_CANARY_EXPECT_TRAINING", true);
const EXPECT_CONNECTED_TRANSPORT =
  process.env.BURN_DRAGON_BROWSER_CANARY_EXPECT_CONNECTED_TRANSPORT?.trim().toLowerCase() ||
  null;
const EXPECT_MIN_DIRECT_PEERS = parseNonnegativeIntegerEnv(
  "BURN_DRAGON_BROWSER_CANARY_EXPECT_MIN_DIRECT_PEERS",
  null,
);
const TRANSIENT_FETCH_RETRIES = parseIntegerEnv(
  "BURN_DRAGON_BROWSER_CANARY_TRANSIENT_FETCH_RETRIES",
  6,
);
const TRANSIENT_FETCH_RETRY_DELAY_MS = parseIntegerEnv(
  "BURN_DRAGON_BROWSER_CANARY_TRANSIENT_FETCH_RETRY_DELAY_MS",
  2_000,
);
const SITE_ASSET_RETRIES = parseIntegerEnv(
  "BURN_DRAGON_BROWSER_CANARY_SITE_ASSET_RETRIES",
  20,
);
const SITE_ASSET_RETRY_DELAY_MS = parseIntegerEnv(
  "BURN_DRAGON_BROWSER_CANARY_SITE_ASSET_RETRY_DELAY_MS",
  3_000,
);
const ARTIFACT_DIR =
  process.env.BURN_DRAGON_BROWSER_CANARY_ARTIFACT_DIR ??
  path.join(os.tmpdir(), `burn-dragon-browser-canary-${Date.now()}`);
const OUTPUT_JSON =
  process.env.BURN_DRAGON_BROWSER_CANARY_OUTPUT_JSON ??
  path.join(ARTIFACT_DIR, "canary-summary.json");
const HEADLESS = process.env.BURN_DRAGON_BROWSER_CANARY_HEADED === "1" ? false : true;
const SITE_OVERRIDE_DIR = process.env.BURN_DRAGON_BROWSER_CANARY_SITE_OVERRIDE_DIR?.trim() || null;
const PENDING_GITHUB_LOGIN_KEY = "burn-dragon-p2p.pending-github-login";
const TRUSTED_CALLBACK_TOKEN_KEY = "burn-dragon-p2p.canary-callback-token";
const DIALABLE_WEBRTC_HOST_PROTOCOLS = new Set([
  "ip4",
  "ip6",
  "dns",
  "dns4",
  "dns6",
  "dnsaddr",
]);

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
const WEBRTC_DIRECT_REQUIRED_CONSOLE_MARKERS = [
  "libp2p webrtc-direct: created browser RTCPeerConnection",
  "libp2p webrtc-direct datachannel: open before-noise",
  "libp2p webrtc-direct: starting Noise handshake over WebRTC datachannel",
  "libp2p webrtc-direct: completed Noise handshake peer=",
];

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

function parseNonnegativeIntegerEnv(name, fallback) {
  const raw = process.env[name];
  if (raw == null || raw.trim() === "") {
    return fallback;
  }
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`invalid non-negative integer environment variable ${name}=${raw}`);
  }
  return parsed;
}

function parseBooleanEnv(name, fallback) {
  const raw = process.env[name];
  if (raw == null || raw.trim() === "") {
    return fallback;
  }
  const normalized = raw.trim().toLowerCase();
  if (["1", "true", "yes", "y"].includes(normalized)) {
    return true;
  }
  if (["0", "false", "no", "n"].includes(normalized)) {
    return false;
  }
  throw new Error(`invalid boolean environment variable ${name}=${raw}`);
}

function validateCanaryMode() {
  if (!["chromium", "firefox", "webkit"].includes(BROWSER_NAME)) {
    throw new Error(`unsupported browser ${BROWSER_NAME}; expected chromium, firefox, or webkit`);
  }
  if (!["auto", "webrtc-direct", "webtransport", "wss"].includes(TRANSPORT_MODE)) {
    throw new Error(
      `unsupported transport mode ${TRANSPORT_MODE}; expected auto, webrtc-direct, webtransport, or wss`,
    );
  }
  if (
    EXPECT_CONNECTED_TRANSPORT != null &&
    !["webrtc-direct", "webtransport", "wss", "ws"].includes(EXPECT_CONNECTED_TRANSPORT)
  ) {
    throw new Error(
      `unsupported expected connected transport ${EXPECT_CONNECTED_TRANSPORT}; expected webrtc-direct, webtransport, wss, or ws`,
    );
  }
  if (EXPECT_TRAINING && BROWSER_NAME !== "chromium") {
    throw new Error("training canary currently requires chromium; use EXPECT_TRAINING=0 for connect-only browser lanes");
  }
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

function shouldRetryTransientFetch(error) {
  const message = String(error?.message ?? error ?? "");
  return (
    /\b(502|503|504)\b/.test(message) ||
    /\bECONNRESET\b/.test(message) ||
    /\bETIMEDOUT\b/.test(message) ||
    /\bfetch failed\b/i.test(message)
  );
}

async function sleep(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchJsonWithTransientRetry(url, options = {}) {
  let attempt = 0;
  let lastError = null;
  while (attempt < TRANSIENT_FETCH_RETRIES) {
    try {
      return await fetchJson(url, options);
    } catch (error) {
      lastError = error;
      attempt += 1;
      if (attempt >= TRANSIENT_FETCH_RETRIES || !shouldRetryTransientFetch(error)) {
        throw error;
      }
      await sleep(TRANSIENT_FETCH_RETRY_DELAY_MS * attempt);
    }
  }
  throw lastError;
}

function snapshotAcceptedReceiptCount(snapshot) {
  const candidates = [
    snapshot?.diagnostics?.accepted_receipts,
    snapshot?.accepted_receipts,
    snapshot?.payload?.diagnostics?.accepted_receipts,
  ];
  for (const candidate of candidates) {
    const value = Number(candidate);
    if (Number.isFinite(value) && value >= 0) {
      return Math.trunc(value);
    }
  }
  return null;
}

function acceptedReceiptIdsFromSubmission(body) {
  if (!body || typeof body !== "object" || !Array.isArray(body.accepted_receipt_ids)) {
    return [];
  }
  return body.accepted_receipt_ids.map((value) => String(value)).filter(Boolean);
}

async function waitForDurableReceiptCount(acceptedReceiptIds, baselineCount) {
  if (baselineCount == null) {
    fail(
      "edge snapshot is missing diagnostics.accepted_receipts; cannot prove durable browser receipt persistence",
    );
  }
  const minimumCount = baselineCount + acceptedReceiptIds.length;
  const deadline = Date.now() + DURABLE_RECEIPT_TIMEOUT_MS;
  let lastSnapshot = null;
  let lastCount = null;
  while (Date.now() < deadline) {
    lastSnapshot = await fetchJsonWithTransientRetry(
      endpoint(EDGE_BASE_URL, "/portal/snapshot"),
      {
        method: "GET",
        headers: {},
      },
    );
    lastCount = snapshotAcceptedReceiptCount(lastSnapshot);
    if (lastCount != null && lastCount >= minimumCount) {
      return {
        accepted_receipt_ids: acceptedReceiptIds,
        baseline_accepted_receipts: baselineCount,
        observed_accepted_receipts: lastCount,
        minimum_expected_accepted_receipts: minimumCount,
        snapshot_captured_at:
          lastSnapshot?.diagnostics?.captured_at ?? lastSnapshot?.captured_at ?? null,
      };
    }
    await sleep(1_000);
  }
  fail(
    `browser receipt was accepted but did not become durable edge state: accepted_ids=${JSON.stringify(acceptedReceiptIds)} baseline=${baselineCount} last_count=${lastCount ?? "missing"} timeout_ms=${DURABLE_RECEIPT_TIMEOUT_MS}`,
  );
}

async function fetchOk(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    ...options,
    headers: {
      "cache-control": "no-cache",
      pragma: "no-cache",
      ...(options.headers ?? {}),
    },
  });
  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`${response.status} ${response.statusText} for ${url}: ${trimPreview(text)}`);
  }
  return {
    url,
    status: response.status,
    content_type: response.headers.get("content-type") ?? null,
    content_length: response.headers.get("content-length") ?? null,
  };
}

async function waitForSiteRuntimeAssets() {
  const relativePaths = [
    "/",
    "/browser-app-config.json",
    "/burn_dragon_p2p_browser.js",
    "/burn_dragon_p2p_browser_bg.wasm",
  ];
  let lastError = null;
  for (let attempt = 1; attempt <= SITE_ASSET_RETRIES; attempt += 1) {
    try {
      return await Promise.all(
        relativePaths.map((relativePath) => fetchOk(endpoint(SITE_BASE_URL, relativePath))),
      );
    } catch (error) {
      lastError = error;
      if (attempt >= SITE_ASSET_RETRIES || !shouldRetryTransientFetch(error)) {
        throw error;
      }
      await sleep(SITE_ASSET_RETRY_DELAY_MS * attempt);
    }
  }
  throw lastError;
}

function trimPreview(text) {
  const normalized = String(text ?? "").trim();
  if (normalized.length <= 240) {
    return normalized;
  }
  return `${normalized.slice(0, 240)}...`;
}

function uniqueStrings(values) {
  return Array.from(new Set(values.filter((value) => value != null).map(String)));
}

function endpoint(baseUrl, relativePath) {
  return new URL(relativePath, baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`).toString();
}

function requestedScopes(experimentId) {
  return ["Connect", "Discover", { Train: { experiment_id: experimentId } }];
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

function pendingBrowserLoginState(edgeBaseUrl, login, requestedScopes) {
  return {
    edge_base_url: edgeBaseUrl.replace(/\/+$/, ""),
    created_at: new Date().toISOString(),
    login,
    requested_scopes: requestedScopes,
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
  return labeledRowsValue(tiles, label);
}

function labeledRowsValue(rows, label) {
  const normalizedLabel = label.trim().toLowerCase();
  for (const row of rows ?? []) {
    if (!Array.isArray(row) || row.length === 0) {
      continue;
    }
    if ((row[0] ?? "").trim().toLowerCase() !== normalizedLabel) {
      continue;
    }
    return row
      .slice(1)
      .map((part) => part ?? "")
      .join(" | ")
      .trim();
  }
  return null;
}

function inferTransportSummary(tiles, metricCards = [], keyValues = []) {
  const directTransportTile = statTileValue(tiles, "transport");
  if (directTransportTile) {
    return directTransportTile;
  }
  const transportKeyValue = labeledRowsValue(keyValues, "transport");
  if (transportKeyValue) {
    return transportKeyValue;
  }
  const networkMetric = labeledRowsValue(metricCards, "network");
  if (networkMetric) {
    return networkMetric;
  }
  const peersTile = statTileValue(tiles, "peers");
  if (/\b(webrtc-direct|webtransport|wss|ws)\b/i.test(peersTile ?? "")) {
    return peersTile;
  }
  return null;
}

function expectedTransportLabel(mode) {
  if (mode === "webrtc-direct") return "webrtc-direct";
  if (mode === "webtransport") return "webtransport";
  if (mode === "wss") return "wss";
  return null;
}

function normalizeTransportLabel(value) {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (!normalized) return null;
  if (normalized.includes("webrtc-direct") || normalized.includes("webrtcdirect")) {
    return "webrtc-direct";
  }
  if (normalized.includes("webtransport")) {
    return "webtransport";
  }
  if (/\bwss\b/.test(normalized) || normalized.includes("wssfallback")) {
    return "wss";
  }
  if (/\bws\b/.test(normalized)) {
    return "ws";
  }
  return normalized;
}

function transportConnectedForMode(summary, mode) {
  const normalized = (summary ?? "").toLowerCase();
  if (!normalized || /\b(target|unavailable|connecting|waiting)\b/.test(normalized)) {
    return false;
  }
  const expected = expectedTransportLabel(mode);
  if (!expected) {
    return /\b(webrtc-direct|webtransport|wss|ws)\b/.test(normalized);
  }
  return normalized.includes(expected);
}

function webRtcDirectConsoleMarkerReport(consoleMessages) {
  const texts = consoleMessages.map((entry) => entry.text ?? "");
  const observed = WEBRTC_DIRECT_REQUIRED_CONSOLE_MARKERS.filter((marker) =>
    texts.some((text) => text.includes(marker)),
  );
  const missing = WEBRTC_DIRECT_REQUIRED_CONSOLE_MARKERS.filter(
    (marker) => !observed.includes(marker),
  );
  return {
    required: WEBRTC_DIRECT_REQUIRED_CONSOLE_MARKERS,
    observed,
    missing,
  };
}

function assertWebRtcDirectTransportPhases(report, consoleMessages) {
  if (report.expected_connected_transport !== "webrtc-direct") {
    return;
  }
  report.webrtc_direct_console_markers = webRtcDirectConsoleMarkerReport(consoleMessages);
  if (report.webrtc_direct_console_markers.missing.length > 0) {
    fail(
      `browser canary connected without complete WebRTC-direct phase evidence: missing=${JSON.stringify(report.webrtc_direct_console_markers.missing)} machine=${JSON.stringify(report.browser_machine_state)}`,
    );
  }
}

function preferredAdvertisedTransport(envelope) {
  const preferred = envelope?.payload?.payload?.transport_policy?.preferred;
  if (!Array.isArray(preferred) || preferred.length === 0) {
    return null;
  }
  for (const value of preferred) {
    const normalized = normalizeTransportLabel(value);
    if (normalized) {
      return normalized;
    }
  }
  return null;
}

function expectedConnectedTransport(mode, envelope) {
  if (EXPECT_CONNECTED_TRANSPORT) {
    return normalizeTransportLabel(EXPECT_CONNECTED_TRANSPORT);
  }
  const explicitTransport = expectedTransportLabel(mode);
  if (explicitTransport || mode === "auto") {
    return explicitTransport;
  }
  return preferredAdvertisedTransport(envelope);
}

function expectedMinimumDirectPeers(expectedTransport) {
  if (EXPECT_MIN_DIRECT_PEERS != null) {
    return EXPECT_MIN_DIRECT_PEERS;
  }
  return ["webrtc-direct", "webtransport"].includes(expectedTransport) ? 1 : 0;
}

function machineStateConnected(report) {
  const state = report.browser_machine_state;
  if (!state || typeof state !== "object") {
    return false;
  }
  const expectedTransport = report.expected_connected_transport;
  const connectedTransport = normalizeTransportLabel(state.connected_transport);
  if (!connectedTransport) {
    return false;
  }
  if (expectedTransport && connectedTransport !== expectedTransport) {
    return false;
  }
  const directPeers = Number(state.direct_peers ?? 0);
  return directPeers >= report.expected_min_direct_peers;
}

function reportConnectedForMode(report, mode) {
  if (report.browser_machine_state) {
    return machineStateConnected(report);
  }
  return transportConnectedForMode(report.transport_summary, mode);
}

function browserConfigSeedNodeUrls(browserConfig) {
  if (!browserConfig || typeof browserConfig !== "object") {
    return [];
  }
  const nested = browserConfig.config?.network?.seed_node_urls;
  if (Array.isArray(nested)) {
    return nested;
  }
  const legacy = browserConfig.seed_node_urls;
  if (Array.isArray(legacy)) {
    return legacy;
  }
  return [];
}

function browserConfigTrainingConfig(browserConfig) {
  if (!browserConfig || typeof browserConfig !== "object") {
    return null;
  }
  const nested = browserConfig.config?.training;
  if (nested && typeof nested === "object") {
    return nested;
  }
  return null;
}

function snapshotAllowsBrowserTraining(snapshot, experimentId) {
  if (!snapshot || !experimentId) {
    return false;
  }
  const entries = snapshot.directory?.entries ?? [];
  const entry = entries.find((candidate) => candidate?.experiment_id === experimentId);
  if (!entry) {
    return false;
  }
  const roles = entry.allowed_roles?.roles ?? [];
  if (roles.includes("BrowserTrainerWgpu")) {
    return true;
  }
  return entry.metadata?.["burn_p2p.revision.browser.role.trainer_wgpu"] === "true";
}

function seedTransportMode(seed) {
  if (isDialableWebRtcSeed(seed)) {
    return "webrtc-direct";
  }
  if (isDialableWebTransportSeed(seed)) {
    return "webtransport";
  }
  if (seed.includes("/wss") || seed.includes("/ws")) {
    return "wss";
  }
  return null;
}

function seedMatchesTransportMode(seed, mode) {
  return mode === "auto" || seedTransportMode(seed) === mode;
}

function transportPolicyForMode(mode) {
  if (mode === "webrtc-direct") {
    return { preferred: ["WebRtcDirect"], allow_fallback_wss: false };
  }
  if (mode === "webtransport") {
    return { preferred: ["WebTransport"], allow_fallback_wss: false };
  }
  if (mode === "wss") {
    return { preferred: ["WssFallback"], allow_fallback_wss: true };
  }
  return null;
}

function filterSignedSeedAdvertisementForTransport(envelope, mode) {
  if (mode === "auto") {
    return envelope;
  }
  const filtered = JSON.parse(JSON.stringify(envelope));
  const payload = filtered?.payload?.payload;
  if (!payload || !Array.isArray(payload.seeds)) {
    return filtered;
  }
  payload.seeds = payload.seeds
    .map((record) => ({
      ...record,
      multiaddrs: (record.multiaddrs ?? []).filter((seed) =>
        seedMatchesTransportMode(seed, mode),
      ),
    }))
    .filter((record) => record.multiaddrs.length > 0);
  const policy = transportPolicyForMode(mode);
  if (policy) {
    payload.transport_policy = policy;
  }
  return filtered;
}

function preferValidatedSignedSeedAdvertisement(envelope, mode) {
  if (mode !== "auto") {
    return envelope;
  }
  const filtered = JSON.parse(JSON.stringify(envelope));
  const payload = filtered?.payload?.payload;
  if (!payload || !Array.isArray(payload.seeds)) {
    return filtered;
  }
  const hasWebRtcDirect = payload.seeds.some((record) =>
    (record.multiaddrs ?? []).some(isDialableWebRtcSeed),
  );
  if (!hasWebRtcDirect) {
    return filtered;
  }
  payload.seeds = payload.seeds
    .map((record) => ({
      ...record,
      multiaddrs: (record.multiaddrs ?? []).filter(isDialableWebRtcSeed),
    }))
    .filter((record) => record.multiaddrs.length > 0);
  payload.transport_policy = { preferred: ["WebRtcDirect"], allow_fallback_wss: false };
  return filtered;
}

function filterBrowserConfigForTransport(browserConfig, mode) {
  if (mode === "auto") {
    return browserConfig;
  }
  const filtered = JSON.parse(JSON.stringify(browserConfig));
  const nested = filtered.config?.network?.seed_node_urls;
  if (Array.isArray(nested)) {
    filtered.config.network.seed_node_urls = nested.filter((seed) =>
      seedMatchesTransportMode(seed, mode),
    );
  }
  if (Array.isArray(filtered.seed_node_urls)) {
    filtered.seed_node_urls = filtered.seed_node_urls.filter((seed) =>
      seedMatchesTransportMode(seed, mode),
    );
  }
  if (filtered.signed_seed_advertisement) {
    filtered.signed_seed_advertisement = filterSignedSeedAdvertisementForTransport(
      filtered.signed_seed_advertisement,
      mode,
    );
  }
  return filtered;
}

function isDialableWebRtcSeed(seed) {
  const segments = seed.split("/").filter(Boolean);
  return (
    DIALABLE_WEBRTC_HOST_PROTOCOLS.has(segments[0]) &&
    segments.includes("webrtc-direct") &&
    segments.includes("certhash")
  );
}

function isDialableWebTransportSeed(seed) {
  return seed.includes("/quic-v1/webtransport") && seed.includes("/certhash/");
}

function isDialableBrowserSeed(seed) {
  return (
    isDialableWebRtcSeed(seed) ||
    isDialableWebTransportSeed(seed) ||
    seed.includes("/wss") ||
    seed.includes("/ws")
  );
}

function browserSeedDnsHost(edgeBaseUrl) {
  const host = new URL(edgeBaseUrl).hostname;
  return host && isIP(host) === 0 ? host : null;
}

function canonicalBrowserSeedUrl(edgeBaseUrl, seed) {
  const edgeHost = browserSeedDnsHost(edgeBaseUrl);
  const segments = seed.split("/").filter(Boolean);
  if (
    !edgeHost ||
    segments.length < 3 ||
    !["ip4", "ip6"].includes(segments[0]) ||
    !isDialableBrowserSeed(seed)
  ) {
    return seed;
  }
  return `/dns4/${edgeHost}/${segments.slice(2).join("/")}`;
}

function canonicalBrowserSeedUrls(edgeBaseUrl, seeds) {
  return Array.from(new Set(seeds.map((seed) => canonicalBrowserSeedUrl(edgeBaseUrl, seed))));
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

async function optionalVisibleText(locator) {
  try {
    if ((await locator.count()) === 0) {
      return null;
    }
    const node = locator.first();
    if (!(await node.isVisible())) {
      return null;
    }
    const text = await node.textContent({ timeout: 1_000 });
    return text?.trim() || null;
  } catch {
    return null;
  }
}

async function optionalText(locator) {
  try {
    if ((await locator.count()) === 0) {
      return null;
    }
    const text = await locator.first().textContent({ timeout: 1_000 });
    return text?.trim() || null;
  } catch {
    return null;
  }
}

async function beginBrowserCanaryLogin(snapshot) {
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
  return {
    login,
    requestedScopes: scopes,
  };
}

async function runCanary() {
  validateCanaryMode();
  ensureDir(ARTIFACT_DIR);
  const siteRuntimeAssets = SITE_OVERRIDE_DIR ? [] : await waitForSiteRuntimeAssets();
  const snapshot = await fetchJsonWithTransientRetry(endpoint(EDGE_BASE_URL, "/portal/snapshot"), {
    method: "GET",
    headers: {},
  });
  const signedSeedsEnvelope = await fetchJsonWithTransientRetry(
    endpoint(
      EDGE_BASE_URL,
      snapshot.paths?.browser_seed_advertisement_path ?? "/browser/seeds/signed",
    ),
    { method: "GET", headers: {} },
  );
  const browserConfig = await fetchJsonWithTransientRetry(endpoint(SITE_BASE_URL, "/browser-app-config.json"), {
    method: "GET",
    headers: {},
  });
  const liveSignedSeeds = signedSeedsEnvelope?.payload?.payload?.seeds?.flatMap(
    (record) => record.multiaddrs ?? [],
  ) ?? [];
  const filteredSignedSeedsEnvelope = preferValidatedSignedSeedAdvertisement(
    filterSignedSeedAdvertisementForTransport(signedSeedsEnvelope, TRANSPORT_MODE),
    TRANSPORT_MODE,
  );
  const filteredBrowserConfig = filterBrowserConfigForTransport(browserConfig, TRANSPORT_MODE);
  const expectedTransport = expectedConnectedTransport(
    TRANSPORT_MODE,
    filteredSignedSeedsEnvelope,
  );
  const expectedMinDirectPeers = expectedMinimumDirectPeers(expectedTransport);

  if (snapshot.transports?.webtransport_gateway) {
    fail("live edge is advertising webtransport_gateway without validated native runtime support");
  }

  const signedSeeds = filteredSignedSeedsEnvelope?.payload?.payload?.seeds?.flatMap(
    (record) => record.multiaddrs ?? [],
  ) ?? [];
  const browserConfigSeeds = browserConfigSeedNodeUrls(filteredBrowserConfig);
  const canonicalSignedSeeds = canonicalBrowserSeedUrls(EDGE_BASE_URL, signedSeeds);
  const canonicalBrowserConfigSeeds = canonicalBrowserSeedUrls(EDGE_BASE_URL, browserConfigSeeds);
  const browserTrainingConfig = browserConfigTrainingConfig(filteredBrowserConfig);
  const acceptedReceiptsBeforeTraining = snapshotAcceptedReceiptCount(snapshot);
  const signedHasWebRtcDirect = signedSeeds.some(isDialableWebRtcSeed);
  const signedHasWebTransport = signedSeeds.some(isDialableWebTransportSeed);
  const signedHasWss = signedSeeds.some((value) => value.includes("/wss"));
  const browserCapableSeedCount = Number(signedHasWebRtcDirect) + Number(signedHasWebTransport) + Number(signedHasWss);
  if (!signedSeeds.length || browserCapableSeedCount === 0) {
    fail(`signed browser seeds are missing browser-capable addresses: ${JSON.stringify(signedSeeds)}`);
  }
  if (
    TRANSPORT_MODE !== "auto" &&
    signedSeeds.some((seed) => !seedMatchesTransportMode(seed, TRANSPORT_MODE))
  ) {
    fail(`transport mode ${TRANSPORT_MODE} left unrelated signed seeds: ${JSON.stringify(signedSeeds)}`);
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
  if (TRANSPORT_MODE === "auto" && Boolean(snapshot.transports?.webrtc_direct) !== signedHasWebRtcDirect) {
    fail(
      `signed browser seeds disagree with snapshot webrtc_direct=${snapshot.transports?.webrtc_direct}: ${JSON.stringify(signedSeeds)}`,
    );
  }
  if (
    TRANSPORT_MODE === "auto" &&
    Boolean(snapshot.transports?.webtransport_gateway) !== signedHasWebTransport
  ) {
    fail(
      `signed browser seeds disagree with snapshot webtransport_gateway=${snapshot.transports?.webtransport_gateway}: ${JSON.stringify(signedSeeds)}`,
    );
  }
  if (TRANSPORT_MODE === "auto" && signedHasWebRtcDirect && signedHasWss) {
    fail(
      `auto browser config retained unvalidated WSS fallback despite a WebRTC-direct seed: ${JSON.stringify(signedSeeds)}`,
    );
  }
  if (JSON.stringify(canonicalBrowserConfigSeeds) !== JSON.stringify(canonicalSignedSeeds)) {
    fail(
      `browser config seeds drifted from signed browser seeds: config=${JSON.stringify(browserConfigSeeds)} signed=${JSON.stringify(signedSeeds)} canonical_config=${JSON.stringify(canonicalBrowserConfigSeeds)} canonical_signed=${JSON.stringify(canonicalSignedSeeds)}`,
    );
  }
  if (
    EXPECT_TRAINING &&
    EXPERIMENT_ID &&
    !browserTrainingConfig &&
    !snapshotAllowsBrowserTraining(snapshot, EXPERIMENT_ID)
  ) {
    fail(
      `browser config and live snapshot are missing browser training for selected experiment ${EXPERIMENT_ID}`,
    );
  }
  if (EXPECT_TRAINING && acceptedReceiptsBeforeTraining == null) {
    fail("portal snapshot did not expose diagnostics.accepted_receipts before training");
  }

  const enrollment = await beginBrowserCanaryLogin(snapshot);
  const provider = (snapshot.login_providers ?? []).find(
    (candidate) => candidate?.callback_path && candidate.callback_path.trim().length > 0,
  );
  if (!provider?.callback_path) {
    fail("edge snapshot did not expose a usable callback path");
  }
  const callbackUrl = endpoint(SITE_BASE_URL, `${provider.callback_path}?code=browser-canary-provider-code`);
  const pendingLoginState = pendingBrowserLoginState(
    EDGE_BASE_URL,
    enrollment.login,
    enrollment.requestedScopes,
  );

  const playwright = await loadPlaywright();
  const browserType = playwright[BROWSER_NAME];
  if (!browserType) {
    fail(`playwright browser type ${BROWSER_NAME} is not available`);
  }
  const launchOptions =
    BROWSER_NAME === "chromium"
      ? {
          headless: HEADLESS,
          args: [
            "--enable-unsafe-webgpu",
            "--use-angle=swiftshader",
            "--enable-features=Vulkan,UseSkiaRenderer,WebGPU",
          ],
        }
      : {
          headless: HEADLESS,
        };
  const browser = await browserType.launch(launchOptions);

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
    browser_name: BROWSER_NAME,
    transport_mode: TRANSPORT_MODE,
    expect_training: EXPECT_TRAINING,
    expected_connected_transport: expectedTransport,
    expected_min_direct_peers: expectedMinDirectPeers,
    network_id: snapshot.network_id,
    transports: snapshot.transports,
    browser_config_seed_node_urls: browserConfigSeeds,
    signed_seed_multiaddrs: signedSeeds,
    live_signed_seed_multiaddrs: liveSignedSeeds,
    signed_seed_transport_preference:
      filteredSignedSeedsEnvelope?.payload?.payload?.transport_policy?.preferred ?? [],
    site_runtime_assets: siteRuntimeAssets,
    connect_clicked: false,
    training_button_visible: false,
    training_button_enabled: false,
    training_button_label: null,
    training_action_detail: null,
    connect_button_visible: false,
    get_started_button_visible: false,
    live_status_label: null,
    live_stat_tiles: [],
    live_metric_cards: [],
    live_keyvalues: [],
    live_panel_detail: null,
    live_notice_label: null,
    live_notice_detail: null,
    transport_summary: null,
    browser_machine_state: null,
    quiet_window_ms: QUIET_WINDOW_MS,
    train_timeout_ms: TRAIN_TIMEOUT_MS,
    quiet_window_control_plane_requests: [],
    artifact_http_fallback_requests: [],
    webrtc_direct_console_markers: null,
    receipt_submission: null,
    accepted_receipts_before_training: acceptedReceiptsBeforeTraining,
    durable_receipt_snapshot: null,
    retained_transport_error: null,
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

    if (SITE_OVERRIDE_DIR || TRANSPORT_MODE !== "auto") {
      const filteredBrowserConfigJson = JSON.stringify(filteredBrowserConfig);
      const filteredSignedSeedsJson = JSON.stringify(filteredSignedSeedsEnvelope);
      const signedSeedPath = snapshot.paths?.browser_seed_advertisement_path ?? "/browser/seeds/signed";
      await context.route("**/*", async (route) => {
        const requestUrl = route.request().url();
        const request = new URL(requestUrl);
        const siteBase = new URL(SITE_BASE_URL);
        const edgeBase = new URL(EDGE_BASE_URL);
        if (
          TRANSPORT_MODE !== "auto" &&
          request.origin === siteBase.origin &&
          request.pathname.endsWith("/browser-app-config.json")
        ) {
          await route.fulfill({
            status: 200,
            body: filteredBrowserConfigJson,
            headers: {
              "content-type": "application/json; charset=utf-8",
              "cache-control": "no-store",
            },
          });
          return;
        }
        if (
          TRANSPORT_MODE !== "auto" &&
          request.origin === edgeBase.origin &&
          request.pathname === signedSeedPath
        ) {
          await route.fulfill({
            status: 200,
            body: filteredSignedSeedsJson,
            headers: {
              "content-type": "application/json; charset=utf-8",
              "cache-control": "no-store",
            },
          });
          return;
        }
        if (!SITE_OVERRIDE_DIR) {
          await route.continue();
          return;
        }
        const overridePath = resolveOverrideAssetPath(SITE_OVERRIDE_DIR, requestUrl, SITE_BASE_URL);
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
      ({ networkId, receiptJson, pendingLoginJson, callbackToken, pendingLoginKey, trustedCallbackTokenKey }) => {
        const receiptKey = `burn-p2p.browser.receipt-outbox.${networkId}`;
        try {
          window.localStorage.setItem(receiptKey, receiptJson);
          window.localStorage.setItem(
            pendingLoginKey,
            pendingLoginJson,
          );
          window.sessionStorage.setItem(trustedCallbackTokenKey, callbackToken);
        } catch (error) {
          console.error("burn-dragon-canary-init-storage-failed", String(error));
        }
      },
      {
        networkId: snapshot.network_id,
        receiptJson: JSON.stringify(emptyReceiptOutbox()),
        pendingLoginJson: JSON.stringify(pendingLoginState),
        callbackToken: CALLBACK_TOKEN,
        pendingLoginKey: PENDING_GITHUB_LOGIN_KEY,
        trustedCallbackTokenKey: TRUSTED_CALLBACK_TOKEN_KEY,
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

    await page.goto(callbackUrl, { waitUntil: "domcontentloaded" });

    const connectButton = page.locator('button:has-text("connect")').first();
    const trainActionButton = page.locator(".dragon-live-actions button").first();
    const getStartedButton = page.locator('button:has-text("get started")').first();
    let quietWindowStartedAt = null;
    const canStartTraining = () => {
      const blockingNotice =
        report.live_notice_label &&
        ["connecting", "syncing", "blocked"].includes(
          report.live_notice_label.toLowerCase(),
        );
      return (
        report.training_button_visible &&
        report.training_button_enabled &&
        report.training_button_label === "run browser training" &&
        !blockingNotice
      );
    };
    const captureLiveStatus = async () => {
      report.training_button_visible = await isVisible(trainActionButton);
      report.training_button_enabled = report.training_button_visible
        ? await trainActionButton.isEnabled().catch(() => false)
        : false;
      report.training_button_label = report.training_button_visible
        ? await optionalVisibleText(trainActionButton)
        : null;
      report.training_action_detail = await optionalVisibleText(
        page.locator(".dragon-live-action-note"),
      );
      report.connect_button_visible = await isVisible(connectButton);
      report.get_started_button_visible = await isVisible(getStartedButton);
      report.live_panel_detail = await optionalVisibleText(
        page.locator(".dragon-live-shell .section-detail"),
      );
      report.live_notice_label = await optionalVisibleText(
        page.locator(".activity-notice-label"),
      );
      report.live_notice_detail = await optionalVisibleText(
        page.locator(".activity-notice-detail"),
      );
      report.live_stat_tiles = await page
        .locator(".dragon-live-stats .stat-tile")
        .evaluateAll((nodes) =>
          nodes.map((node) =>
            Array.from(node.children).map((child) => child.textContent?.trim() ?? ""),
          ),
        )
        .catch(() => []);
      report.live_metric_cards = await page
        .locator(".dragon-metrics-grid .dragon-metric")
        .evaluateAll((nodes) =>
          nodes.map((node) =>
            Array.from(node.children).map((child) => child.textContent?.trim() ?? ""),
          ),
        )
        .catch(() => []);
      report.live_keyvalues = await page
        .locator(".dragon-live-keyvalues .keyvalue-row")
        .evaluateAll((nodes) =>
          nodes.map((node) =>
            Array.from(node.children).map((child) => child.textContent?.trim() ?? ""),
          ),
        )
        .catch(() => []);
      report.live_status_label =
        statTileValue(report.live_stat_tiles, "status") ??
        labeledRowsValue(report.live_metric_cards, "mode");
      report.transport_summary = inferTransportSummary(
        report.live_stat_tiles,
        report.live_metric_cards,
        report.live_keyvalues,
      );
      const machineStateText = await optionalText(
        page.locator(".dragon-live-machine-state, .dragon-machine-state"),
      );
      if (machineStateText) {
        try {
          report.browser_machine_state = JSON.parse(machineStateText);
        } catch (error) {
          fail(`browser machine state was not valid JSON: ${String(error)}: ${machineStateText}`);
        }
      }
    };

    const connectDeadline = Date.now() + CONNECT_TIMEOUT_MS;
    const sessionResumeGraceDeadline = Date.now() + 5_000;
    while (Date.now() < connectDeadline) {
      await captureLiveStatus();
      if (
        (EXPECT_TRAINING && canStartTraining()) ||
        (!EXPECT_TRAINING && reportConnectedForMode(report, TRANSPORT_MODE))
      ) {
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
    if (EXPECT_TRAINING && !canStartTraining()) {
      report.artifact_http_fallback_requests = requests.filter((entry) => entry.artifactFallback);
      fail(
        `browser canary did not become training-ready: status=${report.live_status_label ?? "missing"} transport=${report.transport_summary ?? "missing"} notice=${report.live_notice_detail ?? report.live_panel_detail ?? "none"} action=${report.training_button_label ?? "missing"} action_detail=${report.training_action_detail ?? "none"} button_visible=${report.training_button_visible} button_enabled=${report.training_button_enabled}`,
      );
    }
    if (!EXPECT_TRAINING && !reportConnectedForMode(report, TRANSPORT_MODE)) {
      report.artifact_http_fallback_requests = requests.filter((entry) => entry.artifactFallback);
      fail(
        `browser canary did not connect with expected transport ${report.expected_connected_transport ?? TRANSPORT_MODE}: status=${report.live_status_label ?? "missing"} transport=${report.transport_summary ?? "missing"} machine=${JSON.stringify(report.browser_machine_state)} notice=${report.live_notice_detail ?? report.live_panel_detail ?? "none"}`,
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
    if (EXPECT_TRAINING && report.expected_connected_transport && !reportConnectedForMode(report, TRANSPORT_MODE)) {
      fail(
        `browser canary became training-ready without expected transport ${report.expected_connected_transport}: transport=${report.transport_summary ?? "missing transport signal"} machine=${JSON.stringify(report.browser_machine_state)}`,
      );
    }
    assertWebRtcDirectTransportPhases(report, consoleMessages);
    report.retained_transport_error = reportConnectedForMode(report, TRANSPORT_MODE)
      ? null
      : (report.browser_machine_state?.last_error ?? null);

    if (!EXPECT_TRAINING) {
      report.success = true;
      await page.screenshot({ path: screenshotPath, fullPage: true });
      return report;
    }

    const receiptResponse = await Promise.all([
      page.waitForResponse(
        (response) =>
          response.request().method() === "POST" &&
          new URL(response.url()).host === new URL(EDGE_BASE_URL).host &&
          new URL(response.url()).pathname === "/receipts/browser" &&
          response.status() >= 200 &&
          response.status() < 300,
        { timeout: TRAIN_TIMEOUT_MS },
      ),
      trainActionButton.click(),
    ]).then(([response]) => response);
    const receiptBodyText = await receiptResponse.text();
    let receiptBody = null;
    try {
      receiptBody = receiptBodyText ? JSON.parse(receiptBodyText) : null;
    } catch (error) {
      fail(
        `browser receipt response was not valid JSON: ${String(error)}: ${trimPreview(receiptBodyText)}`,
      );
    }
    const acceptedReceiptIds = acceptedReceiptIdsFromSubmission(receiptBody);
    report.receipt_submission = {
      url: receiptResponse.url(),
      status: receiptResponse.status(),
      accepted_receipt_ids: acceptedReceiptIds,
      pending_receipt_count: receiptBody?.pending_receipt_count ?? null,
      body_preview: trimPreview(receiptBodyText),
    };
    if (acceptedReceiptIds.length === 0) {
      fail(
        `browser receipt submission returned no accepted receipt ids: ${trimPreview(receiptBodyText)}`,
      );
    }
    await page.waitForFunction(
      () =>
        document.body.innerText.includes("Browser training complete:") ||
        document.body.innerText.includes("train loss"),
      { timeout: TRAIN_TIMEOUT_MS },
    );
    report.durable_receipt_snapshot = await waitForDurableReceiptCount(
      acceptedReceiptIds,
      acceptedReceiptsBeforeTraining,
    );
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
    report.console_errors = uniqueStrings(
      consoleMessages.filter((entry) => entry.type === "error").map((entry) => entry.text),
    );
    report.page_errors = uniqueStrings(pageErrors);
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
