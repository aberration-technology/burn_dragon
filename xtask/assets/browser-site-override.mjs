import fs from "node:fs";
import path from "node:path";

export function contentTypeForPath(filePath) {
  if (filePath.endsWith(".html")) return "text/html; charset=utf-8";
  if (filePath.endsWith(".js") || filePath.endsWith(".mjs")) {
    return "text/javascript; charset=utf-8";
  }
  if (filePath.endsWith(".css")) return "text/css; charset=utf-8";
  if (filePath.endsWith(".json") || filePath.endsWith(".map")) {
    return "application/json; charset=utf-8";
  }
  if (filePath.endsWith(".wasm")) return "application/wasm";
  return "application/octet-stream";
}

export function resolveOverrideAssetPath(overrideDir, requestUrl, siteBaseUrl) {
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

export function resolveOverrideFilePath(overrideDir, requestUrl, siteBaseUrl) {
  const assetPath = resolveOverrideAssetPath(overrideDir, requestUrl, siteBaseUrl);
  if (!assetPath) {
    return null;
  }
  if (fs.existsSync(assetPath) && fs.statSync(assetPath).isDirectory()) {
    return path.join(assetPath, "index.html");
  }
  return assetPath;
}
