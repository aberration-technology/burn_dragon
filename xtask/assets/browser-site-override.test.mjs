import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  contentTypeForPath,
  resolveOverrideAssetPath,
  resolveOverrideFilePath,
} from "./browser-site-override.mjs";

function siteFixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "burn-dragon-browser-site-"));
  fs.mkdirSync(path.join(root, "callback", "github"), { recursive: true });
  fs.writeFileSync(path.join(root, "index.html"), "root");
  fs.writeFileSync(path.join(root, "callback", "github", "index.html"), "callback");
  fs.writeFileSync(path.join(root, "browser-app-loader.js"), "export {};");
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  return root;
}

test("callback directory resolves its own index document", (t) => {
  const root = siteFixture(t);
  const callbackUrl = "https://dragon.example/callback/github?code=canary";
  const resolved = resolveOverrideFilePath(
    root,
    callbackUrl,
    "https://dragon.example",
  );
  assert.equal(resolved, path.join(root, "callback", "github", "index.html"));
  const loaderUrl = new URL("../../browser-app-loader.js", callbackUrl).toString();
  assert.equal(
    resolveOverrideFilePath(root, loaderUrl, "https://dragon.example"),
    path.join(root, "browser-app-loader.js"),
  );
});

test("root and static assets remain rooted in the override artifact", (t) => {
  const root = siteFixture(t);
  assert.equal(
    resolveOverrideFilePath(root, "https://dragon.example/", "https://dragon.example"),
    path.join(root, "index.html"),
  );
  assert.equal(
    resolveOverrideFilePath(
      root,
      "https://dragon.example/browser-app-loader.js",
      "https://dragon.example",
    ),
    path.join(root, "browser-app-loader.js"),
  );
});

test("cross-origin requests bypass the site override", (t) => {
  const root = siteFixture(t);
  assert.equal(
    resolveOverrideAssetPath(
      root,
      "https://edge.dragon.example/portal/snapshot",
      "https://dragon.example",
    ),
    null,
  );
});

test("module content types are JavaScript", () => {
  assert.equal(contentTypeForPath("browser-app-loader.js"), "text/javascript; charset=utf-8");
  assert.equal(contentTypeForPath("browser-site-override.mjs"), "text/javascript; charset=utf-8");
});
