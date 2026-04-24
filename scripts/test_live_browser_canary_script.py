from pathlib import Path


def main() -> None:
    script = Path("scripts/live-browser-canary.mjs").read_text()
    required_snippets = [
        "function browserConfigSeedNodeUrls(browserConfig)",
        "function browserConfigTrainingConfig(browserConfig)",
        "browserConfig.config?.network?.seed_node_urls",
        "browserConfig.config?.training",
        "browserConfig.seed_node_urls",
        "const browserConfigSeeds = browserConfigSeedNodeUrls(filteredBrowserConfig);",
        "const browserTrainingConfig = browserConfigTrainingConfig(filteredBrowserConfig);",
        'browser config is missing training payload for selected experiment ${EXPERIMENT_ID}',
        'const PENDING_GITHUB_LOGIN_KEY = "burn-dragon-p2p.pending-github-login";',
        'const TRUSTED_CALLBACK_TOKEN_KEY = "burn-dragon-p2p.canary-callback-token";',
        "const callbackUrl = endpoint(SITE_BASE_URL, `${provider.callback_path}?code=browser-canary-provider-code`);",
        "window.sessionStorage.setItem(trustedCallbackTokenKey, callbackToken);",
        'document.body.innerText.includes("Browser training complete:")',
        "async function beginBrowserCanaryLogin(snapshot) {",
        'const BROWSER_NAME = (',
        'const TRANSPORT_MODE = (',
        'const EXPECT_TRAINING = parseBooleanEnv("BURN_DRAGON_BROWSER_CANARY_EXPECT_TRAINING", true);',
        "function filterSignedSeedAdvertisementForTransport(envelope, mode)",
        "function preferValidatedSignedSeedAdvertisement(envelope, mode)",
        'payload.transport_policy = { preferred: ["WebRtcDirect"], allow_fallback_wss: false };',
        "function filterBrowserConfigForTransport(browserConfig, mode)",
        '["ip4", "ip6", "dns4", "dns6", "dnsaddr"].includes(segments[0])',
        "function reportConnectedForMode(report, mode)",
        "return transportConnectedForMode(report.transport_summary, mode);",
        "function labeledRowsValue(rows, label)",
        '".dragon-metrics-grid .dragon-metric"',
        '".dragon-live-keyvalues .keyvalue-row"',
        '".dragon-live-machine-state, .dragon-machine-state"',
        "WEBRTC_DIRECT_REQUIRED_CONSOLE_MARKERS",
        "libp2p webrtc-direct: completed Noise handshake peer=",
        "function assertWebRtcDirectTransportPhases(report, consoleMessages)",
        "browser canary connected without complete WebRTC-direct phase evidence",
        "report.webrtc_direct_console_markers = webRtcDirectConsoleMarkerReport(consoleMessages);",
        "retained_transport_error: null",
        "report.page_errors = uniqueStrings(pageErrors);",
        "report.console_errors = uniqueStrings(",
        "function uniqueStrings(values)",
        "report.retained_transport_error = reportConnectedForMode(report, TRANSPORT_MODE)",
        "? null",
        ": (report.browser_machine_state?.last_error ?? null);",
        "BURN_DRAGON_BROWSER_CANARY_DURABLE_RECEIPT_TIMEOUT_MS",
        "function snapshotAcceptedReceiptCount(snapshot)",
        "function acceptedReceiptIdsFromSubmission(body)",
        "async function waitForDurableReceiptCount(acceptedReceiptIds, baselineCount)",
        "browser receipt was accepted but did not become durable edge state",
        "accepted_receipts_before_training: acceptedReceiptsBeforeTraining",
        "durable_receipt_snapshot: null",
        "accepted_receipt_ids: acceptedReceiptIds",
        "report.durable_receipt_snapshot = await waitForDurableReceiptCount(",
    ]
    for snippet in required_snippets:
        assert snippet in script, f"live-browser-canary.mjs missing required snippet: {snippet}"
    forbidden_snippets = [
        'const session = await fetchJson(endpoint(EDGE_BASE_URL, callbackPath)',
        'const certificate = await fetchJson(endpoint(EDGE_BASE_URL, snapshot.paths.enroll_path)',
        "SELECTED_EXPERIMENT_ID",
        "/release-manifest.json",
        "browser canary retained a direct transport error after connect",
    ]
    for snippet in forbidden_snippets:
        assert snippet not in script, (
            f"live-browser-canary.mjs should not pre-consume browser callback flow: {snippet}"
        )
    print("live-browser-canary-script-ok")


if __name__ == "__main__":
    main()
