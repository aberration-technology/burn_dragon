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
        "function reportConnectedForMode(report, mode)",
        "return transportConnectedForMode(report.transport_summary, mode);",
    ]
    for snippet in required_snippets:
        assert snippet in script, f"live-browser-canary.mjs missing required snippet: {snippet}"
    forbidden_snippets = [
        'const session = await fetchJson(endpoint(EDGE_BASE_URL, callbackPath)',
        'const certificate = await fetchJson(endpoint(EDGE_BASE_URL, snapshot.paths.enroll_path)',
        "SELECTED_EXPERIMENT_ID",
        "/release-manifest.json",
    ]
    for snippet in forbidden_snippets:
        assert snippet not in script, (
            f"live-browser-canary.mjs should not pre-consume browser callback flow: {snippet}"
        )
    print("live-browser-canary-script-ok")


if __name__ == "__main__":
    main()
