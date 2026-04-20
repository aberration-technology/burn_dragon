from pathlib import Path


def main() -> None:
    script = Path("scripts/live-browser-canary.mjs").read_text()
    required_snippets = [
        "function browserConfigSeedNodeUrls(browserConfig)",
        "function browserConfigTrainingConfig(browserConfig)",
        "browserConfig.config?.network?.seed_node_urls",
        "browserConfig.config?.training",
        "browserConfig.seed_node_urls",
        "const browserConfigSeeds = browserConfigSeedNodeUrls(browserConfig);",
        "const browserTrainingConfig = browserConfigTrainingConfig(browserConfig);",
        'browser config is missing training payload for selected experiment ${SELECTED_EXPERIMENT_ID}',
        'const PENDING_GITHUB_LOGIN_KEY = "burn-dragon-p2p.pending-github-login";',
        'const TRUSTED_CALLBACK_TOKEN_KEY = "burn-dragon-p2p.canary-callback-token";',
        "const callbackUrl = endpoint(SITE_BASE_URL, `${provider.callback_path}?code=browser-canary-provider-code`);",
        "window.sessionStorage.setItem(trustedCallbackTokenKey, callbackToken);",
        'document.body.innerText.includes("Browser training complete:")',
        "async function beginBrowserCanaryLogin(snapshot) {",
    ]
    for snippet in required_snippets:
        assert snippet in script, f"live-browser-canary.mjs missing required snippet: {snippet}"
    forbidden_snippets = [
        'const session = await fetchJson(endpoint(EDGE_BASE_URL, callbackPath)',
        'const certificate = await fetchJson(endpoint(EDGE_BASE_URL, snapshot.paths.enroll_path)',
    ]
    for snippet in forbidden_snippets:
        assert snippet not in script, (
            f"live-browser-canary.mjs should not pre-consume browser callback flow: {snippet}"
        )
    print("live-browser-canary-script-ok")


if __name__ == "__main__":
    main()
