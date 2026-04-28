from pathlib import Path


def main() -> None:
    source = Path("xtask/src/browser_site.rs").read_text()
    required_snippets = [
        "use burn_dragon_p2p::config::DragonBrowserTrainingConfig;",
        "use burn_dragon_p2p::profile::browser_training_config_from_directory_entries;",
        "training: Option<DragonBrowserTrainingConfig>,",
        "let training = resolve_browser_training_config(",
        "training,",
        "signed_seed_advertisement: None,",
        "fn resolve_browser_training_config(",
        "snapshot.directory.entries",
        "Ok(training)",
        "warning: continuing without embedded snapshot because explicit browser seed URLs were provided; the browser app will fetch live edge state at runtime",
    ]
    for snippet in required_snippets:
        assert snippet in source, f"browser_site.rs missing required training-config snippet: {snippet}"
    print("browser-site-training-config-ok")


if __name__ == "__main__":
    main()
