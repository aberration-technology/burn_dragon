from pathlib import Path


def main() -> None:
    source = Path("xtask/src/browser_site.rs").read_text()
    required_snippets = [
        "use burn_dragon_p2p::config::DragonBrowserTrainingConfig;",
        "use burn_dragon_p2p::profile::browser_training_config_from_directory_entries;",
        "training: Option<DragonBrowserTrainingConfig>,",
        "let training = resolve_browser_training_config(",
        "training,",
        "fn resolve_browser_training_config(",
        "snapshot.directory.entries",
        "selected experiment `{selected_experiment_id}` requires an edge snapshot so the browser training profile can be embedded",
        "selected experiment `{selected_experiment_id}` does not publish a browser training profile",
    ]
    for snippet in required_snippets:
        assert snippet in source, f"browser_site.rs missing required training-config snippet: {snippet}"
    print("browser-site-training-config-ok")


if __name__ == "__main__":
    main()
