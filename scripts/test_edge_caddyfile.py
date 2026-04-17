from pathlib import Path


def main() -> None:
    template = Path(
        "crates/burn_dragon_p2p/deploy/terraform/aws/templates/Caddyfile.tftpl"
    ).read_text()
    assert "@browser_shell_get" in template, "missing browser shell redirect matcher"
    assert "header Upgrade websocket" in template, (
        "browser shell redirect must exempt websocket upgrades"
    )
    assert "redir ${browser_app_base_url}{uri} 302" in template, (
        "browser shell redirect target changed unexpectedly"
    )
    print("edge-caddyfile-ok")


if __name__ == "__main__":
    main()
