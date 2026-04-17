from pathlib import Path


def main() -> None:
    template = Path(
        "crates/burn_dragon_p2p/deploy/terraform/aws/templates/Caddyfile.tftpl"
    ).read_text()
    main_tf = Path("crates/burn_dragon_p2p/deploy/terraform/aws/main.tf").read_text()
    assert "@browser_shell_get" in template, "missing browser shell redirect matcher"
    assert "header Upgrade websocket" in template, (
        "browser shell redirect must exempt websocket upgrades"
    )
    assert "@p2p_websocket" in template, "missing websocket matcher for swarm fallback"
    assert "reverse_proxy 127.0.0.1:${p2p_port}" in template, (
        "websocket upgrades must proxy to the native swarm port"
    )
    assert "redir ${browser_app_base_url}{uri} 302" in template, (
        "browser shell redirect target changed unexpectedly"
    )
    assert "p2p_port             = var.p2p_port" in main_tf, (
        "terraform must pass p2p_port into the edge caddy template"
    )
    assert '"/ip4/0.0.0.0/tcp/${var.p2p_port}/ws"' in main_tf, (
        "bootstrap native config must expose websocket listen addresses"
    )
    print("edge-caddyfile-ok")


if __name__ == "__main__":
    main()
