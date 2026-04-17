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
    assert "reverse_proxy 127.0.0.1:${p2p_websocket_port}" in template, (
        "websocket upgrades must proxy to the dedicated local websocket port"
    )
    assert "protocols h1 h2" in template, (
        "edge caddy config must disable HTTP/3 so udp/443 remains available for browser direct transport"
    )
    assert "redir ${browser_app_base_url}{uri} 302" in template, (
        "browser shell redirect target changed unexpectedly"
    )
    assert "p2p_port             = var.p2p_port" in main_tf, (
        "terraform must pass p2p_port into the edge caddy template"
    )
    assert "p2p_websocket_port   = var.p2p_port + 1" in main_tf, (
        "terraform must derive and pass the dedicated websocket port to caddy"
    )
    assert "p2p_webrtc_port              = 443" in main_tf, (
        "terraform must reserve udp/443 for browser webrtc-direct reachability"
    )
    assert '"/ip4/127.0.0.1/tcp/${var.p2p_port + 1}/ws"' in main_tf, (
        "bootstrap native config must bind websocket transport on a dedicated loopback port"
    )
    print("edge-caddyfile-ok")


if __name__ == "__main__":
    main()
