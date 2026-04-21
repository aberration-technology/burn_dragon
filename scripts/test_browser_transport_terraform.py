#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TF_ROOT = ROOT / "crates/burn_dragon_p2p/deploy/terraform/aws"


def require_contains(text: str, snippet: str, label: str) -> None:
    if snippet not in text:
        raise SystemExit(f"missing {label}: {snippet}")


def main() -> None:
    main_tf = (TF_ROOT / "main.tf").read_text()
    outputs_tf = (TF_ROOT / "outputs.tf").read_text()

    require_contains(
        main_tf,
        "p2p_webrtc_port              = 443",
        "browser WebRTC-direct UDP/443 reservation",
    )
    require_contains(
        main_tf,
        '"/ip4/0.0.0.0/udp/${local.p2p_webrtc_port}/webrtc-direct"',
        "bootstrap WebRTC-direct listen multiaddr",
    )
    require_contains(
        main_tf,
        '"/ip4/PUBLIC_IP/udp/${local.p2p_webrtc_port}/webrtc-direct"',
        "bootstrap public WebRTC-direct external multiaddr template",
    )
    require_contains(
        main_tf,
        """ingress {
    from_port   = local.p2p_webrtc_port
    to_port     = local.p2p_webrtc_port
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }""",
        "bootstrap security-group UDP WebRTC-direct ingress",
    )
    require_contains(
        outputs_tf,
        'description = "Deprecated synthetic WebRTC hint. Use the signed browser seed advertisement endpoint instead because dialable browser WebRTC addresses require a runtime certhash."',
        "deprecated synthetic WebRTC output warning",
    )
    require_contains(
        outputs_tf,
        'value       = ""',
        "empty deprecated synthetic WebRTC output",
    )
    print("browser-transport-terraform-ok")


if __name__ == "__main__":
    main()
