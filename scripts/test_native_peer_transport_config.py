from pathlib import Path

main_tf = Path("crates/burn_dragon_p2p/deploy/terraform/aws/main.tf").read_text()
trainer_template = Path(
    "crates/burn_dragon_p2p/deploy/terraform/aws/templates/trainer-user-data.sh.tftpl"
).read_text()
validator_template = Path(
    "crates/burn_dragon_p2p/deploy/terraform/aws/templates/validator-user-data.sh.tftpl"
).read_text()
native_example = Path("crates/burn_dragon_p2p/deploy/native-peer.toml.example").read_text()

required_main_tf_fragments = [
    '"/ip4/PUBLIC_IP/udp/${local.p2p_webrtc_port}/webrtc-direct"',
    "trainer_webrtc_port                = local.p2p_webrtc_port",
    "validator_webrtc_port                = local.p2p_webrtc_port",
    "protocol    = \"udp\"",
    "from_port   = local.p2p_webrtc_port",
]

for fragment in required_main_tf_fragments:
    if fragment not in main_tf:
        raise SystemExit(f"terraform is missing browser-capable native peer transport config: {fragment}")

required_template_fragments = [
    '"/ip4/0.0.0.0/udp/${trainer_webrtc_port}/webrtc-direct"',
    '"/ip4/$${PUBLIC_IPV4}/udp/${trainer_webrtc_port}/webrtc-direct"',
    '"/ip4/0.0.0.0/udp/${validator_webrtc_port}/webrtc-direct"',
    '"/ip4/$${PUBLIC_IPV4}/udp/${validator_webrtc_port}/webrtc-direct"',
]

for fragment in required_template_fragments[:2]:
    if fragment not in trainer_template:
        raise SystemExit(f"trainer template is missing native browser transport config: {fragment}")

for fragment in required_template_fragments[2:]:
    if fragment not in validator_template:
        raise SystemExit(f"validator template is missing native browser transport config: {fragment}")

if "public-ipv4" not in trainer_template or "public-ipv4" not in validator_template:
    raise SystemExit("managed native peer templates do not derive public IPv4 metadata")

if '"/ip4/PUBLIC_IP/udp/443/webrtc-direct"' not in native_example:
    raise SystemExit("native peer example is missing a browser-capable external webrtc-direct address")

bootstrap_user_data = Path(
    "crates/burn_dragon_p2p/deploy/terraform/aws/templates/user-data.sh.tftpl"
).read_text()
for fragment in [
    'BOOTSTRAP_PUBLIC_IPV4="$(imds_get public-ipv4 || true)"',
    'address.replace("PUBLIC_IP", sys.argv[1])',
]:
    if fragment not in bootstrap_user_data:
        raise SystemExit(f"bootstrap user-data is missing public IPv4 bootstrap address rewriting: {fragment}")

print("native-peer-transport-config-ok")
