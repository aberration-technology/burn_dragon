from pathlib import Path
import re

main_tf = Path('crates/burn_dragon_p2p/deploy/terraform/aws/main.tf').read_text()

expected = re.compile(r"bootstrap_head_mirror_seed_node_urls\s*=\s*local\.bootstrap_peer_internal_multiaddrs")
legacy = re.compile(r"bootstrap_head_mirror_seed_node_urls\s*=\s*local\.managed_trainer_seed_node_urls")

if legacy.search(main_tf):
    raise SystemExit('bootstrap head mirror still uses public managed trainer seed URLs')
if not expected.search(main_tf):
    raise SystemExit('bootstrap head mirror is not seeded from bootstrap_peer_internal_multiaddrs')

print('bootstrap-head-mirror-seed-urls-ok')
