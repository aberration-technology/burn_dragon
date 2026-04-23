#!/usr/bin/env bash
set -euo pipefail

repo="${BURN_DRAGON_P2P_REPO:-https://github.com/aberration-technology/burn_p2p.git}"
ref="${BURN_DRAGON_P2P_REF:-main}"
sibling="${BURN_DRAGON_P2P_SIBLING_DIR:-../burn_p2p}"
manifest="${sibling}/crates/burn_p2p/Cargo.toml"

if [[ -f "${manifest}" ]]; then
  echo "burn_p2p sibling already available at ${sibling}"
  exit 0
fi

rm -rf "${sibling}"
git clone --depth 1 --branch "${ref}" "${repo}" "${sibling}" 2>/tmp/burn_p2p_clone.log || {
  rm -rf "${sibling}"
  git clone --depth 1 "${repo}" "${sibling}"
  git -C "${sibling}" fetch --depth 1 origin "${ref}"
  git -C "${sibling}" checkout --detach FETCH_HEAD
}

test -f "${manifest}"
git -C "${sibling}" rev-parse --short HEAD
