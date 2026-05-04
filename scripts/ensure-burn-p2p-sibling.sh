#!/usr/bin/env bash
set -euo pipefail

repo="${BURN_DRAGON_P2P_REPO:-https://github.com/aberration-technology/burn_p2p.git}"
default_ref="87b9aabd49421088fd050695ee7553175d13eeb1"
ref="${BURN_DRAGON_P2P_REF:-${default_ref}}"
sibling="${BURN_DRAGON_P2P_SIBLING_DIR:-../burn_p2p}"
manifest="${sibling}/crates/burn_p2p/Cargo.toml"

if [[ ! "${ref}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "BURN_DRAGON_P2P_REF must be a full 40-character commit SHA, got: ${ref}" >&2
  exit 1
fi

if [[ -f "${manifest}" ]]; then
  echo "burn_p2p sibling already available at ${sibling}"
  exit 0
fi

rm -rf "${sibling}"
git clone --depth 1 "${repo}" "${sibling}"
git -C "${sibling}" fetch --depth 1 origin "${ref}"
git -C "${sibling}" checkout --detach FETCH_HEAD
resolved_ref="$(git -C "${sibling}" rev-parse HEAD)"
if [[ "${resolved_ref}" != "${ref}" ]]; then
  echo "Resolved burn_p2p ref ${resolved_ref} does not match expected ${ref}" >&2
  exit 1
fi

test -f "${manifest}"
git -C "${sibling}" rev-parse --short HEAD
