#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_live_browser_canary.py <report_path>")

    report_path = Path(sys.argv[1])
    if not report_path.is_file():
        raise SystemExit(0)

    report = json.loads(report_path.read_text())
    receipt = report.get("receipt_submission") or {}
    durable_receipt = report.get("durable_receipt_snapshot") or {}
    control_requests = report.get("quiet_window_control_plane_requests") or []
    artifact_fallback = report.get("artifact_http_fallback_requests") or []
    live_status = report.get("live_status_label") or "n/a"
    transport_summary = report.get("transport_summary") or "n/a"
    machine_state = report.get("browser_machine_state") or {}
    webrtc_markers = report.get("webrtc_direct_console_markers") or {}

    print("## live browser canary")
    print()
    print(f"- Success: `{report.get('success', False)}`")
    print(f"- Principal id: `{report.get('principal_id', 'n/a')}`")
    print(f"- Browser: `{report.get('browser_name', 'n/a')}`")
    print(f"- Transport mode: `{report.get('transport_mode', 'n/a')}`")
    print(f"- Expected connected transport: `{report.get('expected_connected_transport') or 'n/a'}`")
    print(f"- Expected minimum direct peers: `{report.get('expected_min_direct_peers', 'n/a')}`")
    print(f"- Expect training: `{report.get('expect_training', 'n/a')}`")
    print(f"- Live status: `{live_status}`")
    print(f"- Transport signal: `{transport_summary}`")
    print(f"- Machine connected transport: `{machine_state.get('connected_transport') or 'n/a'}`")
    print(f"- Machine direct peers: `{machine_state.get('direct_peers', 'n/a')}`")
    print(f"- Machine last error: `{machine_state.get('last_error') or 'none'}`")
    print(
        f"- WebRTC-direct phase evidence: `{len(webrtc_markers.get('observed') or [])}/{len(webrtc_markers.get('required') or [])}`"
    )
    print(
        f"- Missing WebRTC-direct phases: `{', '.join(webrtc_markers.get('missing') or []) or 'none'}`"
    )
    print(
        f"- Signed seed transports: `{', '.join(report.get('signed_seed_transport_preference') or []) or 'none'}`"
    )
    print(f"- Connect clicked: `{report.get('connect_clicked', False)}`")
    print(f"- Training button visible: `{report.get('training_button_visible', False)}`")
    print(f"- Quiet-window control-plane requests: `{len(control_requests)}`")
    print(f"- Edge artifact fallback requests: `{len(artifact_fallback)}`")
    print(f"- Receipt status: `{receipt.get('status', 'n/a')}`")
    print(
        f"- Accepted receipt ids: `{', '.join(receipt.get('accepted_receipt_ids') or []) or 'none'}`"
    )
    print(
        f"- Durable receipts: `{durable_receipt.get('observed_accepted_receipts', 'n/a')}` "
        f"(baseline `{report.get('accepted_receipts_before_training', 'n/a')}`)"
    )
    print(f"- Error: `{report.get('error') or 'none'}`")


if __name__ == "__main__":
    main()
