use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};

use crate::bootstrap_runtime::{
    RuntimeCommandEnv, preserve_current_heads, render_bootstrap_runtime_sync_commands,
};

const BURN_P2P_SIBLING_REF: &str = "750c73dafa7d6fcb32b984b1da00548baea12383";

pub fn run() -> Result<()> {
    repository_has_no_scripts_tree()?;
    runtime_sync_contract()?;
    bootstrap_head_preservation_contract()?;
    workflow_sibling_checkout_contract()?;
    deployment_workflow_contracts()?;
    browser_canary_contracts()?;
    native_canary_contracts()?;
    browser_and_native_transport_contracts()?;
    production_profile_contracts()?;
    browser_site_contracts()?;
    cleanup_workflow_contracts()?;
    agent_task_contracts()?;
    println!("deployment-script-checks-ok");
    Ok(())
}

fn repository_has_no_scripts_tree() -> Result<()> {
    for path in walk(".")? {
        if path
            .components()
            .any(|component| component.as_os_str() == ".git")
            || path
                .components()
                .any(|component| component.as_os_str() == "target")
            || path
                .components()
                .any(|component| component.as_os_str() == "burn_p2p-sibling")
        {
            continue;
        }
        if path.file_name().and_then(|value| value.to_str()) == Some("scripts") {
            bail!("legacy scripts directory remains: {}", path.display());
        }
    }
    Ok(())
}

fn runtime_sync_contract() -> Result<()> {
    let commands = render_bootstrap_runtime_sync_commands(&RuntimeCommandEnv {
        bootstrap_object_uri: "s3://bucket/runtime/bootstrap.json".to_owned(),
        caddy_object_uri: "s3://bucket/runtime/Caddyfile".to_owned(),
        bootstrap_service_unit_object_uri: "s3://bucket/runtime/burn-p2p-bootstrap.service"
            .to_owned(),
        head_mirror_config_object_uri: "s3://bucket/runtime/bootstrap-head-mirror.toml"
            .to_owned(),
        head_mirror_auth_script_object_uri:
            "s3://bucket/runtime/burn-dragon-p2p-fetch-head-mirror-auth-bundle".to_owned(),
        head_mirror_service_object_uri:
            "s3://bucket/runtime/burn-dragon-p2p-head-mirror.service".to_owned(),
        bootstrap_install_source: "crate".to_owned(),
        bootstrap_crate_version: "0.21.6".to_owned(),
        bootstrap_git_repository: "https://github.com/aberration-technology/burn_p2p.git"
            .to_owned(),
        bootstrap_git_ref: String::new(),
        bootstrap_binary_object_uri: String::new(),
        bootstrap_binary_sha256: String::new(),
        bootstrap_features: "admin-http,metrics,metrics-indexer,artifact-publish,artifact-download,artifact-fs,artifact-s3,browser-edge,browser-join,auth-github,rbac,social".to_owned(),
        bootstrap_reinstall: "false".to_owned(),
        dragon_git_repository: "https://github.com/aberration-technology/burn_dragon.git"
            .to_owned(),
        dragon_git_ref: "main".to_owned(),
        head_mirror_binary_object_uri: String::new(),
        head_mirror_binary_sha256: String::new(),
        head_mirror_reinstall: "true".to_owned(),
    })?;
    let joined = commands.join("\n");
    require_contains(
        &joined,
        "timed out waiting for bootstrap runtime sync prerequisites",
        "runtime sync waits for prerequisites",
    )?;
    require_contains(
        &joined,
        "cargo install --locked burn_p2p_bootstrap --version '0.21.6'",
        "crate bootstrap install command",
    )?;
    require_contains(
        &joined,
        "ln -sf /root/.cargo/bin/burn-p2p-bootstrap /usr/local/bin/burn-p2p-bootstrap",
        "bootstrap binary symlink",
    )?;
    require_contains(
        &joined,
        "systemctl enable burn-dragon-p2p-head-mirror",
        "head mirror enabled",
    )?;
    require_absent(
        &joined,
        "systemctl restart burn-dragon-p2p-head-mirror",
        "head mirror restart remains delegated",
    )?;
    require_contains(
        &joined,
        "BOOTSTRAP_PUBLIC_IPV4",
        "bootstrap public ip rewrite",
    )?;
    require_contains(
        &joined,
        "bootstrap_root = Path(\"/var/lib/burn-p2p/bootstrap-peer\")",
        "corrupt state quarantine root",
    )?;
    require_contains(
        &joined,
        "bootstrap_root / \"state\" / \"security-state.json\"",
        "security state quarantine",
    )?;
    require_contains(
        &joined,
        "sorted((bootstrap_root / \"state\" / \"transfers\").glob(\"*.json\"))",
        "transfer state quarantine",
    )?;

    let git_commands = render_bootstrap_runtime_sync_commands(&RuntimeCommandEnv {
        bootstrap_install_source: "git".to_owned(),
        bootstrap_git_ref: "b14acc12".to_owned(),
        bootstrap_reinstall: "true".to_owned(),
        ..dummy_runtime_env()
    })?
    .join("\n");
    require_contains(
        &git_commands,
        "cargo install --locked --git 'https://github.com/aberration-technology/burn_p2p.git' --rev 'b14acc12' burn_p2p_bootstrap",
        "git bootstrap install command",
    )?;

    let prebuilt = render_bootstrap_runtime_sync_commands(&RuntimeCommandEnv {
        bootstrap_binary_object_uri: "s3://bucket/runtime/burn-p2p-bootstrap".to_owned(),
        bootstrap_binary_sha256: "bootstrapsha".to_owned(),
        head_mirror_binary_object_uri: "s3://bucket/runtime/burn_dragon_p2p_native".to_owned(),
        head_mirror_binary_sha256: "headmirrorsha".to_owned(),
        ..dummy_runtime_env()
    })?
    .join("\n");
    require_contains(
        &prebuilt,
        "expected bootstrapsha got $remote_sha",
        "bootstrap checksum check",
    )?;
    require_contains(
        &prebuilt,
        "expected headmirrorsha got $remote_sha",
        "head mirror checksum check",
    )?;
    require_absent(
        &prebuilt,
        "cargo install --locked burn_p2p_bootstrap",
        "prebuilt bootstrap skips cargo install",
    )?;
    Ok(())
}

fn bootstrap_head_preservation_contract() -> Result<()> {
    let mut config = bootstrap_config(None);
    let snapshot = json!({
        "directory": { "entries": [{
            "study_id": "study",
            "experiment_id": "exp",
            "current_revision_id": "rev",
            "current_head_id": "head-live"
        }] },
        "heads": []
    });
    let report = preserve_current_heads(&mut config, &snapshot, false);
    ensure!(
        report["preserved"] == json!(1),
        "expected one preserved head"
    );
    ensure!(
        config["auth"]["directory_entries"][0]["current_head_id"] == json!("head-live"),
        "live current head was not preserved"
    );

    let mut config = bootstrap_config(None);
    let snapshot = json!({
        "directory": { "entries": [] },
        "heads": [
            {
                "study_id": "study",
                "experiment_id": "exp",
                "revision_id": "rev",
                "parent_head_id": null,
                "head_id": "head-old",
                "created_at": "2026-01-01T00:00:00Z"
            },
            {
                "study_id": "study",
                "experiment_id": "exp",
                "revision_id": "rev",
                "parent_head_id": null,
                "head_id": "head-new",
                "created_at": "2026-01-02T00:00:00Z"
            }
        ]
    });
    let report = preserve_current_heads(&mut config, &snapshot, true);
    ensure!(
        report["recovered"] == json!(1),
        "expected one recovered root head"
    );
    ensure!(
        config["auth"]["directory_entries"][0]["current_head_id"] == json!("head-new"),
        "latest visible root was not recovered"
    );
    Ok(())
}

fn workflow_sibling_checkout_contract() -> Result<()> {
    for path in workflow_paths()? {
        let text = read(&path)?;
        if text.contains("repository: aberration-technology/burn_p2p") {
            let ref_snippet = format!("ref: {BURN_P2P_SIBLING_REF}");
            require_contains(
                &text,
                &ref_snippet,
                &format!("{path} pins the burn_p2p sibling revision"),
            )?;
            require_contains(
                &text,
                "path: burn_p2p-sibling",
                &format!("{path} checks burn_p2p out inside workspace"),
            )?;
            require_contains(
                &text,
                "ln -s \"${GITHUB_WORKSPACE}/burn_p2p-sibling\" \"$target\"",
                &format!("{path} links burn_p2p path dependency"),
            )?;
            require_absent(
                &text,
                "path: ../burn_p2p",
                &format!("{path} does not checkout outside workspace"),
            )?;
        }
    }
    Ok(())
}

fn deployment_workflow_contracts() -> Result<()> {
    for path in [
        ".github/workflows/deploy-burn-dragon-p2p-aws.yml",
        ".github/workflows/restore-burn-dragon-p2p-aws.yml",
    ] {
        let text = read(path)?;
        require_contains(
            &text,
            "cargo run -p xtask -- resolve-bootstrap-stack-settings",
            "bootstrap settings are resolved by xtask",
        )?;
        require_contains(
            &text,
            "target/debug/xtask sync-bootstrap-runtime-config",
            "bootstrap runtime sync is xtask owned",
        )?;
        require_contains(
            &text,
            "refresh aws credentials before runtime sync",
            "runtime sync refreshes AWS credentials after long builds",
        )?;
        require_contains(
            &text,
            "bootstrap_binary_path=\"${bootstrap_root}/bin/burn-p2p-bootstrap\"",
            "bootstrap binary path is reused",
        )?;
        require_contains(
            &text,
            "timed out waiting for bootstrap edge enable/start",
            "bootstrap edge startup wait is bounded",
        )?;
        require_contains(&text, "BOOTSTRAP_BINARY_SHA256", "bootstrap checksum env")?;
        require_contains(
            &text,
            "HEAD_MIRROR_BINARY_SHA256",
            "head mirror checksum env",
        )?;
        require_contains(&text, "EDGE_BASE_URL", "edge base url env")?;
        require_absent(
            &text,
            "git-based burn_p2p bootstrap deploys must replace the bootstrap host.",
            "git sync does not require replacement",
        )?;
    }

    let deploy = read(".github/workflows/deploy-burn-dragon-p2p-aws.yml")?;
    let restore = read(".github/workflows/restore-burn-dragon-p2p-aws.yml")?;
    require_contains(
        &deploy,
        "deferring strict head/artifact readiness to the live native canary",
        "forced bootstrap replacement does not require a pre-existing head before the native canary",
    )?;
    for text in [&deploy, &restore] {
        require_contains(
            text,
            "Name=tag:Name,Values=${STACK_NAME}-bootstrap",
            "latest bootstrap tag lookup",
        )?;
        require_contains(
            text,
            "instance_id=\"$(terraform -chdir=\"$TF_ROOT\" output -raw bootstrap_instance_id",
            "terraform bootstrap output fallback",
        )?;
        require_contains(
            text,
            "public_ip=\"$(terraform -chdir=\"$TF_ROOT\" output -raw bootstrap_public_ip",
            "terraform public ip fallback",
        )?;
    }

    let bootstrap_user_data =
        read("crates/burn_dragon_p2p/deploy/terraform/aws/templates/user-data.sh.tftpl")?;
    require_contains(
        &bootstrap_user_data,
        "fallocate -l 4G /swapfile",
        "bootstrap host provisions swap for low-memory edge stability",
    )?;
    require_contains(
        &bootstrap_user_data,
        "vm.swappiness = 20",
        "bootstrap swap swappiness is persisted",
    )?;
    Ok(())
}

fn browser_canary_contracts() -> Result<()> {
    let script = read("xtask/assets/live-browser-canary.mjs")?;
    for snippet in [
        "const EXPECT_TRAINING = parseBooleanEnv(\"BURN_DRAGON_BROWSER_CANARY_EXPECT_TRAINING\", true);",
        "\"BURN_DRAGON_BROWSER_CANARY_EXPECT_CHECKPOINT_SYNC\"",
        "expect_checkpoint_sync: EXPECT_CHECKPOINT_SYNC",
        "libp2p webrtc-direct: completed Noise handshake peer=",
        "function assertWebRtcDirectTransportPhases(report, consoleMessages)",
        "report.webrtc_direct_console_markers = webRtcDirectConsoleMarkerReport(consoleMessages);",
        "BURN_DRAGON_BROWSER_CANARY_DURABLE_RECEIPT_TIMEOUT_MS",
        "BURN_DRAGON_BROWSER_CANARY_MIN_ACCEPTED_RECEIPTS",
        "BURN_DRAGON_BROWSER_CANARY_USE_PRODUCTION_TRAINING_PROFILE",
        "use_production_training_profile: USE_PRODUCTION_TRAINING_PROFILE",
        "function assertBrowserE2eContract(report)",
        "async function loadBrowserConfig()",
        "path.join(SITE_OVERRIDE_DIR, \"browser-app-config.json\")",
        "report.artifact_http_fallback_requests = requests.filter((entry) => entry.artifactFallback);\n    assertBrowserE2eContract(report);",
    ] {
        require_contains(&script, snippet, "live browser canary script")?;
    }
    for forbidden in [
        "consumeCallbackToken",
        "/auth/browser/callback/consume",
        "training.model_config.language_head = { type: \"standard_token_classification\" };",
    ] {
        require_absent(
            &script,
            forbidden,
            "browser canary avoids stale auth or model-profile rewrites",
        )?;
    }

    let workflow = read(".github/workflows/live-browser-canary.yml")?;
    for snippet in [
        "workflow_call:",
        "cargo run -p xtask -- install-playwright-chromium",
        "cargo run -p xtask -- run-live-browser-canary",
        "cargo run -p xtask -- summarize-live-browser-canary \"$report_path\" >>\"$GITHUB_STEP_SUMMARY\"",
        "burn-dragon-live-browser-canary",
        "max-parallel: 2",
        "BURN_DRAGON_BROWSER_CANARY_CONNECT_TIMEOUT_MS: \"240000\"",
        "BURN_DRAGON_BROWSER_CANARY_TRAIN_TIMEOUT_MS: \"300000\"",
        "chromium-webrtc-direct-connect",
        "chromium-webrtc-direct-checkpoint",
        "chromium-webrtc-direct-training",
        "firefox-webrtc-direct-connect",
        "continue-on-error: ${{ matrix.required == '0' }}",
        "BURN_DRAGON_BROWSER_CANARY_BROWSER: ${{ matrix.browser }}",
        "BURN_DRAGON_BROWSER_CANARY_TRANSPORT_MODE: ${{ matrix.transport_mode }}",
        "BURN_DRAGON_BROWSER_CANARY_EXPECT_TRAINING: ${{ matrix.expect_training }}",
        "BURN_DRAGON_BROWSER_CANARY_EXPECT_CHECKPOINT_SYNC: ${{ matrix.expect_checkpoint_sync }}",
        "BURN_DRAGON_BROWSER_CANARY_MIN_ACCEPTED_RECEIPTS: ${{ matrix.min_accepted_receipts }}",
        "BURN_DRAGON_P2P_BROWSER_CANARY_CALLBACK_TOKEN:",
    ] {
        require_contains(&workflow, snippet, "live browser canary workflow")?;
    }
    require_absent(
        &workflow,
        concat!("bash ", "scripts", "/run_live_browser_canary.sh"),
        "browser canary workflow has no legacy script runner",
    )?;
    Ok(())
}

fn native_canary_contracts() -> Result<()> {
    let workflow = read(".github/workflows/live-native-training-canary.yml")?;
    for snippet in [
        "repository: aberration-technology/burn_p2p",
        "cargo build --locked -p burn_dragon_p2p --bin burn_dragon_p2p_native",
        "cargo build --locked -p xtask",
        "target/debug/xtask run-live-native-training-canary",
        "target/debug/xtask summarize-live-native-training-canary",
        "BURN_DRAGON_NATIVE_CANARY_CALLBACK_TOKEN: ${{ secrets.BURN_DRAGON_P2P_BROWSER_CANARY_CALLBACK_TOKEN }}",
        "BURN_DRAGON_NATIVE_CANARY_HEAD_SYNC_TIMEOUT_SECS: \"300\"",
        "BURN_DRAGON_NATIVE_CANARY_CANONICAL_TIMEOUT_SECS: ${{ github.event.inputs.canonical_timeout_secs || '480' }}",
        "BURN_DRAGON_NATIVE_CANARY_P2P_TIMEOUT_SECS: ${{ github.event.inputs.p2p_timeout_secs || '300' }}",
        "BURN_DRAGON_NATIVE_CANARY_COMMAND_TIMEOUT_SECS: ${{ github.event.inputs.command_timeout_secs || '1800' }}",
        "BURN_DRAGON_NATIVE_CANARY_START_VALIDATOR: ${{ github.event.inputs.start_validator || 'true' }}",
        "BURN_DRAGON_NATIVE_CANARY_HTTP_ATTEMPTS: ${{ github.event.inputs.http_attempts || '15' }}",
        "BURN_DRAGON_NATIVE_CANARY_MIRROR_LIVE_HEAD_TO_EDGE: ${{ github.event.inputs.mirror_live_head_to_edge || 'false' }}",
        "BURN_DRAGON_NATIVE_CANARY_REQUIRE_EDGE_HEAD_PROVIDER: ${{ github.event.inputs.require_edge_head_provider || 'true' }}",
        "BURN_DRAGON_NATIVE_CANARY_REPAIR_CURRENT_HEAD_TO_VISIBLE_ROOT: ${{ github.event.inputs.repair_current_head_to_visible_root || 'false' }}",
    ] {
        require_contains(&workflow, snippet, "live native canary workflow")?;
    }

    let native = read("xtask/src/native_canary.rs")?;
    for snippet in [
        "--mirror-live-head-to-edge",
        "--reset-current-head-to-visible-root",
        "--require-head-advanced",
        "fn p2p_bootstrap_addresses",
        "BURN_DRAGON_NATIVE_CANARY_P2P_BOOTSTRAP_ADDRS",
        "BURN_DRAGON_NATIVE_CANARY_HTTP_ATTEMPTS",
        "p2p bootstrap snapshot did not advertise canonical head",
        "fn assert_head_provider_signal",
        "require_edge_provider: bool",
        "BURN_DRAGON_NATIVE_CANARY_MIRROR_LIVE_HEAD_TO_EDGE",
        "BURN_DRAGON_NATIVE_CANARY_REQUIRE_EDGE_HEAD_PROVIDER",
        "BURN_DRAGON_NATIVE_CANARY_REPAIR_CURRENT_HEAD_TO_VISIBLE_ROOT",
        "require_canonical_loss_non_regression",
        "BURN_DRAGON_P2P_NATIVE_STORAGE_ROOT",
        "BURN_DRAGON_NATIVE_CANARY_VALIDATOR_PRINCIPAL_ID",
        "BURN_DRAGON_NATIVE_CANARY_SETTLE_DIFFUSION",
        "BURN_DRAGON_NATIVE_CANARY_DIFFUSION_SETTLE_PASSES",
        "BURN_DRAGON_NATIVE_CANARY_SERVE_AFTER_PUBLISH_SECS",
        "BURN_DRAGON_NATIVE_CANARY_START_VALIDATOR",
        "BURN_DRAGON_NATIVE_CANARY_TRAINING_BATCH_SIZE",
        "BURN_DRAGON_NATIVE_CANARY_TRAINING_MAX_ITERS",
        "BURN_DRAGON_NATIVE_CANARY_EVALUATION_MAX_BATCHES",
    ] {
        require_contains(&native, snippet, "native canary xtask implementation")?;
    }
    require_absent(
        &native,
        "\"--require-head-advanced\",\n                    \"true\",",
        "require-head-advanced remains a presence flag",
    )?;
    Ok(())
}

fn browser_and_native_transport_contracts() -> Result<()> {
    let tf_root = "crates/burn_dragon_p2p/deploy/terraform/aws";
    let main_tf = read(format!("{tf_root}/main.tf"))?;
    let outputs_tf = read(format!("{tf_root}/outputs.tf"))?;
    for snippet in [
        "p2p_webrtc_port              = 443",
        "\"/ip4/0.0.0.0/udp/${local.p2p_webrtc_port}/webrtc-direct\"",
        "\"/ip4/PUBLIC_IP/udp/${local.p2p_webrtc_port}/webrtc-direct\"",
        "from_port   = local.p2p_webrtc_port",
        "trainer_webrtc_port                = local.p2p_webrtc_port",
        "validator_webrtc_port                = local.p2p_webrtc_port",
        "bootstrap_head_mirror_seed_node_urls         = local.bootstrap_peer_internal_multiaddrs",
    ] {
        require_contains(&main_tf, snippet, "terraform browser/native transport")?;
    }
    require_absent(
        &main_tf,
        "bootstrap_head_mirror_seed_node_urls         = local.managed_trainer_seed_node_urls",
        "head mirror avoids public trainer seeds",
    )?;
    require_contains(
        &outputs_tf,
        "Deprecated synthetic WebRTC hint. Use the signed browser seed advertisement endpoint instead because dialable browser WebRTC addresses require a runtime certhash.",
        "deprecated synthetic WebRTC output is documented",
    )?;

    let trainer = read(format!("{tf_root}/templates/trainer-user-data.sh.tftpl"))?;
    let validator = read(format!("{tf_root}/templates/validator-user-data.sh.tftpl"))?;
    require_contains(
        &trainer,
        "\"/ip4/0.0.0.0/udp/${trainer_webrtc_port}/webrtc-direct\"",
        "trainer WebRTC listener",
    )?;
    require_contains(
        &trainer,
        "\"/ip4/$${PUBLIC_IPV4}/udp/${trainer_webrtc_port}/webrtc-direct\"",
        "trainer WebRTC external addr",
    )?;
    require_contains(
        &validator,
        "\"/ip4/0.0.0.0/udp/${validator_webrtc_port}/webrtc-direct\"",
        "validator WebRTC listener",
    )?;
    require_contains(
        &validator,
        "\"/ip4/$${PUBLIC_IPV4}/udp/${validator_webrtc_port}/webrtc-direct\"",
        "validator WebRTC external addr",
    )?;
    let native_example = read("crates/burn_dragon_p2p/deploy/native-peer.toml.example")?;
    require_contains(
        &native_example,
        "\"/ip4/PUBLIC_IP/udp/443/webrtc-direct\"",
        "native peer example WebRTC addr",
    )?;
    Ok(())
}

fn production_profile_contracts() -> Result<()> {
    let variables = read("crates/burn_dragon_p2p/deploy/terraform/aws/variables.tf")?;
    for (variable, expected) in [
        ("bootstrap_install_source", "crate"),
        ("use_retained_bootstrap_data_volume", "false"),
        ("enable_managed_control_plane_redis", "false"),
    ] {
        let actual = terraform_default(&variables, variable)?;
        ensure!(
            actual == expected,
            "Terraform default {variable} = {actual:?}, expected {expected:?}"
        );
    }
    let main_tf = read("crates/burn_dragon_p2p/deploy/terraform/aws/main.tf")?;
    for snippet in [
        "bootstrap_state_storage_mode        = local.use_retained_bootstrap_data_volume ? \"retained-ebs-volume\" : \"root-volume\"",
        "head_artifact_mirror_source_roots = [\n        local.bootstrap_head_mirror_storage_root,\n      ]",
        "preset = \"BootstrapOnly\"",
        "bootstrap_addresses = local.bootstrap_peer_internal_multiaddrs",
        "low_resource_bootstrap_transport_policy = {",
        "target_connected_peers            = 8",
        "max_established_total             = 40",
        "max_relay_circuits                = 16",
        "transport_policy = local.low_resource_bootstrap_transport_policy",
        "bootstrap_peers  = local.bootstrap_peer_internal_multiaddrs",
        "\"/ip4/0.0.0.0/tcp/${var.p2p_port}\"",
        "\"/ip4/0.0.0.0/udp/${local.p2p_webrtc_port}/webrtc-direct\"",
        "authority = null",
        "identity = \"Persistent\"",
        "\"/ip4/PUBLIC_IP/udp/${local.p2p_webrtc_port}/webrtc-direct\"",
        "limit_nofile       = 262144",
    ] {
        require_contains(&main_tf, snippet, "production low-resource p2p config")?;
    }
    let runtime = read("xtask/src/bootstrap_runtime.rs")?;
    require_contains(
        &runtime,
        "admin-http,metrics,metrics-indexer,artifact-publish,artifact-download,artifact-fs,artifact-s3,browser-edge,browser-join,{bootstrap_auth_feature},rbac,social",
        "prod bootstrap burn_p2p feature set",
    )?;
    let service = read(
        "crates/burn_dragon_p2p/deploy/terraform/aws/templates/burn-p2p-bootstrap.service.tftpl",
    )?;
    require_contains(
        &service,
        "LimitNOFILE=${limit_nofile}",
        "bootstrap fd limit",
    )?;
    require_contains(&service, "TimeoutStartSec=90", "bounded bootstrap startup")?;
    let deploy_workflow = read(".github/workflows/deploy-burn-dragon-p2p-aws.yml")?;
    require_contains(
        &deploy_workflow,
        "verify bootstrap p2p handshakes",
        "deploy verifies native and browser-direct p2p handshakes before canaries",
    )?;
    require_contains(
        &deploy_workflow,
        "BURN_DRAGON_NATIVE_CANARY_REPAIR_CURRENT_HEAD_TO_VISIBLE_ROOT: \"true\"",
        "deploy canary repairs stale canonical heads before proving fresh p2p training",
    )?;
    require_contains(
        &deploy_workflow,
        "--address \"$browser_seed\"",
        "deploy probes the signed browser-direct seed",
    )?;

    let profile = read("crates/burn_dragon_p2p/deploy/profiles/nca-r1.profile.json")?;
    let profile: Value = serde_json::from_str(&profile)?;
    let browser = profile
        .get("browser_training")
        .or_else(|| profile.get("browser"))
        .unwrap_or(&Value::Null);
    ensure!(
        browser["batch_size"] == json!(1),
        "browser batch size drifted"
    );
    ensure!(
        browser["max_train_batches"] == json!(8),
        "browser max train batches drifted"
    );
    ensure!(
        browser["max_eval_batches"].as_u64().unwrap_or(u64::MAX) <= 1,
        "browser max eval batches should stay cheap"
    );
    Ok(())
}

fn browser_site_contracts() -> Result<()> {
    let source = read("xtask/src/browser_site.rs")?;
    for snippet in [
        "DragonBrowserTrainingConfig",
        "browser_training_config_from_directory_entries",
        "resolve_browser_training_config",
        "training: Option<DragonBrowserTrainingConfig>",
    ] {
        require_contains(&source, snippet, "browser site config")?;
    }
    let xtask = read("xtask/src/main.rs")?;
    for snippet in [
        "fn local_browser_e2e() -> Result<()>",
        "Self::Wgpu => \"wasm-ui,wasm-peer,wgpu\"",
        "wasm_training_smoke()",
    ] {
        require_contains(&xtask, snippet, "local browser e2e plan")?;
    }
    Ok(())
}

fn cleanup_workflow_contracts() -> Result<()> {
    let text = read(".github/workflows/cleanup-burn-dragon-p2p-aws.yml")?;
    for snippet in [
        "cleanup_route53_health_checks",
        "BURN_DRAGON_P2P_AWS_CLEANUP_ROLE_ARN",
        "BURN_DRAGON_P2P_AWS_ROLE_ARN",
        "broader cleanup still requires the dedicated cleanup role",
        "cleanup_route53_health_checks()",
        "canonical_health_check_name=\"${canonical_stack_name}-edge-primary\"",
        "route53_health_check_inventory",
        "aws route53 delete-health-check --health-check-id \"$health_check_id\"",
    ] {
        require_contains(&text, snippet, "cleanup workflow")?;
    }
    for forbidden in ["aws route53 delete-hosted-zone", "aws s3 rb --force"] {
        require_absent(&text, forbidden, "cleanup workflow avoids broad deletion")?;
    }
    Ok(())
}

fn agent_task_contracts() -> Result<()> {
    let source = read("xtask/src/agent_task.rs")?;
    for snippet in [
        "workflow: \".github/workflows/deploy-pages.yml\"",
        "workflow: \".github/workflows/live-native-training-canary.yml\"",
        "input_env(\"environment\", \"BURN_DRAGON_DEPLOY_PAGES_ENVIRONMENT\")",
        "input_env(\"edge_base_url\", \"BURN_DRAGON_NATIVE_CANARY_EDGE_BASE_URL\")",
        "\"mirror_live_head_to_edge\"",
        "\"repair_current_head_to_visible_root\"",
        "env_u64(\"BURN_DRAGON_NATIVE_CANARY_WATCH_INTERVAL_SECS\", 60)",
    ] {
        require_contains(&source, snippet, "agent task dispatch helpers")?;
    }
    Ok(())
}

fn dummy_runtime_env() -> RuntimeCommandEnv {
    RuntimeCommandEnv {
        bootstrap_object_uri: "s3://bucket/runtime/bootstrap.json".to_owned(),
        caddy_object_uri: "s3://bucket/runtime/Caddyfile".to_owned(),
        bootstrap_service_unit_object_uri: "s3://bucket/runtime/burn-p2p-bootstrap.service"
            .to_owned(),
        head_mirror_config_object_uri: "s3://bucket/runtime/bootstrap-head-mirror.toml"
            .to_owned(),
        head_mirror_auth_script_object_uri:
            "s3://bucket/runtime/burn-dragon-p2p-fetch-head-mirror-auth-bundle".to_owned(),
        head_mirror_service_object_uri:
            "s3://bucket/runtime/burn-dragon-p2p-head-mirror.service".to_owned(),
        bootstrap_install_source: "crate".to_owned(),
        bootstrap_crate_version: "0.21.6".to_owned(),
        bootstrap_git_repository: "https://github.com/aberration-technology/burn_p2p.git"
            .to_owned(),
        bootstrap_git_ref: String::new(),
        bootstrap_binary_object_uri: String::new(),
        bootstrap_binary_sha256: String::new(),
        bootstrap_features: "admin-http,metrics,metrics-indexer,artifact-publish,artifact-download,artifact-fs,artifact-s3,browser-edge,browser-join,auth-github,rbac,social".to_owned(),
        bootstrap_reinstall: "false".to_owned(),
        dragon_git_repository: "https://github.com/aberration-technology/burn_dragon.git"
            .to_owned(),
        dragon_git_ref: "main".to_owned(),
        head_mirror_binary_object_uri: String::new(),
        head_mirror_binary_sha256: String::new(),
        head_mirror_reinstall: "true".to_owned(),
    }
}

fn bootstrap_config(current_head_id: Option<&str>) -> Value {
    let mut entry = json!({
        "study_id": "study",
        "experiment_id": "exp",
        "revision_id": "rev",
    });
    if let Some(head_id) = current_head_id {
        entry["current_head_id"] = json!(head_id);
    }
    json!({
        "auth": {
            "directory_entries": [entry]
        }
    })
}

fn terraform_default(text: &str, variable_name: &str) -> Result<String> {
    let marker = format!("variable \"{variable_name}\"");
    let start = text
        .find(&marker)
        .with_context(|| format!("missing Terraform variable {variable_name}"))?;
    let rest = &text[start..];
    let default = rest
        .lines()
        .map(str::trim)
        .find_map(|line| {
            line.strip_prefix("default")
                .and_then(|value| value.split_once('=').map(|(_, raw)| raw.trim()))
        })
        .with_context(|| format!("missing Terraform default for {variable_name}"))?;
    Ok(default.trim_matches('"').to_owned())
}

fn workflow_paths() -> Result<Vec<String>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(".github/workflows")? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) == Some("yml") {
            paths.push(path.display().to_string());
        }
    }
    Ok(paths)
}

fn walk(path: impl AsRef<Path>) -> Result<Vec<std::path::PathBuf>> {
    let mut paths = Vec::new();
    let path = path.as_ref();
    if path.is_dir() {
        paths.push(path.to_path_buf());
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let child = entry.path();
            let name = child
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default();
            if matches!(name, ".git" | "target") {
                continue;
            }
            paths.extend(walk(child)?);
        }
    }
    Ok(paths)
}

fn read(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))
}

fn require_contains(text: &str, snippet: &str, label: &str) -> Result<()> {
    ensure!(
        text.contains(snippet),
        "{label} missing required snippet: {snippet}"
    );
    Ok(())
}

fn require_absent(text: &str, snippet: &str, label: &str) -> Result<()> {
    ensure!(
        !text.contains(snippet),
        "{label} contains forbidden snippet: {snippet}"
    );
    Ok(())
}
