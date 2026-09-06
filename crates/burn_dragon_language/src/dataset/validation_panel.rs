use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{RuliadValidationPanelMode, TrainingHyperparameters};

use super::{Dataset, RuliadValidationProbeItem, RuliadValidationPromptMode};

const PANEL_SCHEMA_VERSION: u32 = 4;
const PANEL_LOCK_WAIT: Duration = Duration::from_secs(60);

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct RuliadValidationPanelRequest {
    corpus_semantic_fingerprint: Option<String>,
    seed: u64,
    base_items: usize,
    base_difficulty_levels: usize,
    include_training_serialization: bool,
    policy_enabled: bool,
    policy_items: usize,
    policy_task_kind: String,
    policy_difficulty_levels: usize,
    tokenizer_vocab_size: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct RuliadValidationPanelPayload {
    request: RuliadValidationPanelRequest,
    base_items: Vec<RuliadValidationProbeItem>,
    training_serialization_items: Vec<RuliadValidationProbeItem>,
    policy_items: Vec<RuliadValidationProbeItem>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct RuliadValidationPanelManifest {
    schema_version: u32,
    fingerprint_sha256: String,
    payload: RuliadValidationPanelPayload,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedRuliadValidationPanel {
    pub base_items: Vec<RuliadValidationProbeItem>,
    pub training_serialization_items: Vec<RuliadValidationProbeItem>,
    pub policy_items: Vec<RuliadValidationProbeItem>,
    pub fingerprint_sha256: Option<String>,
}

fn panel_request(
    training: &TrainingHyperparameters,
    dataset: &Dataset,
) -> Result<RuliadValidationPanelRequest> {
    Ok(RuliadValidationPanelRequest {
        corpus_semantic_fingerprint: dataset.ruliad_semantic_fingerprint()?,
        seed: training.validation.seed,
        base_items: training.events.ruliad_correctness_probe_items,
        base_difficulty_levels: training.validation.ruliad_panel.base_difficulty_levels,
        include_training_serialization: training.sequence_state_probe.enabled,
        policy_enabled: training.ruliad_policy_probe.enabled,
        policy_items: training.ruliad_policy_probe.items,
        policy_task_kind: burn_dragon_universality::RuliadTaskKind::SelectProofAction
            .label()
            .to_string(),
        policy_difficulty_levels: training.ruliad_policy_probe.stratified_difficulty_levels,
        tokenizer_vocab_size: dataset.tokenizer().len(),
    })
}

fn dynamic_panel(
    dataset: &Dataset,
    training: &TrainingHyperparameters,
    epoch: usize,
    absolute_step: usize,
) -> Result<ResolvedRuliadValidationPanel> {
    let base_items = if training.validation.ruliad_panel.base_difficulty_levels == 0 {
        dataset.sample_ruliad_validation_probe_items(
            epoch,
            absolute_step,
            training.events.ruliad_correctness_probe_items,
        )
    } else {
        dataset.sample_ruliad_validation_probe_items_stratified_fixed(
            training.validation.seed,
            training.events.ruliad_correctness_probe_items,
            training.validation.ruliad_panel.base_difficulty_levels,
            RuliadValidationPromptMode::CanonicalTransfer,
        )
    };
    let training_serialization_items = if training.sequence_state_probe.enabled {
        if training.validation.ruliad_panel.base_difficulty_levels == 0 {
            dataset.sample_ruliad_training_serialization_probe_items(
                epoch,
                absolute_step,
                training.events.ruliad_correctness_probe_items,
            )
        } else {
            dataset.sample_ruliad_validation_probe_items_stratified_fixed(
                training.validation.seed,
                training.events.ruliad_correctness_probe_items,
                training.validation.ruliad_panel.base_difficulty_levels,
                RuliadValidationPromptMode::TrainingSerialization,
            )
        }
    } else {
        Vec::new()
    };
    let policy_items = if !training.ruliad_policy_probe.enabled {
        Vec::new()
    } else if training.ruliad_policy_probe.stratified_difficulty_levels > 0 {
        dataset.sample_ruliad_task_probe_items_fixed(
            training.validation.seed,
            training.ruliad_policy_probe.items,
            burn_dragon_universality::RuliadTaskKind::SelectProofAction.label(),
            training.ruliad_policy_probe.stratified_difficulty_levels,
        )
    } else {
        base_items
            .iter()
            .take(training.ruliad_policy_probe.items)
            .cloned()
            .collect()
    };
    validate_difficulty_strata(
        "base",
        &base_items,
        training.validation.ruliad_panel.base_difficulty_levels,
        training.events.ruliad_correctness_probe_items,
    )?;
    if training.ruliad_policy_probe.enabled {
        validate_difficulty_strata(
            "policy",
            &policy_items,
            training.ruliad_policy_probe.stratified_difficulty_levels,
            training.ruliad_policy_probe.items,
        )?;
    }
    Ok(ResolvedRuliadValidationPanel {
        base_items,
        training_serialization_items,
        policy_items,
        fingerprint_sha256: None,
    })
}

fn materialize_payload(
    dataset: &Dataset,
    request: RuliadValidationPanelRequest,
) -> Result<RuliadValidationPanelPayload> {
    let base_items = if request.base_difficulty_levels == 0 {
        dataset.sample_ruliad_validation_probe_items_fixed(
            request.seed,
            request.base_items,
            RuliadValidationPromptMode::CanonicalTransfer,
        )
    } else {
        dataset.sample_ruliad_validation_probe_items_stratified_fixed(
            request.seed,
            request.base_items,
            request.base_difficulty_levels,
            RuliadValidationPromptMode::CanonicalTransfer,
        )
    };
    let training_serialization_items = if request.include_training_serialization {
        if request.base_difficulty_levels == 0 {
            dataset.sample_ruliad_validation_probe_items_fixed(
                request.seed,
                request.base_items,
                RuliadValidationPromptMode::TrainingSerialization,
            )
        } else {
            dataset.sample_ruliad_validation_probe_items_stratified_fixed(
                request.seed,
                request.base_items,
                request.base_difficulty_levels,
                RuliadValidationPromptMode::TrainingSerialization,
            )
        }
    } else {
        Vec::new()
    };
    let policy_items = if !request.policy_enabled {
        Vec::new()
    } else if request.policy_difficulty_levels > 0 {
        dataset.sample_ruliad_task_probe_items_fixed(
            request.seed,
            request.policy_items,
            &request.policy_task_kind,
            request.policy_difficulty_levels,
        )
    } else {
        base_items
            .iter()
            .take(request.policy_items)
            .cloned()
            .collect()
    };
    if base_items.len() != request.base_items {
        bail!(
            "Ruliad dataset produced {} fixed panel items, expected {}",
            base_items.len(),
            request.base_items
        );
    }
    if request.include_training_serialization
        && training_serialization_items.len() != request.base_items
    {
        bail!(
            "Ruliad dataset produced {} fixed training-serialization items, expected {}",
            training_serialization_items.len(),
            request.base_items
        );
    }
    let expected_policy_items = if request.policy_difficulty_levels > 0 {
        request.policy_items
    } else {
        request.policy_items.min(base_items.len())
    };
    if request.policy_enabled && policy_items.len() != expected_policy_items {
        bail!(
            "Ruliad dataset produced {} fixed policy items, expected {}",
            policy_items.len(),
            expected_policy_items
        );
    }
    validate_difficulty_strata(
        "base",
        &base_items,
        request.base_difficulty_levels,
        request.base_items,
    )?;
    if request.policy_enabled {
        validate_difficulty_strata(
            "policy",
            &policy_items,
            request.policy_difficulty_levels,
            expected_policy_items,
        )?;
    }
    Ok(RuliadValidationPanelPayload {
        request,
        base_items,
        training_serialization_items,
        policy_items,
    })
}

fn validate_difficulty_strata(
    panel: &str,
    items: &[RuliadValidationProbeItem],
    requested_levels: usize,
    requested_items: usize,
) -> Result<()> {
    if requested_levels == 0 {
        return Ok(());
    }
    let expected_levels = requested_levels.min(requested_items);
    let observed = items
        .iter()
        .filter_map(|item| item.item.difficulty_level)
        .collect::<BTreeSet<_>>();
    if observed.len() != expected_levels {
        bail!(
            "Ruliad {panel} panel materialized {} difficulty strata, expected {}: {:?}",
            observed.len(),
            expected_levels,
            observed,
        );
    }
    Ok(())
}

fn payload_fingerprint(payload: &RuliadValidationPanelPayload) -> Result<String> {
    let bytes = serde_json::to_vec(payload).context("serialize Ruliad validation panel payload")?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn validate_manifest(
    manifest: RuliadValidationPanelManifest,
    expected_request: &RuliadValidationPanelRequest,
) -> Result<ResolvedRuliadValidationPanel> {
    if manifest.schema_version != PANEL_SCHEMA_VERSION {
        bail!(
            "unsupported Ruliad validation panel schema {} (expected {})",
            manifest.schema_version,
            PANEL_SCHEMA_VERSION
        );
    }
    if &manifest.payload.request != expected_request {
        bail!(
            "Ruliad validation panel request mismatch: stored={:?} requested={:?}",
            manifest.payload.request,
            expected_request
        );
    }
    let fingerprint = payload_fingerprint(&manifest.payload)?;
    if fingerprint != manifest.fingerprint_sha256 {
        bail!(
            "Ruliad validation panel fingerprint mismatch: stored={} computed={fingerprint}",
            manifest.fingerprint_sha256
        );
    }
    if manifest.payload.base_items.len() != expected_request.base_items {
        bail!(
            "Ruliad validation panel contains {} base items, expected {}",
            manifest.payload.base_items.len(),
            expected_request.base_items
        );
    }
    let RuliadValidationPanelPayload {
        base_items,
        training_serialization_items,
        policy_items,
        ..
    } = manifest.payload;
    validate_difficulty_strata(
        "base",
        &base_items,
        expected_request.base_difficulty_levels,
        expected_request.base_items,
    )?;
    if expected_request.policy_enabled {
        validate_difficulty_strata(
            "policy",
            &policy_items,
            expected_request.policy_difficulty_levels,
            expected_request.policy_items,
        )?;
    }
    Ok(ResolvedRuliadValidationPanel {
        base_items,
        training_serialization_items,
        policy_items,
        fingerprint_sha256: Some(fingerprint),
    })
}

fn load_manifest(
    path: &Path,
    expected_request: &RuliadValidationPanelRequest,
) -> Result<ResolvedRuliadValidationPanel> {
    let bytes = fs::read(path)
        .with_context(|| format!("read Ruliad validation panel {}", path.display()))?;
    let manifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse Ruliad validation panel {}", path.display()))?;
    validate_manifest(manifest, expected_request)
}

fn publish_manifest(path: &Path, payload: RuliadValidationPanelPayload) -> Result<()> {
    let manifest = RuliadValidationPanelManifest {
        schema_version: PANEL_SCHEMA_VERSION,
        fingerprint_sha256: payload_fingerprint(&payload)?,
        payload,
    };
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create Ruliad panel directory {}", parent.display()))?;
    let temporary = sibling_path(path, &format!("{}.tmp", std::process::id()));
    let bytes = serde_json::to_vec_pretty(&manifest).context("serialize Ruliad panel manifest")?;
    fs::write(&temporary, bytes)
        .with_context(|| format!("write temporary Ruliad panel {}", temporary.display()))?;
    fs::rename(&temporary, path).with_context(|| format!("publish Ruliad panel {}", path.display()))
}

fn sibling_path(path: &Path, suffix: &str) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("ruliad-panel.json");
    path.with_file_name(format!(".{file_name}.{suffix}"))
}

fn create_or_reuse(
    path: &Path,
    dataset: &Dataset,
    request: &RuliadValidationPanelRequest,
) -> Result<ResolvedRuliadValidationPanel> {
    if path.is_file() {
        return load_manifest(path, request);
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create Ruliad panel directory {}", parent.display()))?;
    let lock = sibling_path(path, "lock");
    match OpenOptions::new().write(true).create_new(true).open(&lock) {
        Ok(_) => {
            let result = (|| {
                if path.is_file() {
                    return load_manifest(path, request);
                }
                publish_manifest(path, materialize_payload(dataset, request.clone())?)?;
                load_manifest(path, request)
            })();
            let _ = fs::remove_file(&lock);
            result
        }
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {
            let started = Instant::now();
            while started.elapsed() < PANEL_LOCK_WAIT {
                if path.is_file() {
                    return load_manifest(path, request);
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(anyhow!(
                "timed out waiting for Ruliad validation panel publisher at {}",
                path.display()
            ))
        }
        Err(error) => {
            Err(error).with_context(|| format!("acquire Ruliad panel lock {}", lock.display()))
        }
    }
}

pub(crate) fn resolve_ruliad_validation_panel(
    dataset: &Dataset,
    training: &TrainingHyperparameters,
    epoch: usize,
    absolute_step: usize,
) -> Result<ResolvedRuliadValidationPanel> {
    let config = &training.validation.ruliad_panel;
    if matches!(config.mode, RuliadValidationPanelMode::Dynamic) {
        return dynamic_panel(dataset, training, epoch, absolute_step);
    }
    let path = config
        .path
        .as_deref()
        .ok_or_else(|| anyhow!("persisted Ruliad validation panel requires a path"))?;
    let request = panel_request(training, dataset)?;
    match config.mode {
        RuliadValidationPanelMode::Dynamic => unreachable!(),
        RuliadValidationPanelMode::CreateOrReuse => create_or_reuse(path, dataset, &request),
        RuliadValidationPanelMode::RequireExisting => load_manifest(path, &request),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_rejects_tampering_and_request_drift() {
        let request = RuliadValidationPanelRequest {
            corpus_semantic_fingerprint: Some("corpus-v1".into()),
            seed: 7,
            base_items: 0,
            base_difficulty_levels: 4,
            include_training_serialization: false,
            policy_enabled: false,
            policy_items: 0,
            policy_task_kind: "select_proof_action".to_string(),
            policy_difficulty_levels: 0,
            tokenizer_vocab_size: 272,
        };
        let payload = RuliadValidationPanelPayload {
            request: request.clone(),
            base_items: Vec::new(),
            training_serialization_items: Vec::new(),
            policy_items: Vec::new(),
        };
        let fingerprint_sha256 = payload_fingerprint(&payload).expect("fingerprint");
        validate_manifest(
            RuliadValidationPanelManifest {
                schema_version: PANEL_SCHEMA_VERSION,
                fingerprint_sha256,
                payload: payload.clone(),
            },
            &request,
        )
        .expect("valid manifest");

        let mut drifted = request.clone();
        drifted.seed += 1;
        assert!(
            validate_manifest(
                RuliadValidationPanelManifest {
                    schema_version: PANEL_SCHEMA_VERSION,
                    fingerprint_sha256: payload_fingerprint(&payload).expect("fingerprint"),
                    payload: payload.clone(),
                },
                &drifted,
            )
            .is_err()
        );
        let mut changed_corpus = request.clone();
        changed_corpus.corpus_semantic_fingerprint = Some("corpus-v2".into());
        assert!(
            validate_manifest(
                RuliadValidationPanelManifest {
                    schema_version: PANEL_SCHEMA_VERSION,
                    fingerprint_sha256: payload_fingerprint(&payload).unwrap(),
                    payload: payload.clone(),
                },
                &changed_corpus,
            )
            .is_err()
        );
        assert!(
            validate_manifest(
                RuliadValidationPanelManifest {
                    schema_version: PANEL_SCHEMA_VERSION,
                    fingerprint_sha256: "tampered".to_string(),
                    payload,
                },
                &request,
            )
            .is_err()
        );
    }
}
