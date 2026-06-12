use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::manifest::{CorpusKind, UniversalitySampleRecord, load_manifest};
use crate::ruliad::config::LeanMode;
use crate::ruliad::oracles::{RuliadSampleSpec, verify_spec};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadVerificationReport {
    pub count: usize,
    pub ok_count: usize,
    pub failed: Vec<String>,
    pub lean_mode: LeanMode,
    pub lean_checked: bool,
    pub lean_ok: bool,
}

pub fn verify_sample(spec: &RuliadSampleSpec) -> Result<bool> {
    Ok(verify_spec(spec)?.ok)
}

pub fn verify_manifest(
    manifest_path: &Path,
    lean_mode: LeanMode,
    lean_project: Option<&Path>,
) -> Result<RuliadVerificationReport> {
    let manifest = load_manifest(manifest_path)?;
    if manifest.corpus_kind != CorpusKind::Ruliad {
        return Err(anyhow!(
            "manifest {} is {:?}, not ruliad",
            manifest_path.display(),
            manifest.corpus_kind
        ));
    }
    let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let sample_records_path = manifest_dir.join(&manifest.sample_records_path);
    let records = read_sample_records(&sample_records_path)?;
    let mut failed = Vec::new();
    let mut lean_task_seen = false;
    for record in &records {
        let Some(spec_value) = &record.ruliad_spec else {
            failed.push(format!(
                "sample {} missing ruliad_spec",
                record.sample_index
            ));
            continue;
        };
        let spec: RuliadSampleSpec = serde_json::from_value(spec_value.clone())
            .with_context(|| format!("parse sample {} ruliad spec", record.sample_index))?;
        if matches!(spec, RuliadSampleSpec::LeanTask { .. }) {
            lean_task_seen = true;
        }
        let report = verify_spec(&spec)?;
        if !report.ok {
            failed.push(format!(
                "sample {} failed rust verifier ({})",
                record.sample_index, report.oracle_hash
            ));
        }
        if let Some(expected) = &record.oracle_hash
            && *expected != report.oracle_hash
        {
            failed.push(format!(
                "sample {} oracle hash mismatch expected={} actual={}",
                record.sample_index, expected, report.oracle_hash
            ));
        }
    }

    let (lean_checked, lean_ok) = match lean_mode {
        LeanMode::Off => (false, true),
        LeanMode::Optional if !lean_task_seen => (false, true),
        LeanMode::Optional => match verify_lean_project(lean_project) {
            Ok(()) => (true, true),
            Err(_) => (true, false),
        },
        LeanMode::Required if !lean_task_seen => (false, true),
        LeanMode::Required => match verify_lean_project(lean_project) {
            Ok(()) => (true, true),
            Err(error) => {
                failed.push(format!("lean verification failed: {error:#}"));
                (true, false)
            }
        },
    };

    Ok(RuliadVerificationReport {
        count: records.len(),
        ok_count: records.len().saturating_sub(failed.len()),
        failed,
        lean_mode,
        lean_checked,
        lean_ok,
    })
}

fn read_sample_records(path: &Path) -> Result<Vec<UniversalitySampleRecord>> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    contents
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let line = line.trim();
            (!line.is_empty()).then_some((index, line))
        })
        .map(|(index, line)| {
            serde_json::from_str::<UniversalitySampleRecord>(line)
                .with_context(|| format!("failed to parse sample record line {}", index + 1))
        })
        .collect()
}

fn verify_lean_project(project: Option<&Path>) -> Result<()> {
    let project = project
        .map(Path::to_path_buf)
        .unwrap_or_else(default_lean_project_path);
    run_lake_build(&project)
}

fn default_lean_project_path() -> PathBuf {
    PathBuf::from("crates/burn_dragon_universality/lean/ruliad_seed")
}

#[cfg(not(target_arch = "wasm32"))]
fn run_lake_build(project: &Path) -> Result<()> {
    let status = std::process::Command::new(lake_binary())
        .arg("build")
        .current_dir(project)
        .status()
        .with_context(|| format!("failed to launch lake in {}", project.display()))?;
    if status.success() {
        Ok(())
    } else {
        Err(anyhow!(
            "lake build failed in {} with status {}",
            project.display(),
            status
        ))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn lake_binary() -> PathBuf {
    if let Some(path) = std::env::var_os("LAKE").filter(|value| !value.is_empty()) {
        return PathBuf::from(path);
    }
    if let Some(path) = find_on_path("lake") {
        return path;
    }
    if let Some(home) = std::env::var_os("HOME") {
        let candidate = PathBuf::from(home).join(".elan/bin/lake");
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from("lake")
}

#[cfg(not(target_arch = "wasm32"))]
fn find_on_path(binary: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|dir| dir.join(binary))
            .find(|candidate| candidate.is_file())
    })
}

#[cfg(target_arch = "wasm32")]
fn run_lake_build(project: &Path) -> Result<()> {
    Err(anyhow!(
        "lake build is unavailable for wasm target ({})",
        project.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ruliad::config::{
        RuliadCorpusConfig, RuliadFamilyConfig, RuliadFamilyKind, RuliadSerializationConfig,
        RuliadTokenizationConfig,
    };
    use crate::ruliad::generate::generate_ruliad_corpus;
    use tempfile::tempdir;

    #[test]
    fn required_lean_fails_cleanly_when_lake_is_absent_or_project_invalid() {
        let dir = tempdir().expect("tempdir");
        let config = RuliadCorpusConfig {
            output_dir: dir.path().join("out"),
            seed: 4,
            name: "verify-test".to_string(),
            train_samples: 2,
            validation_samples: 1,
            chunk_token_capacity: 512,
            serialization: RuliadSerializationConfig {
                document_tokens: 513,
                preview_samples: 1,
            },
            tokenization: RuliadTokenizationConfig::default(),
            source_selection: crate::ruliad::config::RuliadSourceSelectionConfig::default(),
            families: vec![RuliadFamilyConfig {
                kind: RuliadFamilyKind::LeanTask,
                weight: 1,
                width: None,
                steps: None,
            }],
            proof_tasks: None,
            lean_task_limit: None,
        };
        let report = generate_ruliad_corpus(&config).expect("generate");
        let required = verify_manifest(
            &report.manifest_path,
            LeanMode::Required,
            Some(dir.path().join("missing-lean-project").as_path()),
        )
        .expect("verify report");
        assert!(required.count > 0);
        assert!(!required.failed.is_empty());
    }
}
