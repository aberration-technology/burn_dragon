use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct StackLock {
    schema_version: u32,
    repositories: Vec<LockedRepository>,
}

#[derive(Debug, Deserialize)]
pub struct LockedRepository {
    pub name: String,
    pub path: PathBuf,
    pub url: String,
    pub revision: String,
}

impl StackLock {
    pub fn load(path: &Path) -> Result<Self> {
        let source =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let lock: Self =
            toml::from_str(&source).with_context(|| format!("parsing {}", path.display()))?;
        ensure!(
            lock.schema_version == 1,
            "unsupported stack lock schema {}",
            lock.schema_version
        );
        ensure!(
            !lock.repositories.is_empty(),
            "stack lock contains no repositories"
        );
        for repository in &lock.repositories {
            ensure!(
                repository.revision.len() == 40
                    && repository
                        .revision
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
                "{} must use a full lowercase Git SHA",
                repository.name
            );
            ensure!(
                repository.path.starts_with(".."),
                "{} must remain a sibling path dependency",
                repository.name
            );
            ensure!(
                repository.url.starts_with("https://github.com/"),
                "{} must provide a CI-readable GitHub URL",
                repository.name
            );
        }
        Ok(lock)
    }

    pub fn repository(&self, name: &str) -> Result<&LockedRepository> {
        self.repositories
            .iter()
            .find(|repository| repository.name == name)
            .with_context(|| format!("stack lock does not contain {name}"))
    }
}

pub fn workspace_stack_lock() -> Result<StackLock> {
    StackLock::load(&workspace_root().join("stack.lock.toml"))
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must be inside the workspace")
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_lock_contains_the_complete_dependency_stack() {
        let lock = workspace_stack_lock().expect("stack lock");
        for name in ["burn_ecs", "burn_p2p", "burn_eggroll", "burn_pc"] {
            lock.repository(name).expect("locked repository");
        }
    }
}
