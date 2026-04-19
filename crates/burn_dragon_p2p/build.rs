use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=BURN_DRAGON_GIT_COMMIT");
    println!("cargo:rerun-if-changed=.cargo_vcs_info.json");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    if let Some(commit) = resolve_git_commit(&manifest_dir) {
        println!("cargo:rustc-env=BURN_DRAGON_GIT_COMMIT={commit}");
    }

    for path in git_rerun_paths(&manifest_dir) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn resolve_git_commit(manifest_dir: &Path) -> Option<String> {
    env::var("BURN_DRAGON_GIT_COMMIT")
        .ok()
        .and_then(|value| sanitize_git_commit(&value))
        .or_else(|| git_head_commit(manifest_dir))
        .or_else(|| cargo_vcs_commit(manifest_dir))
}

fn sanitize_git_commit(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

fn git_head_commit(manifest_dir: &Path) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(manifest_dir)
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    sanitize_git_commit(String::from_utf8_lossy(&output.stdout).as_ref())
}

fn cargo_vcs_commit(manifest_dir: &Path) -> Option<String> {
    let path = manifest_dir.join(".cargo_vcs_info.json");
    let contents = fs::read_to_string(path).ok()?;
    let sha1_key = "\"sha1\":\"";
    let start = contents.find(sha1_key)? + sha1_key.len();
    let end = contents[start..].find('"')? + start;
    let sha = contents[start..end].trim();
    let short = sha.get(..12).unwrap_or(sha);
    sanitize_git_commit(short)
}

fn git_rerun_paths(manifest_dir: &Path) -> Vec<PathBuf> {
    let git_common_dir = git_command_output(manifest_dir, ["rev-parse", "--git-common-dir"]);
    let Some(git_common_dir) = git_common_dir else {
        return Vec::new();
    };
    let git_dir = manifest_dir.join(git_common_dir);
    let mut paths = vec![git_dir.join("HEAD"), git_dir.join("packed-refs")];
    if let Some(reference) = git_command_output(manifest_dir, ["symbolic-ref", "-q", "HEAD"]) {
        paths.push(git_dir.join(reference));
    }
    paths
}

fn git_command_output<const N: usize>(manifest_dir: &Path, args: [&str; N]) -> Option<PathBuf> {
    let output = Command::new("git")
        .arg("-C")
        .arg(manifest_dir)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&output.stdout);
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(PathBuf::from(trimmed))
    }
}
