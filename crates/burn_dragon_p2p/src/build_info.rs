use burn_p2p::ClientReleaseManifest;

const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn embedded_git_commit() -> Option<&'static str> {
    option_env!("BURN_DRAGON_GIT_COMMIT").and_then(sanitize_git_commit)
}

pub fn embedded_git_commit_owned() -> Option<String> {
    embedded_git_commit().map(str::to_owned)
}

pub fn embedded_git_commit_or_unknown() -> String {
    embedded_git_commit_owned().unwrap_or_else(|| "unknown".to_owned())
}

pub fn cli_long_version() -> String {
    match embedded_git_commit() {
        Some(commit) => format!("{PACKAGE_VERSION}\nrev: {commit}"),
        None => PACKAGE_VERSION.to_owned(),
    }
}

pub fn footer_build_rev(release_manifest: Option<&ClientReleaseManifest>) -> String {
    release_manifest
        .and_then(|manifest| sanitize_git_commit_owned(&manifest.git_commit))
        .or_else(embedded_git_commit_owned)
        .unwrap_or_else(|| "unknown".to_owned())
}

fn sanitize_git_commit_owned(value: &str) -> Option<String> {
    sanitize_git_commit(value).map(str::to_owned)
}

fn sanitize_git_commit(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed == "unknown" || trimmed == "browser-site" {
        None
    } else {
        Some(trimmed)
    }
}
