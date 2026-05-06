use std::fs::{self, OpenOptions};
use std::io::Write;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};

use crate::workflow_tools::BootstrapStackSettingsMode;

pub fn resolve(mode: BootstrapStackSettingsMode) -> Result<()> {
    match mode {
        BootstrapStackSettingsMode::Deploy => resolve_deploy(),
        BootstrapStackSettingsMode::Restore => resolve_restore(),
    }
}

fn resolve_deploy() -> Result<()> {
    let env_name = required_env("DEPLOY_ENVIRONMENT")?;
    let stack_name = canonical_stack_name(&env_name, "deploy")?;
    let aws_region = env_or("VAR_AWS_REGION", "us-east-2");
    let aws_role_arn =
        required_env("VAR_AWS_ROLE_ARN").context("missing BURN_DRAGON_P2P_AWS_ROLE_ARN")?;
    let network_id = env_or("VAR_NETWORK_ID", "burn-dragon-mainnet");
    let project_family_id = env_or("VAR_PROJECT_FAMILY_ID", "burn-dragon-language");
    let study_id = env_or("VAR_STUDY_ID", "burn-dragon-mainnet");
    let release_train_hash = env_or("VAR_RELEASE_TRAIN_HASH", "burn-dragon-mainnet-train");
    let native_target_artifact_hash =
        env_or("VAR_NATIVE_TARGET_ARTIFACT_HASH", "burn-dragon-native");
    let edge_domain_name = first_env(
        &["INPUT_EDGE_DOMAIN_NAME", "VAR_EDGE_DOMAIN_NAME"],
        "edge.dragon.aberration.technology",
    );
    let route53_zone_name = first_env(
        &["INPUT_ROUTE53_ZONE_NAME", "VAR_ROUTE53_ZONE_NAME"],
        "aberration.technology",
    );
    let browser_app_base_url = trim_url(&env_or(
        "VAR_BROWSER_APP_BASE_URL",
        "https://dragon.aberration.technology",
    ));
    let auth_redirect_base_url =
        trim_url(&env_or("VAR_AUTH_REDIRECT_BASE_URL", &browser_app_base_url));
    let acme_contact_email = env_or(
        "VAR_ACME_CONTACT_EMAIL",
        &format!("admin@{}", route53_zone_name.trim_end_matches('.')),
    );
    let browser_app_pages_domain_target = env_or(
        "VAR_BROWSER_APP_PAGES_DOMAIN_TARGET",
        &format!(
            "{}.github.io",
            env_or("GITHUB_REPOSITORY_OWNER_CONTEXT", "")
        ),
    );
    let browser_app_host = url_host(&browser_app_base_url).unwrap_or_default();
    let disaster_recovery_region = first_env(
        &[
            "INPUT_DISASTER_RECOVERY_REGION",
            "VAR_DISASTER_RECOVERY_REGION",
        ],
        "",
    );
    let secret_parameter_prefix = format!(
        "/{stack_name}/{}/bootstrap",
        required_env("TF_WORKSPACE_NAME")?
    );

    let mut auth_connector_kind = first_env(
        &["INPUT_AUTH_CONNECTOR_KIND", "VAR_AUTH_CONNECTOR_KIND"],
        "github",
    );
    auth_connector_kind = normalize_lower(&auth_connector_kind);
    ensure_auth_connector_kind(&auth_connector_kind)?;
    let auth = resolve_auth_settings(&auth_connector_kind)?;

    let default_dataset_domain_name = if browser_app_host.is_empty() {
        format!("datasets.{edge_domain_name}")
    } else {
        format!("datasets.{browser_app_host}")
    };
    let dataset_domain_name = env_or("VAR_DATASET_DOMAIN_NAME", &default_dataset_domain_name);
    let dataset_bucket_name = env_or("VAR_DATASET_BUCKET_NAME", "");
    let dataset_bucket_path_prefix = env_or("VAR_DATASET_BUCKET_PATH_PREFIX", "dragon-datasets");
    let climbmix_dataset_base_url = first_env(
        &[
            "INPUT_CLIMBMIX_BROWSER_DATASET_BASE_URL",
            "VAR_CLIMBMIX_BROWSER_DATASET_BASE_URL",
        ],
        &format!(
            "https://{dataset_domain_name}/{dataset_bucket_path_prefix}/climbmix-pretraining/climbmix-r1"
        ),
    );

    let mut bootstrap_install_source = first_env(
        &[
            "INPUT_BOOTSTRAP_INSTALL_SOURCE",
            "VAR_BOOTSTRAP_INSTALL_SOURCE",
        ],
        "crate",
    );
    bootstrap_install_source = normalize_lower(&bootstrap_install_source);
    ensure_bootstrap_install_source(&bootstrap_install_source)?;
    let bootstrap_version = first_env(
        &["INPUT_BOOTSTRAP_VERSION", "VAR_BOOTSTRAP_VERSION"],
        "0.21.0-pre.80",
    );
    let bootstrap_git_ref = first_env(&["INPUT_BOOTSTRAP_GIT_REF", "VAR_BOOTSTRAP_GIT_REF"], "");
    validate_bootstrap_install(
        &auth_connector_kind,
        &bootstrap_install_source,
        &bootstrap_version,
        &bootstrap_git_ref,
    )?;

    let dragon_crate_version = workspace_version()?;
    let mut auth_principals = auth_principals_json()?;
    let managed = resolve_managed_trainer(
        &mut auth_principals,
        &secret_parameter_prefix,
        &network_id,
        &stack_name,
        &env_name,
        &dragon_crate_version,
        DeployInputs::default(),
    )?;
    let mirror = add_bootstrap_head_mirror_principal(
        &mut auth_principals,
        &network_id,
        &stack_name,
        &env_name,
    );
    let canary = canary_principals(&required_env("TF_WORKSPACE_NAME")?);
    remove_principals(
        &mut auth_principals,
        &[
            &canary.browser_principal_id,
            &canary.native_principal_id,
            &canary.native_validator_principal_id,
        ],
    );

    let artifact = resolve_artifact_settings(&disaster_recovery_region, false)?;
    let admin = resolve_admin_logins(&auth_connector_kind)?;
    validate_auth_secrets(&auth_connector_kind, &auth, &admin)?;

    let mut lines = base_lines(BaseLines {
        aws_region: &aws_region,
        aws_role_arn: &aws_role_arn,
        stack_name: &stack_name,
        edge_domain_name: &edge_domain_name,
        browser_app_base_url: &browser_app_base_url,
        auth_redirect_base_url: &auth_redirect_base_url,
        acme_contact_email: &acme_contact_email,
        browser_app_pages_domain_target: &browser_app_pages_domain_target,
        route53_zone_name: &route53_zone_name,
        secret_parameter_prefix: &secret_parameter_prefix,
        network_id: &network_id,
        project_family_id: &project_family_id,
        study_id: &study_id,
        release_train_hash: &release_train_hash,
        bootstrap_install_source: &bootstrap_install_source,
        bootstrap_version: &bootstrap_version,
        bootstrap_git_ref: &bootstrap_git_ref,
        dragon_crate_version: &dragon_crate_version,
        native_target_artifact_hash: &native_target_artifact_hash,
        auth_connector_kind: &auth_connector_kind,
        auth_authority_name: &auth.authority_name,
        managed: &managed,
        mirror: &mirror,
        canary: &canary,
        artifact: &artifact,
        dataset_domain_name: &dataset_domain_name,
        dataset_bucket_path_prefix: &dataset_bucket_path_prefix,
    });
    push_optional(
        &mut lines,
        "TF_VAR_artifact_bucket_name",
        &artifact.bucket_name,
    );
    push_optional(
        &mut lines,
        "TF_VAR_artifact_bucket_path_prefix",
        &artifact.bucket_path_prefix,
    );
    push_optional(
        &mut lines,
        "TF_VAR_artifact_replica_bucket_name",
        &artifact.replica_bucket_name,
    );
    push_optional(
        &mut lines,
        "TF_VAR_dataset_bucket_name",
        &dataset_bucket_name,
    );
    push_optional(
        &mut lines,
        "TF_VAR_disaster_recovery_region",
        &disaster_recovery_region,
    );
    push_optional_inputs(&mut lines, DEPLOY_OPTIONAL_ENV);
    auth.append_to(&mut lines);
    if auth_connector_kind == "github" {
        append_github_lines(&mut lines, &auth, &admin, &canary)?;
    }
    lines.push(format!(
        "TF_VAR_climbmix_browser_dataset_base_url={climbmix_dataset_base_url}"
    ));
    append_env_lines(&lines)?;
    append_multiline_env(
        "TF_VAR_auth_principals_json",
        &serde_json::to_string(&auth_principals)?,
    )?;
    Ok(())
}

fn resolve_restore() -> Result<()> {
    let env_name = required_env("DEPLOY_ENVIRONMENT")?;
    let stack_name = canonical_stack_name(&env_name, "restore")?;
    let aws_region = first_env(&["INPUT_AWS_REGION", "VAR_AWS_REGION"], "us-east-2");
    let aws_role_arn =
        required_env("VAR_AWS_ROLE_ARN").context("missing BURN_DRAGON_P2P_AWS_ROLE_ARN")?;
    let network_id = env_or("VAR_NETWORK_ID", "burn-dragon-mainnet");
    let project_family_id = env_or("VAR_PROJECT_FAMILY_ID", "burn-dragon-language");
    let study_id = env_or("VAR_STUDY_ID", "burn-dragon-mainnet");
    let release_train_hash = env_or("VAR_RELEASE_TRAIN_HASH", "burn-dragon-mainnet-train");
    let native_target_artifact_hash =
        env_or("VAR_NATIVE_TARGET_ARTIFACT_HASH", "burn-dragon-native");
    let edge_domain_name = first_env(
        &["INPUT_EDGE_DOMAIN_NAME", "VAR_EDGE_DOMAIN_NAME"],
        "edge.dragon.aberration.technology",
    );
    let route53_zone_name = first_env(
        &["INPUT_ROUTE53_ZONE_NAME", "VAR_ROUTE53_ZONE_NAME"],
        "aberration.technology",
    );
    let browser_app_base_url = trim_url(&env_or(
        "VAR_BROWSER_APP_BASE_URL",
        "https://dragon.aberration.technology",
    ));
    let auth_redirect_base_url =
        trim_url(&env_or("VAR_AUTH_REDIRECT_BASE_URL", &browser_app_base_url));
    let acme_contact_email = env_or(
        "VAR_ACME_CONTACT_EMAIL",
        &format!("admin@{}", route53_zone_name.trim_end_matches('.')),
    );
    let browser_app_pages_domain_target = env_or(
        "VAR_BROWSER_APP_PAGES_DOMAIN_TARGET",
        &format!(
            "{}.github.io",
            env_or("GITHUB_REPOSITORY_OWNER_CONTEXT", "")
        ),
    );
    let browser_app_host = url_host(&browser_app_base_url).unwrap_or_default();
    let secret_parameter_prefix = format!(
        "/{stack_name}/{}/bootstrap",
        required_env("TF_WORKSPACE_NAME")?
    );

    let mut auth_connector_kind = env_or("VAR_AUTH_CONNECTOR_KIND", "github");
    auth_connector_kind = normalize_lower(&auth_connector_kind);
    ensure_auth_connector_kind(&auth_connector_kind)?;
    let auth = resolve_auth_settings(&auth_connector_kind)?;

    let default_dataset_domain_name = if browser_app_host.is_empty() {
        format!("datasets.{edge_domain_name}")
    } else {
        format!("datasets.{browser_app_host}")
    };
    let dataset_domain_name = env_or("VAR_DATASET_DOMAIN_NAME", &default_dataset_domain_name);
    let dataset_bucket_name = env_or("VAR_DATASET_BUCKET_NAME", "");
    let dataset_bucket_path_prefix = env_or("VAR_DATASET_BUCKET_PATH_PREFIX", "dragon-datasets");
    let climbmix_dataset_base_url = env_or(
        "VAR_CLIMBMIX_BROWSER_DATASET_BASE_URL",
        &format!(
            "https://{dataset_domain_name}/{dataset_bucket_path_prefix}/climbmix-pretraining/climbmix-r1"
        ),
    );

    let mut bootstrap_install_source = first_env(
        &[
            "INPUT_BOOTSTRAP_INSTALL_SOURCE",
            "VAR_BOOTSTRAP_INSTALL_SOURCE",
        ],
        "crate",
    );
    bootstrap_install_source = normalize_lower(&bootstrap_install_source);
    ensure_bootstrap_install_source(&bootstrap_install_source)?;
    let bootstrap_version = first_env(
        &["INPUT_BOOTSTRAP_VERSION", "VAR_BOOTSTRAP_VERSION"],
        "0.21.0-pre.80",
    );
    let bootstrap_git_ref = first_env(&["INPUT_BOOTSTRAP_GIT_REF", "VAR_BOOTSTRAP_GIT_REF"], "");
    validate_bootstrap_install(
        &auth_connector_kind,
        &bootstrap_install_source,
        &bootstrap_version,
        &bootstrap_git_ref,
    )?;

    let dragon_crate_version = workspace_version()?;
    let mut auth_principals = auth_principals_json()?;
    let managed = resolve_managed_trainer(
        &mut auth_principals,
        &secret_parameter_prefix,
        &network_id,
        &stack_name,
        &env_name,
        &dragon_crate_version,
        DeployInputs::restore(),
    )?;
    let mirror = add_bootstrap_head_mirror_principal(
        &mut auth_principals,
        &network_id,
        &stack_name,
        &env_name,
    );
    let canary = canary_principals(&required_env("TF_WORKSPACE_NAME")?);
    remove_principals(
        &mut auth_principals,
        &[
            &canary.browser_principal_id,
            &canary.native_principal_id,
            &canary.native_validator_principal_id,
        ],
    );

    let next_disaster_recovery_region = env_or("INPUT_NEXT_DISASTER_RECOVERY_REGION", "");
    let artifact = resolve_artifact_settings(&next_disaster_recovery_region, true)?;
    let snapshot_search_region = env_or("INPUT_SOURCE_SNAPSHOT_REGION", &aws_region);
    let primary_restore_snapshot_id = resolve_restore_snapshot_id(
        &aws_region,
        &snapshot_search_region,
        &stack_name,
        &required_env("TF_WORKSPACE_NAME")?,
    )?;

    let admin = resolve_admin_logins(&auth_connector_kind)?;
    validate_auth_secrets(&auth_connector_kind, &auth, &admin)?;

    let mut lines = base_lines(BaseLines {
        aws_region: &aws_region,
        aws_role_arn: &aws_role_arn,
        stack_name: &stack_name,
        edge_domain_name: &edge_domain_name,
        browser_app_base_url: &browser_app_base_url,
        auth_redirect_base_url: &auth_redirect_base_url,
        acme_contact_email: &acme_contact_email,
        browser_app_pages_domain_target: &browser_app_pages_domain_target,
        route53_zone_name: &route53_zone_name,
        secret_parameter_prefix: &secret_parameter_prefix,
        network_id: &network_id,
        project_family_id: &project_family_id,
        study_id: &study_id,
        release_train_hash: &release_train_hash,
        bootstrap_install_source: &bootstrap_install_source,
        bootstrap_version: &bootstrap_version,
        bootstrap_git_ref: &bootstrap_git_ref,
        dragon_crate_version: &dragon_crate_version,
        native_target_artifact_hash: &native_target_artifact_hash,
        auth_connector_kind: &auth_connector_kind,
        auth_authority_name: &auth.authority_name,
        managed: &managed,
        mirror: &mirror,
        canary: &canary,
        artifact: &artifact,
        dataset_domain_name: &dataset_domain_name,
        dataset_bucket_path_prefix: &dataset_bucket_path_prefix,
    });
    push_optional(
        &mut lines,
        "TF_VAR_artifact_bucket_name",
        &artifact.bucket_name,
    );
    push_optional(
        &mut lines,
        "TF_VAR_artifact_bucket_path_prefix",
        &artifact.bucket_path_prefix,
    );
    push_optional(
        &mut lines,
        "TF_VAR_artifact_replica_bucket_name",
        &artifact.replica_bucket_name,
    );
    lines.push(format!(
        "PRIMARY_RESTORE_SNAPSHOT_ID={primary_restore_snapshot_id}"
    ));
    lines.push(format!("SNAPSHOT_SEARCH_REGION={snapshot_search_region}"));
    push_optional(
        &mut lines,
        "TF_VAR_dataset_bucket_name",
        &dataset_bucket_name,
    );
    push_optional(
        &mut lines,
        "TF_VAR_bootstrap_primary_restore_snapshot_id",
        &primary_restore_snapshot_id,
    );
    push_optional(
        &mut lines,
        "TF_VAR_disaster_recovery_region",
        &next_disaster_recovery_region,
    );
    push_optional_inputs(&mut lines, RESTORE_OPTIONAL_ENV);
    auth.append_to(&mut lines);
    if auth_connector_kind == "github" {
        append_github_lines(&mut lines, &auth, &admin, &canary)?;
    }
    push_optional(
        &mut lines,
        "TF_VAR_climbmix_browser_dataset_base_url",
        &climbmix_dataset_base_url,
    );
    append_env_lines(&lines)?;
    append_multiline_env(
        "TF_VAR_auth_principals_json",
        &serde_json::to_string(&auth_principals)?,
    )?;
    Ok(())
}

#[derive(Clone, Copy)]
struct DeployInputs {
    restore: bool,
}

impl DeployInputs {
    fn default() -> Self {
        Self { restore: false }
    }

    fn restore() -> Self {
        Self { restore: true }
    }
}

struct AuthSettings {
    authority_name: String,
    authorize_base_url: String,
    exchange_url: String,
    token_url: String,
    api_base_url: String,
    userinfo_url: String,
    refresh_url: String,
    revoke_url: String,
    jwks_url: String,
    oidc_issuer: String,
    oauth_provider: String,
    external_authority: String,
    external_trusted_principal_header: String,
    external_trusted_internal_only: String,
    resolved_client_id: String,
    resolved_client_secret: String,
    github_required_org: String,
    github_required_team: String,
    github_required_repo: String,
    github_admin_required_repo_permission: String,
}

impl AuthSettings {
    fn append_to(&self, lines: &mut Vec<String>) {
        push_optional(
            lines,
            "TF_VAR_auth_authorize_base_url",
            &self.authorize_base_url,
        );
        push_optional(lines, "TF_VAR_auth_exchange_url", &self.exchange_url);
        push_optional(lines, "TF_VAR_auth_token_url", &self.token_url);
        push_optional(lines, "TF_VAR_auth_api_base_url", &self.api_base_url);
        push_optional(lines, "TF_VAR_auth_userinfo_url", &self.userinfo_url);
        push_optional(lines, "TF_VAR_auth_refresh_url", &self.refresh_url);
        push_optional(lines, "TF_VAR_auth_revoke_url", &self.revoke_url);
        push_optional(lines, "TF_VAR_auth_jwks_url", &self.jwks_url);
        push_optional(lines, "TF_VAR_auth_oidc_issuer", &self.oidc_issuer);
        push_optional(lines, "TF_VAR_auth_oauth_provider", &self.oauth_provider);
        push_optional(
            lines,
            "TF_VAR_auth_external_authority",
            &self.external_authority,
        );
        push_optional(
            lines,
            "TF_VAR_auth_external_trusted_principal_header",
            &self.external_trusted_principal_header,
        );
        push_optional(
            lines,
            "TF_VAR_auth_external_trusted_internal_only",
            &self.external_trusted_internal_only,
        );
        push_optional(
            lines,
            "TF_VAR_github_admin_required_repo_permission",
            &self.github_admin_required_repo_permission,
        );
    }
}

struct ManagedTrainer {
    desired_capacity: String,
    backend: String,
    experiment_kind: String,
    target: String,
    crate_version: String,
    auth_bundle_parameter_name: String,
    auth_mode: String,
    principal_id: String,
    experiment_id: String,
    revision_id: String,
}

struct MirrorPrincipal {
    principal_id: String,
    experiment_id: String,
    revision_id: String,
    auth_bundle_parameter_name: String,
}

struct CanaryPrincipals {
    browser_principal_id: String,
    browser_experiment_id: String,
    browser_revision_id: String,
    native_principal_id: String,
    native_validator_principal_id: String,
}

struct ArtifactSettings {
    create_bucket: String,
    bucket_name: String,
    bucket_path_prefix: String,
    create_replica_bucket: String,
    replica_bucket_name: String,
}

struct BaseLines<'a> {
    aws_region: &'a str,
    aws_role_arn: &'a str,
    stack_name: &'a str,
    edge_domain_name: &'a str,
    browser_app_base_url: &'a str,
    auth_redirect_base_url: &'a str,
    acme_contact_email: &'a str,
    browser_app_pages_domain_target: &'a str,
    route53_zone_name: &'a str,
    secret_parameter_prefix: &'a str,
    network_id: &'a str,
    project_family_id: &'a str,
    study_id: &'a str,
    release_train_hash: &'a str,
    bootstrap_install_source: &'a str,
    bootstrap_version: &'a str,
    bootstrap_git_ref: &'a str,
    dragon_crate_version: &'a str,
    native_target_artifact_hash: &'a str,
    auth_connector_kind: &'a str,
    auth_authority_name: &'a str,
    managed: &'a ManagedTrainer,
    mirror: &'a MirrorPrincipal,
    canary: &'a CanaryPrincipals,
    artifact: &'a ArtifactSettings,
    dataset_domain_name: &'a str,
    dataset_bucket_path_prefix: &'a str,
}

fn base_lines(input: BaseLines<'_>) -> Vec<String> {
    let dragon_git_repository =
        format!("https://github.com/{}.git", env_or("GITHUB_REPOSITORY", ""));
    let dragon_git_ref = env_or("GITHUB_SHA", "");
    vec![
        format!("AWS_REGION={}", input.aws_region),
        format!("AWS_ROLE_ARN={}", input.aws_role_arn),
        format!("AWS_ACCOUNT_ID={}", aws_account_id(input.aws_role_arn)),
        format!("STACK_NAME={}", input.stack_name),
        format!("EDGE_DOMAIN_NAME={}", input.edge_domain_name),
        format!("BROWSER_APP_BASE_URL={}", input.browser_app_base_url),
        format!("AUTH_REDIRECT_BASE_URL={}", input.auth_redirect_base_url),
        format!("ACME_CONTACT_EMAIL={}", input.acme_contact_email),
        format!(
            "BROWSER_APP_PAGES_DOMAIN_TARGET={}",
            input.browser_app_pages_domain_target
        ),
        format!("ROUTE53_ZONE_NAME={}", input.route53_zone_name),
        format!("SECRET_PARAMETER_PREFIX={}", input.secret_parameter_prefix),
        format!("TF_VAR_aws_region={}", input.aws_region),
        format!("TF_VAR_stack_name={}", input.stack_name),
        format!(
            "TF_VAR_environment_name={}",
            env_or("DEPLOY_ENVIRONMENT", "")
        ),
        format!("TF_VAR_route53_zone_name={}", input.route53_zone_name),
        format!("TF_VAR_edge_domain_name={}", input.edge_domain_name),
        format!("TF_VAR_browser_app_base_url={}", input.browser_app_base_url),
        format!(
            "TF_VAR_auth_redirect_base_url={}",
            input.auth_redirect_base_url
        ),
        format!("TF_VAR_acme_contact_email={}", input.acme_contact_email),
        format!(
            "TF_VAR_browser_app_pages_domain_target={}",
            input.browser_app_pages_domain_target
        ),
        format!(
            "TF_VAR_secret_parameter_prefix={}",
            input.secret_parameter_prefix
        ),
        format!("TF_VAR_network_id={}", input.network_id),
        format!("TF_VAR_project_family_id={}", input.project_family_id),
        format!("TF_VAR_study_id={}", input.study_id),
        format!("TF_VAR_release_train_hash={}", input.release_train_hash),
        format!(
            "TF_VAR_bootstrap_install_source={}",
            input.bootstrap_install_source
        ),
        format!("TF_VAR_bootstrap_crate_version={}", input.bootstrap_version),
        format!("TF_VAR_bootstrap_git_ref={}", input.bootstrap_git_ref),
        format!("TF_VAR_dragon_crate_version={}", input.dragon_crate_version),
        format!("TF_VAR_dragon_git_repository={dragon_git_repository}"),
        format!("TF_VAR_dragon_git_ref={dragon_git_ref}"),
        format!(
            "TF_VAR_bootstrap_head_mirror_auth_bundle_parameter_name={}",
            input.mirror.auth_bundle_parameter_name
        ),
        format!(
            "TF_VAR_create_artifact_bucket={}",
            input.artifact.create_bucket
        ),
        format!("DATASET_DOMAIN_NAME={}", input.dataset_domain_name),
        format!(
            "DATASET_BUCKET_PATH_PREFIX={}",
            input.dataset_bucket_path_prefix
        ),
        format!("TF_VAR_dataset_domain_name={}", input.dataset_domain_name),
        format!(
            "TF_VAR_dataset_bucket_path_prefix={}",
            input.dataset_bucket_path_prefix
        ),
        format!(
            "TF_VAR_managed_trainer_desired_capacity={}",
            input.managed.desired_capacity
        ),
        format!("TF_VAR_managed_trainer_backend={}", input.managed.backend),
        format!(
            "TF_VAR_managed_trainer_experiment_kind={}",
            input.managed.experiment_kind
        ),
        format!("TF_VAR_managed_trainer_target={}", input.managed.target),
        format!(
            "TF_VAR_managed_trainer_crate_version={}",
            input.managed.crate_version
        ),
        format!(
            "TF_VAR_managed_trainer_auth_bundle_parameter_name={}",
            input.managed.auth_bundle_parameter_name
        ),
        format!(
            "TF_VAR_native_target_artifact_hash={}",
            input.native_target_artifact_hash
        ),
        format!(
            "TF_VAR_create_artifact_replica_bucket={}",
            input.artifact.create_replica_bucket
        ),
        format!("AUTH_CONNECTOR_KIND={}", input.auth_connector_kind),
        format!("MANAGED_TRAINER_AUTH_MODE={}", input.managed.auth_mode),
        format!(
            "MANAGED_TRAINER_PRINCIPAL_ID={}",
            input.managed.principal_id
        ),
        format!(
            "MANAGED_TRAINER_EXPERIMENT_ID={}",
            input.managed.experiment_id
        ),
        format!("MANAGED_TRAINER_REVISION_ID={}", input.managed.revision_id),
        format!(
            "BOOTSTRAP_HEAD_MIRROR_PRINCIPAL_ID={}",
            input.mirror.principal_id
        ),
        format!(
            "BOOTSTRAP_HEAD_MIRROR_EXPERIMENT_ID={}",
            input.mirror.experiment_id
        ),
        format!(
            "BOOTSTRAP_HEAD_MIRROR_REVISION_ID={}",
            input.mirror.revision_id
        ),
        format!(
            "BROWSER_CANARY_PRINCIPAL_ID={}",
            input.canary.browser_principal_id
        ),
        format!(
            "BROWSER_CANARY_EXPERIMENT_ID={}",
            input.canary.browser_experiment_id
        ),
        format!(
            "BROWSER_CANARY_REVISION_ID={}",
            input.canary.browser_revision_id
        ),
        format!(
            "NATIVE_CANARY_PRINCIPAL_ID={}",
            input.canary.native_principal_id
        ),
        format!(
            "NATIVE_CANARY_VALIDATOR_PRINCIPAL_ID={}",
            input.canary.native_validator_principal_id
        ),
        format!("TF_VAR_auth_connector_kind={}", input.auth_connector_kind),
        format!("TF_VAR_auth_authority_name={}", input.auth_authority_name),
    ]
}

fn resolve_auth_settings(kind: &str) -> Result<AuthSettings> {
    let resolved_client_id = if kind == "github" {
        first_env(&["AUTH_CLIENT_ID", "GITHUB_CLIENT_ID"], "")
    } else {
        env_or("AUTH_CLIENT_ID", "")
    };
    let resolved_client_secret = if kind == "github" {
        first_env(&["AUTH_CLIENT_SECRET", "GITHUB_CLIENT_SECRET"], "")
    } else {
        env_or("AUTH_CLIENT_SECRET", "")
    };
    Ok(AuthSettings {
        authority_name: env_or("VAR_AUTH_AUTHORITY_NAME", "burn-dragon-auth"),
        authorize_base_url: env_or("VAR_AUTH_AUTHORIZE_BASE_URL", ""),
        exchange_url: env_or("VAR_AUTH_EXCHANGE_URL", ""),
        token_url: env_or("VAR_AUTH_TOKEN_URL", ""),
        api_base_url: env_or("VAR_AUTH_API_BASE_URL", ""),
        userinfo_url: env_or("VAR_AUTH_USERINFO_URL", ""),
        refresh_url: env_or("VAR_AUTH_REFRESH_URL", ""),
        revoke_url: env_or("VAR_AUTH_REVOKE_URL", ""),
        jwks_url: env_or("VAR_AUTH_JWKS_URL", ""),
        oidc_issuer: env_or("VAR_AUTH_OIDC_ISSUER", ""),
        oauth_provider: env_or("VAR_AUTH_OAUTH_PROVIDER", ""),
        external_authority: env_or("VAR_AUTH_EXTERNAL_AUTHORITY", ""),
        external_trusted_principal_header: env_or(
            "VAR_AUTH_EXTERNAL_TRUSTED_PRINCIPAL_HEADER",
            "x-forwarded-user",
        ),
        external_trusted_internal_only: env_or("VAR_AUTH_EXTERNAL_TRUSTED_INTERNAL_ONLY", "true"),
        resolved_client_id,
        resolved_client_secret,
        github_required_org: env_or("VAR_GITHUB_REQUIRED_ORG", ""),
        github_required_team: env_or("VAR_GITHUB_REQUIRED_TEAM", ""),
        github_required_repo: env_or("VAR_GITHUB_REQUIRED_REPO", "mosure/burn_dragon"),
        github_admin_required_repo_permission: env_or(
            "VAR_GITHUB_ADMIN_REQUIRED_REPO_PERMISSION",
            "",
        ),
    })
}

fn resolve_managed_trainer(
    principals: &mut Value,
    secret_prefix: &str,
    network_id: &str,
    stack_name: &str,
    environment: &str,
    dragon_crate_version: &str,
    inputs: DeployInputs,
) -> Result<ManagedTrainer> {
    let desired_capacity = if inputs.restore {
        env_or("VAR_MANAGED_TRAINER_DESIRED_CAPACITY", "0")
    } else {
        first_env(
            &[
                "INPUT_MANAGED_TRAINER_DESIRED_CAPACITY",
                "VAR_MANAGED_TRAINER_DESIRED_CAPACITY",
            ],
            "0",
        )
    };
    let backend = normalize_lower(&if inputs.restore {
        env_or("VAR_MANAGED_TRAINER_BACKEND", "cpu")
    } else {
        first_env(
            &[
                "INPUT_MANAGED_TRAINER_BACKEND",
                "VAR_MANAGED_TRAINER_BACKEND",
            ],
            "cpu",
        )
    });
    let experiment_kind = normalize_lower(&if inputs.restore {
        env_or("VAR_MANAGED_TRAINER_EXPERIMENT_KIND", "nca")
    } else {
        first_env(
            &[
                "INPUT_MANAGED_TRAINER_EXPERIMENT_KIND",
                "VAR_MANAGED_TRAINER_EXPERIMENT_KIND",
            ],
            "nca",
        )
    });
    let target = env_or("VAR_MANAGED_TRAINER_TARGET", "trainer");
    let crate_version = env_or("VAR_MANAGED_TRAINER_CRATE_VERSION", dragon_crate_version);
    let auth_bundle_parameter_name = env_or(
        "VAR_MANAGED_TRAINER_AUTH_BUNDLE_PARAMETER_NAME",
        &format!("{secret_prefix}/trainer_auth_bundle_json"),
    );
    let (experiment_id, revision_id) = match experiment_kind.as_str() {
        "climbmix" => ("climbmix-pretraining", "climbmix-r1"),
        _ => ("nca-prepretraining", "nca-r1"),
    };
    let mut auth_mode = "disabled".to_owned();
    let mut principal_id = String::new();
    if desired_capacity != "0" {
        let suffix = slug(&format!(
            "{}-{experiment_kind}-{backend}",
            required_env("TF_WORKSPACE_NAME")?
        ));
        principal_id = format!("managed-trainer-{}", suffix.trim_end_matches('-'));
        if env_or("TRAINER_AUTH_BUNDLE_JSON", "").is_empty() {
            auth_mode = "bootstrap_static_principal".to_owned();
            upsert_principal(
                principals,
                json!({
                    "principal_id": principal_id,
                    "display_name": format!("burn_dragon managed {experiment_id} {backend} trainer"),
                    "org_memberships": [],
                    "group_memberships": [],
                    "granted_roles": { "roles": [if backend == "cpu" { "TrainerCpu" } else { "TrainerGpu" }, "Archive"] },
                    "granted_scopes": ["Connect", "Discover", { "Train": { "experiment_id": experiment_id } }, { "Archive": { "experiment_id": experiment_id } }],
                    "allowed_networks": [network_id],
                    "custom_claims": {
                        "deployment_profile": environment,
                        "stack": stack_name,
                        "managed_trainer": "true",
                        "managed_trainer_backend": backend,
                        "managed_trainer_experiment_id": experiment_id
                    }
                }),
            );
        } else {
            auth_mode = "provided_bundle".to_owned();
        }
    }
    Ok(ManagedTrainer {
        desired_capacity,
        backend,
        experiment_kind,
        target,
        crate_version,
        auth_bundle_parameter_name,
        auth_mode,
        principal_id,
        experiment_id: experiment_id.to_owned(),
        revision_id: revision_id.to_owned(),
    })
}

fn add_bootstrap_head_mirror_principal(
    principals: &mut Value,
    network_id: &str,
    stack_name: &str,
    environment: &str,
) -> MirrorPrincipal {
    let workspace = env_or("TF_WORKSPACE_NAME", "");
    let principal_id = format!("bootstrap-head-mirror-{workspace}-nca");
    let experiment_id = "nca-prepretraining".to_owned();
    let revision_id = "nca-r1".to_owned();
    upsert_principal(
        principals,
        json!({
            "principal_id": principal_id,
            "display_name": "burn_dragon bootstrap nca head mirror",
            "org_memberships": [],
            "group_memberships": [],
            "granted_roles": { "roles": ["TrainerCpu", "Archive"] },
            "granted_scopes": ["Connect", "Discover", { "Train": { "experiment_id": experiment_id } }, { "Archive": { "experiment_id": experiment_id } }],
            "allowed_networks": [network_id],
            "custom_claims": {
                "deployment_profile": environment,
                "stack": stack_name,
                "bootstrap_head_mirror": "true",
                "bootstrap_head_mirror_experiment_id": experiment_id,
                "admin_capabilities": "register_live_head,rollout_auth_policy"
            }
        }),
    );
    MirrorPrincipal {
        principal_id,
        experiment_id,
        revision_id,
        auth_bundle_parameter_name: format!(
            "/{stack_name}/{workspace}/bootstrap/bootstrap_head_mirror_auth_bundle_json"
        ),
    }
}

fn canary_principals(workspace: &str) -> CanaryPrincipals {
    let native_principal_id = format!("native-canary-{workspace}-nca");
    CanaryPrincipals {
        browser_principal_id: format!("browser-canary-{workspace}-nca"),
        browser_experiment_id: "nca-prepretraining".to_owned(),
        browser_revision_id: "nca-r1".to_owned(),
        native_validator_principal_id: format!("{native_principal_id}-validator"),
        native_principal_id,
    }
}

fn resolve_artifact_settings(
    disaster_recovery_region: &str,
    _restore: bool,
) -> Result<ArtifactSettings> {
    let create_bucket = env_or("INPUT_CREATE_ARTIFACT_BUCKET", "");
    let bucket_name = first_env(
        &["INPUT_ARTIFACT_BUCKET_NAME", "VAR_ARTIFACT_BUCKET_NAME"],
        "",
    );
    let bucket_path_prefix = first_env(
        &[
            "INPUT_ARTIFACT_BUCKET_PATH_PREFIX",
            "VAR_ARTIFACT_BUCKET_PATH_PREFIX",
        ],
        "",
    );
    let create_replica_bucket = first_env(
        &[
            "INPUT_CREATE_ARTIFACT_REPLICA_BUCKET",
            "VAR_CREATE_ARTIFACT_REPLICA_BUCKET",
        ],
        "true",
    );
    let replica_bucket_name = first_env(
        &[
            "INPUT_ARTIFACT_REPLICA_BUCKET_NAME",
            "VAR_ARTIFACT_REPLICA_BUCKET_NAME",
        ],
        "",
    );
    if create_bucket != "true" && bucket_name.is_empty() {
        bail!("artifact_bucket_name is required when create_artifact_bucket=false");
    }
    if !disaster_recovery_region.is_empty()
        && create_replica_bucket != "true"
        && replica_bucket_name.is_empty()
    {
        bail!(
            "artifact_replica_bucket_name is required when disaster recovery is set and create_artifact_replica_bucket=false"
        );
    }
    Ok(ArtifactSettings {
        create_bucket,
        bucket_name,
        bucket_path_prefix,
        create_replica_bucket,
        replica_bucket_name,
    })
}

fn resolve_restore_snapshot_id(
    aws_region: &str,
    snapshot_search_region: &str,
    stack_name: &str,
    workspace: &str,
) -> Result<String> {
    let use_retained = env_or("VAR_USE_RETAINED_BOOTSTRAP_DATA_VOLUME", "false");
    if use_retained != "true" {
        return Ok(String::new());
    }
    let mut snapshot_id = env_or("INPUT_BOOTSTRAP_PRIMARY_RESTORE_SNAPSHOT_ID", "");
    if snapshot_id.is_empty() && env_or("INPUT_RESTORE_FROM_LATEST_SNAPSHOTS", "") == "true" {
        snapshot_id = aws_output(&[
            "ec2",
            "describe-snapshots",
            "--region",
            if snapshot_search_region.is_empty() {
                aws_region
            } else {
                snapshot_search_region
            },
            "--owner-ids",
            "self",
            "--filters",
            &format!("Name=tag:Stack,Values={stack_name}"),
            &format!("Name=tag:TerraformWorkspace,Values={workspace}"),
            "Name=tag:NodeRole,Values=primary",
            "Name=status,Values=completed",
            "--query",
            "reverse(sort_by(Snapshots,&StartTime))[0].SnapshotId",
            "--output",
            "text",
        ])?
        .trim()
        .to_owned();
    }
    if snapshot_id.is_empty() || snapshot_id == "None" {
        bail!(
            "missing bootstrap restore snapshot id; provide one explicitly or enable restore_from_latest_snapshots with matching tagged snapshots"
        );
    }
    Ok(snapshot_id)
}

fn resolve_admin_logins(auth_connector_kind: &str) -> Result<Vec<String>> {
    let raw = if auth_connector_kind == "github" {
        first_env(
            &["INPUT_GITHUB_ADMIN_LOGINS", "VAR_GITHUB_ADMIN_LOGINS"],
            &env_or("GITHUB_ACTOR_CONTEXT", ""),
        )
    } else {
        first_env(
            &["INPUT_GITHUB_ADMIN_LOGINS", "VAR_GITHUB_ADMIN_LOGINS"],
            "",
        )
    };
    let mut values = Vec::new();
    for token in raw.replace('\n', ",").split(',') {
        let login = token.trim().to_ascii_lowercase();
        if !login.is_empty() && !values.contains(&login) {
            values.push(login);
        }
    }
    Ok(values)
}

fn append_github_lines(
    lines: &mut Vec<String>,
    auth: &AuthSettings,
    admin_logins: &[String],
    canary: &CanaryPrincipals,
) -> Result<()> {
    lines.push(format!(
        "TF_VAR_github_required_org={}",
        auth.github_required_org
    ));
    lines.push(format!(
        "TF_VAR_github_required_team={}",
        auth.github_required_team
    ));
    lines.push(format!(
        "TF_VAR_github_required_repo={}",
        auth.github_required_repo
    ));
    lines.push(format!(
        "TF_VAR_github_browser_canary_principal_id={}",
        canary.browser_principal_id
    ));
    lines.push(format!(
        "TF_VAR_github_browser_canary_callback_token={}",
        required_env("BROWSER_CANARY_CALLBACK_TOKEN")?
    ));
    lines.push(format!(
        "TF_VAR_github_native_canary_principal_id={}",
        canary.native_principal_id
    ));
    lines.push(format!(
        "TF_VAR_github_native_canary_validator_principal_id={}",
        canary.native_validator_principal_id
    ));
    lines.push(format!(
        "TF_VAR_github_admin_logins={}",
        serde_json::to_string(admin_logins)?
    ));
    lines.push(format!("ADMIN_LOGINS_CSV={}", admin_logins.join(",")));
    Ok(())
}

fn validate_auth_secrets(kind: &str, auth: &AuthSettings, admin_logins: &[String]) -> Result<()> {
    match kind {
        "github" => {
            ensure!(
                !admin_logins.is_empty(),
                "no admin logins resolved for github deployment"
            );
            ensure!(
                !auth.resolved_client_id.is_empty(),
                "missing BURN_DRAGON_P2P_AUTH_CLIENT_ID or BURN_DRAGON_P2P_GITHUB_CLIENT_ID secret"
            );
            ensure!(
                !auth.resolved_client_secret.is_empty(),
                "missing BURN_DRAGON_P2P_AUTH_CLIENT_SECRET or BURN_DRAGON_P2P_GITHUB_CLIENT_SECRET secret"
            );
            required_env("BROWSER_CANARY_CALLBACK_TOKEN")
                .context("missing BURN_DRAGON_P2P_BROWSER_CANARY_CALLBACK_TOKEN secret")?;
        }
        "oidc" => {
            ensure!(
                !auth.resolved_client_id.is_empty(),
                "missing BURN_DRAGON_P2P_AUTH_CLIENT_ID secret"
            );
            ensure!(
                !auth.resolved_client_secret.is_empty(),
                "missing BURN_DRAGON_P2P_AUTH_CLIENT_SECRET secret"
            );
            ensure!(
                !auth.oidc_issuer.is_empty(),
                "missing BURN_DRAGON_P2P_AUTH_OIDC_ISSUER for oidc deployment"
            );
        }
        "oauth" => {
            ensure!(
                !auth.resolved_client_id.is_empty(),
                "missing BURN_DRAGON_P2P_AUTH_CLIENT_ID secret"
            );
            ensure!(
                !auth.resolved_client_secret.is_empty(),
                "missing BURN_DRAGON_P2P_AUTH_CLIENT_SECRET secret"
            );
            ensure!(
                !auth.oauth_provider.is_empty(),
                "missing BURN_DRAGON_P2P_AUTH_OAUTH_PROVIDER for oauth deployment"
            );
        }
        "external" => {
            ensure!(
                !auth.external_authority.is_empty(),
                "missing BURN_DRAGON_P2P_AUTH_EXTERNAL_AUTHORITY for external deployment"
            );
        }
        _ => {}
    }
    if matches!(kind, "github" | "oidc" | "oauth") {
        append_env_lines(&[
            format!("AUTH_CLIENT_ID_RESOLVED={}", auth.resolved_client_id),
            format!(
                "AUTH_CLIENT_SECRET_RESOLVED={}",
                auth.resolved_client_secret
            ),
        ])?;
    }
    Ok(())
}

fn upsert_principal(principals: &mut Value, principal: Value) {
    let principal_id = principal
        .get("principal_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    if !principals.is_array() {
        *principals = json!([]);
    }
    let array = principals.as_array_mut().expect("principals array");
    array.retain(|item| item.get("principal_id").and_then(Value::as_str) != Some(&principal_id));
    array.push(principal);
}

fn remove_principals(principals: &mut Value, principal_ids: &[&str]) {
    if let Some(array) = principals.as_array_mut() {
        array.retain(|item| {
            item.get("principal_id")
                .and_then(Value::as_str)
                .is_none_or(|id| !principal_ids.contains(&id))
        });
    }
}

fn auth_principals_json() -> Result<Value> {
    let raw = env_or("VAR_AUTH_PRINCIPALS_JSON", "[]");
    serde_json::from_str(&raw).context("VAR_AUTH_PRINCIPALS_JSON is not valid JSON")
}

fn canonical_stack_name(environment: &str, operation: &str) -> Result<String> {
    let canonical = format!("burn-dragon-p2p-{environment}");
    let override_name = env_or("VAR_STACK_NAME", "");
    if !override_name.is_empty() && override_name != canonical {
        bail!(
            "BURN_DRAGON_P2P_STACK_NAME is set to '{override_name}', but managed {environment} {operation}s require '{canonical}'. Update or remove the environment override before rerunning this workflow to avoid duplicate stacks."
        );
    }
    Ok(canonical)
}

fn ensure_auth_connector_kind(kind: &str) -> Result<()> {
    match kind {
        "github" | "oidc" | "oauth" | "static" | "external" => Ok(()),
        _ => bail!("unsupported auth connector kind: {kind}"),
    }
}

fn ensure_bootstrap_install_source(source: &str) -> Result<()> {
    match source {
        "crate" | "git" => Ok(()),
        _ => bail!("unsupported bootstrap_install_source: {source}"),
    }
}

fn validate_bootstrap_install(
    auth_connector_kind: &str,
    install_source: &str,
    version: &str,
    git_ref: &str,
) -> Result<()> {
    if install_source == "git" && git_ref.is_empty() {
        bail!("bootstrap_git_ref is required when bootstrap_install_source=git");
    }
    if auth_connector_kind == "github" && install_source == "crate" && version == "0.21.0-pre.15" {
        bail!(
            "burn_p2p_bootstrap 0.21.0-pre.15 is not deployable with github auth; use bootstrap_install_source=git with a fixed burn_p2p ref or a newer published crate"
        );
    }
    Ok(())
}

fn append_env_lines(lines: &[String]) -> Result<()> {
    let path = required_env("GITHUB_ENV")?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("failed to open {path}"))?;
    for line in lines {
        writeln!(file, "{line}")?;
    }
    Ok(())
}

fn append_multiline_env(name: &str, value: &str) -> Result<()> {
    let path = required_env("GITHUB_ENV")?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("failed to open {path}"))?;
    writeln!(file, "{name}<<__{name}__")?;
    writeln!(file, "{value}")?;
    writeln!(file, "__{name}__")?;
    Ok(())
}

const DEPLOY_OPTIONAL_ENV: &[(&str, &str)] = &[
    ("TF_VAR_instance_type", "VAR_INSTANCE_TYPE"),
    ("TF_VAR_root_volume_size_gib", "VAR_ROOT_VOLUME_SIZE_GIB"),
    ("TF_VAR_data_volume_size_gib", "VAR_DATA_VOLUME_SIZE_GIB"),
    (
        "TF_VAR_use_retained_bootstrap_data_volume",
        "VAR_USE_RETAINED_BOOTSTRAP_DATA_VOLUME",
    ),
    (
        "TF_VAR_managed_trainer_instance_type",
        "VAR_MANAGED_TRAINER_INSTANCE_TYPE",
    ),
    (
        "TF_VAR_managed_trainer_root_volume_size_gib",
        "VAR_MANAGED_TRAINER_ROOT_VOLUME_SIZE_GIB",
    ),
    (
        "TF_VAR_managed_trainer_min_size",
        "VAR_MANAGED_TRAINER_MIN_SIZE",
    ),
    (
        "TF_VAR_managed_trainer_max_size",
        "VAR_MANAGED_TRAINER_MAX_SIZE",
    ),
    (
        "TF_VAR_enable_data_volume_snapshots",
        "VAR_ENABLE_DATA_VOLUME_SNAPSHOTS",
    ),
    (
        "TF_VAR_data_volume_snapshot_retention_days",
        "VAR_DATA_VOLUME_SNAPSHOT_RETENTION_DAYS",
    ),
    (
        "TF_VAR_enable_bootstrap_status_alarms",
        "VAR_ENABLE_BOOTSTRAP_STATUS_ALARMS",
    ),
    ("TF_VAR_alarm_sns_topic_arn", "VAR_ALARM_SNS_TOPIC_ARN"),
    (
        "TF_VAR_enable_control_plane_operational_alarms",
        "VAR_ENABLE_CONTROL_PLANE_OPERATIONAL_ALARMS",
    ),
    (
        "TF_VAR_enable_control_plane_dashboard",
        "VAR_ENABLE_CONTROL_PLANE_DASHBOARD",
    ),
    (
        "TF_VAR_enable_managed_control_plane_redis",
        "VAR_ENABLE_MANAGED_CONTROL_PLANE_REDIS",
    ),
    ("TF_VAR_artifact_bucket_name", "INPUT_ARTIFACT_BUCKET_NAME"),
    (
        "TF_VAR_artifact_bucket_path_prefix",
        "INPUT_ARTIFACT_BUCKET_PATH_PREFIX",
    ),
    (
        "TF_VAR_artifact_replica_bucket_name",
        "INPUT_ARTIFACT_REPLICA_BUCKET_NAME",
    ),
    (
        "TF_VAR_enable_disaster_recovery_snapshot_copies",
        "VAR_ENABLE_DISASTER_RECOVERY_SNAPSHOT_COPIES",
    ),
    (
        "TF_VAR_disaster_recovery_snapshot_retention_days",
        "VAR_DISASTER_RECOVERY_SNAPSHOT_RETENTION_DAYS",
    ),
    (
        "TF_VAR_artifact_replica_bucket_force_destroy",
        "VAR_ARTIFACT_REPLICA_BUCKET_FORCE_DESTROY",
    ),
    (
        "TF_VAR_artifact_bucket_force_destroy",
        "VAR_ARTIFACT_BUCKET_FORCE_DESTROY",
    ),
    (
        "TF_VAR_artifact_bucket_server_side_encryption",
        "VAR_ARTIFACT_BUCKET_SERVER_SIDE_ENCRYPTION",
    ),
];

const RESTORE_OPTIONAL_ENV: &[(&str, &str)] = &[
    ("TF_VAR_instance_type", "VAR_INSTANCE_TYPE"),
    ("TF_VAR_root_volume_size_gib", "VAR_ROOT_VOLUME_SIZE_GIB"),
    ("TF_VAR_data_volume_size_gib", "VAR_DATA_VOLUME_SIZE_GIB"),
    (
        "TF_VAR_use_retained_bootstrap_data_volume",
        "VAR_USE_RETAINED_BOOTSTRAP_DATA_VOLUME",
    ),
    (
        "TF_VAR_managed_trainer_instance_type",
        "VAR_MANAGED_TRAINER_INSTANCE_TYPE",
    ),
    (
        "TF_VAR_managed_trainer_root_volume_size_gib",
        "VAR_MANAGED_TRAINER_ROOT_VOLUME_SIZE_GIB",
    ),
    (
        "TF_VAR_managed_trainer_min_size",
        "VAR_MANAGED_TRAINER_MIN_SIZE",
    ),
    (
        "TF_VAR_managed_trainer_max_size",
        "VAR_MANAGED_TRAINER_MAX_SIZE",
    ),
    (
        "TF_VAR_enable_data_volume_snapshots",
        "VAR_ENABLE_DATA_VOLUME_SNAPSHOTS",
    ),
    (
        "TF_VAR_data_volume_snapshot_retention_days",
        "VAR_DATA_VOLUME_SNAPSHOT_RETENTION_DAYS",
    ),
    (
        "TF_VAR_enable_disaster_recovery_snapshot_copies",
        "VAR_ENABLE_DISASTER_RECOVERY_SNAPSHOT_COPIES",
    ),
    (
        "TF_VAR_disaster_recovery_snapshot_retention_days",
        "VAR_DISASTER_RECOVERY_SNAPSHOT_RETENTION_DAYS",
    ),
    (
        "TF_VAR_enable_bootstrap_status_alarms",
        "VAR_ENABLE_BOOTSTRAP_STATUS_ALARMS",
    ),
    ("TF_VAR_alarm_sns_topic_arn", "VAR_ALARM_SNS_TOPIC_ARN"),
    (
        "TF_VAR_enable_control_plane_operational_alarms",
        "VAR_ENABLE_CONTROL_PLANE_OPERATIONAL_ALARMS",
    ),
    (
        "TF_VAR_enable_control_plane_dashboard",
        "VAR_ENABLE_CONTROL_PLANE_DASHBOARD",
    ),
    (
        "TF_VAR_enable_managed_control_plane_redis",
        "VAR_ENABLE_MANAGED_CONTROL_PLANE_REDIS",
    ),
    ("TF_VAR_artifact_bucket_name", "INPUT_ARTIFACT_BUCKET_NAME"),
    (
        "TF_VAR_artifact_bucket_path_prefix",
        "INPUT_ARTIFACT_BUCKET_PATH_PREFIX",
    ),
    (
        "TF_VAR_artifact_replica_bucket_name",
        "INPUT_ARTIFACT_REPLICA_BUCKET_NAME",
    ),
    (
        "TF_VAR_artifact_bucket_force_destroy",
        "VAR_ARTIFACT_BUCKET_FORCE_DESTROY",
    ),
    (
        "TF_VAR_artifact_replica_bucket_force_destroy",
        "VAR_ARTIFACT_REPLICA_BUCKET_FORCE_DESTROY",
    ),
    (
        "TF_VAR_artifact_bucket_server_side_encryption",
        "VAR_ARTIFACT_BUCKET_SERVER_SIDE_ENCRYPTION",
    ),
];

fn push_optional_inputs(lines: &mut Vec<String>, pairs: &[(&str, &str)]) {
    for (output, input) in pairs {
        push_optional(lines, output, &env_or(input, ""));
    }
}

fn push_optional(lines: &mut Vec<String>, name: &str, value: &str) {
    if !value.is_empty() {
        lines.push(format!("{name}={value}"));
    }
}

fn workspace_version() -> Result<String> {
    let text = fs::read_to_string("Cargo.toml").context("failed to read Cargo.toml")?;
    let mut in_workspace_package = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            in_workspace_package = trimmed == "[workspace.package]";
            continue;
        }
        if in_workspace_package
            && let Some(raw) = trimmed
                .strip_prefix("version")
                .and_then(|value| value.split_once('=').map(|(_, raw)| raw.trim()))
        {
            return Ok(raw.trim_matches('"').to_owned());
        }
    }
    bail!("missing workspace.package.version in Cargo.toml")
}

fn aws_output(args: &[&str]) -> Result<String> {
    let output = Command::new("aws")
        .args(args)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("failed to start aws {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "aws {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn first_env(names: &[&str], default: &str) -> String {
    names
        .iter()
        .find_map(|name| {
            std::env::var(name)
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| default.to_owned())
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_owned())
}

fn required_env(name: &str) -> Result<String> {
    let value = std::env::var(name).with_context(|| format!("{name} is required"))?;
    ensure!(!value.is_empty(), "{name} is required");
    Ok(value)
}

fn trim_url(value: &str) -> String {
    value.trim_end_matches('/').to_owned()
}

fn url_host(value: &str) -> Option<String> {
    reqwest::Url::parse(value)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
}

fn normalize_lower(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

fn slug(value: &str) -> String {
    let mut out = String::new();
    let mut last_dash = false;
    for ch in value.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    out
}

fn aws_account_id(role_arn: &str) -> String {
    role_arn.split(':').nth(4).unwrap_or_default().to_owned()
}
