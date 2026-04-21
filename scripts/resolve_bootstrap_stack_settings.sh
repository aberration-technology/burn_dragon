#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"

if [ -z "$mode" ]; then
  echo "usage: $0 <deploy|restore>" >&2
  exit 1
fi

resolve_deploy() {
  canonical_stack_name="burn-dragon-p2p-${DEPLOY_ENVIRONMENT}"
  if [ -n "$VAR_STACK_NAME" ] && [ "$VAR_STACK_NAME" != "$canonical_stack_name" ]; then
    echo "BURN_DRAGON_P2P_STACK_NAME is set to '$VAR_STACK_NAME', but managed ${DEPLOY_ENVIRONMENT} deploys require '$canonical_stack_name'." >&2
    echo "Update or remove the environment override before rerunning this workflow to avoid duplicate stacks." >&2
    exit 1
  fi
  stack_name="$canonical_stack_name"

  aws_region="$VAR_AWS_REGION"
  aws_role_arn="$VAR_AWS_ROLE_ARN"
  disaster_recovery_region_input="$INPUT_DISASTER_RECOVERY_REGION"
  disaster_recovery_region_var="$VAR_DISASTER_RECOVERY_REGION"
  edge_domain_name_input="$INPUT_EDGE_DOMAIN_NAME"
  edge_domain_name_var="$VAR_EDGE_DOMAIN_NAME"
  route53_zone_name_input="$INPUT_ROUTE53_ZONE_NAME"
  route53_zone_name_var="$VAR_ROUTE53_ZONE_NAME"
  browser_app_base_url_var="$VAR_BROWSER_APP_BASE_URL"
  auth_redirect_base_url_var="$VAR_AUTH_REDIRECT_BASE_URL"
  acme_contact_email_var="$VAR_ACME_CONTACT_EMAIL"
  browser_app_pages_domain_target_var="$VAR_BROWSER_APP_PAGES_DOMAIN_TARGET"
  network_id="$VAR_NETWORK_ID"
  project_family_id="$VAR_PROJECT_FAMILY_ID"
  study_id="$VAR_STUDY_ID"
  release_train_hash="$VAR_RELEASE_TRAIN_HASH"
  native_target_artifact_hash_var="$VAR_NATIVE_TARGET_ARTIFACT_HASH"
  auth_connector_kind_input="$INPUT_AUTH_CONNECTOR_KIND"
  auth_connector_kind_var="$VAR_AUTH_CONNECTOR_KIND"
  auth_authority_name="$VAR_AUTH_AUTHORITY_NAME"
  auth_principals_json="$VAR_AUTH_PRINCIPALS_JSON"
  auth_authorize_base_url="$VAR_AUTH_AUTHORIZE_BASE_URL"
  auth_exchange_url="$VAR_AUTH_EXCHANGE_URL"
  auth_token_url="$VAR_AUTH_TOKEN_URL"
  auth_api_base_url="$VAR_AUTH_API_BASE_URL"
  auth_userinfo_url="$VAR_AUTH_USERINFO_URL"
  auth_refresh_url="$VAR_AUTH_REFRESH_URL"
  auth_revoke_url="$VAR_AUTH_REVOKE_URL"
  auth_jwks_url="$VAR_AUTH_JWKS_URL"
  auth_oidc_issuer="$VAR_AUTH_OIDC_ISSUER"
  auth_oauth_provider="$VAR_AUTH_OAUTH_PROVIDER"
  auth_external_authority="$VAR_AUTH_EXTERNAL_AUTHORITY"
  auth_external_trusted_principal_header="$VAR_AUTH_EXTERNAL_TRUSTED_PRINCIPAL_HEADER"
  auth_external_trusted_internal_only="$VAR_AUTH_EXTERNAL_TRUSTED_INTERNAL_ONLY"
  github_required_org="$VAR_GITHUB_REQUIRED_ORG"
  github_required_team="$VAR_GITHUB_REQUIRED_TEAM"
  github_required_repo="$VAR_GITHUB_REQUIRED_REPO"
  github_admin_required_repo_permission="$VAR_GITHUB_ADMIN_REQUIRED_REPO_PERMISSION"
  github_admin_logins_input="$INPUT_GITHUB_ADMIN_LOGINS"
  github_admin_logins_var="$VAR_GITHUB_ADMIN_LOGINS"
  climbmix_dataset_base_url_input="$INPUT_CLIMBMIX_BROWSER_DATASET_BASE_URL"
  climbmix_dataset_base_url_var="$VAR_CLIMBMIX_BROWSER_DATASET_BASE_URL"
  dataset_domain_name_var="$VAR_DATASET_DOMAIN_NAME"
  dataset_bucket_name_var="$VAR_DATASET_BUCKET_NAME"
  dataset_bucket_path_prefix_var="$VAR_DATASET_BUCKET_PATH_PREFIX"
  bootstrap_install_source_input="$INPUT_BOOTSTRAP_INSTALL_SOURCE"
  bootstrap_install_source_var="$VAR_BOOTSTRAP_INSTALL_SOURCE"
  bootstrap_version_input="$INPUT_BOOTSTRAP_VERSION"
  bootstrap_version_var="$VAR_BOOTSTRAP_VERSION"
  bootstrap_git_ref_input="$INPUT_BOOTSTRAP_GIT_REF"
  bootstrap_git_ref_var="$VAR_BOOTSTRAP_GIT_REF"
  instance_type="$VAR_INSTANCE_TYPE"
  root_volume_size_gib="$VAR_ROOT_VOLUME_SIZE_GIB"
  data_volume_size_gib="$VAR_DATA_VOLUME_SIZE_GIB"
  use_retained_bootstrap_data_volume="$VAR_USE_RETAINED_BOOTSTRAP_DATA_VOLUME"
  managed_trainer_desired_capacity_input="$INPUT_MANAGED_TRAINER_DESIRED_CAPACITY"
  managed_trainer_desired_capacity_var="$VAR_MANAGED_TRAINER_DESIRED_CAPACITY"
  managed_trainer_backend_input="$INPUT_MANAGED_TRAINER_BACKEND"
  managed_trainer_backend_var="$VAR_MANAGED_TRAINER_BACKEND"
  managed_trainer_experiment_kind_input="$INPUT_MANAGED_TRAINER_EXPERIMENT_KIND"
  managed_trainer_experiment_kind_var="$VAR_MANAGED_TRAINER_EXPERIMENT_KIND"
  managed_trainer_instance_type="$VAR_MANAGED_TRAINER_INSTANCE_TYPE"
  managed_trainer_root_volume_size_gib="$VAR_MANAGED_TRAINER_ROOT_VOLUME_SIZE_GIB"
  managed_trainer_min_size="$VAR_MANAGED_TRAINER_MIN_SIZE"
  managed_trainer_max_size="$VAR_MANAGED_TRAINER_MAX_SIZE"
  managed_trainer_target="$VAR_MANAGED_TRAINER_TARGET"
  managed_trainer_crate_version="$VAR_MANAGED_TRAINER_CRATE_VERSION"
  managed_trainer_auth_bundle_parameter_name="$VAR_MANAGED_TRAINER_AUTH_BUNDLE_PARAMETER_NAME"
  enable_data_volume_snapshots="$VAR_ENABLE_DATA_VOLUME_SNAPSHOTS"
  data_volume_snapshot_retention_days="$VAR_DATA_VOLUME_SNAPSHOT_RETENTION_DAYS"
  enable_bootstrap_status_alarms="$VAR_ENABLE_BOOTSTRAP_STATUS_ALARMS"
  alarm_sns_topic_arn="$VAR_ALARM_SNS_TOPIC_ARN"
  enable_control_plane_operational_alarms="$VAR_ENABLE_CONTROL_PLANE_OPERATIONAL_ALARMS"
  enable_control_plane_dashboard="$VAR_ENABLE_CONTROL_PLANE_DASHBOARD"
  enable_managed_control_plane_redis="$VAR_ENABLE_MANAGED_CONTROL_PLANE_REDIS"
  create_artifact_bucket_input="$INPUT_CREATE_ARTIFACT_BUCKET"
  artifact_bucket_name_input="$INPUT_ARTIFACT_BUCKET_NAME"
  artifact_bucket_name_var="$VAR_ARTIFACT_BUCKET_NAME"
  artifact_bucket_path_prefix_input="$INPUT_ARTIFACT_BUCKET_PATH_PREFIX"
  artifact_bucket_path_prefix_var="$VAR_ARTIFACT_BUCKET_PATH_PREFIX"
  create_artifact_replica_bucket_input="$INPUT_CREATE_ARTIFACT_REPLICA_BUCKET"
  create_artifact_replica_bucket_var="$VAR_CREATE_ARTIFACT_REPLICA_BUCKET"
  artifact_replica_bucket_name_input="$INPUT_ARTIFACT_REPLICA_BUCKET_NAME"
  artifact_replica_bucket_name_var="$VAR_ARTIFACT_REPLICA_BUCKET_NAME"
  artifact_replica_bucket_force_destroy="$VAR_ARTIFACT_REPLICA_BUCKET_FORCE_DESTROY"
  enable_disaster_recovery_snapshot_copies="$VAR_ENABLE_DISASTER_RECOVERY_SNAPSHOT_COPIES"
  disaster_recovery_snapshot_retention_days="$VAR_DISASTER_RECOVERY_SNAPSHOT_RETENTION_DAYS"
  artifact_bucket_force_destroy="$VAR_ARTIFACT_BUCKET_FORCE_DESTROY"
  artifact_bucket_server_side_encryption="$VAR_ARTIFACT_BUCKET_SERVER_SIDE_ENCRYPTION"

  if [ -z "$aws_region" ]; then
    aws_region="us-east-2"
  fi
  : "${aws_role_arn:?missing BURN_DRAGON_P2P_AWS_ROLE_ARN}"
  if [ -z "$network_id" ]; then
    network_id="burn-dragon-mainnet"
  fi
  if [ -z "$project_family_id" ]; then
    project_family_id="burn-dragon-language"
  fi
  if [ -z "$study_id" ]; then
    study_id="burn-dragon-mainnet"
  fi
  if [ -z "$release_train_hash" ]; then
    release_train_hash="burn-dragon-mainnet-train"
  fi

  native_target_artifact_hash="$native_target_artifact_hash_var"
  if [ -z "$native_target_artifact_hash" ]; then
    native_target_artifact_hash="burn-dragon-native"
  fi

  edge_domain_name="$edge_domain_name_input"
  if [ -z "$edge_domain_name" ]; then
    edge_domain_name="$edge_domain_name_var"
  fi
  if [ -z "$edge_domain_name" ]; then
    edge_domain_name="edge.dragon.aberration.technology"
  fi

  route53_zone_name="$route53_zone_name_input"
  if [ -z "$route53_zone_name" ]; then
    route53_zone_name="$route53_zone_name_var"
  fi
  if [ -z "$route53_zone_name" ]; then
    route53_zone_name="aberration.technology"
  fi

  browser_app_base_url="$browser_app_base_url_var"
  if [ -z "$browser_app_base_url" ]; then
    browser_app_base_url="https://dragon.aberration.technology"
  fi
  browser_app_base_url="${browser_app_base_url%/}"

  auth_redirect_base_url="$auth_redirect_base_url_var"
  if [ -z "$auth_redirect_base_url" ] && [ -n "$browser_app_base_url" ]; then
    auth_redirect_base_url="$browser_app_base_url"
  fi
  auth_redirect_base_url="${auth_redirect_base_url%/}"

  acme_contact_email="$acme_contact_email_var"
  if [ -z "$acme_contact_email" ] && [ -n "$route53_zone_name" ]; then
    acme_contact_email="admin@${route53_zone_name%.}"
  fi

  browser_app_pages_domain_target="$browser_app_pages_domain_target_var"
  if [ -z "$browser_app_pages_domain_target" ] && [ -n "$browser_app_base_url" ]; then
    browser_app_pages_domain_target="${GITHUB_REPOSITORY_OWNER_CONTEXT}.github.io"
  fi

  browser_app_host=""
  if [ -n "$browser_app_base_url" ]; then
    browser_app_host="$(python3 -c 'from urllib.parse import urlparse; import sys; print(urlparse(sys.argv[1]).hostname or "")' "$browser_app_base_url")"
  fi

  disaster_recovery_region="$disaster_recovery_region_input"
  if [ -z "$disaster_recovery_region" ]; then
    disaster_recovery_region="$disaster_recovery_region_var"
  fi

  secret_parameter_prefix="/${stack_name}/${TF_WORKSPACE_NAME}/bootstrap"
  auth_connector_kind="$auth_connector_kind_input"
  if [ -z "$auth_connector_kind" ]; then
    auth_connector_kind="$auth_connector_kind_var"
  fi
  if [ -z "$auth_connector_kind" ]; then
    auth_connector_kind="github"
  fi
  auth_connector_kind="$(printf '%s' "$auth_connector_kind" | tr '[:upper:]' '[:lower:]' | xargs)"
  case "$auth_connector_kind" in
    github|oidc|oauth|static|external) ;;
    *)
      echo "unsupported auth connector kind: $auth_connector_kind" >&2
      exit 1
      ;;
  esac
  if [ -z "$auth_authority_name" ]; then
    auth_authority_name="burn-dragon-auth"
  fi
  if [ -z "$auth_principals_json" ]; then
    auth_principals_json="[]"
  fi
  if [ -z "$auth_external_trusted_principal_header" ]; then
    auth_external_trusted_principal_header="x-forwarded-user"
  fi
  if [ -z "$auth_external_trusted_internal_only" ]; then
    auth_external_trusted_internal_only="true"
  fi

  resolved_auth_client_id="$AUTH_CLIENT_ID"
  resolved_auth_client_secret="$AUTH_CLIENT_SECRET"
  if [ "$auth_connector_kind" = "github" ]; then
    if [ -z "$resolved_auth_client_id" ]; then
      resolved_auth_client_id="$GITHUB_CLIENT_ID"
    fi
    if [ -z "$resolved_auth_client_secret" ]; then
      resolved_auth_client_secret="$GITHUB_CLIENT_SECRET"
    fi
  fi

  github_admin_logins="$github_admin_logins_input"
  if [ -z "$github_admin_logins" ]; then
    github_admin_logins="$github_admin_logins_var"
  fi
  if [ "$auth_connector_kind" = "github" ] && [ -z "$github_admin_logins" ]; then
    github_admin_logins="${GITHUB_ACTOR_CONTEXT}"
  fi
  dataset_domain_name="$dataset_domain_name_var"
  if [ -z "$dataset_domain_name" ]; then
    if [ -n "$browser_app_host" ]; then
      dataset_domain_name="datasets.${browser_app_host}"
    else
      dataset_domain_name="datasets.${edge_domain_name}"
    fi
  fi
  dataset_bucket_name="$dataset_bucket_name_var"
  dataset_bucket_path_prefix="$dataset_bucket_path_prefix_var"
  if [ -z "$dataset_bucket_path_prefix" ]; then
    dataset_bucket_path_prefix="dragon-datasets"
  fi

  climbmix_dataset_base_url="$climbmix_dataset_base_url_input"
  if [ -z "$climbmix_dataset_base_url" ]; then
    climbmix_dataset_base_url="$climbmix_dataset_base_url_var"
  fi
  if [ -z "$climbmix_dataset_base_url" ]; then
    climbmix_dataset_base_url="https://${dataset_domain_name}/${dataset_bucket_path_prefix}/climbmix-pretraining/climbmix-r1"
  fi

  bootstrap_install_source="$bootstrap_install_source_input"
  if [ -z "$bootstrap_install_source" ]; then
    bootstrap_install_source="$bootstrap_install_source_var"
  fi
  if [ -z "$bootstrap_install_source" ]; then
    bootstrap_install_source="crate"
  fi
  bootstrap_install_source="$(printf '%s' "$bootstrap_install_source" | tr '[:upper:]' '[:lower:]' | xargs)"
  case "$bootstrap_install_source" in
    crate|git) ;;
    *)
      echo "unsupported bootstrap_install_source: $bootstrap_install_source" >&2
      exit 1
      ;;
  esac

  bootstrap_version="$bootstrap_version_input"
  if [ -z "$bootstrap_version" ]; then
    bootstrap_version="$bootstrap_version_var"
  fi
  if [ -z "$bootstrap_version" ]; then
    bootstrap_version="0.21.0-pre.29"
  fi

  bootstrap_git_ref="$bootstrap_git_ref_input"
  if [ -z "$bootstrap_git_ref" ]; then
    bootstrap_git_ref="$bootstrap_git_ref_var"
  fi
  if [ "$bootstrap_install_source" = "git" ] && [ -z "$bootstrap_git_ref" ]; then
    echo "bootstrap_git_ref is required when bootstrap_install_source=git" >&2
    exit 1
  fi
  if [ "$auth_connector_kind" = "github" ] && [ "$bootstrap_install_source" = "crate" ] && [ "$bootstrap_version" = "0.21.0-pre.15" ]; then
    echo "burn_p2p_bootstrap 0.21.0-pre.15 is not deployable with github auth; use bootstrap_install_source=git with a fixed burn_p2p ref or a newer published crate" >&2
    exit 1
  fi

  dragon_crate_version="$(python3 - <<'PY'
import tomllib
with open("Cargo.toml", "rb") as handle:
    print(tomllib.load(handle)["workspace"]["package"]["version"])
PY
)"

  managed_trainer_desired_capacity="$managed_trainer_desired_capacity_input"
  if [ -z "$managed_trainer_desired_capacity" ]; then
    managed_trainer_desired_capacity="$managed_trainer_desired_capacity_var"
  fi
  if [ -z "$managed_trainer_desired_capacity" ]; then
    managed_trainer_desired_capacity="0"
  fi
  managed_trainer_backend="$managed_trainer_backend_input"
  if [ -z "$managed_trainer_backend" ]; then
    managed_trainer_backend="$managed_trainer_backend_var"
  fi
  if [ -z "$managed_trainer_backend" ]; then
    managed_trainer_backend="cpu"
  fi
  managed_trainer_backend="$(printf '%s' "$managed_trainer_backend" | tr '[:upper:]' '[:lower:]' | xargs)"
  managed_trainer_experiment_kind="$managed_trainer_experiment_kind_input"
  if [ -z "$managed_trainer_experiment_kind" ]; then
    managed_trainer_experiment_kind="$managed_trainer_experiment_kind_var"
  fi
  if [ -z "$managed_trainer_experiment_kind" ]; then
    managed_trainer_experiment_kind="nca"
  fi
  managed_trainer_experiment_kind="$(printf '%s' "$managed_trainer_experiment_kind" | tr '[:upper:]' '[:lower:]' | xargs)"
  if [ -z "$managed_trainer_target" ]; then
    managed_trainer_target="trainer"
  fi
  if [ -z "$managed_trainer_crate_version" ]; then
    managed_trainer_crate_version="$dragon_crate_version"
  fi
  if [ -z "$managed_trainer_auth_bundle_parameter_name" ]; then
    managed_trainer_auth_bundle_parameter_name="${secret_parameter_prefix}/trainer_auth_bundle_json"
  fi
  managed_trainer_experiment_id="nca-prepretraining"
  managed_trainer_revision_id="nca-r1"
  if [ "$managed_trainer_experiment_kind" = "climbmix" ]; then
    managed_trainer_experiment_id="climbmix-pretraining"
    managed_trainer_revision_id="climbmix-r1"
  fi
  managed_trainer_auth_mode="disabled"
  managed_trainer_principal_id=""
  if [ "$managed_trainer_desired_capacity" != "0" ]; then
    managed_trainer_principal_suffix="$(printf '%s-%s-%s' "$TF_WORKSPACE_NAME" "$managed_trainer_experiment_kind" "$managed_trainer_backend" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9-' '-')"
    managed_trainer_principal_id="managed-trainer-${managed_trainer_principal_suffix%-}"
    if [ -n "$TRAINER_AUTH_BUNDLE_JSON" ]; then
      managed_trainer_auth_mode="provided_bundle"
    else
      managed_trainer_auth_mode="bootstrap_static_principal"
      auth_principals_json="$(python3 -c '''import json, sys; principals = json.loads(sys.argv[1] or "[]"); principal_id = sys.argv[2]; experiment_id = sys.argv[3]; network_id = sys.argv[4]; backend = sys.argv[5]; stack_name = sys.argv[6]; environment = sys.argv[7]; trainer_role = "TrainerCpu" if backend == "cpu" else "TrainerGpu"; principal = {"principal_id": principal_id, "display_name": f"burn_dragon managed {experiment_id} {backend} trainer", "org_memberships": [], "group_memberships": [], "granted_roles": {"roles": [trainer_role, "Archive"]}, "granted_scopes": ["Connect", "Discover", {"Train": {"experiment_id": experiment_id}}, {"Archive": {"experiment_id": experiment_id}}], "allowed_networks": [network_id], "custom_claims": {"deployment_profile": environment, "stack": stack_name, "managed_trainer": "true", "managed_trainer_backend": backend, "managed_trainer_experiment_id": experiment_id}}; principals = [item for item in principals if item.get("principal_id") != principal_id]; principals.append(principal); print(json.dumps(principals))''' "$auth_principals_json" "$managed_trainer_principal_id" "$managed_trainer_experiment_id" "$network_id" "$managed_trainer_backend" "$stack_name" "$DEPLOY_ENVIRONMENT")"
    fi
  fi
  dragon_git_repository="https://github.com/${GITHUB_REPOSITORY}.git"
  dragon_git_ref="${GITHUB_SHA}"
  bootstrap_head_mirror_experiment_id="nca-prepretraining"
  bootstrap_head_mirror_revision_id="nca-r1"
  bootstrap_head_mirror_principal_id="bootstrap-head-mirror-${TF_WORKSPACE_NAME}-nca"
  bootstrap_head_mirror_auth_bundle_parameter_name="${secret_parameter_prefix}/bootstrap_head_mirror_auth_bundle_json"
  auth_principals_json="$(python3 -c '''import json, sys; principals = json.loads(sys.argv[1] or "[]"); principal_id = sys.argv[2]; experiment_id = sys.argv[3]; network_id = sys.argv[4]; stack_name = sys.argv[5]; environment = sys.argv[6]; principal = {"principal_id": principal_id, "display_name": "burn_dragon bootstrap nca head mirror", "org_memberships": [], "group_memberships": [], "granted_roles": {"roles": ["TrainerCpu", "Archive"]}, "granted_scopes": ["Connect", "Discover", {"Train": {"experiment_id": experiment_id}}, {"Archive": {"experiment_id": experiment_id}}], "allowed_networks": [network_id], "custom_claims": {"deployment_profile": environment, "stack": stack_name, "bootstrap_head_mirror": "true", "bootstrap_head_mirror_experiment_id": experiment_id, "admin_capabilities": "register_live_head,rollout_auth_policy"}}; principals = [item for item in principals if item.get("principal_id") != principal_id]; principals.append(principal); print(json.dumps(principals))''' "$auth_principals_json" "$bootstrap_head_mirror_principal_id" "$bootstrap_head_mirror_experiment_id" "$network_id" "$stack_name" "$DEPLOY_ENVIRONMENT")"
  browser_canary_experiment_id="nca-prepretraining"
  browser_canary_revision_id="nca-r1"
  browser_canary_principal_id="browser-canary-${TF_WORKSPACE_NAME}-nca"
  auth_principals_json="$(python3 -c '''import json, sys; principals = json.loads(sys.argv[1] or "[]"); principal_id = sys.argv[2]; principals = [item for item in principals if item.get("principal_id") != principal_id]; print(json.dumps(principals))''' "$auth_principals_json" "$browser_canary_principal_id")"

  create_artifact_bucket="$create_artifact_bucket_input"
  artifact_bucket_name="$artifact_bucket_name_input"
  if [ -z "$artifact_bucket_name" ]; then
    artifact_bucket_name="$artifact_bucket_name_var"
  fi
  artifact_bucket_path_prefix="$artifact_bucket_path_prefix_input"
  if [ -z "$artifact_bucket_path_prefix" ]; then
    artifact_bucket_path_prefix="$artifact_bucket_path_prefix_var"
  fi
  create_artifact_replica_bucket="$create_artifact_replica_bucket_input"
  if [ -z "$create_artifact_replica_bucket" ]; then
    create_artifact_replica_bucket="$create_artifact_replica_bucket_var"
  fi
  if [ -z "$create_artifact_replica_bucket" ]; then
    create_artifact_replica_bucket="true"
  fi
  artifact_replica_bucket_name="$artifact_replica_bucket_name_input"
  if [ -z "$artifact_replica_bucket_name" ]; then
    artifact_replica_bucket_name="$artifact_replica_bucket_name_var"
  fi
  if [ "$create_artifact_bucket" != "true" ] && [ -z "$artifact_bucket_name" ]; then
    echo "artifact_bucket_name is required when create_artifact_bucket=false" >&2
    exit 1
  fi
  if [ -n "$disaster_recovery_region" ] && [ "$create_artifact_replica_bucket" != "true" ] && [ -z "$artifact_replica_bucket_name" ]; then
    echo "artifact_replica_bucket_name is required when disaster_recovery_region is set and create_artifact_replica_bucket=false" >&2
    exit 1
  fi

  admin_logins_json="$(
    python3 -c 'import json, sys; seen=set(); values=[]; [values.append(login) for token in sys.argv[1].replace("\n", ",").split(",") for login in [token.strip().lower()] if login and login not in seen and not seen.add(login)]; print(json.dumps(values))' "$github_admin_logins"
  )"
  admin_logins_csv="$(
    python3 -c 'import json, sys; print(",".join(json.loads(sys.argv[1])))' "$admin_logins_json"
  )"

  if [ "$auth_connector_kind" = "github" ]; then
    if [ -z "$github_required_repo" ]; then
      github_required_repo="mosure/burn_dragon"
    fi
    : "${admin_logins_csv:?no admin logins resolved for github deployment}"
    : "${resolved_auth_client_id:?missing BURN_DRAGON_P2P_AUTH_CLIENT_ID or BURN_DRAGON_P2P_GITHUB_CLIENT_ID secret}"
    : "${resolved_auth_client_secret:?missing BURN_DRAGON_P2P_AUTH_CLIENT_SECRET or BURN_DRAGON_P2P_GITHUB_CLIENT_SECRET secret}"
    : "${BROWSER_CANARY_CALLBACK_TOKEN:?missing BURN_DRAGON_P2P_BROWSER_CANARY_CALLBACK_TOKEN secret}"
  elif [ "$auth_connector_kind" = "oidc" ] || [ "$auth_connector_kind" = "oauth" ]; then
    : "${resolved_auth_client_id:?missing BURN_DRAGON_P2P_AUTH_CLIENT_ID secret}"
    : "${resolved_auth_client_secret:?missing BURN_DRAGON_P2P_AUTH_CLIENT_SECRET secret}"
  fi

  if [ "$auth_connector_kind" = "oidc" ] && [ -z "$auth_oidc_issuer" ]; then
    echo "missing BURN_DRAGON_P2P_AUTH_OIDC_ISSUER for oidc deployment" >&2
    exit 1
  fi
  if [ "$auth_connector_kind" = "oauth" ] && [ -z "$auth_oauth_provider" ]; then
    echo "missing BURN_DRAGON_P2P_AUTH_OAUTH_PROVIDER for oauth deployment" >&2
    exit 1
  fi
  if [ "$auth_connector_kind" = "external" ] && [ -z "$auth_external_authority" ]; then
    echo "missing BURN_DRAGON_P2P_AUTH_EXTERNAL_AUTHORITY for external deployment" >&2
    exit 1
  fi

  if [ "$auth_connector_kind" != "github" ] && [ -n "$github_admin_logins_input$github_admin_logins_var" ]; then
    echo "github_admin_logins are ignored because auth_connector_kind=$auth_connector_kind"
  fi

  {
    echo "AWS_REGION=$aws_region"
    echo "AWS_ROLE_ARN=$aws_role_arn"
    echo "AWS_ACCOUNT_ID=$(cut -d: -f5 <<<"$aws_role_arn")"
    echo "STACK_NAME=$stack_name"
    echo "EDGE_DOMAIN_NAME=$edge_domain_name"
    echo "BROWSER_APP_BASE_URL=$browser_app_base_url"
    echo "AUTH_REDIRECT_BASE_URL=$auth_redirect_base_url"
    echo "ACME_CONTACT_EMAIL=$acme_contact_email"
    echo "BROWSER_APP_PAGES_DOMAIN_TARGET=$browser_app_pages_domain_target"
    echo "ROUTE53_ZONE_NAME=$route53_zone_name"
    echo "SECRET_PARAMETER_PREFIX=$secret_parameter_prefix"
    echo "TF_VAR_aws_region=$aws_region"
    echo "TF_VAR_stack_name=$stack_name"
    echo "TF_VAR_environment_name=${DEPLOY_ENVIRONMENT}"
    echo "TF_VAR_route53_zone_name=$route53_zone_name"
    echo "TF_VAR_edge_domain_name=$edge_domain_name"
    echo "TF_VAR_browser_app_base_url=$browser_app_base_url"
    echo "TF_VAR_auth_redirect_base_url=$auth_redirect_base_url"
    echo "TF_VAR_acme_contact_email=$acme_contact_email"
    echo "TF_VAR_browser_app_pages_domain_target=$browser_app_pages_domain_target"
    echo "TF_VAR_secret_parameter_prefix=$secret_parameter_prefix"
    echo "TF_VAR_network_id=$network_id"
    echo "TF_VAR_project_family_id=$project_family_id"
    echo "TF_VAR_study_id=$study_id"
    echo "TF_VAR_release_train_hash=$release_train_hash"
    echo "TF_VAR_bootstrap_install_source=$bootstrap_install_source"
    echo "TF_VAR_bootstrap_crate_version=$bootstrap_version"
    echo "TF_VAR_bootstrap_git_ref=$bootstrap_git_ref"
    echo "TF_VAR_dragon_crate_version=$dragon_crate_version"
    echo "TF_VAR_dragon_git_repository=$dragon_git_repository"
    echo "TF_VAR_dragon_git_ref=$dragon_git_ref"
    echo "TF_VAR_bootstrap_head_mirror_auth_bundle_parameter_name=$bootstrap_head_mirror_auth_bundle_parameter_name"
    echo "TF_VAR_create_artifact_bucket=$create_artifact_bucket"
    echo "DATASET_DOMAIN_NAME=$dataset_domain_name"
    echo "DATASET_BUCKET_PATH_PREFIX=$dataset_bucket_path_prefix"
    echo "TF_VAR_dataset_domain_name=$dataset_domain_name"
    echo "TF_VAR_dataset_bucket_path_prefix=$dataset_bucket_path_prefix"
    echo "TF_VAR_managed_trainer_desired_capacity=$managed_trainer_desired_capacity"
    echo "TF_VAR_managed_trainer_backend=$managed_trainer_backend"
    echo "TF_VAR_managed_trainer_experiment_kind=$managed_trainer_experiment_kind"
    echo "TF_VAR_managed_trainer_target=$managed_trainer_target"
    echo "TF_VAR_managed_trainer_crate_version=$managed_trainer_crate_version"
    echo "TF_VAR_managed_trainer_auth_bundle_parameter_name=$managed_trainer_auth_bundle_parameter_name"
    echo "TF_VAR_native_target_artifact_hash=$native_target_artifact_hash"
    echo "TF_VAR_create_artifact_replica_bucket=$create_artifact_replica_bucket"
    echo "AUTH_CONNECTOR_KIND=$auth_connector_kind"
    echo "MANAGED_TRAINER_AUTH_MODE=$managed_trainer_auth_mode"
    echo "MANAGED_TRAINER_PRINCIPAL_ID=$managed_trainer_principal_id"
    echo "MANAGED_TRAINER_EXPERIMENT_ID=$managed_trainer_experiment_id"
    echo "MANAGED_TRAINER_REVISION_ID=$managed_trainer_revision_id"
    echo "BOOTSTRAP_HEAD_MIRROR_PRINCIPAL_ID=$bootstrap_head_mirror_principal_id"
    echo "BOOTSTRAP_HEAD_MIRROR_EXPERIMENT_ID=$bootstrap_head_mirror_experiment_id"
    echo "BOOTSTRAP_HEAD_MIRROR_REVISION_ID=$bootstrap_head_mirror_revision_id"
    echo "BROWSER_CANARY_PRINCIPAL_ID=$browser_canary_principal_id"
    echo "BROWSER_CANARY_EXPERIMENT_ID=$browser_canary_experiment_id"
    echo "BROWSER_CANARY_REVISION_ID=$browser_canary_revision_id"
    echo "TF_VAR_auth_connector_kind=$auth_connector_kind"
    echo "TF_VAR_auth_authority_name=$auth_authority_name"
  } >>"$GITHUB_ENV"
  {
    echo "TF_VAR_auth_principals_json<<__AUTH_PRINCIPALS_JSON__"
    printf '%s\n' "$auth_principals_json"
    echo "__AUTH_PRINCIPALS_JSON__"
  } >>"$GITHUB_ENV"

  if [ "$auth_connector_kind" = "github" ]; then
    {
      echo "TF_VAR_github_required_org=$github_required_org"
      echo "TF_VAR_github_required_team=$github_required_team"
      echo "TF_VAR_github_required_repo=$github_required_repo"
      echo "TF_VAR_github_browser_canary_principal_id=$browser_canary_principal_id"
      echo "TF_VAR_github_browser_canary_callback_token=$BROWSER_CANARY_CALLBACK_TOKEN"
      echo "TF_VAR_github_admin_logins=$admin_logins_json"
      echo "ADMIN_LOGINS_CSV=$admin_logins_csv"
    } >>"$GITHUB_ENV"
  fi

  if [ -n "$auth_authorize_base_url" ]; then echo "TF_VAR_auth_authorize_base_url=$auth_authorize_base_url" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_exchange_url" ]; then echo "TF_VAR_auth_exchange_url=$auth_exchange_url" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_token_url" ]; then echo "TF_VAR_auth_token_url=$auth_token_url" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_api_base_url" ]; then echo "TF_VAR_auth_api_base_url=$auth_api_base_url" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_userinfo_url" ]; then echo "TF_VAR_auth_userinfo_url=$auth_userinfo_url" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_refresh_url" ]; then echo "TF_VAR_auth_refresh_url=$auth_refresh_url" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_revoke_url" ]; then echo "TF_VAR_auth_revoke_url=$auth_revoke_url" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_jwks_url" ]; then echo "TF_VAR_auth_jwks_url=$auth_jwks_url" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_oidc_issuer" ]; then echo "TF_VAR_auth_oidc_issuer=$auth_oidc_issuer" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_oauth_provider" ]; then echo "TF_VAR_auth_oauth_provider=$auth_oauth_provider" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_external_authority" ]; then echo "TF_VAR_auth_external_authority=$auth_external_authority" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_external_trusted_principal_header" ]; then echo "TF_VAR_auth_external_trusted_principal_header=$auth_external_trusted_principal_header" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_external_trusted_internal_only" ]; then echo "TF_VAR_auth_external_trusted_internal_only=$auth_external_trusted_internal_only" >>"$GITHUB_ENV"; fi
  if [ -n "$github_admin_required_repo_permission" ]; then
    echo "TF_VAR_github_admin_required_repo_permission=$github_admin_required_repo_permission" >>"$GITHUB_ENV"
  fi
  if [ "$auth_connector_kind" = "github" ] || [ "$auth_connector_kind" = "oidc" ] || [ "$auth_connector_kind" = "oauth" ]; then
    {
      echo "AUTH_CLIENT_ID_RESOLVED=$resolved_auth_client_id"
      echo "AUTH_CLIENT_SECRET_RESOLVED=$resolved_auth_client_secret"
    } >>"$GITHUB_ENV"
  fi

  echo "TF_VAR_climbmix_browser_dataset_base_url=$climbmix_dataset_base_url" >>"$GITHUB_ENV"

  if [ -n "$instance_type" ]; then
    echo "TF_VAR_instance_type=$instance_type" >>"$GITHUB_ENV"
  fi
  if [ -n "$root_volume_size_gib" ]; then
    echo "TF_VAR_root_volume_size_gib=$root_volume_size_gib" >>"$GITHUB_ENV"
  fi
  if [ -n "$data_volume_size_gib" ]; then
    echo "TF_VAR_data_volume_size_gib=$data_volume_size_gib" >>"$GITHUB_ENV"
  fi
  if [ -n "$use_retained_bootstrap_data_volume" ]; then
    echo "TF_VAR_use_retained_bootstrap_data_volume=$use_retained_bootstrap_data_volume" >>"$GITHUB_ENV"
  fi
  if [ -n "$managed_trainer_instance_type" ]; then
    echo "TF_VAR_managed_trainer_instance_type=$managed_trainer_instance_type" >>"$GITHUB_ENV"
  fi
  if [ -n "$managed_trainer_root_volume_size_gib" ]; then
    echo "TF_VAR_managed_trainer_root_volume_size_gib=$managed_trainer_root_volume_size_gib" >>"$GITHUB_ENV"
  fi
  if [ -n "$managed_trainer_min_size" ]; then
    echo "TF_VAR_managed_trainer_min_size=$managed_trainer_min_size" >>"$GITHUB_ENV"
  fi
  if [ -n "$managed_trainer_max_size" ]; then
    echo "TF_VAR_managed_trainer_max_size=$managed_trainer_max_size" >>"$GITHUB_ENV"
  fi
  if [ -n "$enable_data_volume_snapshots" ]; then
    echo "TF_VAR_enable_data_volume_snapshots=$enable_data_volume_snapshots" >>"$GITHUB_ENV"
  fi
  if [ -n "$data_volume_snapshot_retention_days" ]; then
    echo "TF_VAR_data_volume_snapshot_retention_days=$data_volume_snapshot_retention_days" >>"$GITHUB_ENV"
  fi
  if [ -n "$enable_bootstrap_status_alarms" ]; then
    echo "TF_VAR_enable_bootstrap_status_alarms=$enable_bootstrap_status_alarms" >>"$GITHUB_ENV"
  fi
  if [ -n "$alarm_sns_topic_arn" ]; then
    echo "TF_VAR_alarm_sns_topic_arn=$alarm_sns_topic_arn" >>"$GITHUB_ENV"
  fi
  if [ -n "$enable_control_plane_operational_alarms" ]; then
    echo "TF_VAR_enable_control_plane_operational_alarms=$enable_control_plane_operational_alarms" >>"$GITHUB_ENV"
  fi
  if [ -n "$enable_control_plane_dashboard" ]; then
    echo "TF_VAR_enable_control_plane_dashboard=$enable_control_plane_dashboard" >>"$GITHUB_ENV"
  fi
  if [ -n "$enable_managed_control_plane_redis" ]; then
    echo "TF_VAR_enable_managed_control_plane_redis=$enable_managed_control_plane_redis" >>"$GITHUB_ENV"
  fi
  if [ -n "$artifact_bucket_name" ]; then
    echo "TF_VAR_artifact_bucket_name=$artifact_bucket_name" >>"$GITHUB_ENV"
  fi
  if [ -n "$dataset_bucket_name" ]; then
    echo "TF_VAR_dataset_bucket_name=$dataset_bucket_name" >>"$GITHUB_ENV"
  fi
  if [ -n "$artifact_bucket_path_prefix" ]; then
    echo "TF_VAR_artifact_bucket_path_prefix=$artifact_bucket_path_prefix" >>"$GITHUB_ENV"
  fi
  if [ -n "$disaster_recovery_region" ]; then
    echo "TF_VAR_disaster_recovery_region=$disaster_recovery_region" >>"$GITHUB_ENV"
  fi
  if [ -n "$artifact_replica_bucket_name" ]; then
    echo "TF_VAR_artifact_replica_bucket_name=$artifact_replica_bucket_name" >>"$GITHUB_ENV"
  fi
  if [ -n "$enable_disaster_recovery_snapshot_copies" ]; then
    echo "TF_VAR_enable_disaster_recovery_snapshot_copies=$enable_disaster_recovery_snapshot_copies" >>"$GITHUB_ENV"
  fi
  if [ -n "$disaster_recovery_snapshot_retention_days" ]; then
    echo "TF_VAR_disaster_recovery_snapshot_retention_days=$disaster_recovery_snapshot_retention_days" >>"$GITHUB_ENV"
  fi
  if [ -n "$artifact_replica_bucket_force_destroy" ]; then
    echo "TF_VAR_artifact_replica_bucket_force_destroy=$artifact_replica_bucket_force_destroy" >>"$GITHUB_ENV"
  fi
  if [ -n "$artifact_bucket_force_destroy" ]; then
    echo "TF_VAR_artifact_bucket_force_destroy=$artifact_bucket_force_destroy" >>"$GITHUB_ENV"
  fi
  if [ -n "$artifact_bucket_server_side_encryption" ]; then
    echo "TF_VAR_artifact_bucket_server_side_encryption=$artifact_bucket_server_side_encryption" >>"$GITHUB_ENV"
  fi
}

resolve_restore() {
  canonical_stack_name="burn-dragon-p2p-${DEPLOY_ENVIRONMENT}"
  if [ -n "$VAR_STACK_NAME" ] && [ "$VAR_STACK_NAME" != "$canonical_stack_name" ]; then
    echo "BURN_DRAGON_P2P_STACK_NAME is set to '$VAR_STACK_NAME', but managed ${DEPLOY_ENVIRONMENT} restores require '$canonical_stack_name'." >&2
    echo "Update or remove the environment override before rerunning this workflow to avoid duplicate stacks." >&2
    exit 1
  fi
  stack_name="$canonical_stack_name"

  aws_region_input="$INPUT_AWS_REGION"
  aws_region_var="$VAR_AWS_REGION"
  aws_role_arn="$VAR_AWS_ROLE_ARN"
  edge_domain_name_input="$INPUT_EDGE_DOMAIN_NAME"
  edge_domain_name_var="$VAR_EDGE_DOMAIN_NAME"
  route53_zone_name_input="$INPUT_ROUTE53_ZONE_NAME"
  route53_zone_name_var="$VAR_ROUTE53_ZONE_NAME"
  browser_app_base_url_var="$VAR_BROWSER_APP_BASE_URL"
  auth_redirect_base_url_var="$VAR_AUTH_REDIRECT_BASE_URL"
  acme_contact_email_var="$VAR_ACME_CONTACT_EMAIL"
  browser_app_pages_domain_target_var="$VAR_BROWSER_APP_PAGES_DOMAIN_TARGET"
  network_id="$VAR_NETWORK_ID"
  project_family_id="$VAR_PROJECT_FAMILY_ID"
  study_id="$VAR_STUDY_ID"
  release_train_hash="$VAR_RELEASE_TRAIN_HASH"
  native_target_artifact_hash_var="$VAR_NATIVE_TARGET_ARTIFACT_HASH"
  auth_connector_kind="$VAR_AUTH_CONNECTOR_KIND"
  auth_authority_name="$VAR_AUTH_AUTHORITY_NAME"
  auth_principals_json="$VAR_AUTH_PRINCIPALS_JSON"
  auth_authorize_base_url="$VAR_AUTH_AUTHORIZE_BASE_URL"
  auth_exchange_url="$VAR_AUTH_EXCHANGE_URL"
  auth_token_url="$VAR_AUTH_TOKEN_URL"
  auth_api_base_url="$VAR_AUTH_API_BASE_URL"
  auth_userinfo_url="$VAR_AUTH_USERINFO_URL"
  auth_refresh_url="$VAR_AUTH_REFRESH_URL"
  auth_revoke_url="$VAR_AUTH_REVOKE_URL"
  auth_jwks_url="$VAR_AUTH_JWKS_URL"
  auth_oidc_issuer="$VAR_AUTH_OIDC_ISSUER"
  auth_oauth_provider="$VAR_AUTH_OAUTH_PROVIDER"
  auth_external_authority="$VAR_AUTH_EXTERNAL_AUTHORITY"
  auth_external_trusted_principal_header="$VAR_AUTH_EXTERNAL_TRUSTED_PRINCIPAL_HEADER"
  auth_external_trusted_internal_only="$VAR_AUTH_EXTERNAL_TRUSTED_INTERNAL_ONLY"
  github_required_org="$VAR_GITHUB_REQUIRED_ORG"
  github_required_team="$VAR_GITHUB_REQUIRED_TEAM"
  github_required_repo="$VAR_GITHUB_REQUIRED_REPO"
  github_admin_required_repo_permission="$VAR_GITHUB_ADMIN_REQUIRED_REPO_PERMISSION"
  github_admin_logins="$VAR_GITHUB_ADMIN_LOGINS"
  climbmix_dataset_base_url="$VAR_CLIMBMIX_BROWSER_DATASET_BASE_URL"
  dataset_domain_name_var="$VAR_DATASET_DOMAIN_NAME"
  dataset_bucket_name_var="$VAR_DATASET_BUCKET_NAME"
  dataset_bucket_path_prefix_var="$VAR_DATASET_BUCKET_PATH_PREFIX"
  bootstrap_install_source_input="$INPUT_BOOTSTRAP_INSTALL_SOURCE"
  bootstrap_install_source_var="$VAR_BOOTSTRAP_INSTALL_SOURCE"
  bootstrap_version_input="$INPUT_BOOTSTRAP_VERSION"
  bootstrap_version_var="$VAR_BOOTSTRAP_VERSION"
  bootstrap_git_ref_input="$INPUT_BOOTSTRAP_GIT_REF"
  bootstrap_git_ref_var="$VAR_BOOTSTRAP_GIT_REF"
  instance_type="$VAR_INSTANCE_TYPE"
  root_volume_size_gib="$VAR_ROOT_VOLUME_SIZE_GIB"
  data_volume_size_gib="$VAR_DATA_VOLUME_SIZE_GIB"
  use_retained_bootstrap_data_volume="$VAR_USE_RETAINED_BOOTSTRAP_DATA_VOLUME"
  managed_trainer_desired_capacity="$VAR_MANAGED_TRAINER_DESIRED_CAPACITY"
  managed_trainer_backend="$VAR_MANAGED_TRAINER_BACKEND"
  managed_trainer_experiment_kind="$VAR_MANAGED_TRAINER_EXPERIMENT_KIND"
  managed_trainer_instance_type="$VAR_MANAGED_TRAINER_INSTANCE_TYPE"
  managed_trainer_root_volume_size_gib="$VAR_MANAGED_TRAINER_ROOT_VOLUME_SIZE_GIB"
  managed_trainer_min_size="$VAR_MANAGED_TRAINER_MIN_SIZE"
  managed_trainer_max_size="$VAR_MANAGED_TRAINER_MAX_SIZE"
  managed_trainer_target="$VAR_MANAGED_TRAINER_TARGET"
  managed_trainer_crate_version="$VAR_MANAGED_TRAINER_CRATE_VERSION"
  managed_trainer_auth_bundle_parameter_name="$VAR_MANAGED_TRAINER_AUTH_BUNDLE_PARAMETER_NAME"
  enable_data_volume_snapshots="$VAR_ENABLE_DATA_VOLUME_SNAPSHOTS"
  data_volume_snapshot_retention_days="$VAR_DATA_VOLUME_SNAPSHOT_RETENTION_DAYS"
  enable_disaster_recovery_snapshot_copies="$VAR_ENABLE_DISASTER_RECOVERY_SNAPSHOT_COPIES"
  disaster_recovery_snapshot_retention_days="$VAR_DISASTER_RECOVERY_SNAPSHOT_RETENTION_DAYS"
  enable_bootstrap_status_alarms="$VAR_ENABLE_BOOTSTRAP_STATUS_ALARMS"
  alarm_sns_topic_arn="$VAR_ALARM_SNS_TOPIC_ARN"
  enable_control_plane_operational_alarms="$VAR_ENABLE_CONTROL_PLANE_OPERATIONAL_ALARMS"
  enable_control_plane_dashboard="$VAR_ENABLE_CONTROL_PLANE_DASHBOARD"
  enable_managed_control_plane_redis="$VAR_ENABLE_MANAGED_CONTROL_PLANE_REDIS"
  create_artifact_bucket="$INPUT_CREATE_ARTIFACT_BUCKET"
  artifact_bucket_name="$INPUT_ARTIFACT_BUCKET_NAME"
  artifact_bucket_name_var="$VAR_ARTIFACT_BUCKET_NAME"
  artifact_bucket_path_prefix="$INPUT_ARTIFACT_BUCKET_PATH_PREFIX"
  artifact_bucket_path_prefix_var="$VAR_ARTIFACT_BUCKET_PATH_PREFIX"
  create_artifact_replica_bucket="$INPUT_CREATE_ARTIFACT_REPLICA_BUCKET"
  artifact_replica_bucket_name="$INPUT_ARTIFACT_REPLICA_BUCKET_NAME"
  artifact_replica_bucket_name_var="$VAR_ARTIFACT_REPLICA_BUCKET_NAME"
  artifact_bucket_force_destroy="$VAR_ARTIFACT_BUCKET_FORCE_DESTROY"
  artifact_replica_bucket_force_destroy="$VAR_ARTIFACT_REPLICA_BUCKET_FORCE_DESTROY"
  artifact_bucket_server_side_encryption="$VAR_ARTIFACT_BUCKET_SERVER_SIDE_ENCRYPTION"
  restore_from_latest_snapshots="$INPUT_RESTORE_FROM_LATEST_SNAPSHOTS"
  source_snapshot_region_input="$INPUT_SOURCE_SNAPSHOT_REGION"
  primary_restore_snapshot_id="$INPUT_BOOTSTRAP_PRIMARY_RESTORE_SNAPSHOT_ID"
  next_disaster_recovery_region="$INPUT_NEXT_DISASTER_RECOVERY_REGION"

  aws_region="$aws_region_input"
  if [ -z "$aws_region" ]; then
    aws_region="$aws_region_var"
  fi

  if [ -z "$aws_region" ]; then
    aws_region="us-east-2"
  fi
  : "${aws_role_arn:?missing BURN_DRAGON_P2P_AWS_ROLE_ARN}"
  if [ -z "$network_id" ]; then
    network_id="burn-dragon-mainnet"
  fi
  if [ -z "$project_family_id" ]; then
    project_family_id="burn-dragon-language"
  fi
  if [ -z "$study_id" ]; then
    study_id="burn-dragon-mainnet"
  fi
  if [ -z "$release_train_hash" ]; then
    release_train_hash="burn-dragon-mainnet-train"
  fi

  native_target_artifact_hash="$native_target_artifact_hash_var"
  if [ -z "$native_target_artifact_hash" ]; then
    native_target_artifact_hash="burn-dragon-native"
  fi

  edge_domain_name="$edge_domain_name_input"
  if [ -z "$edge_domain_name" ]; then
    edge_domain_name="$edge_domain_name_var"
  fi
  if [ -z "$edge_domain_name" ]; then
    edge_domain_name="edge.dragon.aberration.technology"
  fi

  route53_zone_name="$route53_zone_name_input"
  if [ -z "$route53_zone_name" ]; then
    route53_zone_name="$route53_zone_name_var"
  fi
  if [ -z "$route53_zone_name" ]; then
    route53_zone_name="aberration.technology"
  fi

  browser_app_base_url="$browser_app_base_url_var"
  if [ -z "$browser_app_base_url" ]; then
    browser_app_base_url="https://dragon.aberration.technology"
  fi
  browser_app_base_url="${browser_app_base_url%/}"

  auth_redirect_base_url="$auth_redirect_base_url_var"
  if [ -z "$auth_redirect_base_url" ] && [ -n "$browser_app_base_url" ]; then
    auth_redirect_base_url="$browser_app_base_url"
  fi
  auth_redirect_base_url="${auth_redirect_base_url%/}"

  acme_contact_email="$acme_contact_email_var"
  if [ -z "$acme_contact_email" ] && [ -n "$route53_zone_name" ]; then
    acme_contact_email="admin@${route53_zone_name%.}"
  fi

  browser_app_pages_domain_target="$browser_app_pages_domain_target_var"
  if [ -z "$browser_app_pages_domain_target" ] && [ -n "$browser_app_base_url" ]; then
    browser_app_pages_domain_target="${GITHUB_REPOSITORY_OWNER_CONTEXT}.github.io"
  fi

  browser_app_host=""
  if [ -n "$browser_app_base_url" ]; then
    browser_app_host="$(python3 -c 'from urllib.parse import urlparse; import sys; print(urlparse(sys.argv[1]).hostname or "")' "$browser_app_base_url")"
  fi

  if [ -z "$auth_connector_kind" ]; then
    auth_connector_kind="github"
  fi
  auth_connector_kind="$(printf '%s' "$auth_connector_kind" | tr '[:upper:]' '[:lower:]' | xargs)"
  if [ -z "$auth_authority_name" ]; then
    auth_authority_name="burn-dragon-auth"
  fi
  if [ -z "$auth_principals_json" ]; then
    auth_principals_json="[]"
  fi
  if [ -z "$auth_external_trusted_principal_header" ]; then
    auth_external_trusted_principal_header="x-forwarded-user"
  fi
  if [ -z "$auth_external_trusted_internal_only" ]; then
    auth_external_trusted_internal_only="true"
  fi
  if [ -z "$github_admin_logins" ] && [ "$auth_connector_kind" = "github" ]; then
    github_admin_logins="$GITHUB_ACTOR_CONTEXT"
  fi
  dataset_domain_name="$dataset_domain_name_var"
  if [ -z "$dataset_domain_name" ]; then
    if [ -n "$browser_app_host" ]; then
      dataset_domain_name="datasets.${browser_app_host}"
    else
      dataset_domain_name="datasets.${edge_domain_name}"
    fi
  fi
  dataset_bucket_name="$dataset_bucket_name_var"
  dataset_bucket_path_prefix="$dataset_bucket_path_prefix_var"
  if [ -z "$dataset_bucket_path_prefix" ]; then
    dataset_bucket_path_prefix="dragon-datasets"
  fi
  if [ -z "$climbmix_dataset_base_url" ]; then
    climbmix_dataset_base_url="https://${dataset_domain_name}/${dataset_bucket_path_prefix}/climbmix-pretraining/climbmix-r1"
  fi
  bootstrap_install_source="$bootstrap_install_source_input"
  if [ -z "$bootstrap_install_source" ]; then
    bootstrap_install_source="$bootstrap_install_source_var"
  fi
  if [ -z "$bootstrap_install_source" ]; then
    bootstrap_install_source="crate"
  fi
  bootstrap_install_source="$(printf '%s' "$bootstrap_install_source" | tr '[:upper:]' '[:lower:]' | xargs)"
  case "$bootstrap_install_source" in
    crate|git) ;;
    *)
      echo "unsupported bootstrap_install_source: $bootstrap_install_source" >&2
      exit 1
      ;;
  esac

  bootstrap_version="$bootstrap_version_input"
  if [ -z "$bootstrap_version" ]; then
    bootstrap_version="$bootstrap_version_var"
  fi
  if [ -z "$bootstrap_version" ]; then
    bootstrap_version="0.21.0-pre.29"
  fi

  bootstrap_git_ref="$bootstrap_git_ref_input"
  if [ -z "$bootstrap_git_ref" ]; then
    bootstrap_git_ref="$bootstrap_git_ref_var"
  fi
  if [ "$bootstrap_install_source" = "git" ] && [ -z "$bootstrap_git_ref" ]; then
    echo "bootstrap_git_ref is required when bootstrap_install_source=git" >&2
    exit 1
  fi
  if [ "$auth_connector_kind" = "github" ] && [ "$bootstrap_install_source" = "crate" ] && [ "$bootstrap_version" = "0.21.0-pre.15" ]; then
    echo "burn_p2p_bootstrap 0.21.0-pre.15 is not deployable with github auth; use bootstrap_install_source=git with a fixed burn_p2p ref or a newer published crate" >&2
    exit 1
  fi
  dragon_crate_version="$(python3 - <<'PY'
import tomllib
with open("Cargo.toml", "rb") as handle:
    print(tomllib.load(handle)["workspace"]["package"]["version"])
PY
)"
  if [ -z "$managed_trainer_desired_capacity" ]; then
    managed_trainer_desired_capacity="0"
  fi
  if [ -z "$managed_trainer_backend" ]; then
    managed_trainer_backend="cpu"
  fi
  if [ -z "$managed_trainer_experiment_kind" ]; then
    managed_trainer_experiment_kind="nca"
  fi
  if [ -z "$managed_trainer_target" ]; then
    managed_trainer_target="trainer"
  fi
  if [ -z "$managed_trainer_crate_version" ]; then
    managed_trainer_crate_version="$dragon_crate_version"
  fi
  if [ -z "$managed_trainer_auth_bundle_parameter_name" ]; then
    managed_trainer_auth_bundle_parameter_name="/${stack_name}/${TF_WORKSPACE_NAME}/bootstrap/trainer_auth_bundle_json"
  fi
  if [ -z "$use_retained_bootstrap_data_volume" ]; then
    use_retained_bootstrap_data_volume="false"
  fi
  if [ -z "$enable_managed_control_plane_redis" ]; then
    enable_managed_control_plane_redis="false"
  fi
  managed_trainer_experiment_id="nca-prepretraining"
  managed_trainer_revision_id="nca-r1"
  if [ "$managed_trainer_experiment_kind" = "climbmix" ]; then
    managed_trainer_experiment_id="climbmix-pretraining"
    managed_trainer_revision_id="climbmix-r1"
  fi
  managed_trainer_auth_mode="disabled"
  managed_trainer_principal_id=""
  if [ "$managed_trainer_desired_capacity" != "0" ]; then
    managed_trainer_principal_suffix="$(printf '%s-%s-%s' "$TF_WORKSPACE_NAME" "$managed_trainer_experiment_kind" "$managed_trainer_backend" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9-' '-')"
    managed_trainer_principal_id="managed-trainer-${managed_trainer_principal_suffix%-}"
    if [ -n "$TRAINER_AUTH_BUNDLE_JSON" ]; then
      managed_trainer_auth_mode="provided_bundle"
    else
      managed_trainer_auth_mode="bootstrap_static_principal"
      auth_principals_json="$(python3 -c '''import json, sys; principals = json.loads(sys.argv[1] or "[]"); principal_id = sys.argv[2]; experiment_id = sys.argv[3]; network_id = sys.argv[4]; backend = sys.argv[5]; stack_name = sys.argv[6]; environment = sys.argv[7]; trainer_role = "TrainerCpu" if backend == "cpu" else "TrainerGpu"; principal = {"principal_id": principal_id, "display_name": f"burn_dragon managed {experiment_id} {backend} trainer", "org_memberships": [], "group_memberships": [], "granted_roles": {"roles": [trainer_role, "Archive"]}, "granted_scopes": ["Connect", "Discover", {"Train": {"experiment_id": experiment_id}}, {"Archive": {"experiment_id": experiment_id}}], "allowed_networks": [network_id], "custom_claims": {"deployment_profile": environment, "stack": stack_name, "managed_trainer": "true", "managed_trainer_backend": backend, "managed_trainer_experiment_id": experiment_id}}; principals = [item for item in principals if item.get("principal_id") != principal_id]; principals.append(principal); print(json.dumps(principals))''' "$auth_principals_json" "$managed_trainer_principal_id" "$managed_trainer_experiment_id" "$network_id" "$managed_trainer_backend" "$stack_name" "$DEPLOY_ENVIRONMENT")"
    fi
  fi
  dragon_git_repository="https://github.com/${GITHUB_REPOSITORY}.git"
  dragon_git_ref="${GITHUB_SHA}"
  bootstrap_head_mirror_experiment_id="nca-prepretraining"
  bootstrap_head_mirror_revision_id="nca-r1"
  bootstrap_head_mirror_principal_id="bootstrap-head-mirror-${TF_WORKSPACE_NAME}-nca"
  bootstrap_head_mirror_auth_bundle_parameter_name="${secret_parameter_prefix}/bootstrap_head_mirror_auth_bundle_json"
  auth_principals_json="$(python3 -c '''import json, sys; principals = json.loads(sys.argv[1] or "[]"); principal_id = sys.argv[2]; experiment_id = sys.argv[3]; network_id = sys.argv[4]; stack_name = sys.argv[5]; environment = sys.argv[6]; principal = {"principal_id": principal_id, "display_name": "burn_dragon bootstrap nca head mirror", "org_memberships": [], "group_memberships": [], "granted_roles": {"roles": ["TrainerCpu", "Archive"]}, "granted_scopes": ["Connect", "Discover", {"Train": {"experiment_id": experiment_id}}, {"Archive": {"experiment_id": experiment_id}}], "allowed_networks": [network_id], "custom_claims": {"deployment_profile": environment, "stack": stack_name, "bootstrap_head_mirror": "true", "bootstrap_head_mirror_experiment_id": experiment_id, "admin_capabilities": "register_live_head,rollout_auth_policy"}}; principals = [item for item in principals if item.get("principal_id") != principal_id]; principals.append(principal); print(json.dumps(principals))''' "$auth_principals_json" "$bootstrap_head_mirror_principal_id" "$bootstrap_head_mirror_experiment_id" "$network_id" "$stack_name" "$DEPLOY_ENVIRONMENT")"
  browser_canary_experiment_id="nca-prepretraining"
  browser_canary_revision_id="nca-r1"
  browser_canary_principal_id="browser-canary-${TF_WORKSPACE_NAME}-nca"
  auth_principals_json="$(python3 -c '''import json, sys; principals = json.loads(sys.argv[1] or "[]"); principal_id = sys.argv[2]; principals = [item for item in principals if item.get("principal_id") != principal_id]; print(json.dumps(principals))''' "$auth_principals_json" "$browser_canary_principal_id")"

  snapshot_search_region="$source_snapshot_region_input"
  if [ -z "$snapshot_search_region" ]; then
    snapshot_search_region="$aws_region"
  fi

  admin_logins_json="$(
    python3 -c 'import json, sys; seen=set(); values=[]; [values.append(login) for token in sys.argv[1].replace("\n", ",").split(",") for login in [token.strip().lower()] if login and login not in seen and not seen.add(login)]; print(json.dumps(values))' "$github_admin_logins"
  )"
  admin_logins_csv="$(
    python3 -c 'import json, sys; print(",".join(json.loads(sys.argv[1])))' "$admin_logins_json"
  )"

  resolved_auth_client_id="$AUTH_CLIENT_ID"
  resolved_auth_client_secret="$AUTH_CLIENT_SECRET"
  if [ "$auth_connector_kind" = "github" ]; then
    if [ -z "$resolved_auth_client_id" ]; then
      resolved_auth_client_id="$GITHUB_CLIENT_ID"
    fi
    if [ -z "$resolved_auth_client_secret" ]; then
      resolved_auth_client_secret="$GITHUB_CLIENT_SECRET"
    fi
  fi

  if [ "$auth_connector_kind" = "github" ]; then
    if [ -z "$github_required_repo" ]; then
      github_required_repo="mosure/burn_dragon"
    fi
    : "${admin_logins_csv:?no admin logins resolved for github deployment}"
    : "${resolved_auth_client_id:?missing BURN_DRAGON_P2P_AUTH_CLIENT_ID or BURN_DRAGON_P2P_GITHUB_CLIENT_ID secret}"
    : "${resolved_auth_client_secret:?missing BURN_DRAGON_P2P_AUTH_CLIENT_SECRET or BURN_DRAGON_P2P_GITHUB_CLIENT_SECRET secret}"
    : "${BROWSER_CANARY_CALLBACK_TOKEN:?missing BURN_DRAGON_P2P_BROWSER_CANARY_CALLBACK_TOKEN secret}"
  elif [ "$auth_connector_kind" = "oidc" ] || [ "$auth_connector_kind" = "oauth" ]; then
    : "${resolved_auth_client_id:?missing BURN_DRAGON_P2P_AUTH_CLIENT_ID secret}"
    : "${resolved_auth_client_secret:?missing BURN_DRAGON_P2P_AUTH_CLIENT_SECRET secret}"
  fi

  if [ -z "$artifact_bucket_name" ]; then
    artifact_bucket_name="$artifact_bucket_name_var"
  fi
  if [ -z "$artifact_bucket_path_prefix" ]; then
    artifact_bucket_path_prefix="$artifact_bucket_path_prefix_var"
  fi
  if [ -z "$artifact_replica_bucket_name" ]; then
    artifact_replica_bucket_name="$artifact_replica_bucket_name_var"
  fi
  if [ "$create_artifact_bucket" != "true" ] && [ -z "$artifact_bucket_name" ]; then
    echo "artifact_bucket_name is required when create_artifact_bucket=false" >&2
    exit 1
  fi
  if [ -n "$next_disaster_recovery_region" ] && [ "$create_artifact_replica_bucket" != "true" ] && [ -z "$artifact_replica_bucket_name" ]; then
    echo "artifact_replica_bucket_name is required when next_disaster_recovery_region is set and create_artifact_replica_bucket=false" >&2
    exit 1
  fi

  if [ "$use_retained_bootstrap_data_volume" = "true" ]; then
    if [ "$restore_from_latest_snapshots" = "true" ]; then
      if [ -z "$primary_restore_snapshot_id" ]; then
        primary_restore_snapshot_id="$(aws ec2 describe-snapshots \
          --region "$snapshot_search_region" \
          --owner-ids self \
          --filters \
            "Name=tag:Stack,Values=$stack_name" \
            "Name=tag:TerraformWorkspace,Values=$TF_WORKSPACE_NAME" \
            "Name=tag:NodeRole,Values=primary" \
            "Name=status,Values=completed" \
          --query 'reverse(sort_by(Snapshots,&StartTime))[0].SnapshotId' \
          --output text)"
      fi
    fi

    if [ -z "$primary_restore_snapshot_id" ] || [ "$primary_restore_snapshot_id" = "None" ]; then
      echo "missing bootstrap restore snapshot id; provide one explicitly or enable restore_from_latest_snapshots with matching tagged snapshots" >&2
      exit 1
    fi
  else
    primary_restore_snapshot_id=""
  fi

  secret_parameter_prefix="/${stack_name}/${TF_WORKSPACE_NAME}/bootstrap"

  {
    echo "AWS_REGION=$aws_region"
    echo "AWS_ROLE_ARN=$aws_role_arn"
    echo "AWS_ACCOUNT_ID=$(cut -d: -f5 <<<"$aws_role_arn")"
    echo "STACK_NAME=$stack_name"
    echo "EDGE_DOMAIN_NAME=$edge_domain_name"
    echo "BROWSER_APP_BASE_URL=$browser_app_base_url"
    echo "AUTH_REDIRECT_BASE_URL=$auth_redirect_base_url"
    echo "ACME_CONTACT_EMAIL=$acme_contact_email"
    echo "BROWSER_APP_PAGES_DOMAIN_TARGET=$browser_app_pages_domain_target"
    echo "ROUTE53_ZONE_NAME=$route53_zone_name"
    echo "SECRET_PARAMETER_PREFIX=$secret_parameter_prefix"
    echo "PRIMARY_RESTORE_SNAPSHOT_ID=$primary_restore_snapshot_id"
    echo "SNAPSHOT_SEARCH_REGION=$snapshot_search_region"
    echo "TF_VAR_aws_region=$aws_region"
    echo "TF_VAR_stack_name=$stack_name"
    echo "TF_VAR_environment_name=${DEPLOY_ENVIRONMENT}"
    echo "TF_VAR_route53_zone_name=$route53_zone_name"
    echo "TF_VAR_edge_domain_name=$edge_domain_name"
    echo "TF_VAR_browser_app_base_url=$browser_app_base_url"
    echo "TF_VAR_auth_redirect_base_url=$auth_redirect_base_url"
    echo "TF_VAR_acme_contact_email=$acme_contact_email"
    echo "TF_VAR_browser_app_pages_domain_target=$browser_app_pages_domain_target"
    echo "TF_VAR_secret_parameter_prefix=$secret_parameter_prefix"
    echo "TF_VAR_network_id=$network_id"
    echo "TF_VAR_project_family_id=$project_family_id"
    echo "TF_VAR_study_id=$study_id"
    echo "TF_VAR_release_train_hash=$release_train_hash"
    echo "TF_VAR_bootstrap_install_source=$bootstrap_install_source"
    echo "TF_VAR_bootstrap_crate_version=$bootstrap_version"
    echo "TF_VAR_bootstrap_git_ref=$bootstrap_git_ref"
    echo "TF_VAR_dragon_crate_version=$dragon_crate_version"
    echo "TF_VAR_dragon_git_repository=$dragon_git_repository"
    echo "TF_VAR_dragon_git_ref=$dragon_git_ref"
    echo "TF_VAR_bootstrap_head_mirror_auth_bundle_parameter_name=$bootstrap_head_mirror_auth_bundle_parameter_name"
    echo "TF_VAR_auth_connector_kind=$auth_connector_kind"
    echo "TF_VAR_managed_trainer_desired_capacity=$managed_trainer_desired_capacity"
    echo "TF_VAR_managed_trainer_backend=$managed_trainer_backend"
    echo "TF_VAR_managed_trainer_experiment_kind=$managed_trainer_experiment_kind"
    echo "TF_VAR_managed_trainer_target=$managed_trainer_target"
    echo "TF_VAR_managed_trainer_crate_version=$managed_trainer_crate_version"
    echo "TF_VAR_managed_trainer_auth_bundle_parameter_name=$managed_trainer_auth_bundle_parameter_name"
    echo "TF_VAR_native_target_artifact_hash=$native_target_artifact_hash"
    echo "TF_VAR_auth_authority_name=$auth_authority_name"
    echo "MANAGED_TRAINER_AUTH_MODE=$managed_trainer_auth_mode"
    echo "MANAGED_TRAINER_PRINCIPAL_ID=$managed_trainer_principal_id"
    echo "MANAGED_TRAINER_EXPERIMENT_ID=$managed_trainer_experiment_id"
    echo "MANAGED_TRAINER_REVISION_ID=$managed_trainer_revision_id"
    echo "BOOTSTRAP_HEAD_MIRROR_PRINCIPAL_ID=$bootstrap_head_mirror_principal_id"
    echo "BOOTSTRAP_HEAD_MIRROR_EXPERIMENT_ID=$bootstrap_head_mirror_experiment_id"
    echo "BOOTSTRAP_HEAD_MIRROR_REVISION_ID=$bootstrap_head_mirror_revision_id"
    echo "BROWSER_CANARY_PRINCIPAL_ID=$browser_canary_principal_id"
    echo "BROWSER_CANARY_EXPERIMENT_ID=$browser_canary_experiment_id"
    echo "BROWSER_CANARY_REVISION_ID=$browser_canary_revision_id"
    echo "TF_VAR_create_artifact_bucket=$create_artifact_bucket"
    echo "DATASET_DOMAIN_NAME=$dataset_domain_name"
    echo "DATASET_BUCKET_PATH_PREFIX=$dataset_bucket_path_prefix"
    echo "TF_VAR_dataset_domain_name=$dataset_domain_name"
    echo "TF_VAR_dataset_bucket_path_prefix=$dataset_bucket_path_prefix"
    echo "TF_VAR_create_artifact_replica_bucket=$create_artifact_replica_bucket"
    echo "AUTH_CONNECTOR_KIND=$auth_connector_kind"
  } >>"$GITHUB_ENV"
  {
    echo "TF_VAR_auth_principals_json<<__AUTH_PRINCIPALS_JSON__"
    printf '%s\n' "$auth_principals_json"
    echo "__AUTH_PRINCIPALS_JSON__"
  } >>"$GITHUB_ENV"

  if [ "$auth_connector_kind" = "github" ]; then
    {
      echo "TF_VAR_github_required_org=$github_required_org"
      echo "TF_VAR_github_required_team=$github_required_team"
      echo "TF_VAR_github_required_repo=$github_required_repo"
      echo "TF_VAR_github_browser_canary_principal_id=$browser_canary_principal_id"
      echo "TF_VAR_github_browser_canary_callback_token=$BROWSER_CANARY_CALLBACK_TOKEN"
      echo "TF_VAR_github_admin_logins=$admin_logins_json"
      echo "ADMIN_LOGINS_CSV=$admin_logins_csv"
    } >>"$GITHUB_ENV"
  fi

  if [ -n "$auth_authorize_base_url" ]; then echo "TF_VAR_auth_authorize_base_url=$auth_authorize_base_url" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_exchange_url" ]; then echo "TF_VAR_auth_exchange_url=$auth_exchange_url" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_token_url" ]; then echo "TF_VAR_auth_token_url=$auth_token_url" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_api_base_url" ]; then echo "TF_VAR_auth_api_base_url=$auth_api_base_url" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_userinfo_url" ]; then echo "TF_VAR_auth_userinfo_url=$auth_userinfo_url" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_refresh_url" ]; then echo "TF_VAR_auth_refresh_url=$auth_refresh_url" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_revoke_url" ]; then echo "TF_VAR_auth_revoke_url=$auth_revoke_url" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_jwks_url" ]; then echo "TF_VAR_auth_jwks_url=$auth_jwks_url" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_oidc_issuer" ]; then echo "TF_VAR_auth_oidc_issuer=$auth_oidc_issuer" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_oauth_provider" ]; then echo "TF_VAR_auth_oauth_provider=$auth_oauth_provider" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_external_authority" ]; then echo "TF_VAR_auth_external_authority=$auth_external_authority" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_external_trusted_principal_header" ]; then echo "TF_VAR_auth_external_trusted_principal_header=$auth_external_trusted_principal_header" >>"$GITHUB_ENV"; fi
  if [ -n "$auth_external_trusted_internal_only" ]; then echo "TF_VAR_auth_external_trusted_internal_only=$auth_external_trusted_internal_only" >>"$GITHUB_ENV"; fi
  if [ -n "$github_admin_required_repo_permission" ]; then echo "TF_VAR_github_admin_required_repo_permission=$github_admin_required_repo_permission" >>"$GITHUB_ENV"; fi
  if [ -n "$instance_type" ]; then echo "TF_VAR_instance_type=$instance_type" >>"$GITHUB_ENV"; fi
  if [ -n "$root_volume_size_gib" ]; then echo "TF_VAR_root_volume_size_gib=$root_volume_size_gib" >>"$GITHUB_ENV"; fi
  if [ -n "$data_volume_size_gib" ]; then echo "TF_VAR_data_volume_size_gib=$data_volume_size_gib" >>"$GITHUB_ENV"; fi
  if [ -n "$use_retained_bootstrap_data_volume" ]; then echo "TF_VAR_use_retained_bootstrap_data_volume=$use_retained_bootstrap_data_volume" >>"$GITHUB_ENV"; fi
  if [ -n "$managed_trainer_instance_type" ]; then echo "TF_VAR_managed_trainer_instance_type=$managed_trainer_instance_type" >>"$GITHUB_ENV"; fi
  if [ -n "$managed_trainer_root_volume_size_gib" ]; then echo "TF_VAR_managed_trainer_root_volume_size_gib=$managed_trainer_root_volume_size_gib" >>"$GITHUB_ENV"; fi
  if [ -n "$managed_trainer_min_size" ]; then echo "TF_VAR_managed_trainer_min_size=$managed_trainer_min_size" >>"$GITHUB_ENV"; fi
  if [ -n "$managed_trainer_max_size" ]; then echo "TF_VAR_managed_trainer_max_size=$managed_trainer_max_size" >>"$GITHUB_ENV"; fi
  if [ -n "$enable_data_volume_snapshots" ]; then echo "TF_VAR_enable_data_volume_snapshots=$enable_data_volume_snapshots" >>"$GITHUB_ENV"; fi
  if [ -n "$data_volume_snapshot_retention_days" ]; then echo "TF_VAR_data_volume_snapshot_retention_days=$data_volume_snapshot_retention_days" >>"$GITHUB_ENV"; fi
  if [ -n "$enable_disaster_recovery_snapshot_copies" ]; then echo "TF_VAR_enable_disaster_recovery_snapshot_copies=$enable_disaster_recovery_snapshot_copies" >>"$GITHUB_ENV"; fi
  if [ -n "$disaster_recovery_snapshot_retention_days" ]; then echo "TF_VAR_disaster_recovery_snapshot_retention_days=$disaster_recovery_snapshot_retention_days" >>"$GITHUB_ENV"; fi
  if [ -n "$enable_bootstrap_status_alarms" ]; then echo "TF_VAR_enable_bootstrap_status_alarms=$enable_bootstrap_status_alarms" >>"$GITHUB_ENV"; fi
  if [ -n "$alarm_sns_topic_arn" ]; then echo "TF_VAR_alarm_sns_topic_arn=$alarm_sns_topic_arn" >>"$GITHUB_ENV"; fi
  if [ -n "$enable_control_plane_operational_alarms" ]; then echo "TF_VAR_enable_control_plane_operational_alarms=$enable_control_plane_operational_alarms" >>"$GITHUB_ENV"; fi
  if [ -n "$enable_control_plane_dashboard" ]; then echo "TF_VAR_enable_control_plane_dashboard=$enable_control_plane_dashboard" >>"$GITHUB_ENV"; fi
  if [ -n "$enable_managed_control_plane_redis" ]; then echo "TF_VAR_enable_managed_control_plane_redis=$enable_managed_control_plane_redis" >>"$GITHUB_ENV"; fi
  if [ -n "$artifact_bucket_name" ]; then echo "TF_VAR_artifact_bucket_name=$artifact_bucket_name" >>"$GITHUB_ENV"; fi
  if [ -n "$dataset_bucket_name" ]; then echo "TF_VAR_dataset_bucket_name=$dataset_bucket_name" >>"$GITHUB_ENV"; fi
  if [ -n "$artifact_bucket_path_prefix" ]; then echo "TF_VAR_artifact_bucket_path_prefix=$artifact_bucket_path_prefix" >>"$GITHUB_ENV"; fi
  if [ -n "$artifact_replica_bucket_name" ]; then echo "TF_VAR_artifact_replica_bucket_name=$artifact_replica_bucket_name" >>"$GITHUB_ENV"; fi
  if [ -n "$artifact_bucket_force_destroy" ]; then echo "TF_VAR_artifact_bucket_force_destroy=$artifact_bucket_force_destroy" >>"$GITHUB_ENV"; fi
  if [ -n "$artifact_replica_bucket_force_destroy" ]; then echo "TF_VAR_artifact_replica_bucket_force_destroy=$artifact_replica_bucket_force_destroy" >>"$GITHUB_ENV"; fi
  if [ -n "$artifact_bucket_server_side_encryption" ]; then echo "TF_VAR_artifact_bucket_server_side_encryption=$artifact_bucket_server_side_encryption" >>"$GITHUB_ENV"; fi
  if [ -n "$primary_restore_snapshot_id" ]; then echo "TF_VAR_bootstrap_primary_restore_snapshot_id=$primary_restore_snapshot_id" >>"$GITHUB_ENV"; fi
  if [ -n "$climbmix_dataset_base_url" ]; then echo "TF_VAR_climbmix_browser_dataset_base_url=$climbmix_dataset_base_url" >>"$GITHUB_ENV"; fi
  if [ -n "$next_disaster_recovery_region" ]; then echo "TF_VAR_disaster_recovery_region=$next_disaster_recovery_region" >>"$GITHUB_ENV"; fi

  if [ "$auth_connector_kind" = "github" ] || [ "$auth_connector_kind" = "oidc" ] || [ "$auth_connector_kind" = "oauth" ]; then
    {
      echo "AUTH_CLIENT_ID_RESOLVED=$resolved_auth_client_id"
      echo "AUTH_CLIENT_SECRET_RESOLVED=$resolved_auth_client_secret"
    } >>"$GITHUB_ENV"
  fi
}

case "$mode" in
  deploy) resolve_deploy ;;
  restore) resolve_restore ;;
  *)
    echo "unsupported resolve mode: $mode" >&2
    exit 1
    ;;
esac
