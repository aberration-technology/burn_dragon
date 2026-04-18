#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
from dataclasses import dataclass


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TERRAFORM_VARIABLES = (
    REPO_ROOT / "crates" / "burn_dragon_p2p" / "deploy" / "terraform" / "aws" / "variables.tf"
)
MONTHLY_HOURS = 730
PUBLIC_IPV4_HOURLY_USD = 0.005
GP3_STORAGE_PER_GIB_MONTH_USD = 0.08
CLOUDWATCH_DASHBOARD_MONTHLY_USD = 3.00
CLOUDWATCH_STANDARD_ALARM_MONTHLY_USD = 0.10
ROUTE53_HEALTH_CHECK_MONTHLY_USD = 0.50
MODEST_S3_STORAGE_RESERVE_MONTHLY_USD = 1.00
FIXED_MONTHLY_COST_CAP_USD = 100.00


INSTANCE_HOURLY_USD = {
    "t3a.small": 0.0188,
    "t3a.medium": 0.0376,
    "t3a.large": 0.0752,
    "m7i.large": 0.1008,
}

REDIS_HOURLY_USD = {
    "cache.t4g.small": 0.0320,
}


def terraform_default(variable_name: str) -> str:
    text = TERRAFORM_VARIABLES.read_text()
    pattern = re.compile(
        rf'variable "{re.escape(variable_name)}" \{{.*?default\s+=\s+([^\n]+)',
        re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise AssertionError(f"missing terraform default for {variable_name}")
    raw = match.group(1).strip()
    if raw.endswith("\r"):
        raw = raw[:-1]
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    return raw


def env_or_default(name: str, variable_name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return terraform_default(variable_name)
    return value


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def parse_int(value: str) -> int:
    return int(str(value).strip())


def monthly_cost_from_hourly(rate: float) -> float:
    return rate * MONTHLY_HOURS


@dataclass
class GuardrailReport:
    environment: str
    operation: str
    workspace: str
    fixed_monthly_cost_usd: float
    cost_breakdown: dict[str, float]
    errors: list[str]
    warnings: list[str]


def build_report(env: dict[str, str] | None = None) -> GuardrailReport:
    env_map = env if env is not None else os.environ
    environment = (env_map.get("DEPLOY_ENVIRONMENT") or "production").strip()
    operation = (env_map.get("DEPLOYMENT_OPERATION") or "deploy").strip()
    workspace = (env_map.get("TF_WORKSPACE_NAME") or "").strip()

    bootstrap_install_source = (
        env_map.get("BOOTSTRAP_INSTALL_SOURCE")
        or env_map.get("TF_VAR_bootstrap_install_source")
        or terraform_default("bootstrap_install_source")
    ).strip()
    instance_type = (
        env_map.get("TF_VAR_instance_type")
        or terraform_default("instance_type")
    ).strip()
    root_volume_size_gib = parse_int(
        env_map.get("TF_VAR_root_volume_size_gib")
        or terraform_default("root_volume_size_gib")
    )
    use_retained_bootstrap_data_volume = parse_bool(
        env_map.get("TF_VAR_use_retained_bootstrap_data_volume")
        or terraform_default("use_retained_bootstrap_data_volume")
    )
    data_volume_size_gib = parse_int(
        env_map.get("TF_VAR_data_volume_size_gib")
        or terraform_default("data_volume_size_gib")
    )
    enable_bootstrap_status_alarms = parse_bool(
        env_map.get("TF_VAR_enable_bootstrap_status_alarms")
        or terraform_default("enable_bootstrap_status_alarms")
    )
    enable_control_plane_operational_alarms = parse_bool(
        env_map.get("TF_VAR_enable_control_plane_operational_alarms")
        or terraform_default("enable_control_plane_operational_alarms")
    )
    enable_control_plane_dashboard = parse_bool(
        env_map.get("TF_VAR_enable_control_plane_dashboard")
        or terraform_default("enable_control_plane_dashboard")
    )
    enable_managed_control_plane_redis = parse_bool(
        env_map.get("TF_VAR_enable_managed_control_plane_redis")
        or terraform_default("enable_managed_control_plane_redis")
    )
    managed_trainer_desired_capacity = parse_int(
        env_map.get("TF_VAR_managed_trainer_desired_capacity")
        or terraform_default("managed_trainer_desired_capacity")
    )
    managed_trainer_instance_type = (
        env_map.get("TF_VAR_managed_trainer_instance_type")
        or terraform_default("managed_trainer_instance_type")
    ).strip()
    managed_trainer_root_volume_size_gib = parse_int(
        env_map.get("TF_VAR_managed_trainer_root_volume_size_gib")
        or terraform_default("managed_trainer_root_volume_size_gib")
    )
    disaster_recovery_region = (
        env_map.get("TF_VAR_disaster_recovery_region")
        or terraform_default("disaster_recovery_region")
    ).strip()
    enable_data_volume_snapshots = parse_bool(
        env_map.get("TF_VAR_enable_data_volume_snapshots")
        or terraform_default("enable_data_volume_snapshots")
    )
    enable_disaster_recovery_snapshot_copies = parse_bool(
        env_map.get("TF_VAR_enable_disaster_recovery_snapshot_copies")
        or terraform_default("enable_disaster_recovery_snapshot_copies")
    )
    alarm_sns_topic_arn = (
        env_map.get("TF_VAR_alarm_sns_topic_arn")
        or env_map.get("BURN_DRAGON_P2P_ALARM_SNS_TOPIC_ARN")
        or terraform_default("alarm_sns_topic_arn")
    ).strip()

    errors: list[str] = []
    warnings: list[str] = []
    breakdown: dict[str, float] = {}

    bootstrap_hourly_usd = INSTANCE_HOURLY_USD.get(instance_type)
    if bootstrap_hourly_usd is None:
        errors.append(
            f"unknown bootstrap instance type `{instance_type}` for cost guardrails; extend scripts/check_deployment_guardrails.py before deploying"
        )
    else:
        breakdown["bootstrap_ec2"] = monthly_cost_from_hourly(bootstrap_hourly_usd)

    breakdown["bootstrap_root_gp3"] = root_volume_size_gib * GP3_STORAGE_PER_GIB_MONTH_USD
    breakdown["bootstrap_public_ipv4"] = monthly_cost_from_hourly(PUBLIC_IPV4_HOURLY_USD)
    breakdown["route53_health_check"] = ROUTE53_HEALTH_CHECK_MONTHLY_USD
    breakdown["modest_s3_storage_reserve"] = MODEST_S3_STORAGE_RESERVE_MONTHLY_USD

    if use_retained_bootstrap_data_volume:
        breakdown["bootstrap_retained_data_gp3"] = (
            data_volume_size_gib * GP3_STORAGE_PER_GIB_MONTH_USD
        )
        if enable_data_volume_snapshots:
            warnings.append(
                "bootstrap data snapshots are enabled; snapshot storage is usage-driven and is not fully modeled in the fixed monthly estimate"
            )

    alarm_count = 0
    if enable_bootstrap_status_alarms:
        alarm_count += 2
    if enable_control_plane_operational_alarms:
        alarm_count += 2
        if enable_managed_control_plane_redis:
            alarm_count += 2
        if managed_trainer_desired_capacity > 0:
            alarm_count += 1
    if alarm_count > 0:
        breakdown["cloudwatch_alarms"] = (
            alarm_count * CLOUDWATCH_STANDARD_ALARM_MONTHLY_USD
        )
    if enable_control_plane_dashboard:
        breakdown["cloudwatch_dashboard"] = CLOUDWATCH_DASHBOARD_MONTHLY_USD

    if enable_managed_control_plane_redis:
        redis_hourly_usd = REDIS_HOURLY_USD.get("cache.t4g.small")
        if redis_hourly_usd is None:
            errors.append("missing Redis hourly price in guardrail model")
        else:
            breakdown["control_plane_redis"] = monthly_cost_from_hourly(redis_hourly_usd)

    if managed_trainer_desired_capacity > 0:
        trainer_hourly_usd = INSTANCE_HOURLY_USD.get(managed_trainer_instance_type)
        if trainer_hourly_usd is None:
            errors.append(
                f"unknown managed trainer instance type `{managed_trainer_instance_type}` for cost guardrails; extend scripts/check_deployment_guardrails.py before deploying"
            )
        else:
            per_trainer = monthly_cost_from_hourly(trainer_hourly_usd) + (
                managed_trainer_root_volume_size_gib * GP3_STORAGE_PER_GIB_MONTH_USD
            )
            breakdown["managed_trainer_pool"] = managed_trainer_desired_capacity * per_trainer

    if disaster_recovery_region:
        warnings.append(
            f"warm DR is enabled for `{disaster_recovery_region}`; replicated storage and snapshot-copy costs are usage-driven and not fully modeled in the fixed monthly estimate"
        )
        if enable_disaster_recovery_snapshot_copies:
            warnings.append(
                "cross-region snapshot copies are enabled; snapshot copy/storage costs are not included in the fixed monthly estimate"
            )

    fixed_monthly_cost_usd = round(sum(breakdown.values()), 2)

    alarms_expected = enable_bootstrap_status_alarms or enable_control_plane_operational_alarms
    if environment == "production":
        if workspace != "mainnet":
            errors.append(
                f"production deployment must use terraform workspace `mainnet`, got `{workspace or 'unset'}`"
            )
        if bootstrap_install_source != "crate":
            errors.append(
                f"production deployment must use bootstrap_install_source=crate, got `{bootstrap_install_source}`"
            )
        if not enable_bootstrap_status_alarms:
            errors.append("production deployment must keep bootstrap status alarms enabled")
        if not enable_control_plane_operational_alarms:
            errors.append("production deployment must keep control-plane operational alarms enabled")
        if not enable_control_plane_dashboard:
            errors.append("production deployment must keep the control-plane dashboard enabled")
        if alarms_expected and not alarm_sns_topic_arn:
            errors.append(
                "production deployment must set BURN_DRAGON_P2P_ALARM_SNS_TOPIC_ARN so CloudWatch alarms route somewhere actionable"
            )

    if fixed_monthly_cost_usd > FIXED_MONTHLY_COST_CAP_USD:
        errors.append(
            f"estimated fixed monthly AWS cost ${fixed_monthly_cost_usd:.2f} exceeds the hard cap of ${FIXED_MONTHLY_COST_CAP_USD:.2f}"
        )

    return GuardrailReport(
        environment=environment,
        operation=operation,
        workspace=workspace,
        fixed_monthly_cost_usd=fixed_monthly_cost_usd,
        cost_breakdown=breakdown,
        errors=errors,
        warnings=warnings,
    )


def write_github_output(report: GuardrailReport) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    path = pathlib.Path(output_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"fixed_monthly_cost_usd={report.fixed_monthly_cost_usd:.2f}\n")
        handle.write(
            "fixed_monthly_cost_breakdown_json="
            + json.dumps(report.cost_breakdown, sort_keys=True)
            + "\n"
        )
        handle.write("warning_count=" + str(len(report.warnings)) + "\n")


def print_report(report: GuardrailReport) -> None:
    print(
        f"[guardrails] {report.operation} {report.environment}/{report.workspace}: estimated fixed monthly AWS cost ${report.fixed_monthly_cost_usd:.2f}"
    )
    for key, value in sorted(report.cost_breakdown.items()):
        print(f"[guardrails]   {key}: ${value:.2f}")
    for warning in report.warnings:
        print(f"[guardrails] warning: {warning}")


def main() -> int:
    report = build_report()
    write_github_output(report)
    print_report(report)
    if report.errors:
        for error in report.errors:
            print(f"[guardrails] error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
