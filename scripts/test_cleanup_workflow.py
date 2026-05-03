from pathlib import Path

import yaml


WORKFLOW_PATH = Path(".github/workflows/cleanup-burn-dragon-p2p-aws.yml")


def main() -> None:
    workflow_text = WORKFLOW_PATH.read_text()
    workflow = yaml.safe_load(workflow_text)
    on_config = workflow.get("on", workflow.get(True))
    assert on_config is not None, "workflow missing on/workflow_dispatch block"
    inputs = on_config["workflow_dispatch"]["inputs"]

    assert "cleanup_force_legacy_buckets" in inputs
    assert inputs["cleanup_force_legacy_buckets"]["default"] is False
    assert "cleanup_duplicate_dataset_certificates" in inputs
    assert inputs["cleanup_duplicate_dataset_certificates"]["default"] is True
    assert "cleanup_route53_health_checks" in inputs
    assert inputs["cleanup_route53_health_checks"]["default"] is True

    required_snippets = [
        "BURN_DRAGON_P2P_AWS_CLEANUP_ROLE_ARN",
        "BURN_DRAGON_P2P_AWS_ROLE_ARN",
        "using deploy role for route53-only cleanup",
        'broader cleanup still requires the dedicated cleanup role',
        'account_id="$(aws sts get-caller-identity --query Account --output text)"',
        'if [ "$account_id" != "$AWS_ACCOUNT_ID" ]; then',
        'legacy_stack_name="dragon-p2p-prod"',
        'force_delete_buckets_by_prefix "$legacy_stack_name"',
        'cleanup_duplicate_dataset_certificates()',
        'local dataset_domain="datasets.dragon.aberration.technology"',
        'aws acm describe-certificate \\',
        'aws cloudfront list-distributions \\',
        'aws acm delete-certificate \\',
        'aws s3 rm "s3://${bucket_name}" --recursive',
        'aws s3api delete-bucket --bucket "$bucket_name"',
        'delete_empty_buckets_by_prefix "$legacy_stack_name"',
        'cleanup_route53_health_checks()',
        'canonical_health_check_name="${canonical_stack_name}-edge-primary"',
        'route53_health_check_inventory',
        'aws route53 list-health-checks',
        'aws route53 list-tags-for-resource',
        'aws route53 delete-health-check --health-check-id "$health_check_id"',
        'preserving Route53 health check',
    ]
    for snippet in required_snippets:
        assert snippet in workflow_text, f"missing required snippet: {snippet}"

    forbidden_snippets = [
        'force_delete_buckets_by_prefix "$canonical_stack_name"',
        'delete_empty_buckets_by_prefix "$canonical_stack_name"',
        'local dataset_domain="edge.dragon.aberration.technology"',
        'local dataset_domain="dragon.aberration.technology"',
    ]
    for snippet in forbidden_snippets:
        assert snippet not in workflow_text, f"unexpected broad cleanup snippet: {snippet}"

    print("cleanup-workflow-ok")


if __name__ == "__main__":
    main()
