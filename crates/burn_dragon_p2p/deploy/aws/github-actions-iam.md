# github actions aws iam

use two roles, not one:

1. `BURN_DRAGON_P2P_AWS_ROLE_ARN`
   normal operations only: deploy, restore, inspect, dataset publication
2. `BURN_DRAGON_P2P_AWS_CLEANUP_ROLE_ARN`
   destructive cleanup only: legacy/orphan teardown

do not attach `AdministratorAccess`, `PowerUserAccess`, or broad `iam:*` policies to either role.

## trust policy

use the same trust policy on both roles and scope it to the repo plus the two managed github environments:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "GitHubOidcTrust",
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "ForAnyValue:StringLike": {
          "token.actions.githubusercontent.com:sub": [
            "repo:aberration-technology/burn_dragon:environment:burn-dragon-p2p-staging",
            "repo:aberration-technology/burn_dragon:environment:burn-dragon-p2p-production"
          ]
        }
      }
    }
  ]
}
```

that keeps random branches, forks, and non-environment workflow runs from assuming the role.

## deploy role

this role is intentionally limited to the services used by:

- `.github/workflows/deploy-burn-dragon-p2p-aws.yml`
- `.github/workflows/restore-burn-dragon-p2p-aws.yml`
- `.github/workflows/inspect-burn-dragon-p2p-aws.yml`
- `.github/workflows/publish-burn-dragon-p2p-dataset.yml`

resource-level scoping is used where AWS supports it cleanly. a smaller set of `Resource: "*"` statements remains necessary for AWS APIs that are list/describe or create-before-arn operations.

replace the placeholders before use:

- `${ACCOUNT_ID}`
- `${PRIMARY_REGION}`
- `${SECRET_PARAMETER_PREFIX}`
- `${ROUTE53_ZONE_ID}`
- `${STATE_BUCKET_NAME}`
- `${STATE_BUCKET_ARN}`
- `${STATE_BUCKET_OBJECT_ARN}`
- `${ARTIFACT_BUCKET_ARN}`
- `${ARTIFACT_BUCKET_OBJECT_ARN}`
- `${DATASET_BUCKET_ARN}`
- `${DATASET_BUCKET_OBJECT_ARN}`
- `${ARTIFACT_REPLICA_BUCKET_ARN}`
- `${ARTIFACT_REPLICA_BUCKET_OBJECT_ARN}`

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadOnlyDiscovery",
      "Effect": "Allow",
      "Action": [
        "acm:DescribeCertificate",
        "acm:ListCertificates",
        "acm:ListTagsForCertificate",
        "autoscaling:Describe*",
        "cloudfront:GetDistribution",
        "cloudfront:GetDistributionConfig",
        "cloudfront:GetOriginAccessControl",
        "cloudfront:GetResponseHeadersPolicy",
        "cloudfront:ListDistributions",
        "cloudfront:ListOriginAccessControls",
        "cloudfront:ListResponseHeadersPolicies",
        "cloudfront:ListTagsForResource",
        "cloudwatch:DescribeAlarms",
        "cloudwatch:GetDashboard",
        "cloudwatch:ListDashboards",
        "dlm:GetLifecyclePolicies",
        "ec2:Describe*",
        "elasticache:Describe*",
        "iam:GetInstanceProfile",
        "iam:GetRole",
        "iam:GetRolePolicy",
        "iam:ListAttachedRolePolicies",
        "iam:ListInstanceProfiles",
        "iam:ListRolePolicies",
        "iam:ListRoles",
        "kms:DescribeKey",
        "kms:ListAliases",
        "route53:GetChange",
        "route53:GetHealthCheck",
        "route53:GetHealthCheckStatus",
        "route53:GetHostedZone",
        "route53:ListHostedZonesByName",
        "route53:ListResourceRecordSets",
        "s3:ListAllMyBuckets",
        "ssm:DescribeInstanceInformation",
        "ssm:DescribeParameters"
      ],
      "Resource": "*"
    },
    {
      "Sid": "TerraformStateBucket",
      "Effect": "Allow",
      "Action": [
        "s3:CreateBucket",
        "s3:DeleteBucket",
        "s3:GetBucketLocation",
        "s3:GetBucketPolicy",
        "s3:GetBucketPublicAccessBlock",
        "s3:GetBucketVersioning",
        "s3:GetEncryptionConfiguration",
        "s3:ListBucket",
        "s3:ListBucketVersions",
        "s3:PutEncryptionConfiguration",
        "s3:PutBucketPolicy",
        "s3:PutBucketPublicAccessBlock",
        "s3:PutBucketVersioning"
      ],
      "Resource": "${STATE_BUCKET_ARN}"
    },
    {
      "Sid": "TerraformStateObjects",
      "Effect": "Allow",
      "Action": [
        "s3:DeleteObject",
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:PutObject"
      ],
      "Resource": "${STATE_BUCKET_OBJECT_ARN}"
    },
    {
      "Sid": "SecretParameterPrefix",
      "Effect": "Allow",
      "Action": [
        "ssm:GetParameter",
        "ssm:GetParameters",
        "ssm:GetParametersByPath",
        "ssm:PutParameter"
      ],
      "Resource": "arn:aws:ssm:${PRIMARY_REGION}:${ACCOUNT_ID}:parameter${SECRET_PARAMETER_PREFIX}/*"
    },
    {
      "Sid": "ScopedIamRolesAndProfiles",
      "Effect": "Allow",
      "Action": [
        "iam:AddRoleToInstanceProfile",
        "iam:AttachRolePolicy",
        "iam:CreateInstanceProfile",
        "iam:CreateRole",
        "iam:DeleteInstanceProfile",
        "iam:DeleteRole",
        "iam:DeleteRolePolicy",
        "iam:DetachRolePolicy",
        "iam:PassRole",
        "iam:PutRolePolicy",
        "iam:RemoveRoleFromInstanceProfile",
        "iam:TagInstanceProfile",
        "iam:TagRole",
        "iam:UntagInstanceProfile",
        "iam:UntagRole",
        "iam:UpdateAssumeRolePolicy"
      ],
      "Resource": [
        "arn:aws:iam::${ACCOUNT_ID}:role/burn-dragon-p2p-*",
        "arn:aws:iam::${ACCOUNT_ID}:instance-profile/burn-dragon-p2p-*"
      ]
    },
    {
      "Sid": "CreateRequiredServiceLinkedRoles",
      "Effect": "Allow",
      "Action": "iam:CreateServiceLinkedRole",
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "iam:AWSServiceName": [
            "autoscaling.amazonaws.com",
            "dlm.amazonaws.com"
          ]
        }
      }
    },
    {
      "Sid": "CoreComputeAndNetwork",
      "Effect": "Allow",
      "Action": [
        "ec2:AllocateAddress",
        "ec2:AssociateAddress",
        "ec2:AssociateRouteTable",
        "ec2:AttachInternetGateway",
        "ec2:AttachVolume",
        "ec2:AuthorizeSecurityGroupEgress",
        "ec2:AuthorizeSecurityGroupIngress",
        "ec2:CreateLaunchTemplate",
        "ec2:CreateLaunchTemplateVersion",
        "ec2:CreateRoute",
        "ec2:CreateRouteTable",
        "ec2:CreateSecurityGroup",
        "ec2:CreateSubnet",
        "ec2:CreateTags",
        "ec2:CreateVolume",
        "ec2:CreateVpc",
        "ec2:DeleteLaunchTemplate",
        "ec2:DeleteLaunchTemplateVersions",
        "ec2:DeleteRoute",
        "ec2:DeleteRouteTable",
        "ec2:DeleteSecurityGroup",
        "ec2:DeleteSubnet",
        "ec2:DeleteTags",
        "ec2:DeleteVolume",
        "ec2:DeleteVpc",
        "ec2:DetachInternetGateway",
        "ec2:DetachVolume",
        "ec2:DisassociateAddress",
        "ec2:DisassociateRouteTable",
        "ec2:ModifyLaunchTemplate",
        "ec2:ModifySubnetAttribute",
        "ec2:ModifyVolume",
        "ec2:ModifyVpcAttribute",
        "ec2:ReleaseAddress",
        "ec2:ReplaceRoute",
        "ec2:RevokeSecurityGroupEgress",
        "ec2:RevokeSecurityGroupIngress",
        "ec2:RunInstances",
        "ec2:StartInstances",
        "ec2:StopInstances",
        "ec2:TerminateInstances"
      ],
      "Resource": "*"
    },
    {
      "Sid": "ManagedTrainerAutoscaling",
      "Effect": "Allow",
      "Action": [
        "autoscaling:CreateAutoScalingGroup",
        "autoscaling:CreateOrUpdateTags",
        "autoscaling:DeleteAutoScalingGroup",
        "autoscaling:DeleteTags",
        "autoscaling:SetDesiredCapacity",
        "autoscaling:SuspendProcesses",
        "autoscaling:UpdateAutoScalingGroup"
      ],
      "Resource": "*"
    },
    {
      "Sid": "ManagedRedis",
      "Effect": "Allow",
      "Action": [
        "elasticache:AddTagsToResource",
        "elasticache:CreateCacheSubnetGroup",
        "elasticache:CreateReplicationGroup",
        "elasticache:DeleteCacheSubnetGroup",
        "elasticache:DeleteReplicationGroup",
        "elasticache:ModifyCacheSubnetGroup",
        "elasticache:ModifyReplicationGroup",
        "elasticache:RemoveTagsFromResource"
      ],
      "Resource": "*"
    },
    {
      "Sid": "ArtifactAndDatasetBuckets",
      "Effect": "Allow",
      "Action": [
        "s3:AbortMultipartUpload",
        "s3:CreateBucket",
        "s3:DeleteBucket",
        "s3:DeleteBucketPolicy",
        "s3:DeleteObject",
        "s3:GetBucketCORS",
        "s3:GetLifecycleConfiguration",
        "s3:GetBucketLocation",
        "s3:GetBucketPolicy",
        "s3:GetBucketPublicAccessBlock",
        "s3:GetBucketVersioning",
        "s3:GetEncryptionConfiguration",
        "s3:GetReplicationConfiguration",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:ListBucketMultipartUploads",
        "s3:ListBucketVersions",
        "s3:PutBucketCORS",
        "s3:PutEncryptionConfiguration",
        "s3:PutLifecycleConfiguration",
        "s3:PutBucketPolicy",
        "s3:PutBucketPublicAccessBlock",
        "s3:PutReplicationConfiguration",
        "s3:PutBucketVersioning",
        "s3:PutObject"
      ],
      "Resource": [
        "${ARTIFACT_BUCKET_ARN}",
        "${ARTIFACT_BUCKET_OBJECT_ARN}",
        "${DATASET_BUCKET_ARN}",
        "${DATASET_BUCKET_OBJECT_ARN}",
        "${ARTIFACT_REPLICA_BUCKET_ARN}",
        "${ARTIFACT_REPLICA_BUCKET_OBJECT_ARN}"
      ]
    },
    {
      "Sid": "Route53HealthChecksAndRecords",
      "Effect": "Allow",
      "Action": [
        "route53:ChangeResourceRecordSets",
        "route53:CreateHealthCheck",
        "route53:DeleteHealthCheck",
        "route53:UpdateHealthCheck"
      ],
      "Resource": [
        "arn:aws:route53:::hostedzone/${ROUTE53_ZONE_ID}",
        "arn:aws:route53:::healthcheck/*"
      ]
    },
    {
      "Sid": "CertificatesAndCloudFront",
      "Effect": "Allow",
      "Action": [
        "acm:AddTagsToCertificate",
        "acm:DeleteCertificate",
        "acm:RemoveTagsFromCertificate",
        "acm:RequestCertificate",
        "cloudfront:CreateDistribution",
        "cloudfront:CreateInvalidation",
        "cloudfront:CreateOriginAccessControl",
        "cloudfront:CreateResponseHeadersPolicy",
        "cloudfront:DeleteDistribution",
        "cloudfront:DeleteOriginAccessControl",
        "cloudfront:DeleteResponseHeadersPolicy",
        "cloudfront:TagResource",
        "cloudfront:UntagResource",
        "cloudfront:UpdateDistribution",
        "cloudfront:UpdateOriginAccessControl",
        "cloudfront:UpdateResponseHeadersPolicy"
      ],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchAndDlm",
      "Effect": "Allow",
      "Action": [
        "cloudwatch:DeleteAlarms",
        "cloudwatch:DeleteDashboards",
        "cloudwatch:PutDashboard",
        "cloudwatch:PutMetricAlarm",
        "dlm:CreateLifecyclePolicy",
        "dlm:DeleteLifecyclePolicy",
        "dlm:TagResource",
        "dlm:UntagResource",
        "dlm:UpdateLifecyclePolicy"
      ],
      "Resource": "*"
    },
    {
      "Sid": "SsmRunCommand",
      "Effect": "Allow",
      "Action": [
        "ssm:GetCommandInvocation",
        "ssm:SendCommand"
      ],
      "Resource": "*"
    }
  ]
}
```

## cleanup role

the cleanup workflow should assume a separate role through `BURN_DRAGON_P2P_AWS_CLEANUP_ROLE_ARN`.

this role is intentionally destructive and should only be used by `.github/workflows/cleanup-burn-dragon-p2p-aws.yml`.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadInventory",
      "Effect": "Allow",
      "Action": [
        "acm:DescribeCertificate",
        "acm:ListCertificates",
        "cloudfront:ListDistributions",
        "cloudwatch:DescribeAlarms",
        "cloudwatch:ListDashboards",
        "ec2:Describe*",
        "iam:GetInstanceProfile",
        "iam:ListAttachedRolePolicies",
        "iam:ListInstanceProfiles",
        "iam:ListRolePolicies",
        "iam:ListRoles",
        "s3:ListAllMyBuckets",
        "s3:ListBucket",
        "s3:ListBucketVersions",
        "ssm:GetParametersByPath"
      ],
      "Resource": "*"
    },
    {
      "Sid": "DeleteLegacyComputeNetworkAndMonitoring",
      "Effect": "Allow",
      "Action": [
        "acm:DeleteCertificate",
        "cloudwatch:DeleteAlarms",
        "cloudwatch:DeleteDashboards",
        "ec2:DeleteInternetGateway",
        "ec2:DeleteNetworkInterface",
        "ec2:DeleteRouteTable",
        "ec2:DeleteSecurityGroup",
        "ec2:DeleteSubnet",
        "ec2:DeleteVolume",
        "ec2:DeleteVpc",
        "ec2:DetachInternetGateway",
        "ec2:DisassociateAddress",
        "ec2:DisassociateRouteTable",
        "ec2:ReleaseAddress",
        "ec2:TerminateInstances"
      ],
      "Resource": "*"
    },
    {
      "Sid": "DeleteLegacyIamPrefixes",
      "Effect": "Allow",
      "Action": [
        "iam:DeleteInstanceProfile",
        "iam:DeleteRole",
        "iam:DeleteRolePolicy",
        "iam:DetachRolePolicy",
        "iam:RemoveRoleFromInstanceProfile"
      ],
      "Resource": [
        "arn:aws:iam::${ACCOUNT_ID}:role/burn-dragon-p2p-*",
        "arn:aws:iam::${ACCOUNT_ID}:role/dragon-p2p-prod*",
        "arn:aws:iam::${ACCOUNT_ID}:instance-profile/burn-dragon-p2p-*",
        "arn:aws:iam::${ACCOUNT_ID}:instance-profile/dragon-p2p-prod*"
      ]
    },
    {
      "Sid": "DeleteScopedParameters",
      "Effect": "Allow",
      "Action": [
        "ssm:DeleteParameter",
        "ssm:DeleteParameters"
      ],
      "Resource": [
        "arn:aws:ssm:${PRIMARY_REGION}:${ACCOUNT_ID}:parameter/burn-dragon-p2p-*",
        "arn:aws:ssm:${PRIMARY_REGION}:${ACCOUNT_ID}:parameter/dragon-p2p-prod*"
      ]
    },
    {
      "Sid": "DeleteLegacyBuckets",
      "Effect": "Allow",
      "Action": [
        "s3:DeleteBucket",
        "s3:DeleteObject",
        "s3:DeleteObjectVersion",
        "s3:GetBucketLocation",
        "s3:ListBucket",
        "s3:ListBucketVersions"
      ],
      "Resource": [
        "arn:aws:s3:::burn-dragon-p2p-*",
        "arn:aws:s3:::burn-dragon-p2p-*/*",
        "arn:aws:s3:::dragon-p2p-prod*",
        "arn:aws:s3:::dragon-p2p-prod*/*"
      ]
    }
  ]
}
```

## notes

- keep the deploy role and cleanup role separate. the cleanup role is a break-glass operator role, not a daily driver.
- if you use pre-existing buckets instead of the managed-bucket path, add those exact bucket arns to the deploy-role `ArtifactAndDatasetBuckets` statement.
- if you use a different hosted zone, replace `${ROUTE53_ZONE_ID}` with the exact zone id. do not grant broad `route53:*` over all hosted zones when one zone is enough.
- some actions remain on `Resource: "*"` because AWS does not support useful resource scoping for them, especially create/list/describe APIs across EC2, CloudFront, CloudWatch alarms, and ElastiCache.
- the workflows now pin the assumed-account id through `allowed-account-ids` and should be run only from the `burn-dragon-p2p-staging` and `burn-dragon-p2p-production` github environments.
