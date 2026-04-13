output "edge_url" {
  description = "Public browser edge URL."
  value       = "https://${var.edge_domain_name}"
}

output "bootstrap_instance_id" {
  description = "EC2 instance id for the bootstrap edge."
  value       = aws_instance.bootstrap.id
}

output "bootstrap_public_ip" {
  description = "Elastic IP attached to the bootstrap edge."
  value       = aws_eip.bootstrap.public_ip
}

output "bootstrap_data_volume_id" {
  description = "Retained EBS volume id carrying bootstrap/auth/publication state. Empty when root-volume-only bootstrap storage is enabled."
  value       = local.use_retained_bootstrap_data_volume ? aws_ebs_volume.bootstrap_data[0].id : ""
}

output "bootstrap_data_mount_path" {
  description = "Mounted data path used for bootstrap/auth/publication state."
  value       = local.bootstrap_data_mount_path
}

output "bootstrap_state_storage_mode" {
  description = "Bootstrap local state storage mode."
  value       = local.bootstrap_state_storage_mode
}

output "artifact_bucket_name" {
  description = "S3 bucket used for durable checkpoint and metric artifact publication."
  value       = local.artifact_bucket_name
}

output "artifact_bucket_uri" {
  description = "S3 URI prefix receiving directly published checkpoint and metric artifacts from the bootstrap host."
  value       = local.artifact_bucket_s3_uri
}

output "dataset_bucket_name" {
  description = "S3 bucket used for managed browser dataset distribution."
  value       = local.dataset_bucket_name
}

output "dataset_bucket_uri" {
  description = "S3 URI prefix backing the managed browser dataset distribution."
  value       = local.dataset_bucket_s3_uri
}

output "dataset_bucket_path_prefix" {
  description = "Key prefix inside the managed browser dataset bucket."
  value       = local.dataset_bucket_path_prefix
}

output "dataset_distribution_domain_name" {
  description = "Public CloudFront hostname serving managed browser datasets."
  value       = local.dataset_domain_name
}

output "dataset_distribution_id" {
  description = "CloudFront distribution id serving managed browser datasets."
  value       = aws_cloudfront_distribution.dataset.id
}

output "managed_climbmix_browser_dataset_base_url" {
  description = "Managed default browser ClimbMix shard-pool base URL published into the ClimbMix profile when no explicit override is supplied."
  value       = local.managed_climbmix_browser_dataset_base_url
}

output "disaster_recovery_region" {
  description = "Configured warm-disaster-recovery AWS region. Empty when warm DR is disabled."
  value       = trimspace(var.disaster_recovery_region)
}

output "artifact_replica_bucket_name" {
  description = "S3 bucket in the disaster-recovery region receiving replicated checkpoint and metric artifacts. Empty when warm DR is disabled."
  value       = local.disaster_recovery_enabled ? local.artifact_replica_bucket_name : ""
}

output "artifact_replica_bucket_uri" {
  description = "S3 URI prefix in the disaster-recovery region receiving replicated checkpoint and metric artifacts. Empty when warm DR is disabled."
  value       = local.disaster_recovery_enabled ? local.artifact_replica_bucket_s3_uri : ""
}

output "control_plane_redis_primary_endpoint" {
  description = "Primary Redis endpoint backing shared operator and auth session state. Empty when local file-backed control-plane state is enabled."
  value       = local.managed_control_plane_redis_enabled ? aws_elasticache_replication_group.control_plane[0].primary_endpoint_address : ""
}

output "control_plane_state_backend" {
  description = "Configured control-plane state backend."
  value       = local.control_plane_state_backend
}

output "control_plane_dashboard_name" {
  description = "CloudWatch dashboard name for the Dragon control plane. Empty when dashboards are disabled."
  value       = var.enable_control_plane_dashboard ? local.control_plane_dashboard_name : ""
}

output "control_plane_dashboard_url" {
  description = "CloudWatch dashboard URL for the Dragon control plane. Empty when dashboards are disabled."
  value       = var.enable_control_plane_dashboard ? local.control_plane_dashboard_url : ""
}

output "managed_trainer_asg_name" {
  description = "Autoscaling group name for the optional managed native trainer pool. Empty when managed trainers are disabled."
  value       = length(aws_autoscaling_group.managed_trainer) > 0 ? aws_autoscaling_group.managed_trainer[0].name : ""
}

output "managed_trainer_desired_capacity" {
  description = "Desired capacity for the optional managed native trainer pool."
  value       = var.managed_trainer_desired_capacity
}

output "managed_trainer_auth_bundle_parameter_name" {
  description = "SSM parameter name expected to contain the managed trainer auth bundle JSON. Empty when managed trainers are disabled."
  value       = local.managed_trainer_enabled ? local.managed_trainer_auth_bundle_parameter_name : ""
}

output "seed_node_tcp_multiaddr" {
  description = "TCP bootstrap multiaddr advertised to native peers."
  value       = "/dns4/${var.edge_domain_name}/tcp/${var.p2p_port}"
}

output "seed_node_quic_multiaddr" {
  description = "QUIC bootstrap multiaddr advertised to native peers."
  value       = "/dns4/${var.edge_domain_name}/udp/${var.p2p_port}/quic-v1"
}

output "secret_parameter_prefix" {
  description = "SSM parameter prefix read by the bootstrap edge at runtime."
  value       = var.secret_parameter_prefix
}
