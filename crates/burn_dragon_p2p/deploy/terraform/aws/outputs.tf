output "edge_url" {
  description = "Public browser edge URL."
  value       = "https://${var.edge_domain_name}"
}

output "bootstrap_instance_id" {
  description = "EC2 instance id for the primary bootstrap edge."
  value       = aws_instance.bootstrap.id
}

output "bootstrap_secondary_instance_id" {
  description = "EC2 instance id for the secondary bootstrap edge."
  value       = aws_instance.bootstrap_secondary.id
}

output "bootstrap_public_ip" {
  description = "Elastic IP attached to the primary bootstrap edge."
  value       = aws_eip.bootstrap.public_ip
}

output "bootstrap_secondary_public_ip" {
  description = "Elastic IP attached to the secondary bootstrap edge."
  value       = aws_eip.bootstrap_secondary.public_ip
}

output "bootstrap_data_volume_id" {
  description = "Retained EBS volume id carrying primary bootstrap/auth/publication state."
  value       = aws_ebs_volume.bootstrap_data.id
}

output "bootstrap_secondary_data_volume_id" {
  description = "Retained EBS volume id carrying secondary bootstrap/auth/publication state."
  value       = aws_ebs_volume.bootstrap_secondary_data.id
}

output "bootstrap_data_mount_path" {
  description = "Mounted data path used for retained bootstrap/auth/publication state."
  value       = local.bootstrap_data_mount_path
}

output "artifact_bucket_name" {
  description = "S3 bucket used for durable checkpoint and metric artifact publication."
  value       = local.artifact_bucket_name
}

output "artifact_bucket_uri" {
  description = "S3 URI prefix receiving directly published checkpoint and metric artifacts from the bootstrap hosts."
  value       = local.artifact_bucket_s3_uri
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
  description = "Primary Redis endpoint backing shared operator and auth session state."
  value       = aws_elasticache_replication_group.control_plane.primary_endpoint_address
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
