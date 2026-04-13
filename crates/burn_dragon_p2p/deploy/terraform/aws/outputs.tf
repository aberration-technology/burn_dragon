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
  description = "Retained EBS volume id carrying bootstrap/auth/publication state."
  value       = aws_ebs_volume.bootstrap_data.id
}

output "bootstrap_data_mount_path" {
  description = "Mounted data path used for retained bootstrap/auth/publication state."
  value       = local.bootstrap_data_mount_path
}

output "bootstrap_publication_root" {
  description = "Local retained publication root where the bootstrap host writes checkpoint and metric artifacts before S3 replication."
  value       = local.bootstrap_publication_root
}

output "artifact_bucket_name" {
  description = "S3 bucket used for durable replicated checkpoint and metric artifacts."
  value       = local.artifact_bucket_name
}

output "artifact_bucket_uri" {
  description = "S3 URI prefix receiving replicated checkpoint and metric artifacts from the bootstrap host."
  value       = local.artifact_bucket_s3_uri
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
