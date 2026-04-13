terraform {
  required_version = ">= 1.8.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

provider "aws" {
  alias  = "dr"
  region = trimspace(var.disaster_recovery_region) != "" ? trimspace(var.disaster_recovery_region) : var.aws_region
}

provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
}
