provider "aws" {
  region = "us-east-1"
}

resource "aws_s3_bucket" "ai_data_lake" {
  bucket = "enterprise-ai-data-lake-v2"
}

resource "aws_emr_cluster" "spark_cluster" {
  name          = "Production-Spark-ML"
  release_label = "emr-7.0.0"
  service_role  = "EMR_DefaultRole"
  applications  = ["Spark", "Hadoop", "Livy"]

  ec2_attributes {
    instance_profile = "EMR_EC2_DefaultRole"
    subnet_id        = "subnet-12345"
  }

  master_instance_group {
    instance_type = "m5.xlarge"
  }

  core_instance_group {
    instance_count = 2
    instance_type  = "m5.xlarge"
  }
}
