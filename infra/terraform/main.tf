provider "aws" {
  region = "us-east-1"
}

resource "aws_bedrock_model_invocation_logging_configuration" "main" {
  logging_config {
    embedding_data_delivery_enabled = true
    image_data_delivery_enabled     = false
    text_data_delivery_enabled      = true
    cloudwatch_config {
      log_group_name = "/aws/bedrock/model-invocations"
      role_arn       = aws_iam_role.bedrock_logging.arn
    }
  }
}

resource "aws_sagemaker_model" "rag_classifier" {
  name               = "rag-classifier-v2"
  execution_role_arn = aws_iam_role.sagemaker.arn

  primary_container {
    image = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-cpu-py310-ubuntu22.04"
    model_data_url = "s3://contoso-model-artifacts/rag-classifier/model.tar.gz"
    environment = {
      HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
      HF_TASK     = "text-generation"
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "rag" {
  name = "rag-classifier-endpoint-config"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.rag_classifier.name
    initial_instance_count = 1
    instance_type          = "ml.g5.xlarge"
  }
}

resource "aws_sagemaker_endpoint" "rag" {
  name                 = "rag-classifier-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.rag.name
}

resource "aws_s3_bucket" "model_artifacts" {
  bucket = "contoso-model-artifacts"
}

resource "aws_s3_bucket" "datasets" {
  bucket = "contoso-dvc-remote"
}

output "sagemaker_endpoint" {
  value = aws_sagemaker_endpoint.rag.name
}
