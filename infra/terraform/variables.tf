# Project Configuration
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "rag-service"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
  
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be one of: development, staging, production."
  }
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "East US"
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
  default     = "rag-service-rg"
}

# Container Configuration
variable "container_image" {
  description = "Container image to deploy"
  type        = string
  default     = "ghcr.io/e1washere/rag-service:latest"
}

variable "container_cpu" {
  description = "CPU allocation for container"
  type        = number
  default     = 1.0
  
  validation {
    condition     = var.container_cpu >= 0.25 && var.container_cpu <= 4.0
    error_message = "Container CPU must be between 0.25 and 4.0."
  }
}

variable "container_memory" {
  description = "Memory allocation for container"
  type        = string
  default     = "2Gi"
}

# Scaling Configuration
variable "min_replicas" {
  description = "Minimum number of replicas"
  type        = number
  default     = 1
  
  validation {
    condition     = var.min_replicas >= 0 && var.min_replicas <= 10
    error_message = "Min replicas must be between 0 and 10."
  }
}

variable "max_replicas" {
  description = "Maximum number of replicas"
  type        = number
  default     = 10
  
  validation {
    condition     = var.max_replicas >= 1 && var.max_replicas <= 30
    error_message = "Max replicas must be between 1 and 30."
  }
}

# Feature Flags
variable "enable_redis" {
  description = "Enable Redis cache"
  type        = bool
  default     = true
}

variable "enable_hybrid_search" {
  description = "Enable hybrid BM25 + embeddings search"
  type        = bool
  default     = true
}

variable "enable_reranker" {
  description = "Enable cross-encoder reranking"
  type        = bool
  default     = false
}

variable "enable_alerts" {
  description = "Enable alerting system"
  type        = bool
  default     = true
}

# API Keys and Secrets
variable "openai_api_key" {
  description = "OpenAI API key"
  type        = string
  default     = ""
  sensitive   = true
}

variable "groq_api_key" {
  description = "Groq API key"
  type        = string
  default     = ""
  sensitive   = true
}

variable "langfuse_secret_key" {
  description = "Langfuse secret key"
  type        = string
  default     = ""
  sensitive   = true
}

variable "langfuse_public_key" {
  description = "Langfuse public key"
  type        = string
  default     = ""
  sensitive   = true
}

variable "alert_webhook_url" {
  description = "Webhook URL for alerts"
  type        = string
  default     = ""
  sensitive   = true
}

# Logging Configuration
variable "log_level" {
  description = "Log level"
  type        = string
  default     = "INFO"
  
  validation {
    condition     = contains(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], var.log_level)
    error_message = "Log level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL."
  }
}

# Tags
variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "rag-service"
    Environment = "production"
    ManagedBy   = "terraform"
  }
}