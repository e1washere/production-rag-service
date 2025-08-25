# Container App Outputs
output "app_url" {
  description = "URL of the deployed application"
  value       = "https://${azurerm_container_app.main.latest_revision_fqdn}"
}

output "app_name" {
  description = "Name of the container app"
  value       = azurerm_container_app.main.name
}

output "app_fqdn" {
  description = "FQDN of the container app"
  value       = azurerm_container_app.main.latest_revision_fqdn
}

# Resource Group
output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.main.name
}

output "resource_group_location" {
  description = "Location of the resource group"
  value       = azurerm_resource_group.main.location
}

# Key Vault
output "key_vault_name" {
  description = "Name of the Key Vault"
  value       = azurerm_key_vault.main.name
}

output "key_vault_uri" {
  description = "URI of the Key Vault"
  value       = azurerm_key_vault.main.vault_uri
}

# Managed Identity
output "managed_identity_id" {
  description = "ID of the managed identity"
  value       = azurerm_user_assigned_identity.main.id
}

output "managed_identity_client_id" {
  description = "Client ID of the managed identity"
  value       = azurerm_user_assigned_identity.main.client_id
}

output "managed_identity_principal_id" {
  description = "Principal ID of the managed identity"
  value       = azurerm_user_assigned_identity.main.principal_id
}

# Application Insights
output "application_insights_connection_string" {
  description = "Application Insights connection string"
  value       = azurerm_application_insights.main.connection_string
  sensitive   = true
}

output "application_insights_instrumentation_key" {
  description = "Application Insights instrumentation key"
  value       = azurerm_application_insights.main.instrumentation_key
  sensitive   = true
}

# Log Analytics
output "log_analytics_workspace_id" {
  description = "Log Analytics workspace ID"
  value       = azurerm_log_analytics_workspace.main.id
}

output "log_analytics_workspace_name" {
  description = "Log Analytics workspace name"
  value       = azurerm_log_analytics_workspace.main.name
}

# Redis Cache (if enabled)
output "redis_hostname" {
  description = "Redis hostname"
  value       = var.enable_redis ? azurerm_redis_cache.main[0].hostname : null
}

output "redis_ssl_port" {
  description = "Redis SSL port"
  value       = var.enable_redis ? azurerm_redis_cache.main[0].ssl_port : null
}

output "redis_primary_access_key" {
  description = "Redis primary access key"
  value       = var.enable_redis ? azurerm_redis_cache.main[0].primary_access_key : null
  sensitive   = true
}

# Storage Account
output "storage_account_name" {
  description = "Name of the storage account"
  value       = azurerm_storage_account.main.name
}

output "storage_account_primary_blob_endpoint" {
  description = "Primary blob endpoint of the storage account"
  value       = azurerm_storage_account.main.primary_blob_endpoint
}

# Container App Environment
output "container_app_environment_id" {
  description = "ID of the Container App Environment"
  value       = azurerm_container_app_environment.main.id
}

output "container_app_environment_name" {
  description = "Name of the Container App Environment"
  value       = azurerm_container_app_environment.main.name
}

# Deployment Information
output "deployment_timestamp" {
  description = "Timestamp of the deployment"
  value       = timestamp()
}

output "terraform_workspace" {
  description = "Terraform workspace used for deployment"
  value       = terraform.workspace
}

# Health Check URLs
output "health_check_url" {
  description = "Health check endpoint URL"
  value       = "https://${azurerm_container_app.main.latest_revision_fqdn}/health"
}

output "api_docs_url" {
  description = "API documentation URL"
  value       = "https://${azurerm_container_app.main.latest_revision_fqdn}/docs"
}

output "stats_url" {
  description = "Statistics endpoint URL"
  value       = "https://${azurerm_container_app.main.latest_revision_fqdn}/stats"
}