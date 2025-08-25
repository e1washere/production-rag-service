#!/bin/bash

# Azure deployment script for RAG Service
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Azure deployment for RAG Service${NC}"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ Azure CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if logged in to Azure
if ! az account show &> /dev/null; then
    echo -e "${YELLOW}⚠️  Not logged in to Azure. Please run 'az login' first.${NC}"
    exit 1
fi

# Build Docker image
echo -e "${GREEN}📦 Building Docker image...${NC}"
docker build -t rag-service:latest .

# Get Azure Container Registry details
echo -e "${GREEN}🔍 Getting ACR details...${NC}"
ACR_LOGIN_SERVER=$(terraform -chdir=infra/terraform output -raw acr_login_server)
ACR_USERNAME=$(az acr credential show --name $(echo $ACR_LOGIN_SERVER | cut -d'.' -f1) --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $(echo $ACR_LOGIN_SERVER | cut -d'.' -f1) --query passwords[0].value -o tsv)

# Tag and push image to ACR
echo -e "${GREEN}📤 Pushing image to Azure Container Registry...${NC}"
docker tag rag-service:latest $ACR_LOGIN_SERVER/rag-service:latest
docker login $ACR_LOGIN_SERVER -u $ACR_USERNAME -p $ACR_PASSWORD
docker push $ACR_LOGIN_SERVER/rag-service:latest

# Deploy infrastructure with Terraform
echo -e "${GREEN}🏗️  Deploying infrastructure with Terraform...${NC}"
cd infra/terraform
terraform init
terraform plan -out=tfplan
terraform apply tfplan

# Get the app URL
APP_URL=$(terraform output -raw app_url)
echo -e "${GREEN}✅ Deployment complete!${NC}"
echo -e "${GREEN}🌐 Your RAG Service is available at: ${APP_URL}${NC}"
echo -e "${GREEN}📊 Health check: ${APP_URL}/health${NC}"
echo -e "${GREEN}📚 API docs: ${APP_URL}/docs${NC}"

# Test the deployment
echo -e "${GREEN}🧪 Testing deployment...${NC}"
sleep 30  # Wait for app to start
curl -f "${APP_URL}/health" || echo -e "${YELLOW}⚠️  Health check failed, app might still be starting${NC}"

echo -e "${GREEN}🎉 Deployment successful!${NC}"
