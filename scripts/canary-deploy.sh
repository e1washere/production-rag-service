#!/bin/bash
# Canary deployment script for RAG Service

set -e

# Configuration
APP_NAME="rag-service"
RESOURCE_GROUP="rag-service-rg"
ACR_NAME="ragserviceregistry"
ENVIRONMENT="rag-service-env"
CANARY_PERCENTAGE=${1:-10}

echo "Starting canary deployment..."

# Validate Azure CLI login
if ! az account show > /dev/null 2>&1; then
    echo "Error: Not logged into Azure CLI. Run 'az login' first."
    exit 1
fi

# Deploy new revision
FQDN=$(az containerapp up \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --environment $ENVIRONMENT \
    --image $ACR_NAME.azurecr.io/$APP_NAME:latest \
    --target-port 8000 \
    --ingress external \
    --registry-server $ACR_NAME.azurecr.io \
    --query properties.configuration.ingress.fqdn -o tsv)

echo "New revision deployed. FQDN: https://$FQDN"

# Get new revision name
NEW_REVISION=$(az containerapp revision list \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query "[?properties.active==false].name" \
    -o tsv | head -1)

if [ -z "$NEW_REVISION" ]; then
    echo "Error: Failed to get new revision name"
    exit 1
fi

# Set traffic splitting
az containerapp revision set-mode \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --mode multiple \
    --revisions "$NEW_REVISION=$CANARY_PERCENTAGE"

echo "Traffic split configured: $CANARY_PERCENTAGE% to $NEW_REVISION"

# Wait for service readiness
for i in {1..30}; do
    if curl -f -s "https://$FQDN/healthz" > /dev/null; then
        echo "Service is ready"
        break
    fi
    echo "Waiting for service... attempt $i/30"
    sleep 10
done

# Run smoke tests
for i in {1..5}; do
    RESPONSE=$(curl -s -X POST "https://$FQDN/query" \
        -H "Content-Type: application/json" \
        -d '{"question": "test", "top_k": 1}')
    
    if echo "$RESPONSE" | grep -q "answer"; then
        echo "Smoke test $i passed"
    else
        echo "Smoke test $i failed"
        exit 1
    fi
    sleep 2
done

echo "Canary deployment completed successfully"
echo "Service URL: https://$FQDN"
echo "Canary revision: $NEW_REVISION ($CANARY_PERCENTAGE% traffic)"
