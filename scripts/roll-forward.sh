#!/bin/bash
# Roll forward script to promote canary to 100% traffic

set -e

APP_NAME="rag-service"
RESOURCE_GROUP="rag-service-rg"

echo "Rolling forward canary to 100% traffic..."

# Get canary revision
CANARY_REVISION=$(az containerapp revision list \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query "[?properties.active==false && properties.trafficWeight > 0].name" \
    -o tsv | head -1)

if [ -z "$CANARY_REVISION" ]; then
    echo "Error: No canary revision found with traffic"
    exit 1
fi

# Promote to 100% traffic
az containerapp revision set-mode \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --mode single \
    --revision $CANARY_REVISION

echo "Canary successfully promoted to 100% traffic"
echo "Active revision: $CANARY_REVISION"
