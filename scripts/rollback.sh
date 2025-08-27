#!/bin/bash
# Rollback script to revert to previous stable revision

set -e

APP_NAME="rag-service"
RESOURCE_GROUP="rag-service-rg"

echo "Rolling back to previous stable revision..."

# Get previous revision
PREV_REVISION=$(az containerapp revision list \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query "[?properties.active==false && properties.trafficWeight == 0].name" \
    -o tsv | head -1)

if [ -z "$PREV_REVISION" ]; then
    echo "Error: No previous stable revision found"
    exit 1
fi

# Confirm rollback
read -p "Rollback to $PREV_REVISION? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Rollback cancelled"
    exit 1
fi

# Execute rollback
az containerapp revision set-mode \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --mode single \
    --revision $PREV_REVISION

# Wait for service readiness
FQDN=$(az containerapp show \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query "properties.configuration.ingress.fqdn" \
    -o tsv)

for i in {1..30}; do
    if curl -f -s "https://$FQDN/healthz" > /dev/null; then
        echo "Service is ready"
        break
    fi
    echo "Waiting for service... attempt $i/30"
    sleep 10
done

echo "Rollback completed successfully"
echo "Active revision: $PREV_REVISION"
echo "Service URL: https://$FQDN"
