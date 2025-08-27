# Operational Runbook

This document provides operational procedures for the Production RAG Service.

## Quick Reference

### Service Information
- **Service Name**: RAG Service
- **Environment**: Azure Container Apps
- **Resource Group**: rag-service-rg
- **Container App**: rag-service
- **Registry**: ragserviceregistry.azurecr.io

### Key Commands
```bash
# Check service status
az containerapp show --name rag-service --resource-group rag-service-rg

# View logs
az containerapp logs show --name rag-service --resource-group rag-service-rg --follow

# Check revisions
az containerapp revision list --name rag-service --resource-group rag-service-rg
```

## Deployment Procedures

### Normal Deployment
```bash
# Deploy new version
az containerapp up \
  --name rag-service \
  --resource-group rag-service-rg \
  --environment rag-service-env \
  --image ragserviceregistry.azurecr.io/rag-service:latest

# Verify deployment
curl -f https://rag-service.rag-service-env.azurecontainerapps.io/healthz
```

### Canary Deployment
```bash
# Deploy with canary
./scripts/canary-deploy.sh 10

# Promote canary
./scripts/roll-forward.sh

# Rollback if needed
./scripts/rollback.sh
```

### Emergency Rollback
```bash
# List revisions
az containerapp revision list \
  --name rag-service \
  --resource-group rag-service-rg \
  --query "[].{Revision:name,Active:properties.active,Traffic:properties.trafficWeight,Created:properties.createdTime}" \
  -o table

# Rollback to previous revision
az containerapp revision set-mode \
  --name rag-service \
  --resource-group rag-service-rg \
  --mode single \
  --revision <REVISION_NAME>
```

## Monitoring and Logging

### Health Checks
```bash
# Service health
curl https://rag-service.rag-service-env.azurecontainerapps.io/healthz

# Metrics endpoint
curl https://rag-service.rag-service-env.azurecontainerapps.io/metrics

# Application Insights
az monitor app-insights query \
  --app rag-service-ai \
  --analytics-query "requests | where timestamp > ago(1h) | summarize count()"
```

### Log Analysis
```bash
# View recent logs
az containerapp logs show \
  --name rag-service \
  --resource-group rag-service-rg \
  --follow

# Filter error logs
az containerapp logs show \
  --name rag-service \
  --resource-group rag-service-rg \
  --query "[?contains(message, 'ERROR')]"
```

### Performance Monitoring
```bash
# Check latency
curl -s https://rag-service.rag-service-env.azurecontainerapps.io/metrics | grep "rag_request_duration"

# Check error rate
curl -s https://rag-service.rag-service-env.azurecontainerapps.io/metrics | grep "rag_requests_total"
```

## Common Issues and Fixes

### Container App Not Starting

**Symptoms**: Container app shows "Failed" status

**Diagnosis**:
```bash
# Check container app status
az containerapp show --name rag-service --resource-group rag-service-rg

# View detailed logs
az containerapp logs show --name rag-service --resource-group rag-service-rg
```

**Common Causes**:
- Missing environment variables
- Invalid image reference
- Resource constraints
- Network connectivity issues

**Resolution**:
1. Check environment variables in Azure portal
2. Verify image exists in ACR
3. Increase CPU/memory limits if needed
4. Check network security groups

### High Latency

**Symptoms**: P95 latency > 1.2s

**Diagnosis**:
```bash
# Check current latency
curl -s https://rag-service.rag-service-env.azurecontainerapps.io/metrics | grep "rag_request_duration"

# Check cache hit rate
curl -s https://rag-service.rag-service-env.azurecontainerapps.io/metrics | grep "cache_hit_rate"
```

**Common Causes**:
- High CPU/memory usage
- Cache miss rate
- External API delays
- Network latency

**Resolution**:
1. Scale up container app resources
2. Check Redis cache performance
3. Monitor external API response times
4. Optimize retrieval pipeline

### High Error Rate

**Symptoms**: 5xx error rate > 0.5%

**Diagnosis**:
```bash
# Check error metrics
curl -s https://rag-service.rag-service-env.azurecontainerapps.io/metrics | grep "rag_requests_total"

# View error logs
az containerapp logs show \
  --name rag-service \
  --resource-group rag-service-rg \
  --query "[?contains(message, 'ERROR')]"
```

**Common Causes**:
- LLM API failures
- Redis connection issues
- Invalid requests
- Resource exhaustion

**Resolution**:
1. Check LLM provider status
2. Verify Redis connectivity
3. Review request validation
4. Scale resources if needed

### Index Issues

**Symptoms**: Retrieval quality degradation

**Diagnosis**:
```bash
# Check index status
curl -X POST https://rag-service.rag-service-env.azurecontainerapps.io/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "top_k": 1}'
```

**Common Causes**:
- Outdated index
- Corrupted index files
- Missing documents

**Resolution**:
1. Rebuild index: `make index`
2. Check document ingestion
3. Verify index file integrity
4. Update document corpus

## Performance Tuning

### Scaling Configuration
```bash
# Check current scaling rules
az containerapp show \
  --name rag-service \
  --resource-group rag-service-rg \
  --query "properties.template.scale"

# Update scaling rules
az containerapp update \
  --name rag-service \
  --resource-group rag-service-rg \
  --min-replicas 2 \
  --max-replicas 10 \
  --scale-rule-name http-scale-rule
```

### Resource Optimization
```bash
# Check resource usage
az containerapp show \
  --name rag-service \
  --resource-group rag-service-rg \
  --query "properties.template.containers[0].resources"

# Update resource limits
az containerapp update \
  --name rag-service \
  --resource-group rag-service-rg \
  --cpu 1.0 \
  --memory 2.0Gi
```

## Incident Response

### Level 1 Incident (On-call Engineer)
1. **Acknowledge** alert within 15 minutes
2. **Assess** impact and scope
3. **Implement** immediate mitigations
4. **Communicate** status to team
5. **Document** actions taken

### Level 2 Incident (Senior Engineer)
1. **Lead** incident response
2. **Coordinate** with stakeholders
3. **Make** technical decisions
4. **Prepare** rollback if necessary
5. **Schedule** post-mortem

### Level 3 Incident (Manager)
1. **Oversee** incident management
2. **Communicate** with customers
3. **Make** strategic decisions
4. **Coordinate** with leadership
5. **Ensure** proper follow-up

## Maintenance Procedures

### Regular Maintenance
```bash
# Weekly health check
curl -f https://rag-service.rag-service-env.azurecontainerapps.io/healthz

# Monthly performance review
# - Review metrics dashboard
# - Analyze error trends
# - Check resource utilization
# - Update documentation

# Quarterly capacity planning
# - Review scaling rules
# - Assess resource requirements
# - Plan infrastructure updates
```

### Backup and Recovery
```bash
# Backup configuration
az containerapp export \
  --name rag-service \
  --resource-group rag-service-rg \
  --file rag-service-backup.json

# Restore configuration
az containerapp create \
  --name rag-service-restored \
  --resource-group rag-service-rg \
  --environment rag-service-env \
  --file rag-service-backup.json
```

## Useful Commands

### Container App Management
```bash
# List all container apps
az containerapp list --resource-group rag-service-rg

# Get container app URL
az containerapp show \
  --name rag-service \
  --resource-group rag-service-rg \
  --query "properties.configuration.ingress.fqdn" \
  -o tsv

# Restart container app
az containerapp restart \
  --name rag-service \
  --resource-group rag-service-rg
```

### Environment Management
```bash
# List environments
az containerapp env list --resource-group rag-service-rg

# Check environment status
az containerapp env show \
  --name rag-service-env \
  --resource-group rag-service-rg
```

### Registry Operations
```bash
# List images in ACR
az acr repository list --name ragserviceregistry

# Check image tags
az acr repository show-tags \
  --name ragserviceregistry \
  --repository rag-service
```

## Contact Information

### Emergency Contacts
- **On-call Engineer**: 24/7 rotation
- **Senior Engineer**: Business hours
- **Engineering Manager**: Escalation contact

### Communication Channels
- **Slack**: #rag-service-alerts
- **Email**: rag-service-ops@company.com
- **PagerDuty**: RAG Service on-call

### Documentation
- **Architecture**: README.md
- **API Documentation**: /docs endpoint
- **Monitoring**: Application Insights dashboard
