#!/bin/bash

# Production deployment script for RAG Service
# Usage: ./scripts/deploy.sh [environment] [image_tag]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_ENVIRONMENT="production"
DEFAULT_IMAGE_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Parse arguments
ENVIRONMENT="${1:-$DEFAULT_ENVIRONMENT}"
IMAGE_TAG="${2:-$DEFAULT_IMAGE_TAG}"

log "Starting deployment to $ENVIRONMENT environment with image tag: $IMAGE_TAG"

# Validate environment
case $ENVIRONMENT in
    production|staging|development)
        ;;
    *)
        error "Invalid environment: $ENVIRONMENT. Must be one of: production, staging, development"
        ;;
esac

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Azure CLI is installed and logged in
    if ! command -v az &> /dev/null; then
        error "Azure CLI is not installed. Please install it first."
    fi
    
    if ! az account show &> /dev/null; then
        error "Not logged into Azure. Please run 'az login' first."
    fi
    
    # Check if Terraform is installed
    if ! command -v terraform &> /dev/null; then
        error "Terraform is not installed. Please install it first."
    fi
    
    # Check if Docker is installed (for building images)
    if ! command -v docker &> /dev/null; then
        warning "Docker is not installed. Skipping image build."
    fi
    
    success "Prerequisites check passed"
}

# Build and push Docker image
build_and_push_image() {
    if command -v docker &> /dev/null; then
        log "Building Docker image..."
        
        cd "$PROJECT_ROOT"
        
        # Build image
        docker build -t "ghcr.io/e1washere/rag-service:$IMAGE_TAG" .
        
        # Push image
        log "Pushing Docker image to registry..."
        docker push "ghcr.io/e1washere/rag-service:$IMAGE_TAG"
        
        success "Docker image built and pushed successfully"
    else
        warning "Docker not available, skipping image build"
    fi
}

# Deploy infrastructure with Terraform
deploy_infrastructure() {
    log "Deploying infrastructure with Terraform..."
    
    cd "$PROJECT_ROOT/infra/terraform"
    
    # Set resource group name based on environment
    case $ENVIRONMENT in
        production)
            RESOURCE_GROUP_NAME="rag-service-rg"
            ;;
        staging)
            RESOURCE_GROUP_NAME="rag-service-staging-rg"
            ;;
        development)
            RESOURCE_GROUP_NAME="rag-service-dev-rg"
            ;;
    esac
    
    # Initialize Terraform
    log "Initializing Terraform..."
    terraform init
    
    # Create Terraform plan
    log "Creating Terraform plan..."
    terraform plan \
        -var="environment=$ENVIRONMENT" \
        -var="resource_group_name=$RESOURCE_GROUP_NAME" \
        -var="container_image=ghcr.io/e1washere/rag-service:$IMAGE_TAG" \
        -var="openai_api_key=${OPENAI_API_KEY:-}" \
        -var="groq_api_key=${GROQ_API_KEY:-}" \
        -var="langfuse_secret_key=${LANGFUSE_SECRET_KEY:-}" \
        -var="langfuse_public_key=${LANGFUSE_PUBLIC_KEY:-}" \
        -var="alert_webhook_url=${ALERT_WEBHOOK_URL:-}" \
        -out=tfplan
    
    # Apply Terraform plan
    log "Applying Terraform plan..."
    terraform apply -auto-approve tfplan
    
    # Get outputs
    APP_URL=$(terraform output -raw app_url)
    HEALTH_CHECK_URL=$(terraform output -raw health_check_url)
    
    success "Infrastructure deployed successfully"
    log "App URL: $APP_URL"
    log "Health Check URL: $HEALTH_CHECK_URL"
}

# Wait for application to be ready
wait_for_app() {
    local health_url="$1"
    local max_attempts=30
    local attempt=1
    
    log "Waiting for application to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$health_url" > /dev/null 2>&1; then
            success "Application is healthy!"
            return 0
        fi
        
        log "Attempt $attempt/$max_attempts: Application not ready yet, waiting 30s..."
        sleep 30
        attempt=$((attempt + 1))
    done
    
    error "Application failed to become healthy after $max_attempts attempts"
}

# Run post-deployment tests
run_smoke_tests() {
    local app_url="$1"
    
    log "Running post-deployment smoke tests..."
    
    # Test health endpoint
    if ! curl -f -s "$app_url/health" > /dev/null; then
        error "Health check failed"
    fi
    
    # Test API docs
    if ! curl -f -s "$app_url/docs" > /dev/null; then
        error "API docs check failed"
    fi
    
    # Test query endpoint
    if ! curl -f -s -X POST "$app_url/query" \
        -H "Content-Type: application/json" \
        -d '{"question": "What is machine learning?", "top_k": 3}' > /dev/null; then
        error "Query endpoint test failed"
    fi
    
    success "All smoke tests passed!"
}

# Main deployment flow
main() {
    log "🚀 Starting RAG Service deployment"
    log "Environment: $ENVIRONMENT"
    log "Image tag: $IMAGE_TAG"
    
    check_prerequisites
    
    if [[ "$IMAGE_TAG" != "latest" ]] || [[ -z "${SKIP_BUILD:-}" ]]; then
        build_and_push_image
    else
        log "Skipping image build (using existing image)"
    fi
    
    deploy_infrastructure
    
    # Get the app URL from Terraform output
    cd "$PROJECT_ROOT/infra/terraform"
    APP_URL=$(terraform output -raw app_url)
    HEALTH_CHECK_URL=$(terraform output -raw health_check_url)
    
    wait_for_app "$HEALTH_CHECK_URL"
    run_smoke_tests "$APP_URL"
    
    success "🎉 Deployment completed successfully!"
    echo
    echo "🔗 Application URLs:"
    echo "   App: $APP_URL"
    echo "   Health: $HEALTH_CHECK_URL"
    echo "   Docs: $APP_URL/docs"
    echo "   Stats: $APP_URL/stats"
    echo
    log "Deployment summary:"
    log "  Environment: $ENVIRONMENT"
    log "  Image: ghcr.io/e1washere/rag-service:$IMAGE_TAG"
    log "  Status: ✅ Healthy"
}

# Trap errors and cleanup
trap 'error "Deployment failed! Check the logs above for details."' ERR

# Run main function
main "$@"
