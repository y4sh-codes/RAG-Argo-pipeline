#!/bin/bash

# RAG Argo Pipeline Deployment Script
# Production-ready deployment with health checks and monitoring

set -e

# Configuration
APP_NAME="rag-argo-pipeline"
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f .env ]; then
        log_warn ".env file not found, copying from .env.example"
        if [ -f .env.example ]; then
            cp .env.example .env
            log_warn "Please configure your .env file before continuing"
            exit 1
        else
            log_error ".env.example not found"
            exit 1
        fi
    fi
    
    log_info "Prerequisites check passed ✓"
}

# Create necessary directories
create_directories() {
    log_info "Creating directories..."
    
    mkdir -p data/pdfs
    mkdir -p outputs
    mkdir -p logs
    mkdir -p ssl
    mkdir -p grafana/dashboards
    mkdir -p grafana/datasources
    
    log_info "Directories created ✓"
}

# Pull latest images
pull_images() {
    log_info "Pulling Docker images..."
    
    docker-compose pull
    
    log_info "Images pulled ✓"
}

# Build application
build_app() {
    log_info "Building application..."
    
    docker-compose build --no-cache rag-api
    
    log_info "Application built ✓"
}

# Start services
start_services() {
    log_info "Starting services..."
    
    # Start infrastructure services first
    docker-compose up -d qdrant redis
    
    # Wait for services to be ready
    log_info "Waiting for infrastructure services..."
    sleep 10
    
    # Start main application
    docker-compose up -d rag-api
    
    # Start monitoring (optional)
    docker-compose up -d prometheus grafana
    
    # Start nginx last
    docker-compose up -d nginx
    
    log_info "Services started ✓"
}

# Health check
health_check() {
    log_info "Performing health checks..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log_info "Health check attempt $attempt/$max_attempts"
        
        if curl -s -f http://localhost/health > /dev/null; then
            log_info "Health check passed ✓"
            return 0
        fi
        
        sleep 10
        ((attempt++))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Show status
show_status() {
    log_info "Service status:"
    docker-compose ps
    
    echo
    log_info "Access points:"
    echo "  • API: http://localhost"
    echo "  • Health: http://localhost/health"
    echo "  • Metrics: http://localhost:9090 (Prometheus)"
    echo "  • Dashboards: http://localhost:3000 (Grafana)"
    echo
    
    log_info "Logs:"
    echo "  • View all logs: docker-compose logs -f"
    echo "  • View API logs: docker-compose logs -f rag-api"
    echo "  • View DB logs: docker-compose logs -f qdrant"
}

# Stop services
stop_services() {
    log_info "Stopping services..."
    docker-compose down
    log_info "Services stopped ✓"
}

# Clean up
cleanup() {
    log_info "Cleaning up..."
    
    # Stop and remove containers
    docker-compose down -v
    
    # Remove unused images
    docker image prune -f
    
    log_info "Cleanup completed ✓"
}

# Backup data
backup_data() {
    log_info "Creating backup..."
    
    local backup_dir="backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup volumes
    docker run --rm -v rag_argo_pipeline_qdrant_data:/data -v $(pwd):/backup alpine tar czf /backup/$backup_dir/qdrant_data.tar.gz -C /data .
    docker run --rm -v rag_argo_pipeline_redis_data:/data -v $(pwd):/backup alpine tar czf /backup/$backup_dir/redis_data.tar.gz -C /data .
    
    # Backup configuration
    cp -r data "$backup_dir/"
    cp -r outputs "$backup_dir/"
    cp .env "$backup_dir/"
    
    log_info "Backup created in $backup_dir ✓"
}

# Update application
update_app() {
    log_info "Updating application..."
    
    # Create backup first
    backup_data
    
    # Pull latest changes (if using git)
    if [ -d .git ]; then
        git pull origin main
    fi
    
    # Rebuild and restart
    build_app
    docker-compose up -d --force-recreate rag-api
    
    # Health check
    health_check
    
    log_info "Application updated ✓"
}

# Show logs
show_logs() {
    local service="${1:-}"
    
    if [ -n "$service" ]; then
        docker-compose logs -f "$service"
    else
        docker-compose logs -f
    fi
}

# Main function
main() {
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            create_directories
            pull_images
            build_app
            start_services
            health_check
            show_status
            ;;
        "start")
            start_services
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            stop_services
            start_services
            ;;
        "status")
            show_status
            ;;
        "health")
            health_check
            ;;
        "logs")
            show_logs "${2:-}"
            ;;
        "backup")
            backup_data
            ;;
        "update")
            update_app
            ;;
        "cleanup")
            cleanup
            ;;
        *)
            echo "Usage: $0 {deploy|start|stop|restart|status|health|logs|backup|update|cleanup}"
            echo
            echo "Commands:"
            echo "  deploy  - Full deployment (default)"
            echo "  start   - Start all services"
            echo "  stop    - Stop all services"
            echo "  restart - Restart all services"
            echo "  status  - Show service status"
            echo "  health  - Check application health"
            echo "  logs    - Show logs (optionally specify service)"
            echo "  backup  - Create data backup"
            echo "  update  - Update and restart application"
            echo "  cleanup - Stop services and clean up"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
