# Docker Deployment Guide

## Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

### 2. Build and Run

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service health
docker-compose ps
```

### 3. Access the Application

- **Streamlit UI**: <http://localhost:8501>
- **FastAPI Docs**: <http://localhost:8000/docs>
- **Health Check**: <http://localhost:8000/health>

## Services

### API (FastAPI)

- **Port**: 8000
- **Workers**: 4 (Gunicorn + Uvicorn)
- **Timeout**: 120s
- **Health Check**: `/health` endpoint

### Frontend (Streamlit)

- **Port**: 8501
- **Health Check**: `/_stcore/health` endpoint

### Redis (Cache)

- **Port**: 6379
- **Persistence**: Append-only file (AOF)

### PostgreSQL (Database)

- **Port**: 5432
- **Database**: mlm
- **User**: postgres

## Production Deployment

### Security Hardening

1. **Change default passwords** in docker-compose.yml
2. **Use secrets management** (Docker Swarm secrets, Kubernetes secrets)
3. **Enable TLS/SSL** with reverse proxy (Nginx, Traefik)
4. **Restrict network access** using firewall rules

### Scaling

```bash
# Scale API workers
docker-compose up -d --scale api=3

# Monitor resource usage
docker stats
```

### Monitoring

- **Logfire Dashboard**: View spans and traces
- **Container Logs**: `docker-compose logs -f api`
- **Health Checks**: Automatic restart on failure

## Maintenance

### Update Application

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

### Database Migrations

```bash
# Run migrations
docker-compose exec api alembic upgrade head

# Create new migration
docker-compose exec api alembic revision --autogenerate -m "description"
```

### Backup Data

```bash
# Backup PostgreSQL
docker-compose exec db pg_dump -U postgres mlm > backup.sql

# Backup Redis
docker-compose exec redis redis-cli SAVE
docker cp mlm_redis:/data/dump.rdb ./redis_backup.rdb
```

### Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (WARNING: deletes data)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## Troubleshooting

### API Not Starting

```bash
# Check logs
docker-compose logs api

# Verify environment variables
docker-compose exec api env | grep OPENROUTER
```

### Database Connection Issues

```bash
# Check database health
docker-compose exec db pg_isready -U postgres

# Restart database
docker-compose restart db
```

### Redis Connection Issues

```bash
# Test Redis connection
docker-compose exec redis redis-cli ping

# Restart Redis
docker-compose restart redis
```

## Development Mode

For local development without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn api.main:app --reload

# Run Frontend (separate terminal)
streamlit run frontend/app.py
```
