# Multi-stage Dockerfile for Market Liquidity Monitor
# Stage 1: Builder - Install dependencies
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Final - Production image
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create Streamlit config directory
RUN mkdir -p /app/.streamlit && \
    chown -R appuser:appuser /app/.streamlit

# Create Streamlit config to disable telemetry and set server address
RUN echo '[server]\n\
    address = "0.0.0.0"\n\
    port = 8501\n\
    headless = true\n\
    \n\
    [browser]\n\
    gatherUsageStats = false\n\
    \n\
    [theme]\n\
    base = "dark"\n' > /app/.streamlit/config.toml

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 8501

# Health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
