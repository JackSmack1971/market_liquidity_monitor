# Dockerfile for Market Liquidity Monitor

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
# 8000 for FastAPI
# 8501 for Streamlit
EXPOSE 8000 8501

# Default command runs the API
# Override with docker run command for frontend
CMD ["python", "-m", "market_liquidity_monitor.api.main"]
