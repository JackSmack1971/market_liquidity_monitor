# Makefile for Market Liquidity Monitor

.PHONY: help install test run-api run-frontend clean lint format

help:
	@echo "Market Liquidity Monitor - Available Commands"
	@echo ""
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run test suite"
	@echo "  make run-api       - Start FastAPI backend"
	@echo "  make run-frontend  - Start Streamlit frontend"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean up cache files"
	@echo ""

install:
	pip install -r requirements.txt
	pip install -r tests/requirements-test.txt
	@echo "✅ Dependencies installed"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env"
	@echo "  2. Add your OPENROUTER_API_KEY to .env"
	@echo "  3. Run 'make run-frontend' or 'make run-api'"

test:
	pytest tests/ -v --cov=market_liquidity_monitor --cov-report=term-missing

run-api:
	python -m market_liquidity_monitor.api.main

run-frontend:
	streamlit run market_liquidity_monitor/frontend/app.py

lint:
	@echo "Running linters..."
	flake8 market_liquidity_monitor/ --max-line-length=100 --ignore=E203,W503 || true
	mypy market_liquidity_monitor/ --ignore-missing-imports || true

format:
	@echo "Formatting code..."
	black market_liquidity_monitor/ tests/
	isort market_liquidity_monitor/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
	@echo "✅ Cleaned cache files"

setup-env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ Created .env from .env.example"; \
		echo "⚠️  Please edit .env and add your OPENROUTER_API_KEY"; \
	else \
		echo "⚠️  .env already exists, skipping"; \
	fi

dev-setup: setup-env install
	@echo ""
	@echo "✅ Development environment setup complete!"
	@echo ""
	@echo "Remember to:"
	@echo "  1. Edit .env and add your OPENROUTER_API_KEY"
	@echo "  2. Run 'make run-frontend' to start the UI"
	@echo ""
