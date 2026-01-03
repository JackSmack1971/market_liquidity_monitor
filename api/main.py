"""
FastAPI application entry point.

Configures the API server with CORS, routes, and lifecycle management.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from ..config import settings
from ..data_engine import exchange_manager
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown tasks.
    """
    # Startup
    print("Starting Market Liquidity Monitor API...")
    print(f"Default exchange: {settings.default_exchange}")
    print(f"LLM model: {settings.default_model}")

    yield

    # Shutdown
    print("Shutting down...")
    await exchange_manager.close_all()
    print("All exchange connections closed")


# Create FastAPI application
app = FastAPI(
    title="Market Liquidity Monitor",
    description="Real-time market liquidity analysis powered by CCXT and LLM reasoning",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["market"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Market Liquidity Monitor API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "market_liquidity_monitor.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
