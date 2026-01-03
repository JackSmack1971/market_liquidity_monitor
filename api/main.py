"""
FastAPI application entry point.

Configures the API server with CORS, routes, and lifecycle management.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import logfire
from contextlib import asynccontextmanager

from config import settings
from data_engine import exchange_manager, cache_manager, stream_manager
from data_engine.database import db_manager
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown tasks.
    """
    # Configure Logfire
    if settings.logfire_token:
        logfire.configure(
            service_name=settings.logfire_service_name,
            token=settings.logfire_token,
            environment=settings.logfire_environment
        )
        logfire.instrument_fastapi(app)
        print("Logfire observability enabled.")
    
    # Startup
    print("Starting Market Liquidity Monitor API...")
    print(f"Default exchange: {getattr(settings, 'default_exchange', 'binance')}")
    print(f"LLM model: {getattr(settings, 'default_model', 'gpt-4o')}")

    # Connect to cache and database
    await cache_manager.connect()
    await db_manager.connect()

    # Pre-load markets for priority exchanges
    priority_exchanges = ["binance", "coinbase", "kraken"]
    print(f"Pre-loading markets for: {', '.join(priority_exchanges)}...")
    await asyncio.gather(
        *[exchange_manager.preload_exchange(ex) for ex in priority_exchanges],
        return_exceptions=True
    )

    yield

    # Shutdown
    print("Shutting down...")
    await stream_manager.stop_all()
    await exchange_manager.close_all()
    await cache_manager.disconnect()
    await db_manager.disconnect()
    print("All connections closed")


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
    allow_origins=getattr(settings, 'cors_origins', ["*"]),
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
        "api.main:app",
        host=getattr(settings, 'api_host', "0.0.0.0"),
        port=getattr(settings, 'api_port', 8000),
        reload=getattr(settings, 'api_reload', False),
    )
