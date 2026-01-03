"""
Market analysis agent using Pydantic-AI and OpenRouter.

The agent combines real-time market data with LLM reasoning to provide
human-readable insights about liquidity and order book depth.
"""

from typing import Optional
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.fallback import FallbackModel

from config.settings import settings
from data_engine.models import LiquidityScorecard
from agents.tools import AGENT_TOOLS
import logfire

# Instrument Pydantic AI for Observability
if settings.logfire_token:
    logfire.instrument_pydantic_ai()


# System prompt for the market analysis agent
SYSTEM_PROMPT = """You are an expert market liquidity analyst with deep knowledge of order books, slippage, and trading dynamics.

Your role is to:
1. Analyze real-time order book data from cryptocurrency exchanges.
2. Identify liquidity risks like thin order books, large bid-ask spreads, or price walls.
3. Provide a structured Liquidity Scorecard (BaseModel) with validated metrics.
4. Estimate market impact and slippage for potential trades.
5. Give clear, actionable assessments through a 1-10 scorecard and risk warnings.

When analyzing order books, consider:
- Bid-ask spread (tight spreads = good liquidity).
- Order book depth (volume at various price levels).
- Presence of large "walls" that could indicate manipulation or strong support/resistance.
- Cumulative volume within 1-2% of best price.
- Potential slippage for different order sizes.
- Volatility: If spreads are wide or depth is unstable, mark as 'VOLATILE'.

USE TOOLS STRATEGICALLY:
- `get_order_book_depth` & `calculate_market_impact` -> Quantitative snapshot.
- `get_historical_metrics` -> Trend analysis, volatility checks, and "liquidity drought" detection.
- `get_market_metadata` -> IMPERATIVE before recommending specific order sizes.

MANDATORY: Return a LiquidityScorecard object.
- liquidity_score: 1-10 (10 is perfect liquidity).
- recommended_max_size: Practical limit for execution based on current depth.
- risk_factors: List of specific issues or 'None' if stable.
- summary_analysis: Concise professional narrative.
- Technical metrics: Taken directly from tool outputs (including `latency_ms` and `circuit_state`).

STRUCTURED VALIDATION:
- `confidence_score`: A float from 0.0 to 1.0. 
    - Reduce if data is stale (old timestamp).
    - Reduce if latency is high (>500ms).
    - Reduce if venue health is 'DEGRADED' or 'OPEN'.
- `system_health_status`: 
    - 'HEALTHY': Latency < 200ms and Circuit CLOSED.
    - 'DEGRADED': Latency > 500ms or Circuit HALF_OPEN.
    - 'CRITICAL': Circuit OPEN or multiple venue failures.

LATENCY & HEALTH PERFORMANCE:
- `latency_ms`: Report the network latency returned by tools. High latency (>500ms) may indicate market stress or REST API throttling.
- `circuit_state`: Report the health status of the venue. If 'OPEN', the venue is unstable and analysis is based on stale or cached data.

HIGH RISK THRESHOLD:
- If `slippage_bps` exceeds 200 bps, mark the situation as 'CRITICAL' in your summary and risk factors.
- Highlight that slippage > 2.0% (200 bps) indicates extreme liquidity drought or price manipulation walls.

CRITICAL INSTRUCTIONS:
1. Always call `calculate_market_impact` if user asks about a specific trade size.
2. Include the returned `MarketImpactReport` in your `LiquidityScorecard`.
3. Highlight the `slippage_bps` and `expected_fill_price` in your summary.
4. PRECISION: Always use exchange precision when recommending prices or amounts.
"""


def create_market_agent(api_key: Optional[str] = None) -> Agent[None, LiquidityScorecard]:
    """
    Create and configure the market analysis agent with FallbackModel.

    Args:
        api_key: OpenRouter API key provided by user (Session State).
    """
    # 1. Primary Model (OpenRouter)
    # Prefer injected key, fallback to settings (for dev/testing)
    openrouter_key = api_key or getattr(settings, 'openrouter_api_key', None)
    
    if not openrouter_key:
        # In Privacy-First mode, this implies strict failure if not provided
        # Use dummy key to allow instantiation (import-time), but requests will fail.
        print("Warning: No OpenRouter Key provided. Using placeholder.")
        openrouter_key = "unconfigured_key_placeholder"

    from pydantic_ai.providers.openai import OpenAIProvider
    from openai import AsyncOpenAI
    
    # Configure OpenRouter provider explicitly
    base_url = getattr(settings, 'openrouter_base_url', "https://openrouter.ai/api/v1")
    if not base_url.endswith("/"):
        base_url += "/"
        
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=openrouter_key,
    )
    # Using openai_client explicitly as per provider source code
    provider = OpenAIProvider(openai_client=client)
    
    primary_model = OpenAIChatModel(
        model_name=getattr(settings, 'default_model', "gpt-4o"),
        provider=provider,
    )

    # Gemini (Google) requires an API key even for initialization
    gemini_key = getattr(settings, 'google_api_key', None) or "dummy-gemini-key"
    from pydantic_ai.providers.google import GoogleProvider
    
    gemini_provider = GoogleProvider(api_key=gemini_key)
    fallback_model = GoogleModel("gemini-1.5-pro", provider=gemini_provider)

    # 3. Create FallbackModel
    combined_model = FallbackModel(primary_model, fallback_model)

    # Create the agent
    agent = Agent(
        combined_model,
        output_type=LiquidityScorecard,
        system_prompt=SYSTEM_PROMPT,
        retries=getattr(settings, 'agent_max_retries', 3),
    )

    # Register tools
    for tool in AGENT_TOOLS:
        agent.tool(tool)

    return agent


class MarketAnalyzer:
    """
    High-level interface for market liquidity analysis.

    Wraps the Pydantic-AI agent with convenience methods.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the market analyzer.
        
        Args:
            api_key: User provided OpenRouter key (Privacy-First).
        """
        self.agent = create_market_agent(api_key=api_key)

    async def analyze_liquidity(
        self,
        query: str,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> LiquidityScorecard:
        """
        Analyze market liquidity based on a natural language query.
        """
        # Build context for the agent
        context_parts = [query]

        if symbol:
            context_parts.append(f"Focus on trading pair: {symbol}")

        if exchange:
            context_parts.append(f"Use exchange: {exchange}")

        full_query = " | ".join(context_parts)

        # Run agent
        result = await self.agent.run(full_query)

        return result.data

    async def quick_check(
        self,
        symbol: str,
        exchange: str = "binance",
    ) -> str:
        """
        Quick liquidity check for a trading pair.
        """
        analysis = await self.analyze_liquidity(
            f"Quick liquidity check for {symbol} on {exchange}",
            symbol=symbol,
            exchange=exchange,
        )

        return f"{analysis.liquidity_score} ({analysis.volatility_rating}): {analysis.reasoning}"

    async def estimate_slippage(
        self,
        symbol: str,
        order_size_usd: float,
        side: str = "buy",
        exchange: str = "binance",
    ) -> LiquidityScorecard:
        """
        Estimate slippage for a potential order.
        """
        query = (
            f"Estimate slippage for a ${order_size_usd:,.0f} {side} order "
            f"of {symbol} on {exchange}"
        )

        return await self.analyze_liquidity(
            query,
            symbol=symbol,
            exchange=exchange,
        )


# Global analyzer instance
market_analyzer = MarketAnalyzer()
