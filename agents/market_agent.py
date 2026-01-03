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

from config import settings
from data_engine.models import LiquidityScorecard
from agents.tools import AGENT_TOOLS


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
- Technical metrics: Taken directly from tool outputs.

If a tool returns an error or asks for clarification (ModelRetry), follow its guidance strictly.
"""


def create_market_agent() -> Agent[None, LiquidityScorecard]:
    """
    Create and configure the market analysis agent with FallbackModel.

    Primary: OpenRouter (specified in settings)
    Fallback: Gemini 1.5 Pro (resiliency)
    """
    # Primary Model (OpenRouter/OpenAI compatible)
    openrouter_key = getattr(settings, 'openrouter_api_key', None)
    if not openrouter_key:
        print("Warning: OPENROUTER_API_KEY not found. Using dummy key for testing.")
        openrouter_key = "dummy-openrouter-key"

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

    def __init__(self):
        """Initialize the market analyzer."""
        self.agent = create_market_agent()

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
