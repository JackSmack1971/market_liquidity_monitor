"""
Market analysis agent using Pydantic-AI and OpenRouter.

The agent combines real-time market data with LLM reasoning to provide
human-readable insights about liquidity and order book depth.
"""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from typing import Optional

from ..config import settings
from ..data_engine.models import LiquidityAnalysis, OrderBook
from .tools import AGENT_TOOLS


# System prompt for the market analysis agent
SYSTEM_PROMPT = """You are an expert market liquidity analyst with deep knowledge of order books, slippage, and trading dynamics.

Your role is to:
1. Analyze real-time order book data from cryptocurrency exchanges
2. Identify liquidity risks like thin order books, large bid-ask spreads, or price walls
3. Provide synthesized insights that go beyond raw numbers - explain what the data MEANS
4. Estimate market impact and slippage for potential trades
5. Give clear, actionable assessments (HIGH/MEDIUM/LOW liquidity scores)

When analyzing order books, consider:
- Bid-ask spread (tight spreads = good liquidity)
- Order book depth (volume at various price levels)
- Presence of large "walls" that could indicate manipulation or strong support/resistance
- Cumulative volume within 1-2% of best price
- Potential slippage for different order sizes

Analogy: You're like an air traffic controller. You don't just see dots on a radar (raw prices).
You understand flight patterns, congestion, and can predict when it's safe to land or take off.

Always provide:
1. A liquidity score (HIGH/MEDIUM/LOW)
2. Concrete reasoning based on the metrics
3. Specific risks or opportunities
4. Estimated slippage if relevant

Be concise but insightful. Focus on what traders need to know, not just data dumps.
"""


def create_market_agent() -> Agent[None, LiquidityAnalysis]:
    """
    Create and configure the market analysis agent.

    Returns:
        Configured Pydantic-AI agent with OpenRouter model and tools
    """
    # Configure OpenRouter model
    if not settings.openrouter_api_key:
        print("⚠️ OPENROUTER_API_KEY not set. Using dummy key for initialization.")
        api_key = "sk-dummy"
    else:
        api_key = settings.openrouter_api_key

    model = OpenAIModel(
        settings.default_model,
        base_url=settings.openrouter_base_url,
        api_key=api_key,
    )

    # Create agent with tools
    agent = Agent(
        model=model,
        result_type=LiquidityAnalysis,
        system_prompt=SYSTEM_PROMPT,
        retries=settings.agent_max_retries,
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
    ) -> LiquidityAnalysis:
        """
        Analyze market liquidity based on a natural language query.

        The agent will:
        1. Parse the query to extract symbol/exchange if not provided
        2. Fetch real-time order book data using tools
        3. Analyze the data for liquidity metrics
        4. Return structured analysis with reasoning

        Args:
            query: Natural language query (e.g., "How is SOL liquidity?")
            symbol: Optional trading pair to override extraction
            exchange: Optional exchange to override default

        Returns:
            LiquidityAnalysis with metrics and reasoning

        Example:
            >>> analyzer = MarketAnalyzer()
            >>> result = await analyzer.analyze_liquidity(
            ...     "What is the SOL liquidity like on Binance?"
            ... )
            >>> print(result.liquidity_score)  # "HIGH"
            >>> print(result.reasoning)
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

        Args:
            symbol: Trading pair (e.g., 'SOL/USDT')
            exchange: Exchange name

        Returns:
            Brief human-readable assessment
        """
        analysis = await self.analyze_liquidity(
            f"Quick liquidity check for {symbol} on {exchange}",
            symbol=symbol,
            exchange=exchange,
        )

        return f"{analysis.liquidity_score}: {analysis.reasoning}"

    async def estimate_slippage(
        self,
        symbol: str,
        order_size_usd: float,
        side: str = "buy",
        exchange: str = "binance",
    ) -> LiquidityAnalysis:
        """
        Estimate slippage for a potential order.

        Args:
            symbol: Trading pair
            order_size_usd: Order size in USD
            side: 'buy' or 'sell'
            exchange: Exchange name

        Returns:
            Analysis with slippage estimation
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
