"""
Streamlit chat interface for market liquidity monitoring.

Provides:
- Natural language chat interface
- Real-time order book visualization
- Interactive analysis with LLM reasoning
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from typing import Optional

# Direct imports (for when backend is running separately)
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from market_liquidity_monitor.agents import market_analyzer
from market_liquidity_monitor.data_engine import exchange_manager
from market_liquidity_monitor.config import settings


# Page configuration
st.set_page_config(
    page_title="Market Liquidity Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "current_orderbook" not in st.session_state:
        st.session_state.current_orderbook = None

    if "last_analysis" not in st.session_state:
        st.session_state.last_analysis = None

    if "current_symbol" not in st.session_state:
        st.session_state.current_symbol = None

    if "current_exchange" not in st.session_state:
        st.session_state.current_exchange = settings.default_exchange


@st.cache_resource
def get_cached_client(exchange_id: str):
    """
    Cache the CCXT exchange client to reuse connections and respect rate limits.
    """
    from market_liquidity_monitor.data_engine.exchange import ExchangeClient
    return ExchangeClient(exchange_id)


def generate_analysis_report():
    """
    Callable for st.download_button to generate report on demand.
    Ensures expensive logic only runs when user clicks.
    """
    analysis = st.session_state.get("last_analysis")
    if not analysis:
        return "No analysis available."

    report = f"""# Liquidity Analysis Report
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Symbol: {analysis.symbol}
Exchange: {analysis.exchange}

## Metrics
- **Liquidity Score:** {analysis.liquidity_score}
- **Spread:** {analysis.spread:.4f} ({analysis.spread_percentage:.3f}%)
- **Bid Depth (10 levels):** {analysis.bid_depth_10:,.2f}
- **Ask Depth (10 levels):** {analysis.ask_depth_10:,.2f}
"""
    if analysis.estimated_slippage_1k:
        report += f"- **Estimated slippage ($1k):** {analysis.estimated_slippage_1k:.3f}%\n"
    if analysis.estimated_slippage_10k:
        report += f"- **Estimated slippage ($10k):** {analysis.estimated_slippage_10k:.3f}%\n"

    report += f"\n## Reasoning\n{analysis.reasoning}\n"
    return report


def create_orderbook_chart(orderbook) -> go.Figure:
    """
    Create interactive order book depth chart.

    Args:
        orderbook: OrderBook object

    Returns:
        Plotly figure
    """
    # Prepare bid data
    bid_prices = [level.price for level in orderbook.bids]
    bid_volumes = [level.amount for level in orderbook.bids]
    bid_cumulative = pd.Series(bid_volumes).cumsum().tolist()

    # Prepare ask data
    ask_prices = [level.price for level in orderbook.asks]
    ask_volumes = [level.amount for level in orderbook.asks]
    ask_cumulative = pd.Series(ask_volumes).cumsum().tolist()

    # Create figure
    fig = go.Figure()

    # Bids (green)
    fig.add_trace(go.Scatter(
        x=bid_prices,
        y=bid_cumulative,
        mode='lines',
        name='Bids',
        line=dict(color='green', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 0, 0.1)',
    ))

    # Asks (red)
    fig.add_trace(go.Scatter(
        x=ask_prices,
        y=ask_cumulative,
        mode='lines',
        name='Asks',
        line=dict(color='red', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.1)',
    ))

    # Layout
    fig.update_layout(
        title=f"Order Book Depth - {orderbook.symbol}",
        xaxis_title="Price",
        yaxis_title="Cumulative Volume",
        hovermode='x unified',
        showlegend=True,
        height=400,
    )

    return fig


def display_orderbook_metrics(orderbook):
    """
    Display key order book metrics.

    Args:
        orderbook: OrderBook object
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Best Bid",
            f"${orderbook.best_bid.price:,.4f}" if orderbook.best_bid else "N/A",
        )

    with col2:
        st.metric(
            "Best Ask",
            f"${orderbook.best_ask.price:,.4f}" if orderbook.best_ask else "N/A",
        )

    with col3:
        st.metric(
            "Spread",
            f"${orderbook.spread:,.4f}" if orderbook.spread else "N/A",
            f"{orderbook.spread_percentage:.3f}%" if orderbook.spread_percentage else None,
        )

    with col4:
        bid_depth = orderbook.get_cumulative_volume("bids", 10)
        st.metric(
            "Bid Depth (10 levels)",
            f"{bid_depth:,.2f}",
        )


@st.fragment(run_every="5s")
def orderbook_visualization():
    """Fragment for real-time order book updates."""
    # Poll for fresh data if we have a symbol
    if st.session_state.current_symbol:
        symbol = st.session_state.current_symbol
        exchange = st.session_state.current_exchange
        
        try:
            client = get_cached_client(exchange)
            # Fetch fresh order book
            orderbook = asyncio.run(client.fetch_order_book(symbol, limit=20))
            st.session_state.current_orderbook = orderbook
        except Exception as e:
            st.error(f"Polling error: {e}")

    if st.session_state.current_orderbook:
        orderbook = st.session_state.current_orderbook

        st.subheader(f"Order Book: {orderbook.symbol}")
        display_orderbook_metrics(orderbook)

        # Chart
        fig = create_orderbook_chart(orderbook)
        st.plotly_chart(fig, use_container_width=True)

        # Timestamp
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")


async def process_query(query: str) -> str:
    """
    Process user query with LLM agent.

    Args:
        query: Natural language query

    Returns:
        Response message
    """
    try:
        # Check if query mentions specific symbol
        symbol = None
        exchange = settings.default_exchange

        # Simple extraction (in production, LLM handles this)
        if "SOL" in query.upper():
            symbol = "SOL/USDT"
        elif "BTC" in query.upper():
            symbol = "BTC/USDT"
        elif "ETH" in query.upper():
            symbol = "ETH/USDT"

        # Get analysis from agent
        analysis = await market_analyzer.analyze_liquidity(
            query=query,
            symbol=symbol,
            exchange=exchange,
        )

        # Store and update current tracking state
        st.session_state.last_analysis = analysis
        st.session_state.current_symbol = symbol
        st.session_state.current_exchange = exchange

        # Also fetch and store order book for visualization
        if symbol:
            client = get_cached_client(exchange)
            orderbook = await client.fetch_order_book(symbol, limit=20)
            st.session_state.current_orderbook = orderbook

        # Format response
        response = f"""
**Liquidity Score:** {analysis.liquidity_score}

**Analysis:**
{analysis.reasoning}

**Metrics:**
- Spread: {analysis.spread:.4f} ({analysis.spread_percentage:.3f}%)
- Bid Depth (10 levels): {analysis.bid_depth_10:,.2f}
- Ask Depth (10 levels): {analysis.ask_depth_10:,.2f}
"""

        if analysis.estimated_slippage_1k:
            response += f"\n- Estimated slippage ($1k order): {analysis.estimated_slippage_1k:.3f}%"

        if analysis.estimated_slippage_10k:
            response += f"\n- Estimated slippage ($10k order): {analysis.estimated_slippage_10k:.3f}%"

        return response

    except Exception as e:
        return f"Error processing query: {str(e)}\n\nPlease check your configuration and ensure the backend is running."


def main():
    """Main Streamlit application."""
    initialize_session_state()

    # Header
    st.title("📊 Market Liquidity Monitor")
    st.markdown("*Powered by CCXT + LLM Reasoning*")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        exchange = st.selectbox(
            "Exchange",
            ["binance", "coinbase", "kraken", "bybit"],
            index=0,
        )

        st.markdown("---")
        st.subheader("Example Queries")
        st.markdown("""
        - "What is the SOL liquidity like?"
        - "Show me BTC order book depth"
        - "Can I execute a $10k ETH sell order?"
        - "Estimate slippage for 5 SOL purchase"
        """)

        st.markdown("---")
        st.caption(f"Model: {settings.default_model}")

        # Download button for latest analysis (using callable for on-demand generation)
        if st.session_state.last_analysis:
            st.markdown("---")
            st.download_button(
                label="📥 Download Analysis Report",
                data=generate_analysis_report,
                file_name=f"liquidity_report_{datetime.now().strftime('%Y%md_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Chat Interface")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about market liquidity..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing market data..."):
                    response = asyncio.run(process_query(prompt))
                    st.markdown(response)

            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Trigger rerun to update visualization
            st.rerun(scope="fragment")

    with col2:
        orderbook_visualization()


if __name__ == "__main__":
    main()
