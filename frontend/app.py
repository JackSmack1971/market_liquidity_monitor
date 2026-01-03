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
from market_liquidity_monitor.data_engine.models import LiquidityScorecard, LiquidityScorecardPartial


async def run_analysis(prompt: str, symbol: str, exchange: str, status, text_placeholder):
    """Async helper to handle streaming analysis."""
    stream_generator = market_analyzer.analyze_liquidity_stream(
        query=prompt,
        symbol=symbol,
        exchange=exchange
    )

    final_result = None
    
    async for stream_result in stream_generator:
        # Check if the stream_result itself yields partials (Pydantic-AI behavior)
        # Note: run_stream returns a StreamedRunResult.
        # We iterate over result.stream() for text/deltas, or get_data() for final.
        # But wait, run_stream context manager yields the result object.
        # My backend yields `result`. So `stream_generator` yields `RunResult`.
        # This wrapper is correct based on my backend implementation.
        
        async for partial in stream_result.stream():
             if isinstance(partial, Exception):
                 continue
             
             try:
                 if hasattr(partial, 'risk_factors') and partial.risk_factors:
                     status.update(label="Identifying risk factors...", state="running")
                 elif hasattr(partial, 'liquidity_score') and partial.liquidity_score:
                      status.update(label=f"Calculating Liquidity Score... ({partial.liquidity_score}/10)", state="running")
                 elif hasattr(partial, 'summary_analysis') and partial.summary_analysis:
                     status.update(label="Drafting analysis...", state="running")
                     text_placeholder.markdown(partial.summary_analysis + "▌")
             except Exception:
                 pass
        
        final_result = await stream_result.get_data()
        usage = stream_result.usage()
    
    return final_result, usage


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
    """
    analysis = st.session_state.get("last_analysis")
    if not analysis:
        return "No analysis available."

    report = f"""# Liquidity Scorecard Report
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Symbol: {analysis.symbol}
Exchange: {analysis.exchange}

## Scorecard
- **Liquidity Score:** {analysis.liquidity_score}/10
- **Est. Slippage:** {analysis.estimated_slippage_percent:.3f}%
- **Max Order Size:** {analysis.recommended_max_size}

## Risk Factors
"""
    if analysis.risk_factors:
        for factor in analysis.risk_factors:
            report += f"- {factor}\n"
    else:
        report += "- None identified (Stable)\n"

    report += f"""
## Technical Metrics
- **Volatility:** {analysis.volatility_rating}
- **Spread:** {analysis.spread_pct:.3f}%
- **Bid Depth:** {analysis.bid_depth_10:,.2f}
- **Ask Depth:** {analysis.ask_depth_10:,.2f}

## Analysis Summary
{analysis.summary_analysis}
"""
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

        # Audio input (Streamlit 1.52+)
        audio_value = st.audio_input("🎤 Voice Query")
        
        # Chat input
        if (prompt := st.chat_input("Ask about market liquidity...")) or audio_value:
            
            # Handle audio input
            if audio_value and not prompt:
                prompt = "Analyze this audio query (simulated transcription)" 
                # Note: In a real app, we'd transcribe `audio_value` here using Whisper/STT.
                # For now, we'll assume the user typed if they didn't, or just generic placeholder.
                # Actually, let's just use the chat input if audio is not fully implemented with STT.
                if audio_value:
                     st.info("Audio received! (Transcription implementation pending)")
                     # prompt = transcribe(audio_value) 

            if not prompt:
                st.stop()
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response
            # Get response
            with st.chat_message("assistant"):
                # Use st.status to show the "Thinking process" (tool calls, streaming)
                with st.status("Analyzing market...", expanded=True) as status:
                    final_result = None
                    text_placeholder = st.empty()
                    
                    try:
                        final_result, usage = asyncio.run(run_analysis(prompt, symbol, exchange, status, text_placeholder))
                        
                        # Display Usage Cost
                        if usage:
                            cost_info = f"Tokens: {usage.total_tokens} (In: {usage.request_tokens}, Out: {usage.response_tokens})"
                            status.update(label=f"Analysis Complete - {cost_info}", state="complete", expanded=False)
                        else:
                            status.update(label="Analysis Complete", state="complete", expanded=False)
                            
                    except Exception as e:
                        status.update(label="Analysis Failed", state="error", expanded=False)
                        st.error(f"Error: {e}")
                    
                    if final_result:
                         text_placeholder.empty() # Clear streaming text
                         
                         # Render Scorecard
                         with st.container(border=True):
                            st.subheader("Liquidity Scorecard")
                            c1, c2, c3 = st.columns(3)
                            
                            c1.metric("Liquidity Score", f"{final_result.liquidity_score}/10")
                            c2.metric("Est. Slippage", f"{final_result.estimated_slippage_percent:.3f}%")
                            c3.metric("Max Order Size", f"{final_result.recommended_max_size}")
                            
                            st.markdown(f"**Analysis:** {final_result.summary_analysis}")
                            
                            if final_result.risk_factors and final_result.risk_factors != ["None"]:
                                st.warning(f"⚠️ **Risk Factors:** {', '.join(final_result.risk_factors)}")

                         # Track response for session state
                         response = final_result.summary_analysis
                         
                         # Store last analysis
                         st.session_state.last_analysis = final_result
                         st.session_state.current_symbol = symbol or final_result.symbol
                         st.session_state.current_exchange = exchange or final_result.exchange

            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Trigger rerun to update visualization
            st.rerun(scope="fragment")

    with col2:
        orderbook_visualization()


if __name__ == "__main__":
    main()
