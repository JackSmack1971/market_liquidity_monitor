"""
Enhanced Streamlit interface with advanced features.

Features:
- Multi-exchange comparison
- Historical tracking
- Real-time alerts
- Advanced visualizations (heatmaps, trends)
- Market impact simulation
"""

import streamlit as st
import asyncio
import sys
from pathlib import Path
from datetime import timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from market_liquidity_monitor.agents import market_analyzer
from market_liquidity_monitor.agents.tools import compare_exchanges, calculate_market_impact
from market_liquidity_monitor.data_engine import exchange_manager
from market_liquidity_monitor.data_engine.historical import historical_tracker
from market_liquidity_monitor.config import settings
from market_liquidity_monitor.frontend.advanced_visualizations import (
    create_liquidity_heatmap,
    create_exchange_comparison_chart,
    create_historical_trend_chart,
    create_alerts_dashboard,
    create_market_impact_chart,
)


# Page configuration
st.set_page_config(
    page_title="Advanced Market Liquidity Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
    # Find the last analysis from messages or session state
    # enhanced_app.py stores messages in st.session_state.messages
    if not st.session_state.messages:
        return "No analysis available."
    
    # Get the last assistant message
    last_assistant_msg = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
    if not last_assistant_msg:
        return "No analysis response found."
        
    report = f"""# Advanced Liquidity Analysis Report
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Symbol: {st.session_state.current_symbol}
Exchange: {st.session_state.current_exchange}

{last_assistant_msg}
"""
    return report


def initialize_session_state():
    """Initialize session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_symbol" not in st.session_state:
        st.session_state.current_symbol = "SOL/USDT"
    if "current_exchange" not in st.session_state:
        st.session_state.current_exchange = "binance"
    if "orderbook" not in st.session_state:
        st.session_state.orderbook = None
    if "comparison_data" not in st.session_state:
        st.session_state.comparison_data = None
    if "alerts" not in st.session_state:
        st.session_state.alerts = []


async def fetch_orderbook(symbol: str, exchange: str):
    """Fetch order book."""
    client = get_cached_client(exchange)
    return await client.fetch_order_book(symbol, limit=50)


async def run_comparison(symbol: str, exchanges: list):
    """Run multi-exchange comparison."""
    return await compare_exchanges(
        symbol=symbol,
        exchanges=exchanges,
        levels=20
    )


async def check_alerts(symbol: str, exchange: str):
    """Check for liquidity alerts."""
    return await historical_tracker.detect_anomalies(
        symbol=symbol,
        exchange=exchange,
        threshold_pct=30.0
    )


def main():
    """Main application."""
    initialize_session_state()

    # Title
    st.title("📊 Advanced Market Liquidity Monitor")
    st.markdown("*Multi-exchange comparison • Historical tracking • Real-time alerts*")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Symbol selection
        symbol = st.selectbox(
            "Trading Pair",
            ["SOL/USDT", "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT"],
            index=0
        )
        st.session_state.current_symbol = symbol

        # Exchange selection
        exchange = st.selectbox(
            "Primary Exchange",
            ["binance", "coinbase", "kraken", "bybit"],
            index=0
        )
        st.session_state.current_exchange = exchange

        st.markdown("---")

        # Feature toggles
        st.subheader("📈 Features")
        show_heatmap = st.checkbox("Liquidity Heatmap", value=True)
        show_comparison = st.checkbox("Multi-Exchange Comparison", value=True)
        show_historical = st.checkbox("Historical Trends", value=False)
        show_alerts = st.checkbox("Alert Dashboard", value=True)

        st.markdown("---")

        # Actions
        st.subheader("🎯 Actions")

        if st.button("🔄 Refresh Data", use_container_width=True):
            with st.spinner("Fetching market data..."):
                orderbook = asyncio.run(fetch_orderbook(symbol, exchange))
                st.session_state.orderbook = orderbook
                st.success("Data refreshed!")
                st.rerun()

        if st.button("⚡ Check Alerts", use_container_width=True):
            with st.spinner("Analyzing liquidity..."):
                alerts = asyncio.run(check_alerts(symbol, exchange))
                st.session_state.alerts = alerts
                if alerts:
                    st.warning(f"Found {len(alerts)} alert(s)!")
                else:
                    st.success("No anomalies detected")
                st.rerun()

        st.markdown("---")
        st.caption(f"Model: {settings.default_model}")

        # Download button for latest analysis
        if any(m["role"] == "assistant" for m in st.session_state.messages):
            st.markdown("---")
            st.download_button(
                label="📥 Download Analysis Report",
                data=generate_analysis_report,
                file_name=f"adv_liquidity_report_{datetime.now().strftime('%Y%md_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chat & Analysis",
        "📊 Order Book",
        "🔍 Multi-Exchange",
        "📈 Historical"
    ])

    # TAB 1: Chat & Analysis
    with tab1:
        st.subheader("Natural Language Analysis")

        # Chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about liquidity..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Process query
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        analysis = asyncio.run(
                            market_analyzer.analyze_liquidity(
                                query=prompt,
                                symbol=symbol,
                                exchange=exchange
                            )
                        )

                        response = f"""
**Liquidity Score:** {analysis.liquidity_score}

**Analysis:**
{analysis.reasoning}

**Metrics:**
- Spread: {analysis.spread:.4f} ({analysis.spread_percentage:.3f}%)
- Bid Depth: {analysis.bid_depth_10:,.2f}
- Ask Depth: {analysis.ask_depth_10:,.2f}
"""
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                        # Fetch orderbook for visualization
                        orderbook = asyncio.run(fetch_orderbook(symbol, exchange))
                        st.session_state.orderbook = orderbook

                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

        # Market Impact Simulator
        st.markdown("---")
        st.subheader("💰 Market Impact Simulator")

        col1, col2, col3 = st.columns(3)
        with col1:
            order_size = st.number_input(
                "Order Size (USD)",
                min_value=100,
                max_value=1000000,
                value=10000,
                step=1000
            )
        with col2:
            side = st.selectbox("Side", ["buy", "sell"])
        with col3:
            if st.button("Calculate Impact", use_container_width=True):
                with st.spinner("Simulating execution..."):
                    impact = asyncio.run(
                        calculate_market_impact(
                            symbol=symbol,
                            order_size_usd=order_size,
                            side=side,
                            exchange=exchange
                        )
                    )

                    if "error" not in impact:
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Slippage", f"{impact['slippage_percentage']:.3f}%")
                        col_b.metric("Avg Price", f"${impact['average_execution_price']:.4f}")
                        col_c.metric("Levels Consumed", impact['levels_consumed'])

                        # Impact visualization
                        fig = create_market_impact_chart(impact)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(impact['error'])

    # TAB 2: Order Book Visualization
    with tab2:
        render_orderbook_fragment()

@st.fragment(run_every="5s")
def render_orderbook_fragment():
    """Fragment for real-time order book updates."""
    symbol = st.session_state.current_symbol
    exchange = st.session_state.current_exchange
    
    # Polling for fresh data
    try:
        orderbook = asyncio.run(fetch_orderbook(symbol, exchange))
        st.session_state.orderbook = orderbook
    except Exception as e:
        st.error(f"Polling error: {e}")

    if st.session_state.orderbook:
        orderbook = st.session_state.orderbook

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Best Bid",
                f"${orderbook.best_bid.price:,.4f}" if orderbook.best_bid else "N/A"
            )
        with col2:
            st.metric(
                "Best Ask",
                f"${orderbook.best_ask.price:,.4f}" if orderbook.best_ask else "N/A"
            )
        with col3:
            st.metric(
                "Spread",
                f"{orderbook.spread_percentage:.3f}%" if orderbook.spread_percentage else "N/A"
            )
        with col4:
            depth = orderbook.get_cumulative_volume("bids", 10)
            st.metric("Bid Depth (10L)", f"{depth:,.2f}")

        # Heatmap
        # (show_heatmap is from the main scope, but since this is a fragment 
        # it will refresh when the main script reruns if needed, 
        # but here we just use the current state)
        st.subheader("🔥 Liquidity Heatmap")
        from market_liquidity_monitor.frontend.advanced_visualizations import create_liquidity_heatmap
        fig = create_liquidity_heatmap(orderbook, levels=30)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No order book data loaded yet.")

    # TAB 3: Multi-Exchange Comparison
    with tab3:
        st.subheader("🌐 Multi-Exchange Liquidity Comparison")

        # Exchange selection
        exchanges_to_compare = st.multiselect(
            "Select Exchanges",
            ["binance", "coinbase", "kraken", "bybit", "okx"],
            default=["binance", "coinbase", "kraken"]
        )

        if st.button("Compare Exchanges", use_container_width=True):
            if len(exchanges_to_compare) < 2:
                st.warning("Select at least 2 exchanges")
            else:
                with st.spinner(f"Fetching from {len(exchanges_to_compare)} exchanges..."):
                    comparison = asyncio.run(run_comparison(symbol, exchanges_to_compare))
                    st.session_state.comparison_data = comparison

        if st.session_state.comparison_data:
            comp = st.session_state.comparison_data

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Bid Exchange", comp.get("best_bid_exchange", "N/A"))
            with col2:
                st.metric("Best Ask Exchange", comp.get("best_ask_exchange", "N/A"))
            with col3:
                st.metric("Tightest Spread", comp.get("tightest_spread_exchange", "N/A"))
            with col4:
                st.metric("Deepest Liquidity", comp.get("deepest_liquidity_exchange", "N/A"))

            # Arbitrage opportunity
            if comp.get("arbitrage_opportunity_pct"):
                st.warning(
                    f"⚡ **ARBITRAGE OPPORTUNITY:** {comp['arbitrage_opportunity_pct']:.2f}%\n\n"
                    f"**Route:** {comp['arbitrage_route']}"
                )

            # Comparison chart
            fig = create_exchange_comparison_chart(comp)
            st.plotly_chart(fig, use_container_width=True)

            # Detailed table
            if st.checkbox("Show detailed comparison table"):
                import pandas as pd
                df = pd.DataFrame(comp.get("order_books", []))
                st.dataframe(df, use_container_width=True)

    # TAB 4: Historical Trends
    with tab4:
        st.subheader("📈 Historical Liquidity Analysis")

        # Time range selection
        col1, col2 = st.columns(2)
        with col1:
            hours = st.slider("Lookback Period (hours)", 1, 168, 24)
        with col2:
            if st.button("Capture Snapshot", use_container_width=True):
                with st.spinner("Capturing snapshot..."):
                    snapshot = asyncio.run(
                        historical_tracker.capture_snapshot(symbol, exchange)
                    )
                    st.success(f"Snapshot captured at {snapshot.timestamp}")

        # Fetch and display historical data
        if show_historical or st.button("Load Historical Data"):
            with st.spinner("Loading historical data..."):
                snapshots = asyncio.run(
                    historical_tracker.get_snapshots(
                        symbol=symbol,
                        exchange=exchange,
                        hours=hours
                    )
                )

                if snapshots:
                    st.info(f"Loaded {len(snapshots)} snapshots")

                    # Trend chart
                    fig = create_historical_trend_chart(snapshots)
                    st.plotly_chart(fig, use_container_width=True)

                    # Baseline metrics
                    baseline = asyncio.run(
                        historical_tracker.get_baseline_metrics(
                            symbol=symbol,
                            exchange=exchange,
                            hours=hours
                        )
                    )

                    if baseline:
                        st.subheader("Baseline Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Avg Spread", f"{baseline.get('avg_spread_pct', 0):.3f}%")
                        col2.metric("Avg Volume", f"{baseline.get('avg_volume', 0):.2f}")
                        col3.metric("Avg Liquidity", f"${baseline.get('avg_liquidity_1pct_usd', 0):,.0f}")
                        col4.metric("Samples", baseline.get('sample_count', 0))

                else:
                    st.warning("No historical data available. Capture some snapshots first!")

    # Alerts sidebar (if enabled)
    if show_alerts and st.session_state.alerts:
        with st.sidebar:
            st.markdown("---")
            st.subheader("🚨 Active Alerts")
            for alert in st.session_state.alerts[:5]:  # Show top 5
                severity_color = {
                    "HIGH": "🔴",
                    "MEDIUM": "🟠",
                    "LOW": "🟡"
                }
                st.warning(
                    f"{severity_color.get(alert.severity, '⚪')} **{alert.alert_type}**\n\n"
                    f"{alert.message}"
                )


if __name__ == "__main__":
    main()
