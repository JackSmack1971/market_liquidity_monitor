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
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from market_liquidity_monitor.agents import market_analyzer
from market_liquidity_monitor.agents.tools import compare_exchanges, calculate_market_impact
from market_liquidity_monitor.data_engine import exchange_manager
from market_liquidity_monitor.data_engine.historical import historical_tracker
from market_liquidity_monitor.config import settings
from market_liquidity_monitor.data_engine.models import LiquidityScorecard, LiquidityScorecardPartial


async def run_analysis_enhanced(prompt: str, symbol: str, exchange: str, status, text_placeholder):
    """Async helper for enhanced streaming analysis."""
    stream_generator = market_analyzer.analyze_liquidity_stream(
        query=prompt,
        symbol=symbol,
        exchange=exchange
    )

    final_result = None
    
    async for stream_result in stream_generator:
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
async def get_cached_client(exchange_id: str):
    """
    Get pooled client from exchange_manager.
    """
    from market_liquidity_monitor.data_engine.exchange import exchange_manager
    return await exchange_manager.get_client(exchange_id)


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


def generate_historical_audit(symbol: str, exchange: str, timeframe: str, days: int) -> str:
    """
    Generate detailed markdown audit of historical liquidity.
    """
    # Re-fetch from cache (fast)
    data = get_historical_data(symbol, exchange, timeframe, days)
    
    if not data:
        return "No historical data available for audit."
        
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Calculate Stats
    start_price = df['close'].iloc[0]
    end_price = df['close'].iloc[-1]
    price_change = ((end_price - start_price) / start_price) * 100
    
    avg_vol = df['volume'].mean()
    max_vol = df['volume'].max()
    volatility = df['close'].std()
    
    # Identify Volume Spikes
    threshold = avg_vol + (2 * df['volume'].std())
    spikes = df[df['volume'] > threshold]
    
    report = f"""# 📜 Historical Liquidity Audit
**Symbol:** {symbol} | **Exchange:** {exchange}
**Period:** Last {days} Days ({timeframe} candles)
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

## 1. Market Stability Summary
- **Volatility (StdDev):** ${volatility:,.2f}
- **Price Change:** {price_change:+.2f}%
- **Start Price:** ${start_price:,.2f}
- **End Price:** ${end_price:,.2f}

## 2. Volume Profile
- **Average Volume:** {avg_vol:,.2f}
- **Peak Volume:** {max_vol:,.2f}
- **Volume Consistency:** {'✅ Stable' if len(spikes) < 5 else '⚠️ Irregular'}

## 3. Liquidity Anomalies (Outliers)
Detected {len(spikes)} periods of abnormal volume activity (> 2σ):

"""
    for ts, row in spikes.iterrows():
        report += f"- **{ts}**: {row['volume']:,.0f} (Price: ${row['close']:,.2f})\n"
        
    report += "\n---\n*Generated by Market Liquidity Monitor*"
    return report




@st.cache_data(ttl=300)
def get_historical_data(symbol: str, exchange: str, timeframe: str, days: int):
    """Fetch historical data for charting (Cached)."""
    # Create a temporary loop for the async call since st.cache_data requires sync
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def _fetch():
        client = await exchange_manager.get_client(exchange)
        # Calculate since
        try:
             now_ms = client.exchange.milliseconds()
        except:
             now_ms = int(asyncio.get_event_loop().time() * 1000)

        since = now_ms - (days * 24 * 60 * 60 * 1000)
        
        if not client.exchange.has['fetchOHLCV']:
            return None
            
        return await client.fetch_ohlcv(symbol, timeframe, since)

    try:
        data = loop.run_until_complete(_fetch())
        return data
    finally:
        loop.close()


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
    client = await get_cached_client(exchange)
    return await client.fetch_order_book(symbol, limit=50)


async def run_comparison(symbol: str, exchanges: list):
    """Run multi-exchange comparison."""
    return await compare_exchanges(
        symbol=symbol,
        exchanges=exchanges,
        levels=20
    )


async def check_alerts(symbol: str, exchange: str):
    """Check for liquidity alerts (Depth + Trend)."""
    alerts = []
    
    # 1. Depth Anomalies (DB/File History)
    try:
        depth_alerts = await historical_tracker.detect_anomalies(
            symbol=symbol,
            exchange=exchange,
            threshold_pct=30.0
        )
        alerts.extend(depth_alerts)
    except Exception as e:
        print(f"Depth alert check failed: {e}")

    # 2. Trend Alerts (OHLCV)
    # Use cached historical data if available for free check
    # default to 1d lookback for immediate trend check
    hist_data = get_historical_data(symbol, exchange, "1h", 7)
    
    if hist_data:
        df = pd.DataFrame(hist_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Volatility Check
        current_volatility = df['close'].std()
        avg_price = df['close'].mean()
        if avg_price > 0:
            vol_pct = (current_volatility / avg_price) * 100
            if vol_pct > 5.0: # >5% deviation is volatile for stable pairs
                 alerts.append(LiquidityAlert(
                     alert_id=f"vol_{symbol}_{len(alerts)}",
                     symbol=symbol,
                     exchange=exchange,
                     timestamp=datetime.now(),
                     alert_type="HIGH_VOLATILITY",
                     severity="MEDIUM",
                     message=f"High price volatility detected ({vol_pct:.1f}% deviation).",
                     value=current_volatility,
                     threshold=avg_price * 0.05,
                     deviation_percentage=vol_pct
                 ))
        
        # Volume Consistency Check
        recent_vol = df['volume'].iloc[-1]
        avg_vol = df['volume'].mean()
        if avg_vol > 0 and recent_vol > (avg_vol * 3):
            alerts.append(LiquidityAlert(
                 alert_id=f"vol_spike_{symbol}_{len(alerts)}",
                 symbol=symbol,
                 exchange=exchange,
                 timestamp=datetime.now(),
                 alert_type="VOLUME_SPIKE",
                 severity="LOW", # Spikes can be good
                 message=f"Volume spike detected ({recent_vol/avg_vol:.1fx} average).",
                 current_value=recent_vol, # mapped for model compatibility if needed
                 value=recent_vol,
                 threshold=avg_vol * 3,
                 deviation_percentage=((recent_vol - avg_vol)/avg_vol)*100
             ))

    return alerts


def main():
    """Main application."""
    initialize_session_state()

    # Title
    st.title("📊 Advanced Market Liquidity Monitor")
    st.markdown("*Multi-exchange comparison • Historical tracking • Real-time alerts*")

    # Alerts Section
    with st.spinner("Checking alerts..."):
        # We need async loop for this
        try:
             loop = asyncio.new_event_loop()
             asyncio.set_event_loop(loop)
             alerts = loop.run_until_complete(check_alerts(st.session_state.current_symbol, st.session_state.current_exchange))
             loop.close()
             st.session_state.alerts = alerts
        except Exception as e:
             st.error(f"Alert check failed: {e}")
             
    if st.session_state.alerts:
        with st.expander(f"🚨 Active Alerts ({len(st.session_state.alerts)})", expanded=True):
             from market_liquidity_monitor.frontend.advanced_visualizations import create_alerts_dashboard
             st.plotly_chart(create_alerts_dashboard(st.session_state.alerts), use_container_width=True)
             
             for alert in st.session_state.alerts:
                  if alert.severity == "HIGH":
                      st.error(f"**{alert.alert_type}**: {alert.message}")
                  else:
                      st.warning(f"**{alert.alert_type}**: {alert.message}")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Symbol selection
        symbol = st.selectbox(
            "Trading Pair",
            ["SOL/USDT", "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT"],
            index=0,
            key="symbol_select"
        )
        st.session_state.current_symbol = symbol

        # Exchange selection
        exchange = st.selectbox(
            "Primary Exchange",
            ["binance", "coinbase", "kraken", "bybit"],
            index=0,
            key="exchange_select"
        )
        st.session_state.current_exchange = exchange

        # System Health Monitor
        st.markdown("---")
        st.subheader("🏥 System Health")
        
        # Check pool status via internal dictionary (Sync check)
        from market_liquidity_monitor.data_engine.exchange import exchange_manager
        client_instance = exchange_manager._clients.get(exchange)
        
        if client_instance:
             status = client_instance.status
             state_color = "green" if status['is_healthy'] else "red"
             if status['state'] == "HALF_OPEN": state_color = "orange"
             
             st.markdown(f"**Status**: :{state_color}[{status['state']}]")
             st.markdown(f"**Failures**: {status['failures']}")
        else:
             st.markdown("**Status**: :grey[IDLE]")

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
            # Audio Input
            audio_val = st.audio_input("🎤 Voice Query")
            
            prompt = st.chat_input("Ask about market liquidity...")
            
            # Handle audio input fallback
            if audio_val and not prompt:
                st.info("Audio received (Transcription pending)")
                # prompt = transcribe(audio_val)

            if prompt or audio_val:
                # Add user message
                if prompt:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

            # Process query
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                        # Use st.status for the thinking/streaming process
                        with st.status("Analyzing market...", expanded=True) as status:
                            text_placeholder = st.empty()
                            try:
                                final_result, usage = asyncio.run(run_analysis_enhanced(
                                    prompt, symbol, exchange, status, text_placeholder
                                ))
                                
                                if usage:
                                     status.update(label=f"Analysis Complete (Tokens: {usage.total_tokens})", state="complete", expanded=False)
                                else:
                                     status.update(label="Analysis Complete", state="complete", expanded=False)

                            except Exception as e:
                                status.update(label="Analysis Failed", state="error", expanded=False)
                                status.update(label="Analysis Failed", state="error", expanded=False)
                        
                        if final_result:
                            text_placeholder.empty()

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

                            response = final_result.summary_analysis
                            st.session_state.messages.append({"role": "assistant", "content": response})

                        # Fetch orderbook for visualization
                        orderbook = asyncio.run(fetch_orderbook(symbol, exchange))
                        st.session_state.orderbook = orderbook


        # Market Impact Simulator
        st.markdown("---")
        st.subheader("💰 Market Impact Simulator")

        col1, col2, col3 = st.columns(3)
        with col1:
             # ... (existing simulator code)
             pass 

        # Historical Analysis Section
        st.markdown("---")
        st.subheader("📈 Historical Liquidity Trends")
        
        hist_col1, hist_col2 = st.columns([1, 3])
        
        with hist_col1:
            timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=0)
            lookback = st.slider("Lookback (Days)", 1, 30, 7)
            
        with hist_col2:
            if st.button("Load Historical Data"):
                with st.spinner("Fetching market history..."):
                    history = get_historical_data(symbol, exchange, timeframe, lookback)
                    
                    if history:
                        # Convert to DataFrame for easier charting
                        df = pd.DataFrame(history, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        
                        tab1, tab2 = st.tabs(["Price Trend", "Volume Profile"])
                        
                        with tab1:
                            st.line_chart(df['close'], color="#00FF00")
                            
                        with tab2:
                            st.bar_chart(df['volume'], color="#0088FF")
                            
                        # Quick metrics
                        volatility = df['close'].std()
                        st.caption(f"Price Volatility (StdDev): ${volatility:,.2f}")
                        
                        # Audit Download
                        st.download_button(
                            label="📜 Download Historical Audit",
                            data=generate_historical_audit(symbol, exchange, timeframe, lookback),
                            file_name=f"liquidity_audit_{symbol.replace('/','-')}_{lookback}d.md",
                            mime="text/markdown",
                            key="hist_download"
                        )
                    else:
                        st.error("Exchange does not support historical data or no data found.")


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
