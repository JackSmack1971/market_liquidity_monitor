"""
Advanced visualizations for liquidity monitoring.

Provides:
- Liquidity heatmaps
- Multi-exchange comparison charts
- Historical trend analysis
- Real-time alerts dashboard
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict
from datetime import datetime
import numpy as np


def create_liquidity_heatmap(orderbook, levels: int = 20) -> go.Figure:
    """
    Create liquidity heatmap showing depth at different price levels.

    The heatmap visualizes liquidity density across price levels,
    making it easy to spot "walls" (large orders) and thin areas.

    Args:
        orderbook: OrderBook object
        levels: Number of levels to visualize

    Returns:
        Plotly figure with heatmap
    """
    # Extract bid and ask data
    bid_prices = [level.price for level in orderbook.bids[:levels]]
    bid_volumes = [level.amount for level in orderbook.bids[:levels]]

    ask_prices = [level.price for level in orderbook.asks[:levels]]
    ask_volumes = [level.amount for level in orderbook.asks[:levels]]

    # Combine and sort by price
    all_prices = bid_prices + ask_prices
    all_volumes = bid_volumes + ask_volumes
    sides = ["BID"] * len(bid_prices) + ["ASK"] * len(ask_prices)

    # Create DataFrame
    df = pd.DataFrame({
        'price': all_prices,
        'volume': all_volumes,
        'side': sides
    }).sort_values('price')

    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Liquidity Heatmap", "Volume Distribution"),
        vertical_spacing=0.12
    )

    # Heatmap data
    # Create price buckets
    price_range = max(all_prices) - min(all_prices)
    num_buckets = 50
    bucket_size = price_range / num_buckets

    # Aggregate volume into buckets
    heatmap_data = []
    for i in range(num_buckets):
        bucket_min = min(all_prices) + i * bucket_size
        bucket_max = bucket_min + bucket_size
        bucket_mid = (bucket_min + bucket_max) / 2

        # Sum volume in this bucket
        bucket_volume = sum(
            vol for price, vol in zip(all_prices, all_volumes)
            if bucket_min <= price < bucket_max
        )

        heatmap_data.append({
            'price': bucket_mid,
            'volume': bucket_volume
        })

    heatmap_df = pd.DataFrame(heatmap_data)

    # Add heatmap
    fig.add_trace(
        go.Heatmap(
            x=heatmap_df['price'],
            y=[0] * len(heatmap_df),
            z=[heatmap_df['volume'].values],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Volume", y=0.85, len=0.6),
        ),
        row=1, col=1
    )

    # Add volume bars
    bid_df = df[df['side'] == 'BID']
    ask_df = df[df['side'] == 'ASK']

    fig.add_trace(
        go.Bar(
            x=bid_df['price'],
            y=bid_df['volume'],
            name='Bids',
            marker_color='green',
            opacity=0.6
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=ask_df['price'],
            y=ask_df['volume'],
            name='Asks',
            marker_color='red',
            opacity=0.6
        ),
        row=2, col=1
    )

    # Add vertical line at mid price
    if orderbook.best_bid and orderbook.best_ask:
        mid_price = (orderbook.best_bid.price + orderbook.best_ask.price) / 2
        fig.add_vline(
            x=mid_price,
            line_dash="dash",
            line_color="white",
            annotation_text="Mid",
            row=1, col=1
        )
        fig.add_vline(
            x=mid_price,
            line_dash="dash",
            line_color="gray",
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        title=f"Liquidity Heatmap - {orderbook.symbol}",
        showlegend=True,
        height=600,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Price", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def create_exchange_comparison_chart(comparison_data: Dict) -> go.Figure:
    """
    Create multi-exchange comparison visualization.

    Args:
        comparison_data: Dictionary from compare_exchanges tool

    Returns:
        Plotly figure with comparison metrics
    """
    exchanges = comparison_data.get("exchanges_compared", [])
    order_books = comparison_data.get("order_books", [])

    if not exchanges or not order_books:
        # Empty figure if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No comparison data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig

    # Extract metrics
    spread_data = [ob.get("spread_pct", 0) for ob in order_books]
    bid_depth_data = [ob.get("bid_depth", 0) for ob in order_books]
    ask_depth_data = [ob.get("ask_depth", 0) for ob in order_books]

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Spread Comparison",
            "Liquidity Depth",
            "Best Prices",
            "Quality Score"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "indicator"}]
        ]
    )

    # 1. Spread comparison
    fig.add_trace(
        go.Bar(
            x=exchanges,
            y=spread_data,
            name="Spread %",
            marker_color='lightblue',
            text=[f"{s:.3f}%" for s in spread_data],
            textposition='auto',
        ),
        row=1, col=1
    )

    # 2. Depth comparison
    fig.add_trace(
        go.Bar(
            x=exchanges,
            y=bid_depth_data,
            name="Bid Depth",
            marker_color='green',
            opacity=0.6
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(
            x=exchanges,
            y=ask_depth_data,
            name="Ask Depth",
            marker_color='red',
            opacity=0.6
        ),
        row=1, col=2
    )

    # 3. Best prices
    best_bid_ex = comparison_data.get("best_bid_exchange")
    best_ask_ex = comparison_data.get("best_ask_exchange")
    best_bid_price = comparison_data.get("best_bid_price", 0)
    best_ask_price = comparison_data.get("best_ask_price", 0)

    price_labels = []
    price_values = []
    price_colors = []

    for ex in exchanges:
        if ex == best_bid_ex:
            price_labels.append(f"{ex}\n(Best Bid)")
            price_values.append(best_bid_price)
            price_colors.append('green')
        elif ex == best_ask_ex:
            price_labels.append(f"{ex}\n(Best Ask)")
            price_values.append(best_ask_price)
            price_colors.append('red')
        else:
            price_labels.append(ex)
            price_values.append((best_bid_price + best_ask_price) / 2)
            price_colors.append('gray')

    fig.add_trace(
        go.Bar(
            x=price_labels,
            y=price_values,
            marker_color=price_colors,
            showlegend=False,
            text=[f"${p:.2f}" for p in price_values],
            textposition='auto',
        ),
        row=2, col=1
    )

    # 4. Overall quality indicator
    best_exchange = comparison_data.get("tightest_spread_exchange", "N/A")
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=len(exchanges),
            title={'text': f"Best: {best_exchange}"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 10]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 3], 'color': "lightgray"},
                    {'range': [3, 7], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 8
                }
            }
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title=f"Multi-Exchange Comparison - {comparison_data.get('symbol', 'N/A')}",
        height=700,
        showlegend=True
    )

    # Add arbitrage annotation if exists
    if comparison_data.get("arbitrage_opportunity_pct"):
        arb_pct = comparison_data["arbitrage_opportunity_pct"]
        arb_route = comparison_data.get("arbitrage_route", "")
        fig.add_annotation(
            text=f"⚠️ ARBITRAGE: {arb_pct:.2f}%<br>{arb_route}",
            xref="paper", yref="paper",
            x=0.5, y=0.95,
            showarrow=False,
            bgcolor="yellow",
            bordercolor="red",
            borderwidth=2,
            font=dict(size=12, color="red")
        )

    return fig


def create_historical_trend_chart(snapshots: List) -> go.Figure:
    """
    Create historical liquidity trend visualization.

    Args:
        snapshots: List of HistoricalSnapshot objects

    Returns:
        Plotly figure with time-series trends
    """
    if not snapshots:
        fig = go.Figure()
        fig.add_annotation(
            text="No historical data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'timestamp': s.timestamp,
            'spread_pct': s.spread_percentage,
            'mid_price': s.mid_price,
            'volume': s.total_volume_20,
            'imbalance': s.imbalance_ratio,
            'liquidity_usd': s.liquidity_1pct_usd
        }
        for s in snapshots
    ])

    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Price & Spread",
            "Order Book Volume",
            "Liquidity (1% depth)",
            "Bid/Ask Imbalance"
        ),
        vertical_spacing=0.08,
        row_heights=[0.3, 0.25, 0.25, 0.2]
    )

    # 1. Price and spread
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['mid_price'],
            name='Mid Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    # Add spread as secondary y-axis would require separate figure
    # Instead, normalize spread to price scale
    spread_scaled = df['spread_pct'] * df['mid_price'].mean() / 100
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=spread_scaled,
            name='Spread (scaled)',
            line=dict(color='orange', width=1, dash='dash'),
            fill='tozeroy',
            fillcolor='rgba(255,165,0,0.1)'
        ),
        row=1, col=1
    )

    # 2. Volume
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['volume'],
            name='Total Volume',
            line=dict(color='purple', width=2),
            fill='tozeroy',
            fillcolor='rgba(128,0,128,0.1)'
        ),
        row=2, col=1
    )

    # 3. Liquidity
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['liquidity_usd'],
            name='Liquidity (USD)',
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.1)'
        ),
        row=3, col=1
    )

    # 4. Imbalance ratio
    # Add reference line at 1.0 (balanced)
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        row=4, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['imbalance'],
            name='Bid/Ask Imbalance',
            line=dict(color='red', width=2),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)'
        ),
        row=4, col=1
    )

    # Update layout
    fig.update_layout(
        title="Historical Liquidity Trends",
        height=900,
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Time", row=4, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="USD", row=3, col=1)
    fig.update_yaxes(title_text="Ratio", row=4, col=1)

    return fig


def create_alerts_dashboard(alerts: List) -> go.Figure:
    """
    Create alerts dashboard visualization.

    Args:
        alerts: List of LiquidityAlert objects

    Returns:
        Plotly figure with alert summary
    """
    if not alerts:
        fig = go.Figure()
        fig.add_annotation(
            text="✅ No active alerts",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="green")
        )
        return fig

    # Group alerts by type and severity
    alert_types = {}
    severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for alert in alerts:
        alert_type = alert.alert_type
        if alert_type not in alert_types:
            alert_types[alert_type] = []
        alert_types[alert_type].append(alert)

        severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1

    # Create figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Alerts by Type", "Alerts by Severity"),
        specs=[[{"type": "bar"}, {"type": "pie"}]]
    )

    # Bar chart by type
    fig.add_trace(
        go.Bar(
            x=list(alert_types.keys()),
            y=[len(alerts) for alerts in alert_types.values()],
            marker_color='red',
            text=[len(alerts) for alerts in alert_types.values()],
            textposition='auto',
        ),
        row=1, col=1
    )

    # Pie chart by severity
    colors = {"HIGH": "red", "MEDIUM": "orange", "LOW": "yellow"}
    fig.add_trace(
        go.Pie(
            labels=list(severity_counts.keys()),
            values=list(severity_counts.values()),
            marker=dict(colors=[colors[k] for k in severity_counts.keys()])
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=f"Alert Dashboard ({len(alerts)} total alerts)",
        height=400,
        showlegend=True
    )

    return fig


def create_market_impact_chart(impact_data: Dict) -> go.Figure:
    """
    Create market impact visualization.

    Args:
        impact_data: Dictionary from calculate_market_impact tool

    Returns:
        Plotly figure showing execution simulation
    """
    # Create gauge chart for slippage
    slippage = impact_data.get("slippage_percentage", 0)
    order_size = impact_data.get("order_size_usd", 0)

    fig = go.Figure()

    # Slippage gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=abs(slippage),
        title={'text': f"Slippage for ${order_size:,.0f} Order"},
        delta={'reference': 0.5, 'suffix': '%'},
        gauge={
            'axis': {'range': [0, 5]},
            'bar': {'color': "darkred" if abs(slippage) > 1 else "orange" if abs(slippage) > 0.5 else "green"},
            'steps': [
                {'range': [0, 0.5], 'color': "lightgreen"},
                {'range': [0.5, 1], 'color': "lightyellow"},
                {'range': [1, 5], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 2
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig
