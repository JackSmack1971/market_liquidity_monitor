import streamlit as st
import asyncio

def credentials_page():
    st.title("🔑 Credential Management")
    st.info("Your keys are stored only in your current session and are never saved to our server.")

    with st.form("credentials_form"):
        # OpenRouter Key for Agent
        or_key = st.text_input(
            "OpenRouter API Key", 
            type="password", 
            value=st.session_state.get("OPENROUTER_API_KEY", ""),
            help="Required for Market Agent reasoning."
        )
        
        # Exchange Keys for Data Engine
        st.subheader("Exchange Connectivity")
        st.caption("Required for private data and high-rate limits.")
        
        ex_key = st.text_input(
            "Exchange API Key (Read-Only)", 
            type="password", 
            value=st.session_state.get("EXCHANGE_API_KEY", "")
        )
        ex_secret = st.text_input(
            "Exchange API Secret", 
            type="password", 
            value=st.session_state.get("EXCHANGE_API_SECRET", "")
        )

        # Anthropic Fallback
        st.subheader("Fallback Redundancy")
        anthro_key = st.text_input(
            "Anthropic API Key (Optional)",
            type="password",
            value=st.session_state.get("ANTHROPIC_API_KEY", ""),
            help="Used if OpenRouter fails."
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button("Save Credentials")
        
        if submitted:
            st.session_state.OPENROUTER_API_KEY = or_key
            st.session_state.EXCHANGE_API_KEY = ex_key
            st.session_state.EXCHANGE_API_SECRET = ex_secret
            st.session_state.ANTHROPIC_API_KEY = anthro_key
            st.success("Credentials updated! Redirecting...")
            st.switch_page("dashboard.py")

    # Connection Status
    st.markdown("---")
    st.subheader("Connection Status")
    
    c1, c2 = st.columns(2)
    
    with c1:
        if st.session_state.get("OPENROUTER_API_KEY"):
            st.success("✅ LLM Agent Configured")
        else:
            st.error("❌ LLM Agent Missing Key")
            
    with c2:
        if st.session_state.get("EXCHANGE_API_KEY"):
            st.success("✅ Exchange Configured") 
        else:
            st.warning("⚠️ Exchange Unconfigured (Public Only)")

    if st.button("🔌 Test Connections", use_container_width=True):
        asyncio.run(run_connection_test())

async def run_connection_test():
    """Verify connections with current session credentials."""
    from market_liquidity_monitor.agents import MarketAnalyzer
    from market_liquidity_monitor.data_engine.exchange import exchange_manager
    
    st.write("---")
    st.subheader("Diagnostic Report")
    
    c1, c2 = st.columns(2)
    
    # 1. Test OpenRouter
    with c1:
        with st.status("Testing LLM Agent...", expanded=True) as status:
            try:
                if not st.session_state.get("OPENROUTER_API_KEY"):
                    status.update(label="Skipped (No Key)", state="error")
                    st.error("No OpenRouter Key provided.")
                else:
                    # Initialize with session key
                    analyzer = MarketAnalyzer(api_key=st.session_state.OPENROUTER_API_KEY)
                    # Simple "Hello" check (cheap)
                    # We can't easily "ping" without cost, but let's try a very short generation
                    # or just rely on initialization check if Pydantic AI validates on init? 
                    # Attempt a short generation check
                    res = await analyzer.agent.run(
                        "Respond with only the word 'OK'.", 
                        model_settings={'max_tokens': 5}
                    )
                    status.update(label="LLM Connected", state="complete")
                    st.success(f"Response: {res.data.summary_analysis[:20]}...") 
            except Exception as e:
                status.update(label="LLM Failed", state="error")
                st.error(f"Error: {str(e)}")

    # 2. Test Exchange
    with c2:
        with st.status("Testing Exchange...", expanded=True) as status:
            try:
                # Use credentials if available
                ak = st.session_state.get("EXCHANGE_API_KEY")
                as_ = st.session_state.get("EXCHANGE_API_SECRET")
                
                # Default to Binance if not set? Or use session state exchange?
                # Using binance as baseline for connectivity
                exchange_id = "binance" 
                
                client = await exchange_manager.get_client(exchange_id, api_key=ak, api_secret=as_)
                await client.exchange.load_markets()
                
                if ak and as_:
                    # Private: Check Balance
                    bal = await client.exchange.fetch_balance()
                    status.update(label="Authenticated", state="complete")
                    st.success(f"Access Confirmed (Balances: {len(bal)})")
                else:
                    # Public: Check Ticker
                    await client.exchange.fetch_ticker("BTC/USDT")
                    status.update(label="Public Access OK", state="complete")
                    st.info("Read-Only Mode (Public Data)")
                    
            except Exception as e:
                status.update(label="Exchange Failed", state="error")
                st.error(f"Error: {str(e)}")



