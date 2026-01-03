import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from frontend.credentials import credentials_page

# Page Config (Global)
st.set_page_config(
    page_title="Market Liquidity Monitor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Navigation
pg = st.navigation([
    st.Page("dashboard.py", title="Liquidity Monitor", icon="📊"),
    st.Page(credentials_page, title="Credentials", icon="🔑")
])

pg.run()
