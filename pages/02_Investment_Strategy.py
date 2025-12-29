import streamlit as st
from src.shared.ui import render_top_navigation

st.set_page_config(page_title="Investment Strategies", layout="wide")
render_top_navigation()

st.title("📈 Systematic Investment Strategies")

st.info("🚧 This project is currently under development.")

st.markdown("""
### Upcoming Features:
* **Backtesting Engine:** Test standard technical indicators (MA, RSI, Bollinger).
* **Portfolio Optimization:** Markowitz Efficient Frontier implementation.
* **Performance Metrics:** Sharpe Ratio, Sortino, Max Drawdown calculation.
""")