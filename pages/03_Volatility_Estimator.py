import streamlit as st
from src.shared.ui import render_top_navigation

st.set_page_config(page_title="Investment Strategies", layout="wide")
render_top_navigation()

st.title("Volatility Estimator")

st.info("🚧 This project is currently under development.")

st.markdown("""
### Upcoming Features:
""")