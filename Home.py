import streamlit as st

# Page Configuration must be the first line
st.set_page_config(
    page_title="Gwendal Quant Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed" # On cache la sidebar par défaut pour un look "Site Web"
)

# Custom CSS for the "Card" look
st.markdown("""
<style>
    .category-card {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        height: 350px;
        display: flex;
        flex-direction: column;
        justify_content: space-between;
    }
    .category-card:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
    }
    .card-desc {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("🧪 Gwendal Quant Lab")
st.markdown("#### Explore my quantitative finance projects powered by Python.")
st.markdown("---")

# --- PROJECT CARDS SECTION ---
st.header("Choose Your Product")
st.write("Each simulator provides comprehensive mathematical modeling with interactive controls and real-time visualizations.")
st.write("") # Spacer

col1, col2, col3 = st.columns(3)

# CARD 1 : Derivatives Pricing (Ton projet actuel)
with col1:
    st.markdown("""
    <div class="category-card">
        <div>
            <div style="font-size: 3rem; margin-bottom: 10px;">💎</div>
            <div class="card-title">Derivatives Pricing & Hedging</div>
            <div class="card-desc">
                Advanced pricing engine for Vanilla and Structured Products (Phoenix Autocall). 
                Includes Monte Carlo simulations, Dynamic Greeks analysis, and Delta-Hedging backtesting.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    if st.button("Explore Pricing Engine", use_container_width=True):
        st.switch_page("pages/01_💎_Derivatives_Pricer.py")

# CARD 2 : Investment Strategy (Futur)
with col2:
    st.markdown("""
    <div class="category-card">
        <div>
            <div style="font-size: 3rem; margin-bottom: 10px;">📈</div>
            <div class="card-title">Investment Strategies</div>
            <div class="card-desc">
                Systematic trading strategy backtester. 
                Test Moving Averages crossovers, RSI strategies, and optimize portfolio allocation using Markowitz Efficient Frontier.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    if st.button("Explore Strategies", use_container_width=True):
        st.switch_page("pages/02_📈_Investment_Strategy.py")

# CARD 3 : Volatility (Futur)
with col3:
    st.markdown("""
    <div class="category-card">
        <div>
            <div style="font-size: 3rem; margin-bottom: 10px;">⚡</div>
            <div class="card-title">Volatility Estimator</div>
            <div class="card-desc">
                Market Risk engine focused on volatility modeling.
                Compare GARCH models vs EWMA, analyze Volatility Surfaces, and perform VaR Stress Tests.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    if st.button("Explore Volatility", use_container_width=True):
        st.switch_page("pages/03_⚡_Volatility_Estimator.py")

# --- FOOTER / CONTACT ---
st.markdown("---")
c1, c2 = st.columns([1,1])
with c1:
    st.markdown("**About this site:** Built with Python & Streamlit.")
with c2:
    st.markdown(f"**Contact:** [LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/)")