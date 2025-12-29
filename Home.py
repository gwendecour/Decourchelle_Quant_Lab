import streamlit as st
from src.shared.ui import render_header

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Gwendal Quant Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- IMPORT DU HEADER ---
render_header()

# --- CSS PERSONNALISÉ (POLICE & BOUTONS) ---
st.markdown("""
<style>
    /* Import de la police 'Lora' (Style académique/finance) */
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;600&display=swap');

    /* Appliquer la police aux titres */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Lora', serif !important;
        color: #1a1a1a;
    }
    
    /* Appliquer une police propre au texte */
    p, .stMarkdown, .stText {
        font-family: 'Inter', sans-serif !important;
        color: #4a4a4a;
        line-height: 1.6;
    }

    /* CUSTOM BUTTON STYLE (Vert élégant comme demandé) */
    div.stButton > button {
        background-color: #2e7d32; /* Vert Finance */
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #1b5e20;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        color: white;
    }
    
    /* Style des cartes (Containers) */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- TITRE PRINCIPAL ---
st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>Choose Your Product</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 50px; color: gray;'>Each simulator provides comprehensive mathematical modeling with interactive controls and real-time visualizations.</p>", unsafe_allow_html=True)

# --- GRILLE DES PROJETS ---
col1, col2, col3 = st.columns(3, gap="medium")

# --- CARTE 1 : PRICING ---
with col1:
    # On utilise st.container(border=True) pour créer la boite
    with st.container(border=True):
        st.markdown("### 💎 Derivatives Pricing")
        st.markdown("""
        <div style="height: 120px;">
        Advanced pricing engine for Vanilla and Structured Products (Phoenix Autocall). 
        Includes Monte Carlo simulations and Delta-Hedging backtesting.
        </div>
        """, unsafe_allow_html=True)
        
        st.write("") # Spacer
        if st.button("Explore Pricing Engine", key="btn_pricing"):
            st.switch_page("pages/01_Pricing&Hedging_Derivatives.py")

# --- CARTE 2 : INVESTMENT ---
with col2:
    with st.container(border=True):
        st.markdown("### 📈 Investment Strategies")
        st.markdown("""
        <div style="height: 120px;">
        Systematic trading strategy backtester. 
        Test Moving Averages crossovers, RSI strategies, and optimize portfolio allocation using Markowitz Frontier.
        </div>
        """, unsafe_allow_html=True)
        
        st.write("") # Spacer
        if st.button("Explore Strategies", key="btn_invest"):
            st.switch_page("pages/02_Investment_Strategy.py")

# --- CARTE 3 : VOLATILITY ---
with col3:
    with st.container(border=True):
        st.markdown("### ⚡ Volatility Estimator")
        st.markdown("""
        <div style="height: 120px;">
        Market Risk engine focused on volatility modeling.
        Compare GARCH models vs EWMA, analyze Volatility Surfaces, and perform VaR Stress Tests.
        </div>
        """, unsafe_allow_html=True)
        
        st.write("") # Spacer
        if st.button("Explore Volatility", key="btn_vol"):
            st.switch_page("pages/03_Volatility_Estimator.py")