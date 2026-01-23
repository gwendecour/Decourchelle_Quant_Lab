import streamlit as st
from src.shared.ui import render_header

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Home",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- IMPORT DU HEADER ---
render_header()

# --- CSS PERSONNALISÉ (POLICE & BOUTONS DUAL STYLE) ---
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

    /* --- STYLE DES BOUTONS --- */
    
    /* 1. Boutons "EXPLORE" (Type Primary) -> VERT FINANCE + TEXTE BLANC */
    div.stButton > button[kind="primary"],
    div.stButton > button[kind="primary"] * {
        background-color: #2e7d32 !important; 
        color: #ffffff !important;            /* FORCE BLANC PARTOUT */
        fill: #ffffff !important;             /* FORCE BLANC POUR LES ICONES SI PRÉSENTES */
        border: none !important;
    }

    /* Gestion du Hover (Survol) */
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[kind="primary"]:hover * {
        background-color: #1b5e20 !important;
        color: #ffffff !important;
    }
    
    /* Force la couleur blanche même si le bouton est actif/cliqué */
    div.stButton > button[kind="primary"]:active, 
    div.stButton > button[kind="primary"]:focus {
        color: #ffffff !important;
        background-color: #1b5e20 !important;
    }

    /* 2. Boutons "METHODOLOGY" (Type Secondary/Défaut) -> GRIS CLAIR */
    div.stButton > button[kind="secondary"] {
        background-color: #f0f2f6 !important; /* Gris clair */
        color: #31333F !important;            /* Noir/Gris foncé */
        border: 1px solid #d0d0d0 !important;
        font-weight: 500 !important;
        border-radius: 6px;
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #e0e2e6 !important;
        border-color: #b0b0b0 !important;
        color: black !important;
    }
    
    /* Style des cartes (Containers) */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: white;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CONTENU DES MODALES (METHODOLOGY)
# ==============================================================================

story_p1 = """
### Why this project?
Derivative valuation is the cornerstone of market finance. I wanted to go beyond Black-Scholes theory to understand real-world implementation challenges: date management, exotic option convexity (Phoenix), and numerical Greek calculation.

### Architecture & Technical Choices
* **Models:** Black-Scholes (Closed-form) for Vanilla options, Monte Carlo for Exotics.
* **Greeks:** Calculated via Finite Differences (Bump & Revalue) to capture non-linear risks.
* **Visualization:** Intensive use of Plotly to visualize the Volatility Smile and P&L surfaces.
"""

story_p2 = """
### Strategic Objective
Simulate the approach of a **"Beta Neutral Trend"** desk. The goal is to capture asset momentum across Equities, Bonds, and Commodities while remaining uncorrelated to the S&P 500.

### The Pipeline
1.  **Multi-Asset Universe:** Selection of liquid ETFs (SPY, TLT, GLD, USO...) for diversification.
2.  **Alpha (Signal):** Continuous Z-Scores on moving averages to detect robust trends.
3.  **Risk Management:**
    * **Volatility Targeting:** Weighting assets by inverse volatility.
    * **Beta Neutralization:** Dynamic hedging (Short SPY) to bring portfolio Beta to 0.

### Key Metrics
Focus on **Sharpe Ratio** and maintaining a **Correlation < 0.2** with the market.
"""

story_p3 = """
### Business Problem
In asset management (e.g., Corporate Bonds), missing data is common. Simply filling gaps with the last known price (Forward Fill) biases volatility downwards and underestimates risk.

### Comparison Arsenal
I compare 5 mathematical methods to recover missing data:
1.  **Baseline:** Forward Fill (The flaw to expose).
2.  **KNN Imputer:** K-Nearest Neighbors based on asset correlation.
3.  **MICE:** Multivariate Imputation by Chained Equations.
4.  **SVD:** Matrix completion assuming low-rank market structure.
5.  **EM Algorithm:** Statistical likelihood maximization.

### Success Metrics
Evaluation using **Frobenius Norm** (distance to Ground Truth) and impact on **Minimum Variance Portfolios**.
"""

# ==============================================================================
# FONCTION DE DIALOGUE (POP-UP)
# ==============================================================================
@st.dialog("Methodology & Backstory")
def show_methodology(title, content):
    st.markdown(f"## {title}")
    st.markdown(content)
    st.markdown("---")
    st.caption("Decourchelle Quant Lab Research")

# ==============================================================================
# PAGE LAYOUT
# ==============================================================================

# --- TITRE PRINCIPAL ---
st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>Choose Your Project</h1>", unsafe_allow_html=True)

# --- NOUVELLE TAGLINE (Sous-titre plus percutant) ---
st.markdown(
    "<p style='text-align: center; margin-bottom: 50px; color: gray; font-size: 1.1rem;'>"
    "Bridging quantitative theory with industrial implementation through interactive research modules."
    "</p>", 
    unsafe_allow_html=True
)

# --- GRILLE DES PROJETS ---
col1, col2, col3 = st.columns(3, gap="medium")

# --- CARTE 1 : PRICING ---
with col1:
    with st.container(border=True):
        # TITRE CENTRÉ VIA HTML
        st.markdown("<h3 style='text-align: center;'>Derivatives Pricing</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="height: 130px; text-align: justify;">
        Advanced pricing engine for Vanilla and Structured Products (Phoenix Autocall). 
        Includes Monte Carlo simulations, full Greeks analysis, and Delta-Hedging backtesting simulations.
        </div>
        """, unsafe_allow_html=True)
        
        # BOUTONS ALIGNÉS COTE A COTE
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("Explore", key="btn_pricing", type="primary", use_container_width=True):
                st.switch_page("pages/01_Pricing&Hedging_Derivatives.py")
        with b_col2:
            if st.button("Methodology", key="story_p1", use_container_width=True):
                show_methodology("Pricing & Hedging Derivatives", story_p1)

# --- CARTE 2 : ALPHATREND ---
with col2:
    with st.container(border=True):
        st.markdown("<h3 style='text-align: center;'>AlphaTrend Strategy</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="height: 130px; text-align: justify;">
        Multi-Asset Trend Following strategy acting as a "Beta Neutral" desk.
        Features dynamic volatility targeting and automatic market beta neutralization (Long/Short).
        </div>
        """, unsafe_allow_html=True)
        
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            # Bouton visuellement "Primary" (Vert) mais logique WIP
            if st.button("Explore", key="btn_invest", type="primary", use_container_width=True):
                st.toast("🚧 Work in Progress! Please check the Methodology for details.", icon="⚠️")
        with b_col2:
            if st.button("Methodology", key="story_p2", use_container_width=True):
                show_methodology("AlphaTrend: Beta Neutral (In Dev)", story_p2)

# --- CARTE 3 : ROBUST COVARIANCE ---
with col3:
    with st.container(border=True):
        st.markdown("<h3 style='text-align: center;'>Robust Covariance</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="height: 130px; text-align: justify;">
        Quantitative research on Missing Data Imputation.
        Compare covariance matrix estimation techniques (KNN, SVD, MICE) on illiquid assets vs Ground Truth.
        </div>
        """, unsafe_allow_html=True)
        
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            # Bouton visuellement "Primary" (Vert) mais logique WIP
            if st.button("Explore", key="btn_vol", type="primary", use_container_width=True):
                st.toast("🚧 Work in Progress! Please check the Methodology for details.", icon="⚠️")
        with b_col2:
            if st.button("Methodology", key="story_p3", use_container_width=True):
                show_methodology("Robust Covariance Estimation (In Dev)", story_p3)