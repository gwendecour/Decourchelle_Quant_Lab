import streamlit as st
from src.shared.ui import render_header

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Home",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- HEADER IMPORT ---
render_header()

# --- CUSTOM CSS (FONT & DUAL STYLE BUTTONS) ---
st.markdown("""
<style>
    /* Import 'Lora' font (Academic/Finance style) */
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;600&display=swap');

    /* Apply font to headers */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Lora', serif !important;
        color: #1a1a1a;
    }
    
    /* Apply clean font to text */
    p, .stMarkdown, .stText {
        font-family: 'Inter', sans-serif !important;
        color: #4a4a4a;
        line-height: 1.6;
    }

    /* --- BUTTON STYLES --- */
    
    /* 1. "EXPLORE" Buttons (Primary Type) -> FINANCE GREEN + WHITE TEXT */
    div.stButton > button[kind="primary"],
    div.stButton > button[kind="primary"] * {
        background-color: #2e7d32 !important; 
        color: #ffffff !important;            /* FORCE WHITE GLOBALLY */
        fill: #ffffff !important;             /* FORCE WHITE FOR ICONS IF PRESENT */
        border: none !important;
    }

    /* Hover State Management */
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[kind="primary"]:hover * {
        background-color: #1b5e20 !important;
        color: #ffffff !important;
    }
    
    /* Force white color even if button is active/clicked */
    div.stButton > button[kind="primary"]:active, 
    div.stButton > button[kind="primary"]:focus {
        color: #ffffff !important;
        background-color: #1b5e20 !important;
    }

    /* 2. "METHODOLOGY" Buttons (Secondary/Default Type) -> LIGHT GREY */
    div.stButton > button[kind="secondary"] {
        background-color: #f0f2f6 !important; /* Light Grey */
        color: #31333F !important;            /* Black/Dark Grey */
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
    
    /* Card Styles (Containers) */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: white;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODAL CONTENT (METHODOLOGY)
# ==============================================================================

story_p1 = r"""
### Strategic Objective
The objective of the **Derivatives Pricing & Hedging** engine is to bridge the gap between theoretical valuation models and industrial trading reality. It provides a robust framework to price, risk-manage, and backtest hedging strategies for both **Vanilla Options** (European Calls/Puts) and **Path-Dependent Structured Products** (Phoenix Autocalls), simulating the daily workflow of an Equity Derivatives desk.

### Pricing Models & Architecture
The engine selects the appropriate valuation method based on the product's complexity:
* **Vanilla Options (Closed-Form):** For European options where the payoff depends solely on $S_T$, we utilize the analytical **Black-Scholes-Merton** formula for instant precision.
$$
C(S,t) = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)
$$
* **Exotic Products (Monte Carlo):** For Phoenix Autocalls, where the payoff is **path-dependent** (barriers are monitored daily/quarterly), we employ a **Monte Carlo Simulation** engine.
    * **Diffusion Process:** We simulate $N$ paths (default $2,000 - 10,000$) using Geometric Brownian Motion (GBM):
    $$ dS_t = (r - q) S_t dt + \sigma S_t dW_t $$
    * **Pricing:** The price is the discounted expectation of the payoff under the risk-neutral measure $\mathbb{Q}$:
    $$ P_0 = e^{-rT} \times \frac{1}{N} \sum_{i=1}^{N} \text{Payoff}(Path_i) $$



### Structured Product Logic (Phoenix Autocall)
The Phoenix is a yield enhancement product defined by three barriers observed at discrete frequencies:
* **Coupon Barrier ($B_{cpn}$):** If $S_t \ge B_{cpn}$, a coupon is paid (Memory effect applies if missed previously).
* **Autocall Barrier ($B_{auto}$):** If $S_t \ge B_{auto}$, the product is redeemed early at Nominal + Coupon.
* **Protection Barrier ($B_{prot}$):** At maturity, if $S_T < B_{prot}$ (and never autocalled), the investor is short a Put option and suffers a capital loss:
$$ \text{Loss} = \text{Nominal} \times \frac{S_T - S_0}{S_0} $$



### Sensitivity Analysis (Numerical Greeks)
Since Exotic products possess discontinuous payoffs with no closed-form derivatives, we calculate sensitivities (Greeks) using **Finite Differences** ("Bump & Revalue"):
* **Delta ($\Delta$):** Sensitivity to Spot. Calculated via **Central Difference** for stability:
$$ \Delta \approx \frac{P(S+\epsilon) - P(S-\epsilon)}{2\epsilon} $$
* **Gamma ($\Gamma$):** Sensitivity to Delta (Convexity). Crucial for hedging frequency:
$$ \Gamma \approx \frac{P(S+\epsilon) - 2P(S) + P(S-\epsilon)}{\epsilon^2} $$
* **Vega ($\nu$):** Sensitivity to Volatility. Calculated via Forward Difference:
$$ \nu \approx P(\sigma + 1\%) - P(\sigma) $$

### Dynamic Hedging Backtest
The module simulates a **Short Volatility** position (Bank selling the option) and manages the Delta-Hedge daily until maturity:
* **Hedge Rebalancing:** At each step $t$, the engine calculates the new $\Delta_t$ and adjusts the inventory of the underlying asset to remain Delta Neutral.
* **Gamma Scalping:** The trading P&L is generated by the mechanical process of "Buying Low / Selling High" (Short Gamma) or vice-versa.
* **P&L Attribution:** The final performance is decomposed into:
    * **Cost of Carry:** Hedge Cost vs Theta (Time Decay).
    * **Vol Spread:** The difference between **Implied Volatility** (priced at inception) and **Realized Volatility** (market reality).
$$ P\&L_{Net} = \text{Premium} - \text{Payout} + \sum (\text{Trading P\&L}) - \text{Costs} $$
"""
story_p2 = r"""
### Strategic Objective
The goal of the **AlphaTrend** strategy is to generate **Absolute Returns (Alpha)** that are uncorrelated with the broader equity market (S&P 500). It simulates a "Quantitative Macro" desk approach by capturing strong trends across diverse asset classes (Equities, Bonds, Commodities) while dynamically hedging systematic market risk.

### Investment Universe
The strategy operates on a liquid, multi-asset universe composed of ETFs representing distinct economic factors:
* **Equities:** `SPY` (Large Cap), `QQQ` (Tech), `IWM` (Small Cap), `XLE` (Energy), `XLF` (Financials)...
* **Fixed Income:** `TLT` (20y+ Treasuries), `IEF` (7-10y Treasuries), `LQD` (Corp Bonds), `TIP` (Inflation-Protected)...
* **Commodities:** `GLD` (Gold), `USO` (Oil), `DBA` (Agriculture)...

### Signal Generation (Alpha Engine)
The strategy evaluates assets using one of three user-selectable momentum methodologies calculated on a rolling lookback window (default $L=126$ days):
* **Risk-Adjusted Momentum (Z-Score) [Default]:** Normalizes returns by their volatility to favor smooth trends.
$$
Score_t = \frac{R_{t, t-L}}{\sigma_{t} \times \sqrt{252}}
$$
* **Distance to Moving Average:** Measures price extension relative to its long-term trend.
$$
Score_t = \frac{P_t}{MA_L(P)} - 1
$$
* **Relative Strength Index (RSI):** Measures the velocity and magnitude of directional price movements.

### Portfolio Construction & Selection Logic
AlphaTrend employs a **Greedy Decorrelation Algorithm** to prevent concentration risk:
* **Regime Filter:** Assets with a **Negative Score** are strictly rejected. If no assets qualify, the strategy allocates to **CASH** (Capital Preservation).
* **Greedy Selection:** Assets are ranked by Score. The best asset is selected first. Subsequent assets are added **only if** their correlation with existing selections is below the `Corr_Threshold` (e.g., 0.6).
* **Weighting Scheme:** Weights are assigned based on **Inverse Volatility** (Risk Parity) between asset classes to balance risk contribution.

### Risk Management & Hedging
To isolate Alpha, the strategy neutralizes its exposure to the market factor (Beta):
* **Beta Calculation:** The portfolio's weighted Beta ($\beta_{port}$) relative to the S&P 500 is calculated dynamically.
* **Dynamic Hedging:** A short position in `SPY` is taken to offset systematic risk:
$$
Hedge_{Size} = - (\beta_{port} \times Portfolio_{NAV})
$$
* **Net Exposure:** The goal is to maintain a **Net Beta $\approx 0$**, ensuring performance is driven by asset selection rather than market direction.

### Key Performance Metrics
* **Alpha ($\alpha$):** The intercept of the regression vs. SPY. Must be positive.
* **Sharpe & Calmar Ratios:** Measures of risk-adjusted return and drawdown resilience.
* **Skewness:** Targeting positive skewness (limiting tail risk).
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
# DIALOG FUNCTION (POP-UP)
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

# --- MAIN TITLE ---
st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>Choose Your Project</h1>", unsafe_allow_html=True)

# --- DYNAMIC SUBTITLE ---
st.markdown(
    "<p style='text-align: center; margin-bottom: 50px; color: gray; font-size: 1.1rem;'>"
    "Bridging quantitative theory with industrial implementation through interactive research modules."
    "</p>", 
    unsafe_allow_html=True
)

# --- PROJECT GRID ---
col1, col2, col3 = st.columns(3, gap="medium")

# --- CARD 1 : PRICING ---
with col1:
    with st.container(border=True):
        # CENTERED TITLE VIA HTML
        st.markdown("<h3 style='text-align: center;'>Derivatives Pricing</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="height: 130px; text-align: justify;">
        Advanced pricing engine for Vanilla and Structured Products (Phoenix Autocall). 
        Includes Monte Carlo simulations, full Greeks analysis, and Delta-Hedging backtesting simulations.
        </div>
        """, unsafe_allow_html=True)
        
        # SIDE-BY-SIDE BUTTONS
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("Explore", key="btn_pricing", type="primary", use_container_width=True):
                st.switch_page("pages/01_Pricing&Hedging_Derivatives.py")
        with b_col2:
            if st.button("Methodology", key="story_p1", use_container_width=True):
                show_methodology("Pricing & Hedging Derivatives", story_p1)

# --- CARD 2 : ALPHATREND ---
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
            # Visually "Primary" Button (Green)
            if st.button("Explore", key="btn_invest", type="primary", use_container_width=True):
                st.switch_page("pages/02_AlphaTrend_Strategy.py")
        with b_col2:
            if st.button("Methodology", key="story_p2", use_container_width=True):
                show_methodology("AlphaTrend: Beta Neutral", story_p2)

# --- CARD 3 : ROBUST COVARIANCE ---
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
            # Visually "Primary" Button (Green)
            if st.button("Explore", key="btn_vol", type="primary", use_container_width=True):
                st.toast("🚧 Work in Progress! Please check the Methodology for details.", icon="⚠️")
        with b_col2:
            if st.button("Methodology", key="story_p3", use_container_width=True):
                show_methodology("Robust Covariance Estimation (In Dev)", story_p3)