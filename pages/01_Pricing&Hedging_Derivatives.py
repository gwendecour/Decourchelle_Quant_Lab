import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# --- MODULE IMPORTS ---
from src.shared.market_data import MarketData
from src.derivatives.instruments import InstrumentFactory
from src.derivatives.pricing_model import EuropeanOption
from src.derivatives.structured_products import PhoenixStructure
from src.derivatives.backtester import DeltaHedgingEngine
import src.derivatives.analytics as analytics
from src.shared.ui import render_header

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Pricing Engine", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>

        .block-container {
            padding-top: 2rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# CSS for inputs/sliders alignment
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
        .stButton button {height: 2.2rem; font-size: 0.8rem;}
        .stNumberInput input {height: 2rem;}
    </style>
""", unsafe_allow_html=True)

render_header()

# ==============================================================================
# STATE & UTILS
# ==============================================================================
defaults = {
    'custom_spot': 100.0, 'custom_vol': 0.20, 'custom_rate': 0.04, 'custom_div': 0.00,
    'ticker_ref': None, 'global_product_type': None, 
    'strike_pct': 100.0, 'barrier_pct': 0.60, 'coupon_barrier_pct': 0.70, 
    'autocall_pct': 1.00, 'coupon_rate': 0.08, 'maturity': 1.0,
    'market_spot': None, 'market_vol': None, 'market_rate': None, 'market_div': None
}
for key, val in defaults.items():
    if key not in st.session_state: st.session_state[key] = val

# Initialize Greeks tab keys to ensure existence
if 'gk_spot' not in st.session_state: st.session_state['gk_spot'] = st.session_state['custom_spot']
if 'gk_vol' not in st.session_state: st.session_state['gk_vol'] = st.session_state['custom_vol']
if 'gk_rate' not in st.session_state: st.session_state['gk_rate'] = st.session_state['custom_rate']

# --- INITIALIZATION & SYNCHRONIZATION HELPERS ---
def update_all_widget_keys(spot=None, vol=None, rate=None, div=None, strike=None, maturity=None):
    """Updates state variables and triggers widget refresh."""
    if spot is not None:
        val = float(spot)
        st.session_state.custom_spot = val
        st.session_state['sl_custom_spot'] = val
        st.session_state['num_custom_spot'] = val
        
        st.session_state['fix_strike_input'] = val
        
        st.session_state['sim_spot_val'] = val
        st.session_state['gk_slider_spot'] = val
        st.session_state['gk_box_spot'] = val

    if vol is not None:
        val = float(vol)
        st.session_state.custom_vol = val
        st.session_state['sl_custom_vol'] = val
        st.session_state['num_custom_vol'] = val
        
        st.session_state['gk_vol_slider'] = val * 100.0

    if rate is not None:
        st.session_state.custom_rate = float(rate)
        st.session_state['sl_custom_rate'] = float(rate)
        st.session_state['num_custom_rate'] = float(rate)
    
    if div is not None:
        st.session_state.custom_div = float(div)
        st.session_state['sl_custom_div'] = float(div)
        st.session_state['num_custom_div'] = float(div)
        
    if strike is not None:
        st.session_state.strike_pct = float(strike)
        if 'sl_strike_pct' in st.session_state: st.session_state['sl_strike_pct'] = float(strike)
        if 'num_strike_pct' in st.session_state: st.session_state['num_strike_pct'] = float(strike)

    if maturity is not None:
        new_mat = max(0.01, float(maturity))
        st.session_state.maturity = new_mat
        if 'num_maturity_pricing' in st.session_state: st.session_state['num_maturity_pricing'] = new_mat
        if 'num_maturity_greeks' in st.session_state: st.session_state['num_maturity_greeks'] = new_mat

def sync_sim_to_strike():
    """Aligns the Simulated Spot with the Fixed Strike value."""
    new_val = st.session_state.fix_strike_input
    st.session_state.sim_spot_val = new_val
    st.session_state.slider_sim_spot = new_val
    st.session_state.box_sim_spot = new_val

# --- Slider <-> Input synchronization functions (TAB 1) ---
def sync_input(master_key, changed_widget_key):
    """Synchronizes Slider <-> Number Input and activates CUSTOM mode."""
    new_value = st.session_state[changed_widget_key]
    st.session_state[master_key] = new_value
    
    if "sl_" in changed_widget_key:
        st.session_state[f"num_{master_key}"] = new_value
    else:
        st.session_state[f"sl_{master_key}"] = new_value
        
    st.session_state.ticker_input = "CUSTOM"

def make_input_group(label, key_base, min_v, max_v, step, format_str="%.2f"):
    """Crée un slider et un input box synchronisés"""
    
    # --- CRUCIAL: FORCE SYNC BEFORE DISPLAY ---
    # After a Reset, key_base changes, but sl_... and num_... do not.
    # Therefore, force widgets to align with the current master value.
    current_master_val = float(st.session_state[key_base])
    st.session_state[f"sl_{key_base}"] = current_master_val
    st.session_state[f"num_{key_base}"] = current_master_val

    col_s, col_i = st.columns([3, 1])
    
    with col_s:
        # Slider
        st.slider(
            label, min_v, max_v, step=step, 
            key=f"sl_{key_base}", 
            on_change=sync_input, 
            args=(key_base, f"sl_{key_base}"), # Arguments passés au callback
            label_visibility="visible"
        )
        
    with col_i:
        # Box
        st.number_input(
            "", min_v, max_v, step=step, format=format_str, 
            key=f"num_{key_base}",
            on_change=sync_input, 
            args=(key_base, f"num_{key_base}"), 
            label_visibility="hidden"
        )

# --- Synchronize from TAB 2 (Greeks) to the rest ---
def sync_from_greeks_tab(): 
    st.session_state.custom_spot = st.session_state.gk_spot
    st.session_state.custom_vol = st.session_state.gk_vol
    st.session_state.custom_rate = st.session_state.gk_rate
    st.session_state['sl_custom_spot'] = st.session_state.gk_spot
    st.session_state['num_custom_spot'] = st.session_state.gk_spot
    st.session_state['sl_custom_vol'] = st.session_state.gk_vol
    st.session_state['num_custom_vol'] = st.session_state.gk_vol
    st.session_state['sl_custom_rate'] = st.session_state.gk_rate
    st.session_state['num_custom_rate'] = st.session_state.gk_rate

def reset_phoenix_props():
    """Resets Phoenix-specific properties to their default values."""
    st.session_state.coupon_rate = 0.08
    st.session_state.autocall_pct = 1.00
    st.session_state.coupon_barrier_pct = 0.70
    st.session_state.barrier_pct = 0.60


# --- Callbacks ---
def switch_to_custom_market():
    """
    Triggered by manual updates to Spot/Vol/Rate.
    Switches ticker to CUSTOM and propagates values to the simulation tab.
    """
    st.session_state.ticker_input = "CUSTOM" 
    
    new_spot = float(st.session_state.custom_spot)
    st.session_state.sim_spot_val = new_spot
    st.session_state.gk_slider_spot = new_spot
    st.session_state.gk_box_spot = new_spot
    
    new_vol = float(st.session_state.custom_vol)
    st.session_state.gk_vol_slider = new_vol * 100.0

def update_market_data():
    ticker = st.session_state.ticker_input
    try:
        spot = MarketData.get_spot(ticker)
        vol = MarketData.get_volatility(ticker, "1y")
        rate = MarketData.get_risk_free_rate() or 0.04
        div = MarketData.get_dividend_yield(ticker) or 0.0
        
        # Save raw market data for reset functionality
        st.session_state.market_spot = float(spot)
        st.session_state.market_vol = float(vol)
        st.session_state.market_rate = float(rate)
        st.session_state.market_div = float(div)
        
        # Update sliders
        update_all_widget_keys(float(spot), float(vol), float(rate), float(div), strike=100.0)
        
    except Exception as e:
        st.error(f"Error: {e}")

def set_pricing_scenario(scenario_type):
    """
    Robust Logic:
    1. Always retrieve the original market data (Reference).
    2. Apply scenarios directly to the reference to avoid accumulation of modifications.
    """
    # Data Retrieval
    ref_spot = st.session_state.get('market_spot')
    ref_vol = st.session_state.get('market_vol')
    ref_rate = st.session_state.get('market_rate')
    ref_div = st.session_state.get('market_div')
    
    # Fallback if no data is loaded
    if ref_spot is None: ref_spot = 100.0
    if ref_vol is None: ref_vol = 0.20
    if ref_rate is None: ref_rate = 0.04
    if ref_div is None: ref_div = 0.00
    
    current_mat = st.session_state.get('maturity', 1.0) 

    # Apply Scenario
    p_type = st.session_state.global_product_type
    
    # --- STRUCTURE SCENARIOS ---
    if scenario_type == "ATM":
        update_all_widget_keys(spot=ref_spot, vol=ref_vol, strike=100.0)
        
    elif scenario_type == "ITM":
        new_strike_pct = 80.0 if "Call" in p_type else 120.0
        update_all_widget_keys(spot=ref_spot, vol=ref_vol, strike=new_strike_pct)
        
    elif scenario_type == "OTM":
        new_strike_pct = 120.0 if "Call" in p_type else 80.0
        skewed_vol = max(ref_vol * 0.9, 0.05) # -10% on reference volatility
        update_all_widget_keys(spot=ref_spot, vol=skewed_vol, strike=new_strike_pct)

    elif scenario_type == "Reset":
        if ref_spot is not None:
            update_all_widget_keys(spot=ref_spot, vol=ref_vol, rate=ref_rate, div=ref_div, strike=100.0)
        else:
            # No market data available, defaulting
            update_all_widget_keys(spot=100.0, vol=0.20, rate=0.04, div=0.00, strike=100.0)

def set_greeks_scenario(scenario_type):
    """
    Handles Tab 2 scenarios (Stress Test Simulation).
    """

    ref_spot = st.session_state.get('fix_strike_input', st.session_state.custom_spot)
    ref_vol = st.session_state.get('market_vol')
    current_mat = st.session_state.get('gk_fix_mat', 1.0)

    # References
    if ref_spot is None: ref_spot = 100.0
    if ref_vol is None: ref_vol = 0.20

    if scenario_type == "Crash":
        new_spot = ref_spot * 0.85
        new_vol_pct = min((ref_vol + 0.20) * 100, 100.0)
        st.session_state.gk_slider_spot = new_spot
        st.session_state.gk_box_spot = new_spot
        st.session_state.gk_vol_slider = new_vol_pct

    elif scenario_type == "Rally":
        new_spot = ref_spot * 1.10
        new_vol_pct = max((ref_vol - 0.05) * 100, 1.0)
        st.session_state.gk_slider_spot = new_spot
        st.session_state.gk_box_spot = new_spot
        st.session_state.gk_vol_slider = new_vol_pct

    elif scenario_type == "TimeBleed":
        # Reduce maturity by 1 month, without affecting spot/vol
        new_mat = max(0.01, current_mat - (1/12))
        st.session_state.gk_fix_mat = new_mat

    elif scenario_type == "Reset":
        st.session_state.gk_slider_spot = ref_spot
        st.session_state.gk_box_spot = ref_spot
        
        real_mkt_vol = st.session_state.get('market_vol')
        if real_mkt_vol is not None:
            st.session_state.gk_vol_slider = float(real_mkt_vol * 100)
        else:
            st.session_state.gk_vol_slider = 20.0
            
        st.session_state.force_pnl_zero = True

# ==============================================================================
# HEADER
# ==============================================================================
TICKERS = {
    "GLE.PA": "SocGen (Bank)",
    "BNP.PA": "BNP Paribas (Bank)",
    "MC.PA": "LVMH (Luxury)",
    "TTE.PA": "TotalEnergies (Energy)",
    "SAN.PA": "Sanofi (Health)",
    "AIR.PA": "Airbus (Indus)",
    "CAP.PA": "Capgemini (Tech)",
    "CUSTOM": "User Defined Data"
}

with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1.5, 1.5, 1.5, 4])
    
    with c1:
        selected_ticker = st.selectbox(
            "Ticker", 
            options=list(TICKERS.keys()), 
            index=None, 
            placeholder="Select Ticker...", 
            format_func=lambda x: f"{x} - {TICKERS[x]}", 
            key="ticker_input",
            label_visibility="collapsed"
        )
        
    with c2:
        selected_product = st.selectbox(
            "Product", 
            options=["Call", "Put", "Phoenix"], 
            index=None, 
            placeholder="Select Product...", 
            key="global_product_type",
            label_visibility="collapsed"
        )
        
    with c3:
        btn_disabled = (selected_ticker is None)
        st.button("Load Market Data", on_click=update_market_data, disabled=btn_disabled, use_container_width=True)
        
    with c4:
        # Verify if market data is loaded
        if st.session_state.get('market_spot') is not None:
            s = st.session_state.custom_spot
            v = st.session_state.custom_vol
            r = st.session_state.custom_rate
            d = st.session_state.custom_div
            
            # Display only if loaded
            display_text = f"Spot: <b style='color:black'>{s:.2f}</b> | Vol: <b style='color:black'>{v:.1%}</b> | r: {r:.1%} | q: {d:.1%}"
            
            st.markdown(f"<div style='text-align:right; padding-top:5px; font-family:monospace; color:gray;'>"
                        f"{display_text}</div>", unsafe_allow_html=True)
        else:
            pass

if not selected_ticker or not selected_product or st.session_state.get('market_spot') is None:
    
    # Guidance message based on missing selections
    if not selected_ticker or not selected_product:
        st.markdown("**Please select a Ticker AND a Product above.**")
    else:
        # Ticker/Product selected but data missing
        st.markdown("**Please click 'Load Market Data' to initialize the Pricing Engine.**")
        
    st.stop()

# ==============================================================================
# TABS
# ==============================================================================
tab_pricing, tab_greeks, tab_backtest = st.tabs(["Pricing & Payoff", "Greeks & Heatmaps", "Delta Hedging"])

S, sigma = st.session_state.custom_spot, st.session_state.custom_vol
r, q = st.session_state.custom_rate, st.session_state.custom_div
p_type = st.session_state.global_product_type

# --- TAB 1: PRICING ---
with tab_pricing:
    layout_col1, layout_col2, layout_col3 = st.columns([1.2, 1, 2], gap="medium")

    # --- MARKET INPUTS ---
    with layout_col1:
        st.markdown("### Market")
        make_input_group("Spot ($)", "custom_spot", 10.0, 700.0, 0.5)
        make_input_group("Vol (σ)", "custom_vol", 0.01, 1.00, 0.005)
        make_input_group("Rate (r)", "custom_rate", 0.00, 0.20, 0.001, "%.3f")
        make_input_group("Div (q)", "custom_div", 0.00, 0.20, 0.001, "%.3f")

    # --- PRODUCT INPUTS ---
    with layout_col2:
        st.markdown(f"### {p_type}")
        maturity = st.number_input("Maturity (Years)",value=float(st.session_state.get("maturity", 1.0)),min_value=0.1, step=0.1, key="maturity")

        if p_type == "Phoenix":
            cpn = st.slider("Cpn", 0.0, 0.20, st.session_state.get('coupon_rate', 0.08), 0.005)
            st.session_state.coupon_rate = cpn
            auto = st.slider("Autocall (%)", 80, 120, int(st.session_state.get('autocall_pct', 1.0)*100), 5)/100
            st.session_state.autocall_pct = auto
            c_bar = st.slider("Cpn Barr (%)", 40, 90, int(st.session_state.get('coupon_barrier_pct', 0.7)*100), 5)/100
            st.session_state.coupon_barrier_pct = c_bar
            p_bar = st.slider("Prot Barr (%)", 30, 80, int(st.session_state.get('barrier_pct', 0.6)*100), 5)/100
            st.session_state.barrier_pct = p_bar
            n_sims = st.selectbox("Sims", [2000, 5000, 10000], index=0)
            
            st.write("")
            st.markdown("### Resets")
            

            st.button("Reset Market Values", on_click=set_pricing_scenario, args=("Reset",), use_container_width=True, help="Resets Spot/Vol/Rate to fetched data")
            st.button("Reset Properties", on_click=reset_phoenix_props, use_container_width=True, help="Resets Barriers and Coupon to default")

        elif p_type in ["Call", "Put"]:
            make_input_group("Moneyness (%)", "strike_pct", 50.0, 150.0, 1.0)
            strike_price = S * (st.session_state.strike_pct / 100.0)
            st.caption(f"Strike: **{strike_price:.2f} €**")
            n_sims = 0

            # --- VANILLA CASE: Structure + Market ---
            
            st.caption("1. Structure / Moneyness")
            b1, b2, b3, b4 = st.columns(4, gap="small")
            with b1: 
                st.button("Reset", on_click=set_pricing_scenario, args=("Reset",), use_container_width=True)
            with b2: 
                st.button("ATM", on_click=set_pricing_scenario, args=("ATM",), use_container_width=True, help="Strike = Spot")
            with b3: 
                st.button("ITM", on_click=set_pricing_scenario, args=("ITM",), use_container_width=True, help="In The Money")
            with b4: 
                st.button("OTM", on_click=set_pricing_scenario, args=("OTM",), use_container_width=True, help="Out The Money")

        else:
            st.error(f"Unknown product: {p_type}")

            
    # --- OUTPUT ---
    with layout_col3:
        st.markdown("### Analysis")
        
        # Instantiation
        if p_type == "Phoenix":
            product = PhoenixStructure(
                S=S, T=maturity, r=r, sigma=sigma, q=q,
                autocall_barrier=st.session_state.autocall_pct, 
                protection_barrier=st.session_state.barrier_pct,
                coupon_barrier=st.session_state.coupon_barrier_pct, 
                coupon_rate=st.session_state.coupon_rate,
                obs_frequency=4, num_simulations=n_sims
            )
            price = product.price()
            fig_main = analytics.plot_payoff(product, spot_range=[S*0.5, S*1.5])
            metric_lbl, metric_val = "Barrier", f"{S*st.session_state.barrier_pct:.2f} €"
        else:
            strike_val = S * (st.session_state.strike_pct / 100.0)
            opt_type = "Call" if "Call" in p_type else "Put"
            product = EuropeanOption(S=S, K=strike_val, T=maturity, r=r, sigma=sigma, q=q, option_type=opt_type)
            price = product.price()
            fig_main = analytics.plot_payoff(product, spot_range=[S*0.6, S*1.4])
            metric_lbl, metric_val = "Moneyness", f"{(S/strike_val)*100:.1f}%"

        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("Price", f"{price:.2f} €")
        k2.metric("% Nominal", f"{(price/S)*100:.2f} %")
        k3.metric(metric_lbl, metric_val)

        
        with st.expander("Model & Algorithm Details (How is the price calculated?)"):
            if p_type == "Phoenix":
                st.markdown("""
        **Pricing Engine: Monte Carlo Simulation**
        
        Since Phoenix Autocalls are **Path-Dependent** structures (the outcome depends on the daily/monthly history of the spot, not just the final value), closed-form formulas (like Black-Scholes) cannot be used directly.
        
        **Algorithm Steps:**
        *   **Diffusion:** We simulate **2,000 to 10,000 paths** of the underlying asset using Geometric Brownian Motion (GBM).
            * $dS_t = r S_t dt + \sigma S_t dW_t$
        *   **Observation:** For each path, we check the barriers (Coupon, Protection, Autocall) at every observation date.
        *   **Payoff Calculation:** We determine the cash flows for each specific path (Coupons paid, early redemption, or final payout).
        *   **Discounting:** We discount the average payoff back to present value using the risk-free rate $r$.
            * $Price = e^{-rT} \cdot \mathbb{E}^{\mathbb{Q}}[\text{Payoff}]$
        """)
            else:
        # Define text based on Option type
                if "Call" in p_type:
                    direction = "Call"
                    formula_latex = r"C(S,t) = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)"
                    desc = "The price represents the spot price weighted by probability minus the discounted strike."
                else:
                    direction = "Put"
                    formula_latex = r"P(S,t) = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)"
                    desc = "The price represents the discounted strike weighted by probability minus the spot price."

                st.markdown(f"""
                **Pricing Engine: Black-Scholes-Merton (Closed Form)**
        
                For Vanilla European **{direction}** Options, the payoff depends only on the spot price at maturity $T$. We use the analytical Black-Scholes formula.
        
                **Formula ({direction}):**
                * ${formula_latex}$
        
                *{desc}*
        
                Where:
                * $d_1 = \\frac{{\ln(S/K) + (r - q + \sigma^2/2)T}}{{\sigma \sqrt{{T}}}}$
                * $d_2 = d_1 - \sigma \sqrt{{T}}$
                """)
        st.divider()

        # Payoff Chart
        st.plotly_chart(fig_main, use_container_width=True)

        if p_type == "Phoenix":
            # Retrieve thresholds for display
            p_lvl = S * st.session_state.barrier_pct
            c_lvl = S * st.session_state.coupon_barrier_pct
            a_lvl = S * st.session_state.autocall_pct
            
            st.markdown(f"""
            **Phoenix Payoff Zones (at Maturity):**
            
            **Downside Risk (< {p_lvl:.2f} €):** Below the **Protection Barrier**, the capital protection disappears. You are fully exposed to the stock's fall (1:1 loss like holding the stock).
            
            **Coupon Zone ({c_lvl:.2f} € - {a_lvl:.2f} €):** Between the Coupon Barrier and Autocall level. You recover your **100% Capital** + Potential Coupons (Memory effect usually applies).
            
            **Autocall / Cap (> {a_lvl:.2f} €):** Above the Autocall level. You get **100% Capital + Coupon**. The performance is capped (you don't benefit from the stock's rise beyond the coupon).
            """)
            
        elif p_type == "Call":
             k_val = S * (st.session_state.strike_pct / 100.0)
             st.markdown(f"**Call Payoff:** Client profit if Spot > Strike (**{k_val:.2f} €**). It's the opposite for the bank Short Call")
             
        elif p_type == "Put":
             k_val = S * (st.session_state.strike_pct / 100.0)
             st.markdown(f"**Put Payoff:** Client profit if Spot < Strike (**{k_val:.2f} €**). It's the opposite for the bank Short Put")
        
    # --- ROW 2: SENSITIVITY ANALYSIS ---
    st.divider()
    
    st.markdown("### Sensitivity Analysis")
    
    graph_col1, graph_col2 = st.columns(2, gap="medium")

    with graph_col1:
        st.markdown("**Price Sensitivity to Strike**")
        with st.spinner("Computing..."):
            fig_struct = analytics.plot_price_vs_strike(product, current_spot=S)
            st.plotly_chart(fig_struct, use_container_width=True, config={'displayModeBar': False})

            if p_type == "Call":
                note = "**Trend:** Decreasing. Higher strike decreases probability of exercise."
            elif p_type == "Put":
                note = "**Trend:** Increasing. Higher strike increases probability of exercise."
            elif p_type == "Phoenix":
                note = "**Trend:** Sharp rise below Protection Barrier, steady in Coupon Zone, flat above Autocall."
            st.caption(note)
            
    with graph_col2:
        st.markdown("**Price Sensitivity to Volatility**")
        with st.spinner("Computing..."):
            fig_vol = analytics.plot_price_vs_vol(product, current_vol=sigma)
            st.plotly_chart(fig_vol, use_container_width=True, config={'displayModeBar': False})
            if p_type in ["Call", "Put"]:
                note_vol = "**Trend:** Positive Vega. Long options benefit from higher uncertainty/volatility."
            elif p_type == "Phoenix":
                note_vol = "**Trend:** Negative Vega. Higher volatility increases the risk of hitting the downside barrier, lowering the price."
            st.caption(note_vol)



# --- TAB 2: GREEKS & HEATMAPS ---
with tab_greeks:
    st.subheader("Greeks Sensitivity Analysis")

    # Layout: 2 Columns
    col_params, col_metrics = st.columns([1.3, 1], gap="large")

    # LEFT COLUMN: PARAMETERS & SIMULATION
    
    with col_params:
        
        # --- CONTRACT SETUP (FIXED) ---
        st.markdown("**Contract Setup (Fixed)**")
        
        c_def1, c_def2 = st.columns(2)
        with c_def1:
            fixed_maturity = st.number_input("Maturity (Years)", 0.01, 10.0, 1.0, 0.1, key="gk_fix_mat")
        
        with c_def2:
            if p_type == "Phoenix":
                st.markdown(f"**Ref Spot:** {S:.2f} €")
                ref_value = S
            else:
                # Callback: Realign simulation if fixed strike is changed
                def sync_sim_to_strike():
                    val = st.session_state.fix_strike_input
                    st.session_state.sim_spot_val = val
                    st.session_state.gk_slider_spot = val
                    st.session_state.gk_box_spot = val

                # Robust default defaults to avoid yellow warnings
                def_k = float(st.session_state.get('fix_strike_input', S))
                
                fixed_strike = st.number_input("Strike (€)", value=def_k, step=1.0, format="%.2f", 
                                               key="fix_strike_input", 
                                               on_change=sync_sim_to_strike)
                ref_value = fixed_strike

        st.divider()

        # --- MARKET SIMULATION (VARIABLE) ---
        st.markdown("**Market Simulation**")
        
        # Robust Initialization
        if 'sim_spot_val' not in st.session_state: 
            st.session_state.sim_spot_val = float(S)
        
        # Volatility Initialization
        if 'gk_vol_slider' not in st.session_state:
            st.session_state.gk_vol_slider = float(sigma * 100)

        # Ensure specific widget keys exist
        if 'gk_slider_spot' not in st.session_state:
            st.session_state.gk_slider_spot = st.session_state.sim_spot_val
        if 'gk_box_spot' not in st.session_state:
            st.session_state.gk_box_spot = st.session_state.sim_spot_val

        # CROSS-SYNC Callbacks (Box <-> Slider)
        def update_slider():
            val = st.session_state.gk_slider_spot
            st.session_state.sim_spot_val = val
            st.session_state.gk_box_spot = val # Force box

        def update_box():
            val = st.session_state.gk_box_spot
            st.session_state.sim_spot_val = val
            st.session_state.gk_slider_spot = val # Force slider

        # Range definition
        max_spot = float(ref_value * 2.0) if ref_value > 0 else 100.0
        
        # Slider / Box Display
        c_sim1, c_sim2 = st.columns([3, 1])
        with c_sim1:
            st.slider("Spot Range", 0.0, max_spot, key="gk_slider_spot", 
                      on_change=update_slider, label_visibility="collapsed")
        with c_sim2:
            st.number_input("Spot", 0.0, max_spot, key="gk_box_spot", 
                            on_change=update_box, label_visibility="collapsed")
        
        # DYNAMIC Variable for calculation
        dyn_spot = st.session_state.sim_spot_val 
        
        # Visual feedback (% move)
        pct_move = (dyn_spot / ref_value - 1) * 100 if ref_value > 0 else 0
        st.caption(f"Simulated Spot: **{dyn_spot:.2f} €** ({pct_move:+.2f}%)")

        # Volatility
        st.write("")
        st.slider("Volatility (%)", 1.0, 100.0, key="gk_vol_slider")
        
        # Reading state to ensure correct post-Reset values
        dyn_vol = st.session_state.gk_vol_slider / 100.0

        st.divider()
        
        # Scenario Buttons (Tab 2 Only)
        st.caption("Quick Scenarios")
        b1, b2, b3, b4 = st.columns(4)
        
        with b1: st.button("Crash", on_click=set_greeks_scenario, args=("Crash",), use_container_width=True, help="**Market Crash:**\n- Spot: -15%\n- Volatility: +20 pts (Fear spike)\n\nSimulates a sudden market drop panic.")
        with b2: st.button("Rally", on_click=set_greeks_scenario, args=("Rally",), use_container_width=True, help="**Bull Rally:**\n- Spot: +10%\n- Volatility: -5 pts (Calm)\n\nSimulates a steady market rise.")
        with b3: st.button("Bleed", on_click=set_greeks_scenario, args=("TimeBleed",), use_container_width=True, help="**Time Decay:**\n- Maturity: -1 Month\n- Spot/Vol: Unchanged\n\nIsolates the effect of Theta (Time passing).")
        with b4: st.button("Reset", on_click=set_greeks_scenario, args=("Reset",), use_container_width=True, help="**Reset:**\nReverts all parameters (Spot, Vol, Time) to the initial Market Data values.")
    
    # RIGHT COLUMN: METRICS & P&L
    with col_metrics:
        st.markdown("#### Greeks (Bank View)")
        
        # PRICING WITH DYNAMIC SPOT (dyn_spot)
        if p_type == "Phoenix":
            prod_gk = PhoenixStructure(
                S=dyn_spot,       # Slider Spot
                T=fixed_maturity, 
                r=r, 
                sigma=dyn_vol,    # Slider Vol
                q=q,
                autocall_barrier=st.session_state.autocall_pct,
                protection_barrier=st.session_state.barrier_pct,
                coupon_barrier=st.session_state.coupon_barrier_pct,
                coupon_rate=st.session_state.coupon_rate, 
                obs_frequency=4, 
                num_simulations=n_sims
            )
            
            # Delta Live
            with st.spinner("Calc Delta..."):
                client_delta = prod_gk.calculate_delta_quick(n_sims=2000)
            
            greeks = {'delta': -client_delta, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}

            # Full Greeks on demand
            if st.button("Compute Full Greeks"):
                c_greeks = prod_gk.greeks()
                greeks = {k: -v for k, v in c_greeks.items()}

        else:
            # Vanilla Case
            prod_gk = EuropeanOption(S=dyn_spot, K=fixed_strike, T=fixed_maturity, r=r, sigma=dyn_vol, q=q, option_type=p_type)
            cg = prod_gk.greeks()
            greeks = {k: -v for k, v in cg.items()}

        # GREEKS DISPLAY
        m1, m2 = st.columns(2)
        m1.metric("Delta (Δ)", f"{greeks.get('delta',0):.4f}")
        m1.metric("Gamma (Γ)", f"{greeks.get('gamma',0):.4f}")
        m2.metric("Vega (ν)", f"{greeks.get('vega',0):.4f}")
        m2.metric("Theta (Θ)", f"{greeks.get('theta',0):.4f}")

        
        with st.expander("How are Greeks calculated? (Methodology)"):
    
            # CAS 1 : PRODUIT COMPLEXE (PHOENIX)
            if p_type == "Phoenix":
                st.markdown(r"""
        **Method: Finite Differences ("Bump & Revalue")**
        
        Since the Phoenix Autocall has a **path-dependent** payoff (barriers, memory effect) and no closed-form solution, we cannot calculate derivatives analytically. Instead, we perform a numerical approximation by slightly shifting market parameters and re-running the Monte Carlo simulation.
        
        * **Delta ($\Delta$):** We use a **Central Difference** approach. We re-price the product with Spot $\pm 1\%$ ($\epsilon$).
            $$ \Delta \approx \frac{P(S+\epsilon) - P(S-\epsilon)}{2\epsilon} $$
            
        * **Gamma ($\Gamma$):** Derived from the "curvature" of the price (second derivative).
            $$ \Gamma \approx \frac{P(S+\epsilon) - 2P(S) + P(S-\epsilon)}{\epsilon^2} $$
            
        * **Vega ($\nu$):** We shift the entire volatility surface by $+1\%$ (Forward Difference).
            $$ \nu \approx P(\sigma + 1\%) - P(\sigma) $$

        **Why isn't it instant? (Computational Cost)**
        Unlike Vanilla options where formulas are executed in milliseconds, each term $P(\cdot)$ in the equations above represents a **full Monte Carlo simulation** (e.g., 10,000 simulations). 
        To calculate Delta, Gamma, and Vega, the engine must therefore run **4 to 5 separate Monte Carlo simulations** in sequence. This explains the slight delay compared to Vanilla options.
        
        """)

    # CAS 2 : PRODUIT VANILLA (CALL / PUT)
            else:
                st.markdown(r"""
        **Method: Analytical Formulas (Black-Scholes-Merton)**
        
        For European Vanilla options (Call & Put), the payoff depends solely on the final spot price. We do not need simulation; we use the exact **closed-form partial derivatives** from the Black-Scholes model.
        
        * **Delta ($\Delta$):** Represents the probability of the option expiring in-the-money (adjusted).
            $$ \Delta_{call} = N(d_1) \quad ; \quad \Delta_{put} = N(d_1) - 1 $$
            
        * **Gamma ($\Gamma$):** Identical for Call and Put (measure of convexity).
            $$ \Gamma = \frac{N'(d_1)}{S \sigma \sqrt{T}} $$
            
        * **Vega ($\nu$):** Sensitivity to volatility (identical for Call and Put).
            $$ \nu = S \sqrt{T} N'(d_1) $$
            
        *(Where $N(x)$ is the standard normal cumulative distribution function and $d_1$ is the standard BS factor).*
        """)

        st.divider()

        # P&L DECOMPOSITION
        st.markdown("#### P&L Attribution")
        
        # Differentials
        d_spot = dyn_spot - ref_value
        d_vol = dyn_vol - sigma
        
        # Taylor
        pnl_delta = greeks.get('delta', 0) * d_spot
        pnl_gamma = 0.5 * greeks.get('gamma', 0) * (d_spot ** 2)
        pnl_vega = greeks.get('vega', 0) * (d_vol * 100) # Assuming Vega per 1%
        
        taylor_pnl = pnl_delta + pnl_gamma + pnl_vega
        
        # Real P&L (Repricing)
        if p_type == "Phoenix":
            prod_ref = PhoenixStructure(S=ref_value, T=fixed_maturity, r=r, sigma=sigma, q=q,
                                      autocall_barrier=st.session_state.autocall_pct,
                                      protection_barrier=st.session_state.barrier_pct,
                                      coupon_barrier=st.session_state.coupon_barrier_pct,
                                      coupon_rate=st.session_state.coupon_rate, obs_frequency=4, num_simulations=2000)
        else:
            prod_ref = EuropeanOption(S=ref_value, K=fixed_strike, T=fixed_maturity, r=r, sigma=sigma, q=q, option_type=p_type)

        real_pnl = - (prod_gk.price() - prod_ref.price())

        # If the Reset button has just been clicked, force visual cleanup
        if st.session_state.get('force_pnl_zero', False):
            real_pnl = 0.0
            taylor_pnl = 0.0
            pnl_delta = 0.0
            pnl_gamma = 0.0
            pnl_vega = 0.0
            # Turn off the flag so that next movements calculate normally
            st.session_state.force_pnl_zero = False

        # P&L Display
        c_pnl1, c_pnl2 = st.columns(2)
        color = "normal" if real_pnl >= 0 else "inverse"
        c_pnl1.metric("ACTUAL P&L", f"{real_pnl:+.2f} €", delta_color=color)
        c_pnl2.metric("Taylor Est.", f"{taylor_pnl:+.2f} €", delta=f"{taylor_pnl-real_pnl:.2f} err", delta_color="off")

        cols = st.columns(3)
        cols[0].metric("Delta P&L", f"{pnl_delta:+.2f}")
        cols[1].metric("Gamma P&L", f"{pnl_gamma:+.2f}")
        cols[2].metric("Vega P&L", f"{pnl_vega:+.2f}")

        # ... (Greeks calculation and P&L Attribution done just before) ...
    
        # --- DYNAMIC EXPLANATION LOGIC ---
        st.subheader("P&L Attribution Analysis")
    
        # Detect the simulated movement
        spot_move = st.session_state.sim_spot_val - S # S = Initial Spot
        vol_move = (st.session_state.gk_vol_slider/100.0) - sigma # sigma = Initial Vol
    
        explanation = []
    
        # Product Analysis (BANK / SELLER View)
        if p_type == "Call":
            role = "Short Call"
            explanation.append(f"**Position:** You are **{role}** (Bank View). You are Short Delta, Short Gamma, Short Vega, Long Theta.")
        
            # Delta/Gamma Analysis
            if spot_move > 0:
                explanation.append(f"**Spot (+):** Market went UP. Being Short Delta, you **lost money** on Delta.")
                explanation.append(f"**Gamma Impact:** As Spot rose, your negative Delta became even more negative (Short Gamma). **Losses accelerated**.")
            elif spot_move < 0:
                explanation.append(f"**Spot (-):** Market went DOWN. Being Short Delta, you **made money** on Delta.")
                explanation.append(f"**Gamma Impact:** Short Gamma worked in your favor here (cushioning losses or accelerating gains).")
            
            # Vega Analysis
            if vol_move > 0:
                explanation.append(f"**Vol (+):** Implied Vol rose. Being Short Vega, the option price increased, so you **lost money** (Mark-to-Market).")
            elif vol_move < 0:
                explanation.append(f"**Vol (-):** Vol dropped. Being Short Vega, you **gained money**.")

        elif p_type == "Put":
            role = "Short Put"
            explanation.append(f"**Position:** You are **{role}**. You are Long Delta (Bullish), Short Gamma, Short Vega, Long Theta.")
        
            # Delta Analysis
            if spot_move > 0:
                explanation.append(f"**Spot (+):** Market went UP. Being Long Delta, you **made money** (Put value dropped).")
            elif spot_move < 0:
                explanation.append(f"**Spot (-):** Market went DOWN. Being Long Delta, you **lost money**.")
            
            # Vega Analysis (Same as Call)
            if vol_move > 0:
                explanation.append(f"**Vol (+):** Vol rose. Short Vega -> **Loss**.")
            elif vol_move < 0:
                explanation.append(f"**Vol (-):** Vol dropped. Short Vega -> **Gain**.")

        elif p_type == "Phoenix":
            role = "Short Phoenix (Issuer)"
            explanation.append(f"**Position:** You are **{role}**. Generally Long Vega (unlike vanilla), Long Theta, and Mixed Delta/Gamma.")
        
            # Vega Specificity for Phoenix
            if vol_move > 0:
                explanation.append(f"**Vol (+):** Uniquely here, Vol rising often helps the Issuer (Long Vega). Higher Vol increases the probability of hitting the downside barrier (Knock-In), lowering the product's value (your liability). -> **Gain**.")
            elif vol_move < 0:
                explanation.append(f"**Vol (-):** Vol dropping makes the product safer for the client. Its value rises. -> **Loss**.")
            
            # Delta/Spot
            if spot_move < 0:
                explanation.append(f"**Spot (-):** Market drop. The product gets closer to the risk barrier. Its value drops heavily. You **gain**.")
            elif spot_move > 0:
                explanation.append(f"**Spot (+):** Market rise. The product gets closer to Autocall (paying 100% + Cpn). Its value rises towards Par. You **lose** (or gain less).")

        
        # Clean display
        st.markdown("\n\n".join(explanation))    

    # ==========================================================================
    # PART 3: HEATMAPS 
    # ==========================================================================
    st.divider()
    st.subheader("Risk Heatmaps (Scenario Analysis)")
    
    # Heatmap specific controls
    hm_c1, hm_c2, hm_c3 = st.columns(3)
    with hm_c1: 
        hm_spot_rng = st.slider("Matrix Spot Range (%)", 5, 50, 15, 5) / 100
    with hm_c2: 
        # Vol Slider WITHOUT KEY or with unique key to avoid conflicts
        hm_vol_rng = st.slider("Matrix Vol Range (pts)", 5, 50, 10, 5) / 100
    with hm_c3:
        hm_mode = st.radio("View Mode", ["2D Matrix", "3D Surface"], horizontal=True)
        if p_type == "Phoenix":
            mc_prec = st.select_slider("MC Precision", [500, 1000, 2000], value=1000)
        else:
            mc_prec = 0

    if hm_mode == "2D Matrix":
        with st.spinner("Computing Scenarios..."):
            mat_u, mat_h, x_m, y_m = prod_gk.compute_scenario_matrices(
                spot_range_pct=hm_spot_rng, vol_range_abs=hm_vol_rng, n_spot=5, n_vol=5, matrix_sims=mc_prec
            )
        
        x_lab = [f"{x:+.0%}" for x in x_m]
        y_lab = [f"{y:+.0%}" for y in y_m]
        
        fig_hm = make_subplots(rows=1, cols=2, subplot_titles=("Unhedged P&L", "Delta-Hedged P&L"))
        fig_hm.add_trace(go.Heatmap(z=mat_u, x=x_lab, y=y_lab, colorscale='RdYlGn', zmid=0, text=np.round(mat_u, 2), texttemplate="%{text}", showscale=False), row=1, col=1)
        fig_hm.add_trace(go.Heatmap(z=mat_h, x=x_lab, y=y_lab, colorscale='RdYlGn', zmid=0, text=np.round(mat_h, 2), texttemplate="%{text}", showscale=True), row=1, col=2)
        fig_hm.update_layout(height=400, margin=dict(t=50, b=50))
        fig_hm.update_xaxes(title_text="Spot Variation (%)")
        fig_hm.update_yaxes(title_text="Vol Variation (pts %)")
        st.plotly_chart(fig_hm, use_container_width=True)
    
    else:
        with st.spinner("Generating Surface..."):
            n_g = 15 if p_type != "Phoenix" else 9
            mat_u, _, x_m, y_m = prod_gk.compute_scenario_matrices(
                spot_range_pct=hm_spot_rng, vol_range_abs=hm_vol_rng, n_spot=n_g, n_vol=n_g, matrix_sims=mc_prec
            )
        
        X_pct, Y_pct = np.meshgrid(x_m * 100, y_m * 100)
        fig_3d = go.Figure(data=[go.Surface(z=mat_u, x=X_pct, y=Y_pct, colorscale='Viridis', opacity=0.9)])
        fig_3d.update_layout(title="P&L Surface", scene=dict(xaxis_title='Spot Move (%)', yaxis_title='Vol Move (pts %)', zaxis_title='P&L (€)'), height=400)
        st.plotly_chart(fig_3d, use_container_width=True)
    

    st.divider()
    st.subheader("Structural Analysis: Greeks vs Spot")

    if p_type in ["Call", "Put"]:
        with st.spinner("Computing Greeks Profile..."):
            # prod_gk object already contains dyn_spot (red point) 
            # and fixed_strike (dotted line)
            fig_structure = analytics.plot_greeks_profile(prod_gk)
            st.plotly_chart(fig_structure, use_container_width=True)
            
    elif p_type == "Phoenix":
        st.markdown("Structural Analysis graphs are disabled for Phoenix (Computationally too heavy).")

# ==============================================================================
# TAB 3: BACKTEST
# ==============================================================================

with tab_backtest:
    st.subheader("Dynamic Hedging Simulation")
    
    with st.container(border=True):
        
        # PRODUCT CONFIGURATION
        st.markdown(f"#### {p_type} Configuration")
        
        # Init variables
        bt_autocall, bt_coupon_bar, bt_protection, bt_coupon_rate = 0, 0, 0, 0
        bt_strike_pct = 1.0
        bt_maturity = 1.0 

        if p_type == "Phoenix":
            # --- PHOENIX SETUP ---
            phx_c1, phx_c2, phx_c3, phx_c4, phx_c5 = st.columns(5)
            
            with phx_c1:
                val_ac = st.number_input("Autocall Barrier (%)", value=100.0, step=5.0, key="bt_ac_input")
                bt_autocall = val_ac / 100.0
            with phx_c2:
                val_cb = st.number_input("Coupon Barrier (%)", value=60.0, step=5.0, key="bt_cb_input")
                bt_coupon_bar = val_cb / 100.0
            with phx_c3:
                val_pb = st.number_input("Protection Barrier (%)", value=60.0, step=5.0, key="bt_pb_input")
                bt_protection = val_pb / 100.0
            with phx_c4:
                val_cr = st.number_input("Annual Coupon (%)", value=8.0, step=0.5, key="bt_cr_input")
                bt_coupon_rate = val_cr / 100.0
            with phx_c5:
                bt_maturity = st.number_input("Maturity (Years)", value=5.0, step=1.0, min_value=0.5, key="bt_mat_phx")
                
        else:
            # --- VANILLA SETUP ---
            vanilla_c1, vanilla_c2 = st.columns([2, 2])
            with vanilla_c1:
                bt_strike_pct = st.number_input("Strike % Init Spot", value=1.0, step=0.05, key="bt_strike_input")
            with vanilla_c2:
                bt_maturity = st.number_input("Maturity (Years)", value=1.0, step=0.25, min_value=0.1, key="bt_mat_vanilla")

        # MARKET & SIMULATION SETTINGS
        st.markdown("#### Market & Execution Settings")
        
        sim_c1, sim_c2, sim_c3 = st.columns(3)
        
        with sim_c1:
            rebal_freq = st.selectbox("Rebalancing Freq", ["Daily", "Weekly"], index=0, key="bt_freq_input")
            
        with sim_c2:
            tc_val = st.number_input("Transaction Cost (%)", value=0.10, step=0.05, format="%.2f", key="bt_tc_input")
            transaction_cost_pct = tc_val / 100.0
            
        with sim_c3:
            period_choice = st.selectbox(
                "Historical Period", 
                ["Last 3 Months", "Last 6 Months", "Last 1 Year", "Last 2 Years", "YTD"],
                index=2, 
                key="bt_period_input"
            )
            
            today = datetime.date.today()
            if period_choice == "Last 3 Months": start_d_calc = today - datetime.timedelta(days=90)
            elif period_choice == "Last 6 Months": start_d_calc = today - datetime.timedelta(days=180)
            elif period_choice == "Last 1 Year": start_d_calc = today - datetime.timedelta(days=365)
            elif period_choice == "Last 2 Years": start_d_calc = today - datetime.timedelta(days=730)
            else: start_d_calc = datetime.date(today.year, 1, 1)
            
            st.caption(f"{start_d_calc} -> {today}")
            date_range = (start_d_calc, today)

    # --------------------------------------------------------------------------
    # C. EXECUTION & ANALYSIS
    # --------------------------------------------------------------------------
    if st.button("Run Backtest (instant for Vanilla option, up to 15sec for Phoenix option)", type="primary", use_container_width=True):
        start_d, end_d = date_range
        lookback_start = start_d - datetime.timedelta(days=365)
        
        with st.spinner("1/3 Calibrating Historical Volatility..."):
            try:
                # --- CALIBRATION ---
                md_calib = MarketData()
                df_calib = md_calib.get_historical_data(st.session_state.ticker_input, lookback_start.strftime("%Y-%m-%d"), start_d.strftime("%Y-%m-%d"))
                
                sold_vol = sigma 
                if df_calib is not None and not df_calib.empty:
                    log_rets = np.log(df_calib['Close'] / df_calib['Close'].shift(1)).dropna()
                    sold_vol = log_rets.std() * np.sqrt(252)
                    st.toast(f"Calibration Done: Sold Volatility = {sold_vol:.2%}")
                
                # --- BACKTEST DATA ---
                md_bt = MarketData()
                hist_data = md_bt.get_historical_data(st.session_state.ticker_input, start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d"))
                
                if hist_data is None or hist_data.empty:
                    st.error("No data found for the simulation period.")
                else:
                    init_spot = hist_data['Close'].iloc[0]
                    
                    # --- INSTANTIATION ---
                    if p_type == "Phoenix":
                        opt_hedge = PhoenixStructure(
                            S=init_spot, 
                            T=bt_maturity, 
                            r=r, 
                            sigma=sold_vol, 
                            q=q,
                            autocall_barrier=bt_autocall,       
                            protection_barrier=bt_protection,   
                            coupon_barrier=bt_coupon_bar,       
                            coupon_rate=bt_coupon_rate,         
                            obs_frequency=4, 
                            num_simulations=2000
                        )
                    else:
                        strike_bt = init_spot * bt_strike_pct
                        is_call = "Call" in p_type
                        opt_hedge = EuropeanOption(
                            S=init_spot, 
                            K=strike_bt, 
                            T=bt_maturity, 
                            r=r, 
                            sigma=sold_vol, 
                            q=q, 
                            option_type="Call" if is_call else "Put"
                        )

                    # --- ENGINE EXECUTION ---
                    hedging_engine = DeltaHedgingEngine(
                        option=opt_hedge, 
                        market_data=hist_data,
                        risk_free_rate=r, 
                        dividend_yield=q, 
                        volatility=sold_vol,
                        transaction_cost=transaction_cost_pct
                    )

                    res, met = hedging_engine.run_backtest()

                    # --- POST-ANALYSIS & ACCOUNTING ---
                    
                    final_date_str = met['Final Date']
                    duration = met['Duration (Months)']
                    status = met['Status']
                    final_S = met['Final Spot']
                    
                    # Product Status
                    if p_type == "Phoenix":
                        coupons = met['Coupons Paid']
                        st.markdown(f"Status: Product {status} on {final_date_str} "
                                f"(Duration: {duration:.1f} months). "
                                f"The client received {coupons} coupons.")
                    else:
                        strike_val = init_spot * bt_strike_pct
                        is_call = "Call" in p_type
                        
                        if is_call:
                            is_itm = final_S > strike_val
                            condition = "Spot > Strike" if is_itm else "Spot < Strike"
                            payout_msg = f"Payout: {max(final_S - strike_val, 0):.2f}" if is_itm else "No Payout (Expired Worthless)"
                        else: # Put
                            is_itm = final_S < strike_val
                            condition = "Spot < Strike" if is_itm else "Spot > Strike"
                            payout_msg = f"Payout: {max(strike_val - final_S, 0):.2f}" if is_itm else "No Payout (Expired Worthless)"
                        
                        st.markdown(f"Status: Option Matured on {final_date_str} ({duration:.1f} months). "
                                f"Final Spot: {final_S:.2f} vs Strike: {strike_val:.2f} ({condition}). "
                                f"Bank Liability: {payout_msg}")

                    # Financials Breakdown
                    premium = met['Option Premium']
                    costs = met['Total Transaction Costs']
                    
                    if p_type == "Phoenix":
                        total_payout_paid = met['Phoenix Payouts Included']
                        net_pnl = met['Engine P&L'] 
                        # Trading P&L approximation for Phoenix
                        trading_result = net_pnl - premium + total_payout_paid + costs
                    else:
                        # Vanilla: Re-calculate payout (since Engine ignores it)
                        if is_call: vanilla_payout = max(final_S - strike_val, 0.0)
                        else:       vanilla_payout = max(strike_val - final_S, 0.0)
                        
                        total_payout_paid = vanilla_payout
                        net_pnl = met['Engine P&L'] - vanilla_payout
                        # Trading P&L = Engine Result - Premium + Costs
                        trading_result = met['Engine P&L'] - premium + costs

                    # --- KPI DISPLAY ---
                    st.subheader("Performance Breakdown")
                    
                    # Volatility Spread Logic
                    # Spread = Priced (Sold) - Realized
                    vol_spread = met['Pricing Volatility'] - met['Realized Volatility']
                    
                    if p_type == "Phoenix":
                        # Phoenix (Issuer is Long Vega):
                        # We benefit if Realized > Priced (Spread is Negative).
                        # Negative = Good -> We need "inverse" mode.
                        vol_color_mode = "inverse"
                        help_vol = "For Phoenix (Bank Long Vega): Green if Realized Volatility is higher than Pricing Volatility (Negative spread is profitable)."
                    else:
                        # Vanilla Short (Issuer is Short Vega):
                        # We benefit if Realized < Priced (Spread is Positive).
                        # Positive = Good -> We need "normal" mode.
                        vol_color_mode = "normal"
                        help_vol = "For Short Option (Bank Short Vega): Green if Realized Volatility is lower than Pricing Volatility (Positive spread means Theta gain)."

                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("Premium Received", f"+{premium:.2f}", help="Cash received at inception (Short Position)")
                    kpi2.metric("Payout Paid", f"-{total_payout_paid:.2f}", help="Cash paid to client at maturity (or accumulated coupons)")
                    kpi3.metric("Trans. Costs", f"-{costs:.2f}", help="Accumulated transaction costs from rebalancing")
                    
                    # Corrected Vol Spread Metric
                    kpi4.metric("Vol Spread", f"{vol_spread*100:+.2f} pts", 
                                delta=vol_spread, 
                                delta_color=vol_color_mode, # Fixed based on Product Type
                                help=help_vol)

                    st.divider()
                    
                    # NET P&L
                    c_final1, c_final2 = st.columns([1, 1])
                    
                    with c_final1:
                        st.markdown("##### Trading Activity (Gamma P&L)")
                        st.metric("Trading Result", f"{trading_result:+.2f}", 
                                  help="Pure result from dynamic hedging (Buy Low / Sell High). Independent of the premium.")
                    
                    with c_final2:
                        st.markdown("##### Net Profit / Loss")
                        color = "normal" if net_pnl >= 0 else "inverse"
                        st.metric("Total Net P&L", f"{net_pnl:+.2f}", delta=net_pnl, delta_color=color,
                                  help="Net P&L = Premium + Trading Result - Payout - Costs")
                                
                    st.caption("Equation: Net P&L = Premium + Trading P&L - Payouts - Costs")
                    
                    # --- METHODOLOGY EXPANDER (UPDATED) ---
                    with st.expander("Backtesting Methodology & Assumptions"):
                        st.markdown(r"""
                        **Initialization**
                        * **Short Position:** The simulation assumes the bank sells the option at $t_0$.
                        * **Pricing:** The premium is collected based on the **Implied Volatility** (calibrated on the 12-month historical window prior to start).

                        **Trading P&L (Gamma Scalping)**
                        * **Mechanism:** The engine simulates a Daily/Weekly delta-hedging strategy.
                        * **Gamma Bleed:** Being Short Gamma (Short Option), the bank must buy high and sell low to stay Delta Neutral. This generates a trading loss (cost) that is theoretically offset by the Theta (time decay) collected.
                        * **Visualization:** The charts strictly display this rebalancing activity. Entry and Exit cash flows are excluded from the charts to preserve scale.

                        **Net P&L Calculation**
                        * The final result combines two distinct components:
                            * **Fixed Flows:** Premium received - Final Payout (Liability).
                            * **Trading Flows:** The cumulative result of the hedging process - Transaction Costs.
                        """)
                        
                    t1, t2 = st.tabs(["Analysis Dashboard", "Delta History"])
                    with t1:
                        fig_bt = analytics.plot_pnl(hedging_engine)
                        if fig_bt: st.plotly_chart(fig_bt, use_container_width=True, key="chart_pnl_unique")
                        else: st.warning("No data.")
                    with t2:
                        fig_d = go.Figure(go.Scatter(x=res.index, y=res['Delta'], fill='tozeroy', name='Delta', line=dict(color='purple')))
                        fig_d.update_layout(title="Hedge Ratio Evolution", template="plotly_dark", height=400)
                        st.plotly_chart(fig_d, use_container_width=True, key="chart_delta_unique")

            except Exception as e:
                st.error(f"Backtest Error: {str(e)}")