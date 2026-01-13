import streamlit as st
import numpy as np
import pandas as pd # Ajout nécessaire pour le backtest
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from scipy.stats import norm

# --- IMPORTS MODULES (Tes propres modules) ---
from src.shared.market_data import MarketData
from src.derivatives.instruments import InstrumentFactory
from src.derivatives.pricing_model import EuropeanOption
from src.derivatives.structured_products import PhoenixStructure
from src.derivatives.backtester import DeltaHedgingEngine
from src.shared.ui import render_header

# --- CONFIGURATION PAGE & CSS ---
st.set_page_config(
    page_title="Pricing Engine",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        div[data-testid="column"] {padding: 0rem;}
        .stButton button {height: 2.2rem; padding-top: 0; padding-bottom: 0; font-size: 0.8rem;}
        h3 {font-size: 1.2rem !important; margin-bottom: 0.5rem !important;}
        .stNumberInput, .stSlider {margin-bottom: -15px !important;}
        /* Style pour les boutons de scénarios compacts */
        div[data-testid="stHorizontalBlock"] button {width: 100%;}
    </style>
""", unsafe_allow_html=True)

render_header()

# ==============================================================================
# 1. STATE & CALLBACKS (LOGIQUE)
# ==============================================================================

# Initialisation des états par défaut
defaults = {
    'market_spot': None, 'market_rate': None, 'market_vol': None, 'market_div': None,
    'custom_spot': 100.0, 'custom_vol': 0.20, 'custom_rate': 0.04, 'custom_div': 0.00,
    'ticker_ref': "GLE.PA",
    'product_type': "European Call",
    'strike_pct': 100.0,
    'barrier_pct': 0.60,
    'coupon_barrier_pct': 0.70,
    'autocall_pct': 1.00
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- Callback: Chargement Market Data ---
def update_market_data():
    ticker = st.session_state.ticker_input # Lié au selectbox
    try:
        spot = MarketData.get_spot(ticker)
        rate = MarketData.get_risk_free_rate()
        vol = MarketData.get_volatility(ticker, "1y")
        div = MarketData.get_dividend_yield(ticker)

        # Mise à jour des données de référence
        st.session_state.market_spot = spot
        st.session_state.market_rate = rate
        st.session_state.market_vol = vol
        st.session_state.market_div = div

        # Mise à jour des données "Custom" (Sliders)
        st.session_state.custom_spot = float(spot)
        st.session_state.custom_vol = float(vol)
        st.session_state.custom_rate = float(rate)
        st.session_state.custom_div = float(div)
        
        # Reset Strike ATM
        st.session_state.strike_pct = 100.0
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# --- Callback: Scénarios ---
def set_scenario(scenario_type):
    # On travaille sur la base du Spot actuel (Custom)
    spot = st.session_state.custom_spot
    p_type = st.session_state.get('product_type', 'European Call')
    
    if scenario_type == "ATM":
        st.session_state.strike_pct = 100.0
        # Remet la vol par défaut ou celle du marché
        if st.session_state.market_vol:
            st.session_state.custom_vol = st.session_state.market_vol
        else:
            st.session_state.custom_vol = 0.20

    elif scenario_type == "Deep ITM":
        # In The Money : Strike < Spot pour Call, Strike > Spot pour Put
        if "Call" in p_type:
            st.session_state.strike_pct = 80.0  # Strike = 80% du Spot
        elif "Put" in p_type:
            st.session_state.strike_pct = 120.0 # Strike = 120% du Spot
        else:
            st.session_state.strike_pct = 80.0 
        
    elif scenario_type == "Deep OTM":
        # Out The Money : Strike > Spot pour Call, Strike < Spot pour Put
        if "Call" in p_type:
            st.session_state.strike_pct = 120.0 
        elif "Put" in p_type:
            st.session_state.strike_pct = 80.0  
        else:
            st.session_state.strike_pct = 120.0
            
        # Souvent la vol est plus faible ou différente OTM
        st.session_state.custom_vol = max(st.session_state.custom_vol * 0.9, 0.05)

    elif scenario_type == "Reset":
        if st.session_state.market_spot:
            # Retour aux données marché
            st.session_state.custom_spot = st.session_state.market_spot
            st.session_state.custom_vol = st.session_state.market_vol
            st.session_state.custom_rate = st.session_state.market_rate
            st.session_state.custom_div = st.session_state.market_div
        else:
            # Retour hard-coded
            st.session_state.custom_spot = 100.0
            st.session_state.custom_vol = 0.20
            st.session_state.custom_rate = 0.04
            st.session_state.custom_div = 0.00
        st.session_state.strike_pct = 100.0


# ==============================================================================
# 2. HEADER & MARKET CONNECTION
# ==============================================================================
TICKERS = {
    "GLE.PA": "Societe Generale", "TTE.PA": "TotalEnergies", 
    "MC.PA": "LVMH", "AIR.PA": "Airbus", "SAN.PA": "Sanofi",
    "RMS.PA": "Hermes Intl"
}

with st.container(border=True):
    c1, c2, c3 = st.columns([2, 2, 5])
    with c1:
        st.selectbox("Ticker", list(TICKERS.keys()), 
                     format_func=lambda x: f"{x} - {TICKERS[x]}", 
                     key="ticker_input", label_visibility="collapsed")
    with c2:
        # Bouton avec CALLBACK
        st.button("📡 Load Market Data", on_click=update_market_data, use_container_width=True)
    with c3:
        vals = [st.session_state[k] for k in ['market_spot', 'market_vol', 'market_rate', 'market_div']]
        labels = [f"**{v:.2f}**" if v is not None else "--" for v in vals]
        if vals[1]: labels[1] = f"**{vals[1]:.2%}**"
        if vals[2]: labels[2] = f"**{vals[2]:.2%}**"
        if vals[3]: labels[3] = f"**{vals[3]:.2%}**"
        
        st.markdown(f"Spot: {labels[0]} | Vol: {labels[1]} | RiskFree: {labels[2]} | Div: {labels[3]}", 
                    help="Reference Market Data")

# ==============================================================================
# 3. MAIN UI (TABS)
# ==============================================================================
tab_pricing, tab_greeks, tab_backtest = st.tabs(["Pricing & Payoff", "Greeks Analysis", "Delta Hedging"])

# --- VARIABLES GLOBALES ---
S = st.session_state.custom_spot
sigma = st.session_state.custom_vol
r = st.session_state.custom_rate
q = st.session_state.custom_div

# --- TAB 1: PRICING (Inchangé) ---
with tab_pricing:
    layout_col1, layout_col2, layout_col3 = st.columns([1, 1, 2], gap="medium")

    # --- COLUMN 1: MARKET PARAMETERS ---
    with layout_col1:
        st.markdown("### 1. Market")
        
        # Spot
        c_s1, c_s2 = st.columns([2, 1])
        with c_s1: st.slider("Spot", 10.0, 1000.0, key="custom_spot", label_visibility="collapsed")
        with c_s2: st.number_input("S", value=st.session_state.custom_spot, key="num_spot_display", disabled=True, label_visibility="collapsed")
        st.caption("Spot Price ($)")

        # Vol
        c_v1, c_v2 = st.columns([2, 1])
        with c_v1: st.slider("Vol", 0.05, 1.00, step=0.01, key="custom_vol", label_visibility="collapsed")
        with c_v2: st.write(f"{st.session_state.custom_vol:.1%}")
        st.caption("Volatility (σ)")

        # Rate
        c_r1, c_r2 = st.columns([2, 1])
        with c_r1: st.slider("Rate", 0.00, 0.15, step=0.001, key="custom_rate", label_visibility="collapsed")
        with c_r2: st.write(f"{st.session_state.custom_rate:.1%}")
        st.caption("Risk-Free Rate (r)")

        # Div
        c_d1, c_d2 = st.columns([2, 1])
        with c_d1: st.slider("Div", 0.00, 0.10, step=0.001, key="custom_div", label_visibility="collapsed")
        with c_d2: st.write(f"{st.session_state.custom_div:.1%}")
        st.caption("Dividend Yield (q)")
        
        # Mise à jour des variables locales
        S, sigma, r, q = st.session_state.custom_spot, st.session_state.custom_vol, st.session_state.custom_rate, st.session_state.custom_div

    # --- COLUMN 2: PRODUCT PARAMETERS & SCENARIOS ---
    with layout_col2:
        st.markdown("### 2. Product")
        
        product_type = st.selectbox("Type", ["Phoenix Autocall", "European Call", "European Put"], key="product_type", label_visibility="collapsed")
        st.caption("Product Type")
        
        maturity_years = st.number_input("Maturity (Years)", value=1.0, step=0.5)
        
        if product_type == "Phoenix Autocall":
            # Phoenix Params
            c_cp1, c_cp2 = st.columns([2, 1])
            with c_cp1: coup_sl = st.slider("Cpn", 0.0, 0.20, 0.08, 0.005, label_visibility="collapsed")
            with c_cp2: st.markdown(f"**{coup_sl:.1%}** Cpn")
            coupon_rate = coup_sl
            
            autocall_pct = st.slider("Autocall (%)", 80, 120, 100, step=5) / 100
            coupon_barrier_pct = st.slider("Cpn Barr (%)", 40, 90, 70, step=5) / 100
            barrier_pct = st.slider("Prot Barr (%)", 30, 80, 60, step=5) / 100
            n_sims = st.selectbox("Sims", [2000, 5000, 10000], index=0)
        else:
            # Option Params
            st.slider("Moneyness (%)", 50.0, 150.0, step=1.0, key="strike_pct")
            
            strike_price = S * (st.session_state.strike_pct / 100.0)
            st.caption(f"Strike Price: **{strike_price:.2f} €** ({st.session_state.strike_pct:.0f}%)")
            n_sims = 0

        st.write("") 
        st.write("") 
        
        # --- SCENARIOS ---
        st.markdown("###### Scenarios")
        s1, s2, s3, s4 = st.columns(4, gap="small")
        with s1: st.button("ATM", on_click=set_scenario, args=("ATM",), use_container_width=True, help="At The Money")
        with s2: st.button("Deep ITM", on_click=set_scenario, args=("Deep ITM",), use_container_width=True, help="Deep In The Money")
        with s3: st.button("Deep OTM", on_click=set_scenario, args=("Deep OTM",), use_container_width=True, help="Deep Out of The Money")
        with s4: st.button("Reset", on_click=set_scenario, args=("Reset",), use_container_width=True)


    # --- COLUMN 3: ANALYSIS ---
    with layout_col3:
        st.markdown("### 3. Analysis")
        
        if product_type == "Phoenix Autocall":
            phoenix = PhoenixStructure(
                S=S, T=maturity_years, r=r, sigma=sigma, q=q,
                autocall_barrier=autocall_pct, protection_barrier=barrier_pct,
                coupon_barrier=coupon_barrier_pct, coupon_rate=coupon_rate,
                obs_frequency=12, num_simulations=n_sims
            )
            price = phoenix.price()
            fig_payoff = phoenix.plot_payoff(spot_range=[S*0.5, S*1.5])
            
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Price", f"{price:.2f} €")
            kpi2.metric("% Nominal", f"{(price/S)*100:.2f} %")
            kpi3.metric("Barrier", f"{S*barrier_pct:.2f} €")
            
        else:
            # Option Vanilla
            strike_price = S * (st.session_state.strike_pct / 100.0)
            option = EuropeanOption(
                S=S, K=strike_price, T=maturity_years, r=r, sigma=sigma, q=q,
                option_type=product_type.split(" ")[1]
            )
            price = option.price()
            fig_payoff = option.plot_payoff(spot_range=[S*0.6, S*1.4])
            
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Price", f"{price:.2f} €")
            kpi2.metric("% Nominal", f"{(price/S)*100:.2f} %")
            kpi3.metric("Moneyness", f"{(S/strike_price)*100:.1f}%")

        fig_payoff.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_payoff, use_container_width=True)


# ==============================================================================
# TAB 2: GREEKS ANALYSIS (Refondu)
# ==============================================================================
# ==============================================================================
# TAB 2: GREEKS ANALYSIS (Corrigé)
# ==============================================================================
with tab_greeks:
    st.subheader("📊 Greeks & Risk Analysis")
    
    # Paramètres de l'option pour les Grecs (Basé sur la selection courante)
    if product_type == "Phoenix Autocall":
        st.info("⚠️ Analysis based on a standard European Call with current market parameters.")
        g_strike = S
        g_type = "Call"
    else:
        g_strike = S * (st.session_state.strike_pct / 100.0)
        g_type = product_type.split(" ")[1] # "Call" ou "Put"

    # --- 1. INSTANTANEOUS GREEKS ---
    # CORRECTION ICI : Utilisation d'arguments nommés (keyword arguments)
    opt_greeks = EuropeanOption(
        S=S, 
        K=g_strike, 
        T=maturity_years, 
        r=r, 
        sigma=sigma, 
        q=q, 
        option_type=g_type
    )
    
    # Récupération des Grecs via la méthode existante de la classe
    try:
        g_vals = opt_greeks.greeks() 
    except Exception as e:
        st.error(f"Erreur calcul Grecs: {e}")
        g_vals = {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}

    # Affichage Métriques
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Delta (Δ)", f"{g_vals.get('delta', 0):.2f}")
    c2.metric("Gamma (Γ)", f"{g_vals.get('gamma', 0):.4f}")
    c3.metric("Vega (ν)", f"{g_vals.get('vega', 0):.2f}")
    c4.metric("Theta (Θ)", f"{g_vals.get('theta', 0):.3f}")

    st.divider()

    # --- 2. VISUALIZATION (Surface & Heatmap) ---
    st.subheader("Risk Surface & Heatmap")
    
    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    with col_ctrl1:
        viz_greek = st.selectbox("Greek", ["Delta", "Gamma", "Vega"])
        viz_mode = st.radio("View", ["3D Surface", "2D Heatmap"])
        
    with col_ctrl2:
        c_r1, c_r2 = st.columns(2)
        with c_r1: spot_range_pct = st.slider("Spot Range (+/- %)", 10, 50, 30, 5)
        with c_r2: vol_range_pct = st.slider("Vol Range (+/- %)", 10, 100, 50, 10)

    # Calcul des Grilles
    spots = np.linspace(S * (1 - spot_range_pct/100), S * (1 + spot_range_pct/100), 25)
    vols = np.linspace(sigma * (1 - vol_range_pct/100), sigma * (1 + vol_range_pct/100), 25)
    S_mesh, V_mesh = np.meshgrid(spots, vols)

    # Calcul Z (Grec choisi) sur la grille (Formule simplifiée BS pour affichage rapide)
    T_val = maturity_years
    d1_mesh = (np.log(S_mesh / g_strike) + (r - q + 0.5 * V_mesh**2) * T_val) / (V_mesh * np.sqrt(T_val))
    
    if viz_greek == "Delta":
        if g_type == "Call" or g_type == "call":
            Z_mesh = np.exp(-q * T_val) * norm.cdf(d1_mesh)
        else:
            Z_mesh = -np.exp(-q * T_val) * norm.cdf(-d1_mesh)
    elif viz_greek == "Gamma":
        Z_mesh = np.exp(-q * T_val) * norm.pdf(d1_mesh) / (S_mesh * V_mesh * np.sqrt(T_val))
    elif viz_greek == "Vega":
        Z_mesh = S_mesh * np.exp(-q * T_val) * norm.pdf(d1_mesh) * np.sqrt(T_val) / 100

    # Plotting
    if viz_mode == "3D Surface":
        fig = go.Figure(data=[go.Surface(z=Z_mesh, x=S_mesh, y=V_mesh, colorscale='Viridis')])
        fig.update_layout(title=f'{viz_greek} Surface', autosize=True,
                          scene=dict(xaxis_title='Spot', yaxis_title='Vol', zaxis_title=viz_greek),
                          height=500, margin=dict(l=0, r=0, t=30, b=0))
    else:
        fig = go.Figure(data=go.Heatmap(z=Z_mesh, x=spots, y=vols, colorscale='RdBu_r'))
        fig.update_layout(title=f'{viz_greek} Heatmap', xaxis_title='Spot Price', yaxis_title='Volatility', height=500)
        
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    
    # --- 3. P&L SIMULATOR ---
    st.subheader("P&L Attribution Simulator")
    pl1, pl2, pl3 = st.columns(3)
    with pl1: dS = st.number_input("Spot Move (%)", -20.0, 20.0, 1.0, step=0.5) / 100
    with pl2: dVol = st.number_input("Vol Move (pts)", -10.0, 10.0, 0.0, step=0.5) / 100
    with pl3: dT = st.number_input("Days Passed", 0, 30, 1) / 365.0

    # Pricing t0 vs t1
    price_t0 = opt_greeks.price()
    
    # CORRECTION ICI AUSSI : Arguments nommés pour opt_t1
    opt_t1 = EuropeanOption(
        S=S * (1+dS), 
        K=g_strike, 
        T=max(0.01, maturity_years - dT), 
        r=r, 
        sigma=sigma + dVol, 
        q=q, 
        option_type=g_type
    )
    price_t1 = opt_t1.price()
    
    pnl = price_t1 - price_t0
    
    # Approx via Grecs
    pnl_delta = g_vals['delta'] * (S * dS)
    pnl_gamma = 0.5 * g_vals['gamma'] * (S * dS)**2
    pnl_vega = g_vals['vega'] * (dVol * 100)
    pnl_theta = g_vals['theta'] * (dT * 365)
    
    # Graphique Waterfall
    fig_pnl = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = ["relative", "relative", "relative", "relative", "total"],
        x = ["Delta", "Gamma", "Vega", "Theta", "Total P&L"],
        textposition = "outside",
        text = [f"{v:.2f}" for v in [pnl_delta, pnl_gamma, pnl_vega, pnl_theta, pnl]],
        y = [pnl_delta, pnl_gamma, pnl_vega, pnl_theta, pnl],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    fig_pnl.update_layout(title = "Estimated P&L Attribution", height=350)
    st.plotly_chart(fig_pnl, use_container_width=True)


# ==============================================================================
# TAB 3: DELTA-HEDGING BACKTEST (Corrigé & Amélioré)
# ==============================================================================
with tab_backtest:
    st.subheader("Dynamic Hedging Simulation")
    
    with st.container(border=True):
        bt_col1, bt_col2, bt_col3 = st.columns(3)
        with bt_col1:
            bt_option_type = st.selectbox("Instrument", ["European Call"])
            bt_strike_pct = st.number_input("Strike % Init Spot", value=1.0, step=0.05)
        with bt_col2:
            rebal_freq = st.selectbox("Rebalancing", ["Daily", "Weekly"], index=0)
            transaction_cost_pct = st.number_input("Trans. Cost (%)", value=0.10, step=0.05) / 100
        with bt_col3:
            today = datetime.date.today()
            start_date_default = today - datetime.timedelta(days=365)
            date_range = st.date_input("Period", value=(start_date_default, today))

    if st.button("⚙️ Run Backtest", type="primary", use_container_width=True):
        if len(date_range) != 2:
            st.error("Select start and end date.")
        else:
            start_d, end_d = date_range
            with st.spinner("Simulating..."):
                try:
                    # CORRECTION ICI: Instanciation sans arguments
                    md_bt = MarketData() 
                    # Récupération des données via l'instance
                    hist_data = md_bt.get_historical_data(
                        st.session_state.ticker_input, # On passe le ticker ici
                        start_d.strftime("%Y-%m-%d"), 
                        end_d.strftime("%Y-%m-%d")
                    )
                    
                    if hist_data.empty:
                        st.error("No data found.")
                    else:
                        initial_spot_bt = hist_data['Close'].iloc[0]
                        strike_bt = initial_spot_bt * bt_strike_pct
                        maturity_bt = (end_d - start_d).days / 365.0

                        option_to_hedge = InstrumentFactory.create_instrument(
                            "European Call", strike=strike_bt, maturity=maturity_bt, is_call=True
                        )

                        hedging_engine = DeltaHedgingEngine(
                            option=option_to_hedge,
                            market_data=hist_data,
                            risk_free_rate=r,       
                            dividend_yield=q,      
                            volatility=sigma,       
                            transaction_cost=transaction_cost_pct,
                            rebalancing_freq=rebal_freq.lower()
                        )

                        results_df, metrics = hedging_engine.run_backtest()

                        # Metrics
                        met_c1, met_c2, met_c3, met_c4 = st.columns(4)
                        met_c1.metric("Total P&L", f"{metrics['Total P&L']:.2f} €", delta=metrics['Total P&L'])
                        met_c2.metric("Hedge Error", f"{metrics['Hedge Error Std']:.2f}")
                        met_c3.metric("Costs", f"{metrics['Total Transaction Costs']:.2f} €")
                        met_c4.metric("Avg Delta", f"{results_df['Delta'].abs().mean():.2f}")

                        # Graphiques
                        tab_res1, tab_res2 = st.tabs(["Performance P&L", "Greeks & Dynamics"])
                        
                        with tab_res1:
                            fig_bt = make_subplots(specs=[[{"secondary_y": True}]])
                            fig_bt.add_trace(go.Scatter(x=results_df.index, y=results_df['Cumulative P&L'], name='Hedged P&L', line=dict(color='green')), secondary_y=False)
                            fig_bt.add_trace(go.Scatter(x=results_df.index, y=results_df['Spot'], name='Spot', line=dict(color='grey', dash='dot'), opacity=0.5), secondary_y=True)
                            fig_bt.update_layout(height=400, title_text="Hedged P&L vs Spot")
                            st.plotly_chart(fig_bt, use_container_width=True)
                        
                        with tab_res2:
                            fig_d = go.Figure()
                            fig_d.add_trace(go.Scatter(x=results_df.index, y=results_df['Delta'], name='Delta', fill='tozeroy'))
                            fig_d.update_layout(height=400, title="Delta Evolution")
                            st.plotly_chart(fig_d, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Backtest error: {str(e)}")
