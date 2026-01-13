import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime

# Import de tes modules customs
from src.shared.market_data import MarketData
from src.derivatives.instruments import InstrumentFactory
from src.derivatives.pricing_model import BlackScholesPricer, MonteCarloEngine
from src.derivatives.structured_products import PhoenixStructure
from src.derivatives.backtester import DeltaHedgingEngine

# --- IMPORT DU HEADER ---
from src.shared.ui import render_header

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Pricing & Hedging Engine",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed" # On cache la sidebar native
)

# Affiche le menu du haut
render_header()

st.title("💎 Derivatives Pricing & Hedging Engine")
st.markdown("Analyze structured products, calculate Greeks, and backtest dynamic hedging strategies.")

# ==============================================================================
# 1. MARKET DATA PANEL (TOP HORIZONTAL BAR)
# ==============================================================================
st.subheader("⚡ Market Environment & Asset Specs")

# On crée une bordure visuelle pour ce panneau de contrôle
with st.container(border=True):
    # On utilise 6 colonnes pour aligner les inputs horizontalement
    md_col1, md_col2, md_col3, md_col4, md_col5, md_col6 = st.columns(6)

    with md_col1:
        st.markdown("**Underlying Asset**")
        ticker = st.text_input("Yahoo Ticker", value="GLE.PA")
        
        col_fetch1, col_fetch2 = st.columns(2)
        with col_fetch1:
             if st.button("Get Price", use_container_width=True):
                try:
                    md = MarketData(ticker)
                    last_price = md.get_last_price()
                    st.session_state['fetched_spot'] = last_price
                    st.success(f"Spot: {last_price:.2f}")
                except Exception as e:
                    st.error(str(e))
        with col_fetch2:
            if st.button("Est. Vol", use_container_width=True):
                try:
                     md = MarketData(ticker)
                     # On prend une fenêtre d'un an par défaut pour l'estimation rapide
                     start_date_vol = datetime.date.today() - datetime.timedelta(days=365)
                     hist_data = md.get_historical_data(start_date_vol.strftime("%Y-%m-%d"), datetime.date.today().strftime("%Y-%m-%d"))
                     estimated_vol = md.calculate_historical_volatility(hist_data)
                     st.session_state['fetched_vol'] = estimated_vol
                     st.success(f"Vol: {estimated_vol:.2%}")
                except Exception as e:
                    st.error("Need Data")

    # Valeurs par défaut si non fetchées
    default_spot = st.session_state.get('fetched_spot', 100.0)
    default_vol = st.session_state.get('fetched_vol', 0.20)

    with md_col2:
        st.markdown("**Spot Price ($S_0$)**")
        spot_price = st.number_input("Current Price", value=default_spot, format="%.2f")

    with md_col3:
        st.markdown("**Volatility ($\sigma$)**")
        volatility = st.number_input("Annual Vol", value=default_vol, step=0.01, format="%.2f")

    with md_col4:
        st.markdown("**Risk-Free Rate ($r$)**")
        risk_free_rate = st.number_input("Annual Rate", value=0.03, step=0.005, format="%.3f")

    with md_col5:
        st.markdown("**Dividend Yield ($q$)**")
        dividend_yield = st.number_input("Annual Yield", value=0.01, step=0.005, format="%.3f")
        
    with md_col6:
         st.markdown("**Parameters**")
         maturity_years = st.number_input("Maturity (Years)", value=1.0, step=0.25)
         drift = risk_free_rate - dividend_yield # Calcul automatique du drift

# Séparateur visuel
st.divider()

# ==============================================================================
# 2. TABS FOR DIFFERENT ANALYSES
# ==============================================================================
tab1, tab2, tab3 = st.tabs(["📐 Structuring & Pricing", "📊 Greeks Analysis", "⚙️ Delta-Hedging Backtest"])

# --- TAB 1: STRUCTURING & PRICING (Phoenix) ---
with tab1:
    st.subheader("Product Specifications: Phoenix Autocallable")
    
    # Inputs spécifiques au produit, alignés horizontalement
    with st.container(border=True):
        prod_col1, prod_col2, prod_col3, prod_col4 = st.columns(4)
        with prod_col1:
            coupon_rate = st.number_input("Annual Coupon Rate", value=0.08, step=0.01, format="%.2f")
        with prod_col2:
            barrier_level_pct = st.number_input("Barrier Level (%)", value=0.70, step=0.05, format="%.2f")
        with prod_col3:
            autocall_level_pct = st.number_input("Autocall Level (%)", value=1.00, step=0.05, format="%.2f", help="Initial Autocall level, stays constant in this simplified version.")
        with prod_col4:
            n_sims_pricing = st.selectbox("MC Simulations (Pricing)", options=[1000, 5000, 10000, 50000], index=2)

    # Conversion des pourcentages en niveaux absolus
    barrier_level = spot_price * barrier_level_pct
    autocall_level = spot_price * autocall_level_pct

    st.write("") # Spacer
    
    # Bouton d'action principal
    if st.button("🚀 Launch Phoenix Pricing (Monte Carlo)", type="primary", use_container_width=True):
        with st.spinner("Running Monte Carlo simulations..."):
            # 1. Instanciation du produit Structured
            # NOTE: Pour simplifier ici, on met le strike initial = strike autocall = spot actuel
            strike_price_struct = spot_price
            
            phoenix = PhoenixStructure(
                maturity=maturity_years,
                strike=strike_price_struct,
                barrier_level=barrier_level,
                autocall_level=autocall_level,
                coupon_rate=coupon_rate,
                observation_frequency='monthly' # Fixé pour l'instant
            )
            
            # 2. Moteur Monte Carlo
            mc_engine = MonteCarloEngine(
                spot=spot_price,
                vol=volatility,
                rate=risk_free_rate,
                div=dividend_yield,
                n_sims=n_sims_pricing,
                n_steps=int(maturity_years * 252) # steps journaliers
            )
            
            # 3. Pricing
            price, std_error = phoenix.price(mc_engine)
            
            # 4. Affichage Résultats
            st.markdown("### Pricing Results")
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                 st.metric("Fair Value (Price)", f"{price:.2f} €")
            with res_col2:
                 st.metric("Price in % of Notional", f"{(price/spot_price)*100:.2f} %")
            with res_col3:
                 st.metric("MC Standard Error", f"{std_error:.4f}")

            st.info(f"Note: Product notionally based on Spot Price ({spot_price}). Barrier at {barrier_level:.2f}, Autocall at {autocall_level:.2f}.")
            st.success("Pricing complete based on risk-neutral measure evaluation of future cashflows.")


# --- TAB 2: GREEKS ANALYSIS (Vanilla Focus) ---
with tab2:
    st.subheader("Vanilla Option Greeks Surface")
    st.markdown("Analyze sensitivity of a standard European Call option.")

    # Inputs pour les ranges de plot
    with st.container(border=True):
        g_col1, g_col2, g_col3 = st.columns(3)
        with g_col1:
            greek_strike = st.number_input("Option Strike K", value=spot_price, format="%.2f")
        with g_col2:
            spot_range_pct = st.slider("Spot Range (+/- %)", min_value=10, max_value=50, value=30, step=5)
        with g_col3:
            vol_range_pct = st.slider("Vol Range (+/- %)", min_value=10, max_value=100, value=50, step=10)

    # Préparation des données pour la surface
    spots = np.linspace(spot_price * (1 - spot_range_pct/100), spot_price * (1 + spot_range_pct/100), 30)
    vols = np.linspace(volatility * (1 - vol_range_pct/100), volatility * (1 + vol_range_pct/100), 30)
    S_mesh, V_mesh = np.meshgrid(spots, vols)

    # Calcul du Delta sur la grille (Call Européen)
    # d1 = (ln(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    # Delta Call = exp(-qT) * N(d1)
    from scipy.stats import norm
    T = maturity_years
    r = risk_free_rate
    q = dividend_yield
    K = greek_strike

    d1 = (np.log(S_mesh / K) + (r - q + 0.5 * V_mesh**2) * T) / (V_mesh * np.sqrt(T))
    delta_mesh = np.exp(-q * T) * norm.cdf(d1)

    # Plotly 3D Surface
    fig = go.Figure(data=[go.Surface(z=delta_mesh, x=S_mesh, y=V_mesh, colorscale='Viridis')])
    fig.update_layout(title='Call Delta Surface', autosize=True,
                      scene=dict(
                          xaxis_title='Spot Price',
                          yaxis_title='Volatility',
                          zaxis_title='Delta'
                      ),
                      height=600)
    st.plotly_chart(fig, use_container_width=True)


# --- TAB 3: DELTA-HEDGING BACKTEST ---
with tab3:
    st.subheader("Dynamic Hedging Simulation (Call Option)")
    st.markdown("Backtest a Delta-Hedging strategy on a European Call option over the historical period.")

    # Paramètres du backtest, alignés horizontalement
    with st.container(border=True):
        bt_col1, bt_col2, bt_col3 = st.columns(3)
        with bt_col1:
            bt_option_type = st.selectbox("Instrument to Hedge", ["European Call"])
            bt_strike_pct = st.number_input("Strike % of initial Spot", value=1.0, step=0.05)
        with bt_col2:
            rebal_freq = st.selectbox("Rebalancing Frequency", options=["Daily", "Weekly"], index=0)
            transaction_cost_pct = st.number_input("Transaction Cost (%)", value=0.10, step=0.05, format="%.2f") / 100
        with bt_col3:
             # Sélection des dates pour le backtest historique
            today = datetime.date.today()
            start_date_default = today - datetime.timedelta(days=365)
            date_range = st.date_input("Historical Period", value=(start_date_default, today))


    st.write("") # Spacer

    if st.button("⚙️ Run Hedging Backtest", type="primary", use_container_width=True):
        if len(date_range) != 2:
            st.error("Please select a start and end date.")
        else:
            start_d, end_d = date_range
            
            with st.spinner("Fetching historical data and running simulation..."):
                # 1. Récupération des données historiques réelles
                md_bt = MarketData(ticker)
                hist_data = md_bt.get_historical_data(start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d"))
                
                if hist_data.empty:
                    st.error("No historical data found for this period.")
                else:
                    # 2. Setup de l'option à hedger
                    initial_spot_bt = hist_data['Close'].iloc[0]
                    strike_bt = initial_spot_bt * bt_strike_pct
                    
                    # Durée réelle du backtest en années
                    days_in_period = (end_d - start_d).days
                    maturity_bt = days_in_period / 365.0

                    # On utilise l'instrument Factory (pour l'instant juste Call)
                    option_to_hedge = InstrumentFactory.create_instrument(
                        "European Call",
                        strike=strike_bt,
                        maturity=maturity_bt,
                        is_call=True
                    )

                    # NOTE IMPORTANTE SUR LA VOLATILITÉ DANS UN BACKTEST :
                    # Idéalement, il faudrait utiliser la volatilité implicite historique jour par jour.
                    # Pour simplifier ici, on utilise la volatilité constante définie en haut de page.
                    vol_for_hedging = volatility 

                    # 3. Moteur de Backtest
                    hedging_engine = DeltaHedgingEngine(
                        option=option_to_hedge,
                        market_data=hist_data,
                        risk_free_rate=risk_free_rate,
                        dividend_yield=dividend_yield,
                        volatility=vol_for_hedging,
                        transaction_cost=transaction_cost_pct,
                        rebalancing_freq=rebal_freq.lower()
                    )

                    # 4. Exécution
                    results_df, metrics = hedging_engine.run_backtest()

                    # 5. Affichage des résultats
                    st.markdown("### Backtest Performance")
                    
                    met_c1, met_c2, met_c3, met_c4 = st.columns(4)
                    met_c1.metric("Total P&L", f"{metrics['Total P&L']:.2f} €", help="Final P&L of the hedged portfolio at maturity.")
                    met_c2.metric("Hedge Error (Std Dev)", f"{metrics['Hedge Error Std']:.2f}")
                    met_c3.metric("Total Transaction Costs", f"{metrics['Total Transaction Costs']:.2f} €")
                    met_c4.metric("Avg Abs Delta", f"{results_df['Delta'].abs().mean():.2f}")

                    #Graphiques
                    tab_g1, tab_g2 = st.tabs(["P&L Attribution", "Delta & Spot Dynamics"])
                    
                    with tab_g1:
                        # P&L cumulé du portefeuille hedgé vs Option non hedgée
                        fig_pnl = go.Figure()
                        fig_pnl.add_trace(go.Scatter(x=results_df.index, y=results_df['Cumulative P&L'], name='Hedged Portfolio P&L', line=dict(color='green')))
                        # Pour comparaison : valeur théorique de l'option seule (approximatif ici sans recalculer le prix chaque jour)
                        # On montre juste le P&L du hedge pour l'instant pour rester simple.
                        fig_pnl.update_layout(title="Cumulative Hedged P&L over Time", xaxis_title="Date", yaxis_title="P&L (€)")
                        st.plotly_chart(fig_pnl, use_container_width=True)

                    with tab_g2:
                        # Graphique double axe : Spot vs Delta
                        from plotly.subplots import make_subplots
                        fig_delta = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_delta.add_trace(go.Scatter(x=results_df.index, y=results_df['Spot'], name='Underlying Spot'), secondary_y=False)
                        fig_delta.add_trace(go.Scatter(x=results_df.index, y=results_df['Delta'], name='Option Delta', line=dict(color='orange')), secondary_y=True)
                        fig_delta.update_layout(title="Spot Price vs Hedge Ratio (Delta)", xaxis_title="Date")
                        fig_delta.update_yaxes(title_text="Spot Price", secondary_y=False)
                        fig_delta.update_yaxes(title_text="Delta", secondary_y=True)
                        st.plotly_chart(fig_delta, use_container_width=True)