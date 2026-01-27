import streamlit as st
import pandas as pd
import datetime

# Imports de tes modules (Assure-toi que les fichiers existent dans src/alpha/ et src/shared/)
from src.shared.market_data import MarketData
from src.alpha.universe import get_universe, ASSET_POOLS
from src.alpha.backtester import BacktestEngine
import src.alpha.analytics as analytics

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="AlphaStream Strategy", layout="wide", initial_sidebar_state="collapsed")

# --- CSS CUSTOM (Style "Pricing Engine") ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
        }
        /* Style compact pour les boutons et inputs */
        .stButton button {
            height: 2.6rem; 
            font-size: 0.9rem; 
            font-weight: bold;
        }
        .stSelectbox div[data-baseweb="select"] > div {
            min-height: 2.2rem;
        }
        /* Titres des onglets un peu plus gros */
        button[data-baseweb="tab"] {
            font-size: 1rem;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
def render_header():
    st.markdown("### AlphaTrend: Multi-Asset Momentum Strategy")

render_header()

# --- INITIALISATION SESSION STATE ---
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'benchmark_data' not in st.session_state:
    st.session_state.benchmark_data = None
if 'params' not in st.session_state:
    st.session_state.params = {}

# --- CONTROL BOX (TOP CONTAINER) ---
with st.container(border=True):
    c1, c2, c3, c4, c5, c6 = st.columns([0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
    
    with c1:
        st.caption("Investment Universe")
        selected_preset = st.selectbox(
            "Universe Preset",
            options=["Standard (12)", "Large (24)", "No Commodities"],
            index=0,
            label_visibility="collapsed",
            key="univ_input"
        )

    with c2:
        st.caption("Benchmark to Beat") # <--- NOUVEAU
        selected_benchmark = st.selectbox(
            "Benchmark",
            options=["SPY (S&P 500)", "Risk Parity (Multi-Asset)"],
            index=0,
            label_visibility="collapsed",
            key="bench_input"
        )

    with c3:
        st.caption("Signal Method")
        selected_signal = st.selectbox(
            "Signal",
            options=["z_score", "rsi", "distance_ma"],
            format_func=lambda x: x.replace("_", " ").upper(),
            index=0,
            label_visibility="collapsed",
            key="signal_input"
        )

    with c4:
        st.caption("Settings")
        # On met Top N et Freq sur la même "case" visuelle ou l'un sous l'autre ?
        # Ici on utilise des colonnes imbriquées pour gagner de la place
        sc1, sc2 = st.columns(2)
        with sc1:
            top_n = st.number_input("Top N", min_value=1, max_value=5, value=2, label_visibility="collapsed", key="top_n_input")
        with sc2:
            rebal_freq = st.selectbox("Freq", options=["ME", "W-FRI", "QE"], index=0, label_visibility="collapsed")

    with c5:
        st.caption("Risk Mgmt")
        # Une checkbox stylisée ou simple
        use_hedge = st.toggle("Beta Hedge", value=True)
        # Date de départ
        start_date = st.date_input("Start", value=datetime.date(2022, 1, 1), label_visibility="collapsed")

    with c6:
        # Le bouton prend toute la hauteur/largeur dispo dans sa colonne
        st.caption("&nbsp;") # Spacer pour aligner le bouton en bas
        btn_run = st.button("RUN BACKTEST", use_container_width=True, type="primary")

# --- EXECUTION LOGIC ---
if btn_run:
    with st.spinner("Fetching Market Data & Running Simulation..."):
        try:
            # 1. Récupération de l'univers
            universe_dict = get_universe(selected_preset)
            # On aplatit la liste pour le téléchargement (+ SPY pour benchmark)
            all_tickers = [t for sublist in universe_dict.values() for t in sublist] + ['SPY']
            all_tickers = list(set(all_tickers)) # Dédoublonnage
            
            # 2. Téléchargement Data (On prend large pour le lookback)
            data_start = start_date - datetime.timedelta(days=365*2) 
            data_end = datetime.date.today()
            
            market_df, meta = MarketData.get_clean_multiticker_data(all_tickers, data_start, data_end)
            
            
            if market_df is None or market_df.empty:
                st.error("No data fetched. Please check tickers or internet connection.")
            else:
                # 3. Lancement du Moteur
                engine = BacktestEngine(market_df, universe_dict, initial_capital=100000)
                
                # Conversion des dates pour pandas
                sim_start = pd.Timestamp(start_date)
                top_n = st.session_state.top_n_input
                
                results = engine.run(
                    start_date=sim_start,
                    freq=rebal_freq,
                    signal_method=selected_signal,
                    top_n=top_n,
                    hedge_on=use_hedge,
                    # On fixe les paramètres avancés en dur ou on pourrait les ajouter à l'interface
                    lookback=126, 
                    corr_threshold=0.6
                )
                
                # 4. GESTION DU BENCHMARK SÉLECTIONNÉ
                bench_series = None
                bench_name = ""
            
                if selected_benchmark == "SPY (S&P 500)":
                    bench_name = "SPY"
                    if 'SPY' in market_df.columns:
                        bench_series = market_df['SPY']
                    elif 'Adj Close' in market_df.columns and 'SPY' in market_df['Adj Close'].columns:
                        bench_series = market_df['Adj Close']['SPY']
                    else:
                        bench_series = market_df.iloc[:, 0] # Fallback
            
                elif selected_benchmark == "Risk Parity (Multi-Asset)":
                    bench_name = "Risk Parity Index"
                    with st.spinner("Calculating Risk Parity Benchmark..."):
                        # On lance le calcul du Risk Parity
                        bench_series = engine.run_risk_parity_benchmark(start_date=sim_start)
            
                # 5. Stockage en Session
                st.session_state.backtest_results = results
                st.session_state.benchmark_data = bench_series
                st.session_state.benchmark_name = bench_name # On stocke le nom pour l'affichage

                st.session_state.params = {
                    "univ": selected_preset,
                    "signal": selected_signal,
                    "hedge": use_hedge
                }
                
                st.success("Backtest Completed Successfully!")
                
        except Exception as e:
            st.error(f"Critical Error during execution: {e}")
            # Pour le debug, on peut afficher la trace : st.exception(e)

# --- TABS DISPLAY ---
tab_overview, tab_allocation, tab_signals, tab_risk = st.tabs([
    " Overview & Performance", 
    " Asset Allocation", 
    " Signals & Selection", 
    " Risk & Hedge"
])

# --- TAB 1: OVERVIEW ---
with tab_overview:
    if st.session_state.backtest_results is not None:
        results = st.session_state.backtest_results
        bench_data = st.session_state.benchmark_data
        # RÉCUPÉRATION DU NOM DYNAMIQUE
        bench_label = st.session_state.get('benchmark_name', 'Benchmark') 
        
        # A. KPIs Section
        # On passe bench_label pour avoir la bonne colonne dans le tableau
        kpi_df = analytics.calculate_kpis(results['NAV'], bench_data, benchmark_label=bench_label)
        
        def get_val(metric):
            return kpi_df.loc[kpi_df['Metric'] == metric, 'Strategy'].values[0]
        def get_delta(metric):
            return kpi_df.loc[kpi_df['Metric'] == metric, 'Alpha (Diff)'].values[0]

        # Ligne de Métriques
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Return", get_val("Total Return"), get_delta("Total Return"))
        m2.metric("CAGR", get_val("CAGR"), get_delta("CAGR"))
        m3.metric("Sharpe Ratio", get_val("Sharpe Ratio"), get_delta("Sharpe Ratio"))
        m4.metric("Volatility", get_val("Annual Volatility"), get_delta("Annual Volatility"), delta_color="inverse")
        m5.metric("Max Drawdown", get_val("Max Drawdown"), get_delta("Max Drawdown"), delta_color="inverse")
        
        st.markdown("---")
        
        # B. Graphs Section
        col_g1, col_g2 = st.columns([2, 1])
        
        with col_g1:
            # 1. Equity Curve
            st.subheader(f"Equity Curve vs {bench_label}") # Titre dynamique
            fig_equity = analytics.plot_equity_curve(
                results['NAV'], 
                bench_data, 
                benchmark_ticker=bench_label # Légende dynamique
            )
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # 2. Rolling Sharpe
            st.subheader("Rolling 6-Month Sharpe Ratio")
            fig_sharpe = analytics.plot_rolling_sharpe(
                results['NAV'], 
                bench_data, 
                benchmark_label=bench_label # Légende dynamique
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)

        with col_g2:
            # 3. Returns Distribution
            st.subheader("Risk Profile Comparison")
            fig_dist = analytics.plot_returns_distribution(
                results['NAV'], 
                bench_data, 
                benchmark_label=bench_label # Légende dynamique
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # 4. Underwater Plot (Reste le même)
            st.subheader("Strategy Drawdown")
            fig_dd = analytics.plot_drawdown_underwater(results['NAV'])
            st.plotly_chart(fig_dd, use_container_width=True)
            
        # C. Detailed Table
        with st.expander("View Detailed Performance Table"):
            st.dataframe(kpi_df, use_container_width=True)

    else:
        st.info("Please configure the strategy in the top panel and click 'RUN BACKTEST'.")