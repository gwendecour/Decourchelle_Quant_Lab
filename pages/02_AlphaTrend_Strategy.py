import streamlit as st
import pandas as pd
import datetime
from src.alpha.universe import get_asset_name
import src.alpha.universe as universe

# Imports de tes modules (Assure-toi que les fichiers existent dans src/alpha/ et src/shared/)
from src.shared.market_data import MarketData
from src.alpha.universe import get_universe, ASSET_POOLS
from src.alpha.backtester import BacktestEngine
import src.alpha.analytics as analytics
from src.shared.ui import render_header

if 'shared_corr_threshold' not in st.session_state:
    st.session_state.shared_corr_threshold = 0.60
if 'slider_top_key' not in st.session_state:
    st.session_state.slider_top_key = st.session_state.shared_corr_threshold
if 'slider_bottom_key' not in st.session_state:
    st.session_state.slider_bottom_key = st.session_state.shared_corr_threshold

CORR_LOOKBACK_WINDOW = 60

# --- FONCTIONS CALLBACK  ---
def update_corr_from_top():
    new_val = st.session_state.slider_top_key
    st.session_state.shared_corr_threshold = new_val
    st.session_state.slider_bottom_key = new_val 
    st.session_state.run_trigger = True 

def update_corr_from_bottom():
    new_val = st.session_state.slider_bottom_key
    st.session_state.shared_corr_threshold = new_val
    st.session_state.slider_top_key = new_val 
    st.session_state.run_trigger = True

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="AlphaStream Strategy", layout="wide", initial_sidebar_state="collapsed")

# --- CSS CUSTOM (Style "Pricing Engine") ---
st.markdown("""
    <style>

        .block-container {
            padding-top: 2rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# CSS pour l'alignement des inputs/sliders
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
        .stButton button {height: 2.2rem; font-size: 0.8rem;}
        .stNumberInput input {height: 2rem;}
    </style>
""", unsafe_allow_html=True)

render_header()

# --- INITIALISATION SESSION STATE ---
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'benchmark_data' not in st.session_state:
    st.session_state.benchmark_data = None
if 'params' not in st.session_state:
    st.session_state.params = {}

def run_backtest_logic():
    # Cette fonction sera appelée dès que le slider bouge
    st.session_state.run_trigger = True

if 'run_trigger' not in st.session_state:
    st.session_state.run_trigger = False

# --- CONTROL BOX (TOP CONTAINER) ---
with st.container(border=True):
    st.markdown("Backtest Configuration")

    col1, col2 = st.columns([5, 1])
    
    with col1:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.caption("Universe Preset")
            selected_preset = st.selectbox("Universe Preset", options=["Standard (12)", "Large (24)", "No Commodities", "Global Macro (Max)"], index=0, label_visibility="collapsed", key="univ_input")
        with c2:
            st.caption("Benchmark to Beat")
            selected_benchmark = st.selectbox("Benchmark", options=["SPY (S&P 500)", "Risk Parity (Multi-Asset)"], index=0, label_visibility="collapsed", key="bench_input")
        with c3:
            st.caption("Start Date")
            start_date = st.date_input("Start", value=datetime.date(2022, 1, 1), label_visibility="collapsed", key="start_date_input")
        with c4:
            freq_map = {"ME": "Monthly", "W-FRI": "Weekly (Fri)", "QE": "Quarterly"}
            st.caption("Rebalance Freq")
            rebal_freq = st.selectbox("Freq", options=list(freq_map.keys()), format_func=lambda x: freq_map.get(x, x), index=0, label_visibility="collapsed")
        c5, c6, c7 = st.columns([1,2,1])
        with c5:
            st.caption("Signal Method")
            selected_signal = st.selectbox("Signal", options=["z_score", "rsi", "distance_ma"], format_func=lambda x: x.replace("_", " ").upper(), index=0, label_visibility="collapsed", key="signal_input")
        with c6:
            st.caption("6. Max Correlation")
            corr_limit = st.slider("Correlation Threshold", min_value=0.5, max_value=0.99, value=st.session_state.shared_corr_threshold, step=0.05, label_visibility="collapsed",key="slider_top_key", on_change=update_corr_from_top, help="Assets with correlation higher than this will be skipped.")
        with c7:
            st.caption("Top N (per Class)")
            top_n = st.number_input("Top N", min_value=1, max_value=5, value=2, label_visibility="collapsed", key="top_n_input")
    
    with col2:
        st.caption("&nbsp;")
        btn_run = st.button("RUN BACKTEST", use_container_width=True, type="primary")
        use_hedge = st.toggle("Beta Hedge", value=True)

# --- EXECUTION LOGIC UNIFIÉE ---

# On lance le backtest si le bouton est cliqué OU si le trigger est activé (par le slider)
if btn_run or st.session_state.run_trigger:
    
    # On reset le trigger immédiatement pour éviter une boucle infinie au prochain rechargement
    st.session_state.run_trigger = False
    current_threshold = st.session_state.shared_corr_threshold
    
    with st.spinner(f"Fetching Market Data & Running Simulation"):
        try:
            # 1. Récupération de l'univers
            universe_dict = get_universe(selected_preset)
            all_tickers = [t for sublist in universe_dict.values() for t in sublist] + ['SPY']
            all_tickers = list(set(all_tickers)) 
            
            # 2. Téléchargement Data
            data_start = start_date - datetime.timedelta(days=365*2) 
            data_end = datetime.date.today()
            
            # Vérification si on a déjà les données en cache pour ne pas retélécharger si on change juste le slider
            # (Optionnel mais accélère grandement l'expérience utilisateur)
            if 'market_data' in st.session_state and st.session_state.market_data is not None:
                market_df = st.session_state.market_data
            else:
                market_df, meta = MarketData.get_clean_multiticker_data(all_tickers, data_start, data_end)
            
            
            if market_df is None or market_df.empty:
                st.error("No data fetched. Please check tickers or internet connection.")
            else:
                # 3. Lancement du Moteur
                engine = BacktestEngine(market_df, universe_dict, initial_capital=100000)
                
                sim_start = pd.Timestamp(start_date)
                
                # --- C'EST ICI LA CORRECTION MAJEURE ---
                results = engine.run(
                    start_date=sim_start,
                    freq=rebal_freq,
                    signal_method=selected_signal,
                    top_n=top_n, # Utilise la variable locale définie plus haut
                    hedge_on=use_hedge,
                    lookback=126, 
                    corr_threshold=current_threshold,
                    corr_lookback=CORR_LOOKBACK_WINDOW
                )
                
                # 4. GESTION DU BENCHMARK
                bench_series = None
                bench_name = ""
            
                if selected_benchmark == "SPY (S&P 500)":
                    bench_name = "SPY"
                    # Logique robuste pour trouver SPY
                    if isinstance(market_df.columns, pd.MultiIndex):
                        if 'Adj Close' in market_df.columns and 'SPY' in market_df['Adj Close'].columns:
                            bench_series = market_df['Adj Close']['SPY']
                        elif 'SPY' in market_df.columns: # Cas rare où multiindex mais SPY au niveau 0
                             bench_series = market_df['SPY']
                    else:
                        bench_series = market_df['SPY'] if 'SPY' in market_df.columns else market_df.iloc[:, 0]
            
                elif selected_benchmark == "Risk Parity (Multi-Asset)":
                    bench_name = "Risk Parity Index"
                    with st.spinner("Calculating Risk Parity Benchmark..."):
                        bench_series = engine.run_risk_parity_benchmark(start_date=sim_start)
            
                # 5. Stockage en Session
                st.session_state.backtest_results = results
                st.session_state.benchmark_data = bench_series
                st.session_state.benchmark_name = bench_name
                st.session_state.market_data = market_df # On stocke pour le cache

                # On sauvegarde TOUS les paramètres importants pour l'affichage
                st.session_state.params = {
                    "univ": selected_preset,
                    "signal": selected_signal,
                    "hedge": use_hedge,
                    "top_n": top_n,
                    "corr_threshold": current_threshold
                }
                
        except Exception as e:
            st.error(f"Critical Error during execution: {e}")
            # st.exception(e) # Décommenter pour voir la stack trace complète

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
        results_dict = st.session_state.backtest_results
        nav_series = results_dict['NAV']
        weights_df = results_dict['Weights']
        bench_data = st.session_state.benchmark_data
        # RÉCUPÉRATION DU NOM DYNAMIQUE
        bench_label = st.session_state.get('benchmark_name', 'Benchmark') 
        
        # A. KPIs Section
        # On passe bench_label pour avoir la bonne colonne dans le tableau
        kpi_df = analytics.calculate_kpis(nav_series, bench_data, benchmark_label=bench_label)
        
        def get_val(metric):
            return kpi_df.loc[kpi_df['Metric'] == metric, 'Strategy'].values[0]
        def get_delta(metric):
            return kpi_df.loc[kpi_df['Metric'] == metric, 'Alpha (Diff)'].values[0]

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        
        m1.metric("Total Return", get_val("Total Return"), get_delta("Total Return"))
        m2.metric("Annual Average", get_val("CAGR"), get_delta("CAGR"))
        m3.metric("Sharpe Ratio", get_val("Sharpe Ratio"), get_delta("Sharpe Ratio"))
        m4.metric("Volatility", get_val("Annual Volatility"), get_delta("Annual Volatility"), delta_color="inverse")
        m5.metric("Max Drawdown", get_val("Max Drawdown"), get_delta("Max Drawdown"), help="The maximum observed loss between two peaks (indicator of downside risk)")
        m6.metric("Calmar Ratio", get_val("Calmar Ratio"), get_delta("Calmar Ratio"), help="(Annual Average Return / Max Drawdown). Higher is better.")

        st.markdown("---")
        
        # B. Graphs Section
        col_g1, col_g2 = st.columns([3, 2])
        
        with col_g1:
            # 1. Equity Curve
            st.subheader(f"Equity Curve vs {bench_label}") # Titre dynamique
            fig_equity = analytics.plot_equity_curve(
                nav_series, 
                bench_data, 
                benchmark_ticker=bench_label # Légende dynamique
            )
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # 2. Rolling Sharpe
            st.subheader("Rolling 6-Month Sharpe Ratio")
            fig_sharpe = analytics.plot_rolling_sharpe(
                nav_series, 
                bench_data, 
                benchmark_label=bench_label # Légende dynamique
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)

        with col_g2:
            # 3. Returns Distribution
            st.subheader("Risk Profile Comparison")
            fig_dist = analytics.plot_returns_distribution(
                nav_series, 
                bench_data, 
                benchmark_label=bench_label # Légende dynamique
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            with st.expander("How to interpret the Risk Profile (Distribution)?"):
                st.markdown("""
                This chart visualizes the "personality" of the returns compared to the Benchmark.
        
                * **The Peak (Mean/Avg):** We want the Green curve's peak to be **shifted to the right** of the Red curve. This indicates higher average returns.
                * **The Width (Volatility):** We want a **narrower, taller** Green curve. A wide, flat curve means erratic performance and uncertainty.
                * **Skewness (Crash Risk):** The S&P 500 often has a "long left tail" (negative skew), meaning it crashes fast. We want your Strategy to have a **higher Skewness** (ideally > -0.5 or positive), indicating fewer sudden catastrophic losses.
                * **Kurtosis (Fat Tails):** Lower is generally better. High kurtosis means "Black Swan" events (extreme outliers) happen more frequently.
                """)
            
            # 4. Underwater Plot
            st.subheader("Drawdown Comparison")
            fig_dd = analytics.plot_drawdown_underwater(nav_series, bench_data, benchmark_label=bench_label)
            st.plotly_chart(fig_dd, use_container_width=True)

            
        
        st.markdown("---")
        st.subheader("Alpha Proof: Can you beat the market?")
    
        bench_series = st.session_state.get('benchmark_data')
        nav_series = st.session_state.backtest_results['NAV']
    
        if bench_series is not None and not bench_series.empty:
            # On garde tes deux colonnes, mais on change le contenu
            col_alpha1, col_alpha2 = st.columns(2)
        
            with col_alpha1:
                # Graphique 1 : Vue Globale (Nuage entier)
                fig_full = analytics.plot_alpha_beta_scatter(nav_series, bench_series, view_mode='full')
                if fig_full:
                    st.plotly_chart(fig_full, use_container_width=True)
                else:
                    st.info("Insufficient data.")
                
            with col_alpha2:
                # Graphique 2 : Vue Zoomée (Le Juge de Paix)
                # On appelle la même fonction mais en mode 'zoomed'
                fig_zoom = analytics.plot_alpha_beta_scatter(nav_series, bench_series, view_mode='zoomed')
                if fig_zoom:
                    st.plotly_chart(fig_zoom, use_container_width=True)
        else:
            st.warning("Please verify that a Benchmark is selected.")

        with st.expander("Understanding CAPM & Alpha Generation"):
            st.markdown("""
            We use the **Capital Asset Pricing Model (CAPM)** to isolate your "True Skill" (Alpha) from "Market Luck" (Beta).
        
            $$R_{Strategy} = \\alpha + \\beta \\times R_{Benchmark} + \\epsilon$$
        
            * **The Intercept ($\\alpha$):** Look at where the red line crosses the vertical axis.
                * **Above 0: Positive Alpha.** You generate excess returns that cannot be explained by market movements. You are adding value.
                * **Below 0: Negative Alpha.** You are underperforming the risk you are taking.
            * **The Slope ($\\beta$):** * **< 1.0:** Defensive profile (less volatile than the market).
                * **> 1.0:** Aggressive profile (amplifies market moves).
            """)

        # C. Detailed Table
        with st.expander("View Detailed Performance Table"):
            st.dataframe(kpi_df, use_container_width=True)

    else:
        st.info("Please configure the strategy in the top panel and click 'RUN BACKTEST'.")

# --- TAB 2: ASSET ALLOCATION ---
with tab_allocation:
    if st.session_state.backtest_results is not None:
        results_dict = st.session_state.backtest_results
        weights_df = results_dict['Weights']
        nav_series = results_dict['NAV']
        market_df = st.session_state.get('market_data', None)
        
        # Récupération de l'univers
        univ_preset = st.session_state.params.get("univ", "Standard (12)")
        universe_dict = get_universe(univ_preset)
        
        # --- SECTION 1 : Vision Historique (Full Width) ---
        st.markdown("### Historical Allocation Evolution")
        fig_alloc = analytics.plot_dynamic_allocation(weights_df, universe_dict)
        st.plotly_chart(fig_alloc, use_container_width=True)
        fig_heatmap = analytics.plot_asset_rotation_heatmap(weights_df, top_n_display=12)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.caption("What drove the monthly returns? (Weighted Contribution by Asset Class)")
        
        if market_df is not None:
            # ON APPELLE LA NOUVELLE FONCTION ICI
            fig_contrib = analytics.plot_monthly_contribution(weights_df, market_df, universe_dict)
            st.plotly_chart(fig_contrib, use_container_width=True)
            
            st.info("""
            **Analyse Tactique :**
            Ce graphique montre l'impact réel de chaque classe sur la performance du mois.
            - **Exemple Momentum :** Si les Actions (Vert) font une grosse barre négative en Janvier, 
            vous devriez voir la barre Verte disparaître ou réduire drastiquement en Février (l'algo a coupé le risque).
            """)
        else:
            st.warning("Market data missing for attribution analysis. Please re-run the backtest.")

        st.markdown("---")

        # --- SECTION 2 : Inspection Ponctuelle (Calendrier) ---
        st.markdown("### Historical Portfolio Inspector")
        
        # Bornes pour le calendrier
        min_date = weights_df.index[0].date()
        max_date = weights_df.index[-1].date()
        
        # Layout Input
        col_input, col_void = st.columns([1, 3])
        with col_input:
            target_date = st.date_input(
                "Select a specific date to inspect:",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )

        # Logique de récupération "Smart" (Gestion des Weekends/Fériés)
        # On cherche l'index le plus proche dans le passé si la date exacte n'existe pas
        try:
            # Conversion en Timestamp pour pandas
            target_ts = pd.Timestamp(target_date)
            
            # get_indexer avec method='pad' trouve l'index précédent le plus proche
            idx_pos = weights_df.index.get_indexer([target_ts], method='pad')[0]
            
            # Si idx_pos est -1, c'est que la date est avant le début du backtest
            if idx_pos == -1:
                st.warning("Date selected is before the start of the simulation.")
            else:
                snapshot_weights = weights_df.iloc[idx_pos]
                snapshot_nav = nav_series.iloc[idx_pos]

                snapshot_combined = snapshot_weights.copy()
                snapshot_combined['NAV'] = snapshot_nav
                display_date = weights_df.index[idx_pos]

            col_b1, col_b2 = st.columns([1, 2])
            
            with col_b1:
                # Le Donut
                fig_pie = analytics.plot_allocation_donut(snapshot_combined)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_b2:
                # Le Tableau
                st.subheader(f"Holdings on {display_date.strftime('%Y-%m-%d')}")
                holdings_df = analytics.get_holdings_table(snapshot_combined, universe_dict)
                
                st.dataframe(
                    holdings_df,
                    column_config={
                        "Asset": st.column_config.TextColumn("Asset"),
                        "Class": st.column_config.TextColumn("Class"),
                        "Weight": st.column_config.ProgressColumn(
                            "Weight",
                            format="%.2f%%",
                            min_value=0,
                            max_value=1,
                        ),
                        "Value ($)": st.column_config.NumberColumn(
                            "Value ($)",
                            format="$%d"
                        )
                    },
                    use_container_width=True,
                    hide_index=True,
                    height=300
                )

        except Exception as e:
            st.error(f"Could not retrieve data for this date: {e}")

    else:
        st.info("Please configure the strategy in the top panel and click 'RUN BACKTEST'.")

# --- TAB 3: SIGNALS & SELECTION ---
with tab_signals:
    # 1. Récupération des données marché brutes
    market_df = st.session_state.get('market_data', None)
    results_dict = st.session_state.backtest_results # On récupère le dict global
    
    if market_df is not None:
        params = st.session_state.get('params', {})
        used_signal = params.get("signal", "z_score")
        lookback_period = 126 
                
        # Calcul des signaux
        with st.spinner(f"Computing historical {used_signal} values..."):
            signals_df = analytics.calculate_all_signals(market_df, used_signal, lookback=lookback_period)
        
        # --- SECTION 1 : La Course (Evolution) ---
        st.markdown("### Signal Evolution (The Race)")
        st.caption("Spotlight View: Select assets to compare. Others remain gray for context.")
        
        all_tickers = list(signals_df.columns)
        default_selection = all_tickers[:5] if len(all_tickers) > 5 else all_tickers
        
        selected_tickers = st.multiselect(
            "Select Assets to Highlight:",
            options=all_tickers,
            default=default_selection,
            format_func=get_asset_name,
            key="signal_race_multiselect"
        )
        
        fig_race = analytics.plot_signal_race(
            signals_df, 
            highlight_assets=selected_tickers,
            signal_method=used_signal
        )
        st.plotly_chart(fig_race, use_container_width=True)
        
        with st.expander("About Signal Evolution"):
            st.markdown("""
            This "Spaghetti Chart" helps visualize momentum trends over time.
            * **Top Lines:** Assets currently leading the market (Strongest Momentum).
            * **Rising Lines:** Assets gaining strength (Potential buy candidates).
            * **Falling Lines:** Assets losing momentum (Potential sell candidates).
            Use this to spot regime changes (e.g., when Energy crosses above Tech).
            """)

        st.markdown("---")
        
        # --- SECTION 2 : Snapshot Inspector ---
        st.markdown("### Ranking Snapshot")
        # --- MISSING DATE PICKER ADDED HERE ---
        min_date = signals_df.index[0].date()
        max_date = signals_df.index[-1].date()
        
        col_date, col_void = st.columns([1, 3])
        with col_date:
            target_date_sig = st.date_input(
                "Select date to inspect ranking:",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="signal_snapshot_date_picker"
            )
        # --------------------------------------

        # --- 1. PRÉPARATION DES DONNÉES ---
        ts = pd.Timestamp(target_date_sig)
        
        # Récupération historique (inchangé)
        chosen_tickers_for_date = []
        selections_history = results_dict.get('Selections', {})
        if selections_history:
            rebalance_dates = pd.DatetimeIndex(selections_history.keys()).sort_values()
            past_dates = rebalance_dates[rebalance_dates <= ts]
            if not past_dates.empty:
                chosen_tickers_for_date = selections_history[past_dates[-1]]

        # Récupération Signaux (inchangé)
        try:
            current_signals = signals_df.loc[:ts].iloc[-1].dropna()
            full_ranking = current_signals.sort_values(ascending=False)
        except Exception:
            full_ranking = pd.Series()

        # --- 2. LOGIQUE D'AFFICHAGE PAR CLASSE (Simplifiée) ---
        if not full_ranking.empty:
            
            st.markdown("---")
            st.subheader("Asset Class Analysis")
            
            thresh = st.slider("Correlation Alert Threshold", min_value=0.5, max_value=0.99, step=0.05, value=st.session_state.shared_corr_threshold, key="slider_bottom_key", on_change=update_corr_from_bottom)
            current_thresh = st.session_state.shared_corr_threshold

            # Définition des onglets basés sur tes classes réelles
            # On utilise universe.get_asset_class pour le filtre
            tabs_classes = st.tabs(["All", "Actions", "Bonds", "Commodities"])
            
            # Dictionnaire de filtres utilisant TA source de vérité
            categories = {
                "All":        lambda t: True,
                "Actions":     lambda t: universe.get_asset_class(t) == "Actions",
                "Bonds":       lambda t: universe.get_asset_class(t) == "Bonds",
                "Commodities": lambda t: universe.get_asset_class(t) == "Commodities"
            }
            
            # BOUCLE D'AFFICHAGE
            for tab, (cat_name, filter_func) in zip(tabs_classes, categories.items()):
                with tab:
                    # A. FILTRAGE
                    # On garde uniquement les tickers qui matchent la catégorie via universe.py
                    filtered_tickers = [t for t in full_ranking.index if filter_func(t)]
                    
                    if not filtered_tickers:
                        st.info(f"No assets found for class: {cat_name} in current selection.")
                        continue
                        
                    subset_ranking = full_ranking[filtered_tickers]
                    subset_ranking_df = subset_ranking.to_frame(name='Score')

                    col_rank, col_matrix = st.columns([2, 1])
                    
                    # B. RANKING BAR CHART
                    with col_rank:
                        st.markdown(f"**{cat_name} Signals**")
                        subset_selected = [t for t in chosen_tickers_for_date if t in filtered_tickers]
                        
                        # Création du DF temporaire pour le bar chart
                        temp_signals_df = pd.DataFrame(index=[ts], data=[subset_ranking.to_dict()])
                        
                        fig_rank = analytics.plot_signal_ranking_bar(
                            temp_signals_df, 
                            target_date=ts, 
                            actual_selections=subset_selected
                        )
                        st.plotly_chart(fig_rank, use_container_width=True, key=f"rank_{cat_name}")
                        with st.expander("Ranking Logic"):
                            st.markdown("""
                            This is the **Selection Brain** of the algorithm for the chosen date.
                            * **Green Bars:** Assets that were **bought** for the portfolio.
                            * **Gray Bars:** Assets that had a good score but were **skipped**.
                            * **Why Skipped?** Usually because they were too highly correlated with a stronger asset already selected (Diversification Filter).
                            """)


                    # C. CORRELATION MATRIX
                    with col_matrix:
                        st.markdown(f"**{cat_name} Correlation**")
                        
                        if len(filtered_tickers) < 2:
                            st.caption("Need at least 2 assets to calculate correlation.")
                        else:
                            market_df = st.session_state.get('market_data')
                            if market_df is not None:
                                fig_corr = analytics.plot_correlation_matrix(
                                    market_df, 
                                    subset_ranking_df, 
                                    threshold=current_thresh,
                                    window=CORR_LOOKBACK_WINDOW
                                )
                                if fig_corr:
                                    st.plotly_chart(fig_corr, use_container_width=True, key=f"corr_{cat_name}")
                        with st.expander("Correlation Filter"):
                            st.markdown("""
                            This matrix protects you from buying duplicates (e.g., buying 3 different Tech ETFs).
                            * **Red Cells:** High correlation (> Threshold). The algorithm will pick the winner and discard the others.
                            * **White Cells:** Low correlation. These assets provide true diversification benefit.
                            """)
        

        # --- SECTION 3 : Deep Dive (Single Asset) ---
        st.markdown("---")
        st.markdown("### Signal vs Price Deep Dive")
        st.caption("Validate the signal effectiveness. Does the price react when the signal spikes?")
        
        # Ensure we have a valid default asset for the deep dive
        # Either the first selected in the multiselect, or the first available ticker
        deep_dive_default = selected_tickers[0] if selected_tickers else all_tickers[0]

        target_asset = st.selectbox(
            "Select Single Asset to Inspect:",
            options=all_tickers,
            index=all_tickers.index(deep_dive_default) if deep_dive_default in all_tickers else 0,
            format_func=get_asset_name,
            key="deep_dive_asset_selector"
        )
        
        if target_asset:
            fig_deep = analytics.plot_signal_vs_price(market_df, signals_df, target_asset)
            st.plotly_chart(fig_deep, use_container_width=True)

            with st.expander("Analysis: Signal vs Price Action"):
                st.markdown("""
                This chart allows you to verify the reactivity of the algorithm:
                * **Blue Line (Score):** Represents the calculated momentum. A positive value indicates an upward trend.
                * **Black Line (Price):** The asset price rebased to 100.
                
                **What to look for:**
                1. **Leading Indicator:** Does the score turn positive *before* a major rally?
                2. **Lag:** Does the score take too long to turn red during a crash?
                3. **Noise:** If the score flickers around 0 frequently, it might generate false signals (whipsaws).
                """)

        else:
            st.info("Select an asset to visualize.")

    else:
        st.warning("No data found. Please go to the Overview tab and click 'RUN BACKTEST' first.")

# --- TAB 4: HEDGING ---
with tab_risk:
    results_dict = st.session_state.backtest_results
    market_df = st.session_state.get('market_data', None)

    if results_dict is not None and market_df is not None:
        # 1. On récupère les poids (nettoyés)
        weights_df = results_dict.get('Weights')
        
        # 2. On récupère le Hedge Ratio
        hedge_series = results_dict.get('Hedge Ratio')

        # --- DÉBUT DE L'AFFICHAGE PROPRE ---
        st.markdown("### Portfolio Protection Analysis")
        
        # Vérification : Est-ce que hedge_series existe et contient des données actives ?
        # On vérifie si la somme (en valeur absolue pour être sûr) est différente de 0
        if hedge_series is not None and not hedge_series.empty and hedge_series.abs().sum() != 0:
            
            # --- SECTION 1: Ratio Evolution ---
            st.markdown("#### Hedge Activation")
            st.caption("This chart shows when the algorithm decided to protect the portfolio and with what intensity.")
            
            # C'est ICI qu'on trace le graphique (une seule fois)
            fig_ratio = analytics.plot_hedge_ratio(hedge_series)
            st.plotly_chart(fig_ratio, use_container_width=True, key="chart_hedge_activation")
            
            col_h1, col_h2 = st.columns([4,1])
            
            with col_h1:
                # --- SECTION 2: Financial Impact ---
                st.markdown("#### Hedge Financial Impact")
                st.caption("Cumulative profit or loss from the short position.")
                
                fig_impact = analytics.plot_hedge_impact(hedge_series, market_df)
                st.plotly_chart(fig_impact, use_container_width=True, key="chart_hedge_impact")
                
            with col_h2:
                # --- SECTION 3: Stats & Explanation ---
                # On utilise .abs() pour les stats aussi car tes valeurs sont négatives
                avg_hedge = hedge_series.abs().mean()
                max_hedge = hedge_series.abs().max()
                
                st.markdown("#### Protection Statistics")
                st.metric("Average Hedge Level", f"{avg_hedge:.1%}")
                st.metric("Peak Protection", f"{max_hedge:.1%}")
            with st.expander("Hedge Logic"):
                st.markdown(f"""
                The strategy shorts the benchmark to neutralize market Beta.
                - **Up Market:** The hedge usually costs money (insurance premium).
                - **Down Market:** The hedge should offset losses from long positions.
                """)
        else:
            st.warning("Hedging was disabled or inactive for this backtest run.")
            
    else:
        st.info("Please run a backtest with 'Apply Beta Hedge' enabled to see this analysis.")