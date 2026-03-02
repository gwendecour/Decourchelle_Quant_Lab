import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from src.shared.ui import render_header
from src.shared.universe import get_universe, get_asset_name
from src.covariance.data_pipeline import MarketDataProvider, DataCorruptor
from src.covariance.imputers import (
    ForwardFillImputer, KNNImputerModel, MICEImputerModel, 
    SVDImputerModel, EMImputerModel
)
from src.covariance.evaluator import CovarianceMatrixEstimator, ErrorMetrics, PortfolioOptimizer
from src.covariance.analytics import QuantAnalytics

st.set_page_config(page_title="Investment Strategies", page_icon="assets/logo.png", layout="wide")
render_header()

st.title("Robust Covariance & Missing Data Imputation")
with st.expander("Pourquoi ce projet ?"):
    st.markdown("""
    ### The Liquidity Illusion in Portfolio Engineering
    In quantitative finance and systematic trading, missing data is unavoidable (non-synchronous trading, isolated holidays, micro-structure holes). 
    The standard industry practice is to routinely **Forward-Fill** the last known price.
    However, this artificially forces the realized volatility towards zero and silently destroys the global correlation structure.
    
    **The result?** The Minimum Variance Portfolio Optimizer will be actively misled into over-allocating to these "falsely stable" assets, generating a massive *Ex-Post Risk* that wasn't priced in. 
    This dashboard compares advanced mathematical imputations (KNN, Matrix Reconstruction, Expectation-Maximization) against the naïve industry baseline.
    """)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Data Universe & Corruption")

# Mutualized Asset Universe selection
preset_options = ["Standard (12)", "Large (24)", "No Commodities", "Global Macro (Max)"]
pool_choice = st.sidebar.selectbox("Select Asset Universe", preset_options, index=0)
universe_dict = get_universe(pool_choice)

# Flatten the dictionary into a list of tickers
tickers = []
for cat, t_list in universe_dict.items():
    tickers.extend(t_list)

st.sidebar.markdown("---")
monte_carlo_runs = st.sidebar.slider("Monte-Carlo Runs (Stability Analysis)", 2, 20, 5, step=1, help="Impacts specifically the Boxplot computation time in Tab 2.")

st.markdown("### Illiquidity Simulation Parameters")
col_p1, col_p2, col_p3 = st.columns([1, 1, 1])

with col_p1:
    target_illiquid = st.multiselect("Target Illiquid Assets", tickers, default=tickers[:2] if len(tickers)>1 else tickers, format_func=get_asset_name, help="Assets that will suffer from randomly missing prices.")
with col_p2:
    missing_rate = st.slider("Missing Data Probability (%)", 0, 80, 20, step=5) / 100.0
with col_p3:
    duration_str = st.selectbox(
        "Backtest Duration", 
        ["1 Year", "3 Years", "5 Years"],
        index=0,
        help="Longer durations increase the robustness of the historical covariance but significantly increase computation time."
    )
    
duration_map = {"1 Year": 1, "3 Years": 3, "5 Years": 5}
years = duration_map[duration_str]

if years > 1:
    st.warning(f"⚠️ {years} Years backtest computation may take significantly longer.")

end_date = pd.to_datetime("2024-01-01")
start_date = end_date - pd.DateOffset(years=years)

# Force purely random missing days (1 day by 1 day)
corruption_tag = "MCAR" 

st.markdown("---")

# --- CORE DATA PROCESSING (HEAVILY CACHED) ---
@st.cache_data(show_spinner=False)
def fetch_ground_truth(tickers_list, start, end):
    provider = MarketDataProvider(tickers_list, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    return provider.fetch_data()

@st.cache_resource(show_spinner=False)
def get_models():
    return {
        "Forward-Fill (Baseline)": ForwardFillImputer(),
        "K-Nearest Neighbors": KNNImputerModel(n_neighbors=5),
        "MICE (Bayesian Ridge)": MICEImputerModel(max_iter=10),
        "Matrix Completion (SVD)": SVDImputerModel(rank=max(1, len(tickers)//2), max_iter=50),
        "Expectation-Maximization": EMImputerModel(max_iter=50)
    }

@st.cache_data(show_spinner="Phase 1: Running Core Imputation Engines...")
def compute_base_metrics(tickers_list, start, end, target_tickers, miss_rate, m_method):
    gt_df = fetch_ground_truth(tickers_list, start, end)
    models = get_models()
    estimator = CovarianceMatrixEstimator(ann_factor=252)
    
    corruptor = DataCorruptor(miss_rate, method=m_method, target_tickers=target_tickers)
    corrupted_df_visual = corruptor.corrupt(gt_df, random_state=42)
    
    true_cov = estimator.estimate(gt_df)
    
    imputed_prices_df = {}
    imputed_covs = {}
    frob_distances = {}
    
    for name, model in models.items():
        filled_df = model.fit_transform(corrupted_df_visual.copy())
        imputed_prices_df[name] = filled_df
        cov_est = estimator.estimate(filled_df)
        imputed_covs[name] = cov_est
        frob_distances[name] = ErrorMetrics.frobenius_norm(cov_est, true_cov)
        
    vis_col_name = target_tickers[0] if (target_tickers and target_tickers[0] in gt_df.columns) else gt_df.columns[0]
    
    return gt_df, corrupted_df_visual, true_cov, imputed_prices_df, imputed_covs, frob_distances, vis_col_name

@st.cache_data(show_spinner="Phase 2A: Computing Phase Transition Margin...")
def compute_phase_metrics(tickers_list, start, end, target_tickers, miss_rate, m_method):
    gt_df = fetch_ground_truth(tickers_list, start, end)
    models = get_models()
    estimator = CovarianceMatrixEstimator(ann_factor=252)
    
    # Fast fetch from cache
    _, _, true_cov, _, _, _, _ = compute_base_metrics(tickers_list, start, end, target_tickers, miss_rate, m_method)
    
    phase_rates = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    phase_results = []
    for rate in phase_rates:
        t_corr = DataCorruptor(rate, method=m_method, target_tickers=target_tickers)
        t_df = t_corr.corrupt(gt_df, random_state=42)
        row = {"Missing Rate": rate}
        for name, model in models.items():
            t_filled = model.fit_transform(t_df.copy())
            t_cov = estimator.estimate(t_filled)
            row[name] = ErrorMetrics.frobenius_norm(t_cov, true_cov)
        phase_results.append(row)
        
    return pd.DataFrame(phase_results)

@st.cache_data(show_spinner="Phase 2B: Running Heavy Monte-Carlo Stability...")
def compute_stability_metrics(tickers_list, start, end, target_tickers, miss_rate, m_method, mc_runs):
    gt_df = fetch_ground_truth(tickers_list, start, end)
    models = get_models()
    estimator = CovarianceMatrixEstimator(ann_factor=252)
    
    # Fast fetch from cache
    _, _, true_cov, _, _, _, _ = compute_base_metrics(tickers_list, start, end, target_tickers, miss_rate, m_method)
    
    mc_corruptor = DataCorruptor(miss_rate, method=m_method, target_tickers=target_tickers)
    mc_results = {name: [] for name in models.keys()}
    for mc_run in range(mc_runs):
        rnd_df = mc_corruptor.corrupt(gt_df, random_state=np.random.randint(1, 100000))
        for name, model in models.items():
            rnd_filled = model.fit_transform(rnd_df.copy())
            rnd_cov = estimator.estimate(rnd_filled)
            mc_results[name].append(ErrorMetrics.frobenius_norm(rnd_cov, true_cov))
            
    return pd.DataFrame(mc_results)

@st.cache_data(show_spinner="Phase 3: Solving Optimal Portfolio Allocations...")
def compute_portfolio_metrics(tickers_list, start, end, target_tickers, miss_rate, m_method, port_strategy):
    models = get_models()
    optimizer = PortfolioOptimizer()
    
    # Fast fetch from cache
    _, _, true_cov, _, imputed_covs, _, _ = compute_base_metrics(tickers_list, start, end, target_tickers, miss_rate, m_method)
    
    def optimize(cov, strat):
        if "Risk Parity" in strat:
            return optimizer.risk_parity_portfolio(cov)
        elif "Maximum Diversification" in strat:
            return optimizer.max_diversification_portfolio(cov)
        else:
            return optimizer.min_variance_portfolio(cov)
            
    true_weights = optimize(true_cov, port_strategy)
    
    imputed_weights = {}
    risk_metrics = {}
    
    for name in models.keys():
        cov_est = imputed_covs[name]
        weights_est = optimize(cov_est, port_strategy)
        imputed_weights[name] = weights_est
        risk_metrics[name] = optimizer.evaluate_risk(weights_est, true_cov, cov_est)
        
    return true_weights, imputed_weights, risk_metrics

if tickers and start_date < end_date:
    # --- DATA COMPUTATION ---
    try:
        gt_df, corrupted_df, true_cov, imputed_prices, imputed_covs, frobenius_dist, vis_col_name = compute_base_metrics(
            tuple(tickers), start_date, end_date, tuple(target_illiquid), missing_rate, corruption_tag
        )
        all_model_names = list(imputed_prices.keys())
        imputed_series = {k: v[vis_col_name] for k, v in imputed_prices.items()}
    except Exception as e:
        import traceback
        st.error(f"Error executing base data pipeline: {e}")
        st.code(traceback.format_exc())
        st.stop()
        
    # --- TABS DISPLAY ---
    st.markdown("---")
    tab_micro, tab_systemic, tab_convergence, tab_business = st.tabs([
        " 1. Microscopic Truth", 
        " 2. Systemic Structure", 
        " 3. Algorithmic Convergence", 
        " 4. Business Impact"
    ])

with tab_micro:
    st.markdown(f"Focusing on **{vis_col_name}**. See how the models try to reconstruct missing data and its temporal impact.")
        
    st.markdown("Select Imputation Models to display:")
    selected_ts = st.multiselect(
        "Models Config", 
        all_model_names, 
        default=["Forward-Fill (Baseline)", "Matrix Completion (SVD)"], 
        key='ts_multi'
    )
        
    if selected_ts:
        fig_ts = QuantAnalytics.plot_time_series_reconstruction(
            vis_col_name, gt_df[vis_col_name],
            corrupted_df[vis_col_name],
            {k: imputed_series[k] for k in selected_ts}
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        with st.expander("Understanding the Time Series Models"):
            st.write("""
            **What to observe:** We have artificially masked segments of the data to simulate deep liquidity holes. Below is how the algorithms try to fill them:
            * **Forward-Fill (FFill):** Extends the last known price forward, creating flat horizontal steps (forcing a sequence of 0% returns).
            * **K-Nearest Neighbors (KNN):** Searches historical data for the 'k' days where the rest of the market behaved most similarly to the blackout day, and averages their returns to infer the missing price.
            * **Matrix Completion (SVD):** Decomposes the market into systemic risk factors (eigenvectors). It reconstructs the missing price by assuming the asset's exposure to these macro factors remains constant during the blackout period.
            * **MICE (Bayesian Ridge):** Iteratively predicts the missing values using Bayesian linear regression on all other observable assets, adding a layer of probabilistic uncertainty to prevent overconfidence.
            * **Expectation-Maximization (EM):** An iterative process that guesses the missing prices, uses them to calculate a full covariance matrix, and then uses that matrix to refine its guesses again until the system converges to a logical state.
            """)
            
        col_vol, col_acf = st.columns([1, 1])
        with col_vol:
            fig_roll, stats_df = QuantAnalytics.plot_rolling_volatility(
                gt_df[vis_col_name],
                {k: imputed_prices[k][vis_col_name] for k in selected_ts},
                selected_ts, window=30
            )
            st.plotly_chart(fig_roll, use_container_width=True)
            
            st.dataframe(
                stats_df,
                column_config={
                    "Mean Vol (bps)": st.column_config.NumberColumn(format="%.4f bps"),
                    "Noise Std (bps)": st.column_config.NumberColumn(format="%.4f bps")
                },  
                use_container_width=True, hide_index=True
            )
            
            with st.expander("The Volatility Paradox"):
                st.write("""
                **Wait, why is Forward-Fill volatility sometimes HIGHER than Ground Truth?**
                When a price remains artificially flat for several days, it accumulates a large, unrecorded discrepancy with reality. On the first day it trades again, it 'catches up' to the true market price in a single, massive, violent jump.
                
                Because statistical variance squares the returns ($R^2$), a single explosive 2.0% gap-up geometrically dominates four smooth 0.5% daily returns. 
                As seen in the legend's Noise Standard Deviation ($\sigma$), this mathematical dynamic artificially inflates the sample variance, rendering the rolling volatility highly erratic, noisy, and utterly unreliable for risk management.
                """)
                
        with col_acf:
            fig_acf = QuantAnalytics.plot_autocorrelation(
                gt_df[vis_col_name],
                {k: imputed_prices[k][vis_col_name] for k in selected_ts},
                selected_ts, lags=15
            )
            st.plotly_chart(fig_acf, use_container_width=True)
            with st.expander("The Memory Effect (ACF)"):
                st.write("""
                **What to observe:** The Efficient Market Hypothesis states that past returns should not predict future returns (ACF $\\approx$ 0).
                However, artificial imputation methods inject mathematical memory structural artifacts into the series:
                * **FFill** forces chains of exact 0% returns, creating a lingering structural autocorrelation that violates white noise assumptions.
                * Other methods can also introduce slight structural patterns due to linear interpolation approximations.
                
                *You can use the toggle button on the chart to switch to **Absolute |ACF|** to strictly visualize the magnitude of the deviation away from 0 for all methods.*
                """)
        
    st.markdown("---")
    st.markdown("### Return Distribution Distortion (Macro Effect)")
    single_model = st.selectbox("Select Model for Return Distribution", all_model_names, index=0)
    fig_kde = QuantAnalytics.plot_returns_distribution(
        gt_df[vis_col_name], imputed_prices[single_model][vis_col_name], single_model
    )
    st.plotly_chart(fig_kde, use_container_width=True)
    with st.expander("Return Distribution Distortion"):
        st.write("""
        **What to observe:** This curve plots the density of daily log-returns. A healthy asset typically forms a bell-shaped curve (normal-ish distribution).
        
        **N.B.** Notice the massive, unnatural central spike precisely at **0.0%** for Forward-Fill. This corresponds exactly to the artificially flat days. 
        Furthermore, to computationally compensate for these 0s, the large 'catch-up' jumps push the outer edges of the distribution further away, artificially inflating the **Fat Tails** (kurtosis) of the asset. A robust imputation method restores the natural bell shape.
        """)
            
    # Smart Background Pre-computation
    with st.spinner("Advanced Analytics pre-computing in background..."):
        compute_phase_metrics(tuple(tickers), start_date, end_date, tuple(target_illiquid), missing_rate, corruption_tag)
        compute_stability_metrics(tuple(tickers), start_date, end_date, tuple(target_illiquid), missing_rate, corruption_tag, monte_carlo_runs)
        compute_portfolio_metrics(tuple(tickers), start_date, end_date, tuple(target_illiquid), missing_rate, corruption_tag, "Minimum Variance")
with tab_systemic:
    st.header("The Systemic Shock: Cross-Asset Relationships")
    st.markdown("Missing data breaks the co-movement matrix of the market. See how the structure is destroyed.")
        
    col_select, _ = st.columns([1, 1])
    with col_select:
        single_sys_model = st.selectbox("In-Depth View Model", all_model_names, index=0, key="sys_model")
            
    st.markdown(f"### Covariance Error Heatmap")
    st.markdown(f"Distance Score (Frobenius Norm): **{frobenius_dist[single_sys_model]:.4f}**")
    fig_hm = QuantAnalytics.plot_error_heatmap(true_cov, imputed_covs[single_sys_model], single_sys_model)
    st.plotly_chart(fig_hm, use_container_width=True)
    with st.expander("Covariance Error Heatmap"):
        st.write("""
        **What to observe:** This heatmap shows the literal difference between the True Covariance Matrix and the Estimated one. 
        Deep red or blue spots indicate that the algorithm has completely misunderstood the relationship between two specific assets. Forward-Fill typically destroys covariance because artificial 0-returns on one asset don't correlate with the moves of another.
        """)
    st.markdown("---")
    st.markdown("### Multi-Model Scatter Cross-Correlation")
    selected_scatter = st.multiselect(
        "Select Models to compare against Perfect Correlation", 
        all_model_names, 
        default=["Forward-Fill (Baseline)", "Expectation-Maximization"],
        key='scat_multi'
    )
    if selected_scatter:
        fig_scat2 = QuantAnalytics.plot_cross_correlation_scatter(
            true_cov, 
            {k: imputed_covs[k] for k in selected_scatter}, 
            selected_scatter
        )
        st.plotly_chart(fig_scat2, use_container_width=True)
        with st.expander("Scatter Cross-Correlation & The Epps Effect"):
            st.write("""
            **What to observe:** Perfect imputation aligns all dots on the black dashed `y=x` diagonal. 
            
            Notice how most imputation methods actually 'pull' or 'dampen' the pairwise correlations towards 0. This creates a flatter, horizontal cloud with a linear trendline Slope < 1. 
            This statistical phenomenon is known in quantitative finance as the **Epps Effect**. Forward-Fill is especially guilty of this correlation destruction, because the flat 0% flat-lines of one asset will mathematically fail to correlate with the active, daily movements of another.
            """)
                
    # Smart background
    with st.spinner("🚀 Advanced Analytics pre-computing in background..."):
        compute_phase_metrics(tuple(tickers), start_date, end_date, tuple(target_illiquid), missing_rate, corruption_tag)
        compute_stability_metrics(tuple(tickers), start_date, end_date, tuple(target_illiquid), missing_rate, corruption_tag, monte_carlo_runs)
        compute_portfolio_metrics(tuple(tickers), start_date, end_date, tuple(target_illiquid), missing_rate, corruption_tag, "Minimum Variance")

with tab_convergence:
    try:
        degradation_df = compute_phase_metrics(tuple(tickers), start_date, end_date, tuple(target_illiquid), missing_rate, corruption_tag)
        stability_df = compute_stability_metrics(tuple(tickers), start_date, end_date, tuple(target_illiquid), missing_rate, corruption_tag, monte_carlo_runs)
    except Exception as e:
        st.error(f"Error executing Monte Carlo: {e}")
        st.stop()
            
    st.header("Algorithmic Convergence Math (Phase Transition)")
    st.markdown("Choose algorithms to compare their breaking points and statistical variance under Monte-Carlo.")
        
    selected_metrics = st.multiselect(
        "Compare Models", 
        all_model_names, 
        default=["Forward-Fill (Baseline)", "MICE (Bayesian Ridge)", "Matrix Completion (SVD)", "Expectation-Maximization"],
        key='metrics_multi'
    )
        
    if selected_metrics:
        col_phase, col_box = st.columns([1, 1])
        with col_phase:
            fig_phase = QuantAnalytics.plot_phase_transition(degradation_df, selected_metrics)
            st.plotly_chart(fig_phase, use_container_width=True)
            with st.expander("Phase Transition"):
                st.write(""" 
            The **Phase Transition** graph shows how rapidly the algorithms fail as the missing rate increases. MICE and Expectation-Maximization are the most robust algorithms, maintaining a relatively linear and contained slope of degradation up to high missing rates, whereas simpler methods might degrade much faster.   
            """)
        with col_box:
            fig_box = QuantAnalytics.plot_stability_boxplots(stability_df, selected_metrics)
            st.plotly_chart(fig_box, use_container_width=True)
            with st.expander("Phase Transition & Stability"):
                st.write("""
            The **Stability Boxplots** shows the variance of this error over many Monte Carlo randomized scenarios. A reliable algorithm isn't just accurate on average, it must be consistently stable under random systemic shocks (thin box).
            """)

    # Smart background
    with st.spinner("Resolving Optimal Portfolio structures in background..."):
        compute_portfolio_metrics(tuple(tickers), start_date, end_date, tuple(target_illiquid), missing_rate, corruption_tag, "Minimum Variance")

with tab_business:
    st.markdown("### Portfolio Specifications")
    portfolio_strategy = st.selectbox(
        "Select Portfolio Allocation Strategy:", 
        ["Minimum Variance", "Risk Parity (Equal Risk Contribution)", "Maximum Diversification"],
        index=0,
        help="Determines how the weights are assigned based on the covariance matrix."
    )
    st.markdown("---")
        
    try:
        true_weights, imputed_weights, risk_metrics = compute_portfolio_metrics(tuple(tickers), start_date, end_date, tuple(target_illiquid), missing_rate, corruption_tag, portfolio_strategy)
    except Exception as e:
        st.error(f"Error executing Portfolio solver: {e}")
        st.stop()
            
    st.header(f"Logical Allocation Error: {portfolio_strategy.split('(')[0].strip()}")
    st.markdown("How a matrix error translates into a massive misallocation of capital and an unexpected explosion of risk.")
        
    selected_portfolios = st.multiselect(
        "Select Imputed Portfolios to compare against True Allocation", 
        all_model_names, 
        default=["Forward-Fill (Baseline)", "Expectation-Maximization"],
        key='port_multi'
    )
        
    if selected_portfolios:
        # Top Row: Turnover Friction & Volatility Illusion
        col_turn, col_scat = st.columns([1, 1])
        
        with col_turn:
            fig_turn = QuantAnalytics.plot_turnover_penalty(
                true_weights, {k: imputed_weights[k] for k in selected_portfolios}, selected_portfolios
            )
            st.plotly_chart(fig_turn, use_container_width=True)
            with st.expander("Turnover Friction Penalty"):
                st.write("""
                **What to observe:** This bar chart shows the total % of your capital that was allocated to the wrong assets. 
                A 20% turnover penalty means that 20% of your portfolio is invested differently than the optimal Ground Truth portfolio, exposing you to hidden risks and dead-weight trading transition costs.
                
                **N.B:** If you observe 0% turnover for all models, it means the corruption level is too low to force the algorithms into different allocations. Try increasing the **Missing Rate** in the sidebar.
                """)
                
        with col_scat:
            true_port_vol = np.sqrt(np.dot(true_weights.values.T, np.dot(true_cov.values, true_weights.values)))
            fig_illusion = QuantAnalytics.plot_volatility_illusion(risk_metrics, true_port_vol, selected_portfolios)
            st.plotly_chart(fig_illusion, use_container_width=True)
            with st.expander("The Volatility Illusion & Ex-Post Risk"):
                st.write("""
                **What to observe:** Models in the red 'Danger Zone' (above the dotted line) have a higher *real* risk than what they estimated *mathematically*. 
                Forward-Fill often creates a massive illusion because it thinks the flat prices are risk-free. The bubble size represents the magnitude of this hidden risk.
                """)
                
                risk_data = []
                for name in selected_portfolios:
                    metrics = risk_metrics[name]
                    risk_data.append({
                        "Model": name,
                        "Ex-Ante Risk": f"{metrics['estimated_volatility']*100:.2f}%",
                        "Real Risk (Ex-Post)": f"{metrics['true_volatility']*100:.2f}%",
                        "Hidden Jump (Error)": f"{metrics['volatility_underestimation']*100:.2f}%"
                    })
                st.dataframe(pd.DataFrame(risk_data).set_index("Model"), use_container_width=True)
                
        st.markdown("---")
        
        # Bottom Row: Tracking Error Penalty
        fig_pnl = QuantAnalytics.plot_portfolio_tracking_error(
            true_weights,
            {k: imputed_weights[k] for k in selected_portfolios},
            gt_df,
            selected_portfolios
        )
        st.plotly_chart(fig_pnl, use_container_width=True)
        with st.expander("Cumulative Tracking Error Penalty"):
            st.write("""
            **What to observe:** It is possible for a badly allocated portfolio to get 'lucky' and make more money than the optimal one if it accidentally overweights an asset that happens to surge. However, in quantitative finance, we measure the **penalty of being wrong**. 
            
            This chart plots the **Cumulative Squared Tracking Error**. The Optimal Ground Truth is exactly 0. You want an algorithm that stays as close to 0 as possible (a horizontal flat line).
            Notice how Forward-Fill drifts massively away, accumulating tracking error penalty over time because its weights are fundamentally wrong, translating into uncompensated risks.
            
            **N.B:** If you observe that all models overlap the optimal 0 line, try increasing the **Missing Rate** in the sidebar to simulate a more severe liquidity crisis.
            """)
        
# Replaced fully by the chunk above
