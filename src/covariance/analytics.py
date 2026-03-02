import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import scipy.stats as stats

class QuantAnalytics:
    """
    Class generating advanced visualizations for the 'Robust Covariance' presentation.
    It builds complex Plotly objects to demonstrate the impact of missing data.
    """
    
    @staticmethod
    def plot_time_series_reconstruction(
        ticker: str, 
        true_series: pd.Series, 
        corrupted_series: pd.Series, 
        imputed_results: dict[str, pd.Series]
    ) -> go.Figure:
        """
        Visualizes how different algorithms filled the same liquidity holes.
        """
        fig = go.Figure()
        
        # Ground Truth
        fig.add_trace(go.Scatter(
            x=true_series.index, y=true_series.values, 
            mode='lines', name='Ground Truth', 
            line=dict(color='lightgrey', width=4, dash='dash'),
            opacity=0.7
        ))
        
        # Plot Imputed Models
        colors = px.colors.qualitative.Plotly
        for i, (model_name, series) in enumerate(imputed_results.items()):
            mask = corrupted_series.isna()
            y_filled_only = series.copy()
            y_filled_only[~mask] = np.nan 
            
            fig.add_trace(go.Scatter(
                x=series.index, y=y_filled_only.values, 
                mode='markers', name=f'{model_name} (Fill)',
                marker=dict(size=8, symbol='x', color=colors[i % len(colors)], line=dict(width=1, color='DarkSlateGrey'))
            ))
            
        # The Corrupted data
        fig.add_trace(go.Scatter(
            x=corrupted_series.index, y=corrupted_series.values, 
            mode='markers+lines', name='Observed Data', 
            line=dict(color='black', width=2)
        ))
        
        fig.update_layout(
            title=f"Time Series Reconstruction - {ticker}",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            template="plotly_white",
            height=500
        )
        return fig

    @staticmethod
    def plot_returns_distribution(true_prices: pd.Series, imputed_prices: pd.Series, model_name: str) -> go.Figure:
        """
        KDE/Histogram plot of daily log-returns to visualize the volatility collapse (spike at 0).
        """
        true_returns = np.log(true_prices / true_prices.shift(1)).dropna()
        imputed_returns = np.log(imputed_prices / imputed_prices.shift(1)).dropna()
        
        fig = go.Figure()

        # True Returns
        fig.add_trace(go.Histogram(
            x=true_returns, histnorm='probability density', 
            name='True Returns', opacity=0.5, marker_color='grey',
            nbinsx=150
        ))
        
        # Imputed Returns
        fig.add_trace(go.Histogram(
            x=imputed_returns, histnorm='probability density', 
            name=f'{model_name} Returns', opacity=0.5, marker_color='red',
            nbinsx=150
        ))
        
        # Smoothed KDE Curves
        try:
            x_grid = np.linspace(min(true_returns.min(), imputed_returns.min()), max(true_returns.max(), imputed_returns.max()), 500)
            kde_true = stats.gaussian_kde(true_returns)
            kde_imp = stats.gaussian_kde(imputed_returns)
            
            fig.add_trace(go.Scatter(
                x=x_grid, y=kde_true(x_grid), mode='lines', name='True KDE',
                line=dict(color='black', width=2), visible=False
            ))
            
            fig.add_trace(go.Scatter(
                x=x_grid, y=kde_imp(x_grid), mode='lines', name=f'{model_name} KDE',
                line=dict(color='darkred', width=2), visible=False
            ))
        except np.linalg.LinAlgError:
            pass # Skip KDE if data is completely singular (e.g. all 0)

        fig.update_layout(
            barmode='overlay',
            title=f"Return Distribution Distortion ({model_name})",
            xaxis_title="Daily Log-Returns",
            yaxis_title="Density",
            template="plotly_white",
            height=400,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=1.0, y=1.15,
                    buttons=list([
                        dict(args=[{"visible": [True, True, False, False]}], label="Histograms", method="update"),
                        dict(args=[{"visible": [False, False, True, True]}], label="Smoothed Curves", method="update"),
                        dict(args=[{"visible": [True, True, True, True]}], label="Both", method="update")
                    ]),
                )
            ]
        )
        return fig

    @staticmethod
    def plot_error_heatmap(true_cov: pd.DataFrame, imputed_cov: pd.DataFrame, model_name: str) -> go.Figure:
        """
        Visual Absolute Difference Heatmap (|True - Imputed|).
        Highlights which cross-asset correlations were the most destroyed.
        """
        diff_matrix = np.abs(true_cov - imputed_cov)
        max_err = np.max(diff_matrix.values) + 1e-6
        
        fig = px.imshow(
            diff_matrix.values, 
            x=diff_matrix.columns, 
            y=diff_matrix.index,
            text_auto='.4f', 
            color_continuous_scale='Reds',
            zmin=0, zmax=max_err
        )
        
        fig.update_layout(
            title=f"Absolute Covariance Error (|True - {model_name}|)",
            xaxis_title=None, yaxis_title=None,
            height=600, width=600
        )
        return fig

    @staticmethod
    def plot_phase_transition(degradation_results: pd.DataFrame, selected_models: list[str]) -> go.Figure:
        """
        Plots the Frobenius Norm (Y) against the Missing Rate percentage (X).
        Only plots the models selected by the user.
        """
        fig = go.Figure()
        
        for model in selected_models:
            if model in degradation_results.columns:
                fig.add_trace(go.Scatter(
                    x=degradation_results['Missing Rate'],
                    y=degradation_results[model],
                    mode='lines+markers',
                    name=model,
                    hovertemplate="Rate: %{x:.0%}<br>Error: %{y:.4f}<extra></extra>"
                ))
            
        fig.update_layout(
            title="Algorithm Degradation Curve (Phase Transition)",
            xaxis_title="Missing Data Percentage (%)",
            yaxis_title="Covariance Error (Frobenius Norm)",
            xaxis_tickformat='%',
            hovermode="x unified",
            template="plotly_white",
            height=500
        )
        return fig

    @staticmethod
    def plot_stability_boxplots(stability_results: pd.DataFrame, selected_models: list[str]) -> go.Figure:
        """
        Boxplots for Monte Carlo stability. Only plots the selected models.
        """
        fig = go.Figure()
        
        for model in selected_models:
            if model in stability_results.columns:
                fig.add_trace(go.Box(
                    y=stability_results[model],
                    name=model,
                    boxpoints='all',
                    jitter=0.5,
                    whiskerwidth=0.2,
                    marker_size=4,
                    line_width=2
                ))
            
        fig.update_layout(
            title="Algorithm Convergence Variance (Monte Carlo runs)",
            yaxis_title="Covariance Error (Frobenius Norm)",
            xaxis_title="Imputation Model",
            template="plotly_white",
            showlegend=False,
            height=500
        )
        return fig

    @staticmethod
    def plot_portfolio_weights(true_weights: pd.Series, imputed_weights_dict: dict[str, pd.Series], selected_models: list[str]) -> go.Figure:
        """
        Bar chart comparing the optimal weights of the 'True' minimum variance portfolio
        vs the portfolios computed using the flawed covariance matrices (for selected models).
        """
        tickers = true_weights.index.tolist()
        fig = go.Figure()
        
        # Add True Portfolio
        fig.add_trace(go.Bar(
            name='Ideal Portfolio (Ground Truth)',
            x=tickers,
            y=true_weights.values,
            marker_color='black'
        ))
        
        # Add Each Selected Imputed Portfolio
        colors = px.colors.qualitative.Plotly
        for i, model_name in enumerate(selected_models):
            if model_name in imputed_weights_dict:
                fig.add_trace(go.Bar(
                    name=f'{model_name} Portfolio',
                    x=tickers,
                    y=imputed_weights_dict[model_name].values,
                    marker_color=colors[i % len(colors)]
                ))
            
        fig.update_layout(
            barmode='group',
            title="Asset Allocation Distortion (Minimum Variance)",
            xaxis_title="Assets",
            yaxis_title="Allocation Weight",
            yaxis_tickformat='.1%',
            template="plotly_white",
            height=400
        )
        return fig

    @staticmethod
    def plot_volatility_illusion(risk_metrics: dict[str, dict], true_vol: float, selected_models: list[str]) -> go.Figure:
        """
        Scatter Plot: Ex-Ante Volatility (X) vs Ex-Post Volatility (Y).
        Shows the 'Volatility Illusion' where a model promises low risk but delivers high risk.
        """
        fig = go.Figure()

        # Dynamic Zoom Bounds
        all_ex_ante = [m['estimated_volatility'] for k, m in risk_metrics.items() if k in selected_models] + [true_vol]
        all_ex_post = [m['true_volatility'] for k, m in risk_metrics.items() if k in selected_models] + [true_vol]
        
        # Add margin to avoid dots being cut off and allow a better global view
        vol_range = max(max(all_ex_ante + all_ex_post) - min(all_ex_ante + all_ex_post), 0.001)
        min_vol = max(0, min(all_ex_ante + all_ex_post) - vol_range * 0.25)
        max_vol = max(all_ex_ante + all_ex_post) + vol_range * 0.25
        
        # The 'Truth' Line (Ex-Ante == Ex-Post)
        fig.add_trace(go.Scatter(
            x=[min_vol, max_vol], y=[min_vol, max_vol],
            mode='lines', name='Perfect Estimation (x=y)',
            line=dict(color='black', dash='dash')
        ))

        # Ground Truth Marker
        fig.add_trace(go.Scatter(
            x=[true_vol], y=[true_vol],
            mode='markers', name='Ideal Portfolio',
            marker=dict(color='black', size=15, symbol='star'),
            hovertemplate="Truth<br>Ex-Ante: %{x:.2%}<br>Ex-Post: %{y:.2%}<extra></extra>"
        ))
        
        colors = px.colors.qualitative.Plotly
        for i, model_name in enumerate(selected_models):
            if model_name in risk_metrics:
                metrics = risk_metrics[model_name]
                error = max(0, metrics['volatility_underestimation'])
                
                fig.add_trace(go.Scatter(
                    x=[metrics['estimated_volatility']], 
                    y=[metrics['true_volatility']],
                    mode='markers', name=model_name,
                    marker=dict(
                        color=colors[i % len(colors)], 
                        size=15 + (error * 500), # Boost size for visual effect
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    hovertemplate=f"<b>{model_name}</b><br>Ex-Ante (Estimated): %{{x:.2%}}<br>Ex-Post (Real): %{{y:.2%}}<br>Underestimation Error: {error:.2%}<extra></extra>"
                ))
            
        fig.update_layout(
            title="The Volatility Illusion (Ex-Ante vs Ex-Post Risk)",
            xaxis_title="Estimated Volatility (Ex-Ante) -> The model's promise",
            yaxis_title="Real Volatility (Ex-Post) -> What you actually pay",
            xaxis_tickformat='.2%', yaxis_tickformat='.2%',
            template="plotly_white",
            height=500,
            xaxis=dict(range=[min_vol, max_vol]),
            yaxis=dict(range=[min_vol, max_vol])
        )
        
        # Add a shaded region for the "Illusion Area" (Ex-post > Ex-ante)
        fig.add_shape(
            type="path",
            path=f"M {min_vol} {min_vol} L {max_vol} {max_vol} L {min_vol} {max_vol} Z",
            fillcolor="red",
            opacity=0.1,
            line_width=0,
            layer="below"
        )
        # Add a text annotation
        fig.add_annotation(
            x=min_vol + (max_vol-min_vol)*0.2,
            y=min_vol + (max_vol-min_vol)*0.8,
            text="Danger Zone<br>(Hidden Risk)",
            showarrow=False,
            font=dict(color="red", size=14)
        )

        return fig

    @staticmethod
    def plot_rolling_volatility(true_prices: pd.Series, imputed_prices_dict: dict[str, pd.Series], selected_models: list[str], window: int = 30) -> tuple[go.Figure, pd.DataFrame]:
        """
        Plots the rolling annualized volatility and calculates the mean and standard deviation (noise) of the rolling series.
        Returns a tuple: (Plotly Figure, DataFrame containing summary stats)
        """
        fig = go.Figure()
        
        true_returns = np.log(true_prices / true_prices.shift(1)).dropna()
        true_roll_vol = true_returns.rolling(window=window).std() * np.sqrt(252)
        true_mean_bps = true_roll_vol.mean() * 10000
        true_noise_bps = true_roll_vol.std() * 10000
        
        stats_data = [{"Method": "Ground Truth", "Mean Vol (bps)": true_mean_bps, "Noise Std (bps)": true_noise_bps}]
        
        fig.add_trace(go.Scatter(
            x=true_roll_vol.index, y=true_roll_vol.values,
            mode='lines', name='Ground Truth',
            line=dict(color='black', width=3, dash='dash')
        ))
        
        colors = px.colors.qualitative.Plotly
        for i, model_name in enumerate(selected_models):
            if model_name in imputed_prices_dict:
                imp_series = imputed_prices_dict[model_name]
                imp_returns = np.log(imp_series / imp_series.shift(1)).dropna()
                imp_roll_vol = imp_returns.rolling(window=window).std() * np.sqrt(252)
                imp_mean_bps = imp_roll_vol.mean() * 10000
                imp_noise_bps = imp_roll_vol.std() * 10000
                
                stats_data.append({"Method": model_name, "Mean Vol (bps)": imp_mean_bps, "Noise Std (bps)": imp_noise_bps})
                
                fig.add_trace(go.Scatter(
                    x=imp_roll_vol.index, y=imp_roll_vol.values,
                    mode='lines', name=model_name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    opacity=0.8
                ))
                
        fig.update_layout(
            title=f"{window}-Day Rolling Annualized Volatility",
            xaxis_title="Date",
            yaxis_title="Volatility",
            yaxis_tickformat='.1%',
            hovermode="x unified",
            template="plotly_white",
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        stats_df = pd.DataFrame(stats_data)
        return fig, stats_df

    @staticmethod
    def plot_autocorrelation(true_prices: pd.Series, imputed_prices_dict: dict[str, pd.Series], selected_models: list[str], lags: int = 20) -> go.Figure:
        """
        Plots the AutoCorrelation Function (ACF) to show how FFill creates artificial memory (violating Efficient Market Hypothesis).
        """
        fig = go.Figure()
        
        true_returns = np.log(true_prices / true_prices.shift(1)).dropna()
        acf_true = [true_returns.autocorr(lag=i) for i in range(1, lags + 1)]
        acf_true_abs = [abs(x) for x in acf_true]
        
        fig.add_trace(go.Scatter(
            x=list(range(1, lags + 1)), y=acf_true,
            mode='lines+markers', name='Ground Truth',
            line=dict(color='black', width=3, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(1, lags + 1)), y=acf_true_abs,
            mode='lines+markers', name='Ground Truth',
            line=dict(color='black', width=3, dash='dash'), visible=False
        ))
        
        colors = px.colors.qualitative.Plotly
        
        vis_std = [True, False]
        vis_abs = [False, True]
        
        for i, model_name in enumerate(selected_models):
            if model_name in imputed_prices_dict:
                imp_series = imputed_prices_dict[model_name]
                imp_returns = np.log(imp_series / imp_series.shift(1)).dropna()
                acf_imp = [imp_returns.autocorr(lag=i) for i in range(1, lags + 1)]
                acf_imp_abs = [abs(x) for x in acf_imp]
                
                fig.add_trace(go.Scatter(
                    x=list(range(1, lags + 1)), y=acf_imp,
                    mode='lines+markers', name=model_name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    opacity=0.8
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(1, lags + 1)), y=acf_imp_abs,
                    mode='lines+markers', name=model_name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    opacity=0.8, visible=False
                ))
                vis_std.extend([True, False])
                vis_abs.extend([False, True])
                
        fig.update_layout(
            title="Return Autocorrelation (ACF) - Memory Effect Test",
            xaxis_title="Lag (Days)",
            yaxis_title="Autocorrelation Coefficient",
            template="plotly_white",
            height=400,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=1.0, y=1.15,
                    buttons=list([
                        dict(args=[{"visible": vis_std}], label="Standard ACF", method="update"),
                        dict(args=[{"visible": vis_abs}, {"yaxis.title": "Absolute Autocorrelation |ACF|"}], label="Absolute |ACF|", method="update")
                    ]),
                )
            ]
        )
        return fig

    @staticmethod
    def plot_cross_correlation_scatter(true_cov: pd.DataFrame, imputed_covs_dict: dict[str, pd.DataFrame], selected_models: list[str]) -> go.Figure:
        """
        Scatter plot of True Pairwise Correlation vs Imputed Pairwise Correlation.
        Perfect imputation lies on the y=x diagonal. Focuses on slope decay (Epps Effect).
        """
        true_corr = true_cov.corr()
        mask = np.triu(np.ones(true_corr.shape), k=1).astype(bool)
        true_pairs = true_corr.where(mask).stack().values
        
        fig = go.Figure()
        
        # Perfect line y = x
        fig.add_trace(go.Scatter(
            x=[-1, 1], y=[-1, 1],
            mode='lines', name='Perfect Alignment (y=x)',
            line=dict(color='black', dash='dash')
        ))
        
        colors = px.colors.qualitative.Plotly
        for i, model_name in enumerate(selected_models):
            if model_name in imputed_covs_dict:
                imp_corr = imputed_covs_dict[model_name].corr()
                imp_pairs = imp_corr.where(mask).stack().values
                
                # Scatter points
                fig.add_trace(go.Scatter(
                    x=true_pairs, y=imp_pairs,
                    mode='markers', name=model_name,
                    marker=dict(size=8, color=colors[i % len(colors)], opacity=0.7)
                ))
                
                # Computed Linear Regression to show Epps Effect (beta decay)
                if len(true_pairs) > 1:
                    slope, intercept, _, _, _ = stats.linregress(true_pairs.astype(float), imp_pairs.astype(float))
                    line_x = np.array([-1.1, 1.1])
                    line_y = slope * line_x + intercept
                    fig.add_trace(go.Scatter(
                        x=line_x, y=line_y,
                        mode='lines', name=f'{model_name} Trend (Slope: {slope:.2f})',
                        line=dict(color=colors[i % len(colors)], width=3, dash='dot')
                    ))
                
        fig.update_layout(
            title="Cross-Correlation Structure Preservation & Trendlines (Epps Effect)",
            xaxis_title="True Cross-Correlation (Ground Truth)",
            yaxis_title="Imputed Cross-Correlation",
            template="plotly_white",
            height=500,
            xaxis=dict(range=[-1.1, 1.1]),
            yaxis=dict(range=[-1.1, 1.1])
        )
        return fig

    @staticmethod
    def plot_correlation_dendrogram(cov_df: pd.DataFrame, title: str) -> go.Figure:
        """
        Plots a hierarchical clustering dendrogram of the correlation matrix.
        Shows how missing data can falsely regroup assets.
        """
        import plotly.figure_factory as ff
        corr = cov_df.corr()
        
        # Create dendrogram using correlation matrix features
        fig = ff.create_dendrogram(corr.values, labels=corr.columns)
        
        fig.update_layout(
            title=title,
            xaxis_title="Assets",
            yaxis_title="Distance",
            template="plotly_white",
            height=400
        )
        return fig

    @staticmethod
    def plot_portfolio_tracking_error(true_weights: pd.Series, imputed_weights_dict: dict[str, pd.Series], true_prices: pd.DataFrame, selected_models: list[str]) -> go.Figure:
        """
        Calculates the Cumulative Squared Tracking Error. The penalty of deviating from 
        the optimal Ground Truth portfolio. Proves that bad imputation always incurs an error.
        """
        returns = true_prices.pct_change().dropna()
        
        fig = go.Figure()
        
        # True Portfolio Returns
        true_port_ret = returns.dot(true_weights)
        
        fig.add_trace(go.Scatter(
            x=true_port_ret.index, y=np.zeros(len(true_port_ret)),
            mode='lines', name='Ideal Portfolio (Ground Truth)',
            line=dict(color='black', width=3, dash='dash')
        ))
        
        colors = px.colors.qualitative.Plotly
        for i, model_name in enumerate(selected_models):
            if model_name in imputed_weights_dict:
                imp_weights = imputed_weights_dict[model_name]
                imp_port_ret = returns.dot(imp_weights)
                
                # Cumulative Squared Error (scaled for readability)
                sq_error = (imp_port_ret - true_port_ret)**2
                cum_sq_error = sq_error.cumsum() * 10000 
                
                fig.add_trace(go.Scatter(
                    x=cum_sq_error.index, y=cum_sq_error.values,
                    mode='lines', name=f'{model_name} Penalty',
                    line=dict(color=colors[i % len(colors)], width=2),
                    opacity=0.8
                ))
                
        fig.update_layout(
            title="Cumulative Tracking Error Penalty (Deviation from Optimal)",
            xaxis_title="Date",
            yaxis_title="Cumulative Squared Error Penalty",
            hovermode="x unified",
            template="plotly_white",
            height=500
        )
        return fig

    @staticmethod
    def plot_turnover_penalty(true_weights: pd.Series, imputed_weights_dict: dict[str, pd.Series], selected_models: list[str]) -> go.Figure:
        """
        Bar chart showing the Turn-Over penalty (sum of absolute weight differences)
        from the ideal portfolio. Corresponds to dead-weight trading costs.
        """
        fig = go.Figure()
        
        models = []
        turnovers = []
        
        colors = px.colors.qualitative.Plotly
        bar_colors = []
        
        for i, model_name in enumerate(selected_models):
            if model_name in imputed_weights_dict:
                imp_weights = imputed_weights_dict[model_name]
                turnover = np.sum(np.abs(imp_weights - true_weights))
                models.append(model_name)
                turnovers.append(turnover)
                bar_colors.append(colors[i % len(colors)])
                
        fig.add_trace(go.Bar(
            x=models, y=turnovers,
            marker_color=bar_colors,
            text=[f"{t:.1%}" for t in turnovers],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Turnover Friction Penalty (Absolute Weight Reallocation)",
            xaxis_title="Model",
            yaxis_title="Excess Turnover (%) vs Ideal",
            yaxis_tickformat='.1%',
            template="plotly_white",
            height=400
        )
        return fig
