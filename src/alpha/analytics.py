import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from src.alpha.universe import get_asset_name
import plotly.express as px
import scipy.stats as stats

# ==============================================================================
# TAB 1: OVERVIEW & PERFORMANCE
# ==============================================================================

def calculate_kpis(portfolio_nav, benchmark_data, benchmark_label="Benchmark"):
    """
    Computes standard performance and risk metrics (CAGR, Volatility, Sharpe, Max Drawdown).
    """
    common_index = portfolio_nav.index.intersection(benchmark_data.index)
    port = portfolio_nav.loc[common_index]
    bench = benchmark_data.loc[common_index]
    
    port_rets = port.pct_change().dropna()
    bench_rets = bench.pct_change().dropna()
    
    def get_metrics(series, rets):
        if series.empty: return 0, 0, 0, 0, 0, 0 
        
        total_ret = (series.iloc[-1] / series.iloc[0]) - 1
        
        days = (series.index[-1] - series.index[0]).days
        years = days / 365.25
        cagr = (series.iloc[-1] / series.iloc[0])**(1/years) - 1 if years > 0 else 0
        
        vol = rets.std() * np.sqrt(252)
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() != 0 else 0
        
        rolling_max = series.cummax()
        drawdown = (series / rolling_max) - 1
        max_dd = drawdown.min()
        
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        return total_ret, cagr, vol, sharpe, max_dd, calmar

    p_tot, p_cagr, p_vol, p_sharpe, p_dd, p_calmar = get_metrics(port, port_rets)
    b_tot, b_cagr, b_vol, b_sharpe, b_dd, b_calmar = get_metrics(bench, bench_rets)
    
    metrics = {
        'Metric': ['Total Return', 'CAGR', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio'],
        'Strategy': [f"{p_tot:.2%}", f"{p_cagr:.2%}", f"{p_vol:.2%}", f"{p_sharpe:.2f}", f"{p_dd:.2%}", f"{p_calmar:.2f}"],
        benchmark_label: [f"{b_tot:.2%}", f"{b_cagr:.2%}", f"{b_vol:.2%}", f"{b_sharpe:.2f}", f"{b_dd:.2%}", f"{b_calmar:.2f}"],
        'Alpha (Diff)': [
            f"{(p_tot - b_tot):.2%}", 
            f"{(p_cagr - b_cagr):.2%}", 
            f"{(p_vol - b_vol):.2%}", 
            f"{(p_sharpe - b_sharpe):.2f}", 
            f"{(p_dd - b_dd):.2%}",
            f"{(p_calmar - b_calmar):.2f}"
        ]
    }
    
    return pd.DataFrame(metrics)

def plot_equity_curve(portfolio_nav, benchmark_data, benchmark_ticker="Benchmark"):
    """
    Plots the cumulative return of the strategy vs a rebased benchmark.
    """
    common_index = portfolio_nav.index.intersection(benchmark_data.index)
    port_series = portfolio_nav.loc[common_index]
    bench_series_raw = benchmark_data.loc[common_index]
    
    initial_capital = port_series.iloc[0]
    initial_bench_price = bench_series_raw.iloc[0]
    scale_factor = initial_capital / initial_bench_price
    bench_series_scaled = bench_series_raw * scale_factor

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=port_series.index, y=port_series,
        mode='lines', name='AlphaStream Strategy',
        line=dict(color='#00CC96', width=2),
        hovertemplate='$%{y:,.0f} (Strategy)<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=bench_series_scaled.index, y=bench_series_scaled,
        mode='lines', name=benchmark_ticker, 
        line=dict(color='#EF553B', width=2, dash='dot'),
        hovertemplate=f'$%{{y:,.0f}} ({benchmark_ticker})<extra></extra>'
    ))

    fig.update_layout(
        title={'text': f"<b>Equity Curve vs {benchmark_ticker}</b> (Rebased)", 'y':0.9, 'x':0.5, 'xanchor': 'center'},
        xaxis_title="Date", yaxis_title="Portfolio Value ($)",
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig

def plot_returns_distribution(portfolio_nav, benchmark_data, benchmark_label="Benchmark"):
    """
    Visualizes the return distribution (KDE) linking moments (Mean, Vol, Skew, Kurtosis).
    """
    common_index = portfolio_nav.index.intersection(benchmark_data.index)
    port_rets = portfolio_nav.loc[common_index].pct_change().dropna()
    bench_rets = benchmark_data.loc[common_index].pct_change().dropna()

    stats_dict = {
        'Strat': {'Mean': port_rets.mean(), 'Vol': port_rets.std(), 'Skew': port_rets.skew(), 'Kurt': port_rets.kurtosis()},
        'Bench': {'Mean': bench_rets.mean(), 'Vol': bench_rets.std(), 'Skew': bench_rets.skew(), 'Kurt': bench_rets.kurtosis()}
    }

    min_x = min(port_rets.min(), bench_rets.min()) - 0.01
    max_x = max(port_rets.max(), bench_rets.max()) + 0.01
    x_range = np.linspace(min_x, max_x, 500)

    kde_strat = stats.gaussian_kde(port_rets)(x_range)
    kde_bench = stats.gaussian_kde(bench_rets)(x_range)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_range, y=kde_bench, mode='lines', name=f'{benchmark_label} (KDE)',
        line=dict(color='#EF553B', width=2), fill='tozeroy', fillcolor='rgba(239, 85, 59, 0.2)' 
    ))

    fig.add_trace(go.Scatter(
        x=x_range, y=kde_strat, mode='lines', name='Strategy (KDE)',
        line=dict(color='#00CC96', width=3), fill='tozeroy', fillcolor='rgba(0, 204, 150, 0.3)'
    ))

    fig.add_vline(x=stats_dict['Bench']['Mean'], line_dash="dot", line_color="#EF553B", opacity=0.8)
    fig.add_vline(x=stats_dict['Strat']['Mean'], line_dash="dash", line_color="#00CC96", opacity=1.0)

    stats_text = (
        f"<b>STATISTICS</b><br>"
        f"<span style='color:#00CC96'>Strategy</span> vs <span style='color:#EF553B'>Bench</span><br>"
        f"-----------------------<br>"
        f"<b>Mean:</b>  {stats_dict['Strat']['Mean']:.2%}  |  {stats_dict['Bench']['Mean']:.2%}<br>"
        f"<b>Vol:</b>   {stats_dict['Strat']['Vol']:.2%}  |  {stats_dict['Bench']['Vol']:.2%}<br>"
        f"<b>Skew:</b>  {stats_dict['Strat']['Skew']:.2f}   |  {stats_dict['Bench']['Skew']:.2f}<br>"
        f"<b>Kurt:</b>  {stats_dict['Strat']['Kurt']:.2f}   |  {stats_dict['Bench']['Kurt']:.2f}"
    )

    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.98, xanchor="right", yanchor="top",
        text=stats_text, showarrow=False, align="left",
        font=dict(family="Courier New, monospace", size=12, color="black"),
        bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="black", borderwidth=1
    )

    fig.update_layout(
        title={'text': "<b>Return Distribution Analysis (KDE)</b>", 'y':0.9, 'x':0.05, 'xanchor': 'left'},
        xaxis_title="Daily Return", yaxis_title="Probability Density",
        template="plotly_white",
        xaxis=dict(tickformat=".1%", range=[min_x, max_x], zeroline=True, zerolinewidth=1, zerolinecolor='grey'),
        yaxis=dict(showticklabels=False), 
        legend=dict(x=0.01, y=0.99), margin=dict(l=20, r=20, t=60, b=20), height=450
    )
    return fig

def plot_drawdown_underwater(portfolio_nav, benchmark_data, benchmark_label="Benchmark"):
    """
    Plots the depth and duration of historical drawdowns.
    """
    common_index = portfolio_nav.index.intersection(benchmark_data.index)
    port_series = portfolio_nav.loc[common_index]
    bench_series = benchmark_data.loc[common_index]

    p_rolling_max = port_series.cummax()
    p_drawdown = (port_series / p_rolling_max) - 1

    b_rolling_max = bench_series.cummax()
    b_drawdown = (bench_series / b_rolling_max) - 1

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=p_drawdown.index, y=p_drawdown, mode='lines', name='Strategy',
        fill='tozeroy', line=dict(color='#00CC96', width=1.5), 
        fillcolor='rgba(0, 204, 150, 0.2)', hovertemplate='Strategy: %{y:.2%}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=b_drawdown.index, y=b_drawdown, mode='lines', name=f'{benchmark_label}',
        line=dict(color='#EF553B', width=2, dash='dot'), hovertemplate=f'{benchmark_label}: %{{y:.2%}}<extra></extra>'
    ))

    fig.update_layout(
        title={'text': f"<b>Underwater Plot</b> (Drawdown vs {benchmark_label})", 'y':0.9, 'x':0.35, 'xanchor': 'center'},
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        template="plotly_white", yaxis=dict(tickformat=".0%"), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig

def plot_alpha_beta_scatter(nav_series, benchmark_series, view_mode='full'):
    """
    Evaluates Strategy Alpha & Beta generation using CAPM linear regression.
    """
    strat_ret = nav_series.pct_change().dropna()
    bench_ret = benchmark_series.pct_change().dropna()
    
    common_idx = strat_ret.index.intersection(bench_ret.index)
    strat_ret, bench_ret = strat_ret.loc[common_idx], bench_ret.loc[common_idx]
    
    if len(common_idx) < 30: return None

    beta, alpha_daily, r_value, p_value, std_err = stats.linregress(bench_ret, strat_ret)
    alpha_annual = (1 + alpha_daily)**252 - 1
    
    x_min, x_max = bench_ret.min(), bench_ret.max()
    x_range = np.linspace(x_min - 0.01, x_max + 0.01, 100)
    y_pred = beta * x_range + alpha_daily

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bench_ret, y=strat_ret, mode='markers', name='Daily Returns',
        marker=dict(color='rgba(100, 149, 237, 0.5)', size=6), 
        hovertemplate='Bench: %{x:.2%}<br>Strat: %{y:.2%}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=x_range, y=y_pred, mode='lines', name=f'Regression (β={beta:.2f})',
        line=dict(color='#FF3131', width=3) 
    ))

    fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.5)
    fig.add_vline(x=0, line_width=1, line_color="black", opacity=0.5)

    if view_mode == 'zoomed':
        zoom_range = [-0.002, 0.002]
        fig.update_layout(
            title="<b> Alpha Zoom (Intercept Check)</b>",
            xaxis=dict(range=zoom_range, title="Benchmark Return", zeroline=False),
            yaxis=dict(range=zoom_range, title="Strategy Return", zeroline=False),
            showlegend=False, margin=dict(l=20, r=20, t=40, b=20), height=400
        )
        
        arrow_col, txt, ay_offset = ("#000000", "<b>POSITIVE ALPHA</b>", 40) if alpha_daily > 0 else ("#EF553B", "Negative Alpha", -40)
            
        fig.add_annotation(
            x=0, y=alpha_daily, xref="x", yref="y", text=txt,
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=arrow_col,
            ax=0, ay=ay_offset, font=dict(color=arrow_col, size=12)
        )
    else:
        fig.update_layout(
            title="<b> Alpha Generation (Full View)</b>",
            xaxis_title="Benchmark Return", yaxis_title="Strategy Return",
            template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.8)')
        )
        
        fig.add_annotation(
            xref="paper", yref="paper", x=0.98, y=0.02, xanchor="right", yanchor="bottom",
            text=f"<b>Alpha (Ann): {alpha_annual:+.2%}</b><br>Beta: {beta:.2f}",
            showarrow=False, font=dict(size=13, color="black"),
            bgcolor="#DDDDDD", bordercolor="#000000", borderwidth=2, opacity=0.9
        )
    return fig

def plot_rolling_sharpe(portfolio_nav, benchmark_data, window_months=6, benchmark_label="Benchmark"):
    """
    Tracks Alpha Decay by plotting rolling annualized Sharpe ratio.
    """
    common_index = portfolio_nav.index.intersection(benchmark_data.index)
    port_rets = portfolio_nav.loc[common_index].pct_change().dropna()
    bench_rets = benchmark_data.loc[common_index].pct_change().dropna()
    
    window = window_months * 21
    
    def get_rolling_sharpe(series):
        return (series.rolling(window).mean() / series.rolling(window).std()) * np.sqrt(252)

    roll_sharpe_port = get_rolling_sharpe(port_rets).dropna()
    roll_sharpe_bench = get_rolling_sharpe(bench_rets).dropna()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=roll_sharpe_port.index, y=roll_sharpe_port, mode='lines', name='Strategy', line=dict(color='#00CC96', width=2)))
    fig.add_trace(go.Scatter(x=roll_sharpe_bench.index, y=roll_sharpe_bench, mode='lines', name=f'{benchmark_label}', line=dict(color='#EF553B', width=1.5, dash='dot')))

    fig.add_hline(y=0, line_width=1, line_color="black", line_dash="solid")
    fig.add_hline(y=1, line_width=1, line_color="gray", line_dash="dot", annotation_text="Good (>1)", annotation_position="bottom right")

    fig.update_layout(
        title={'text': f"<b>{window_months}-Month Rolling Sharpe Ratio</b> (Stability)", 'y':0.9, 'x':0.5, 'xanchor': 'center'},
        xaxis_title="Date", yaxis_title="Sharpe Ratio", template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig


# ==============================================================================
# TAB 2: ASSET ALLOCATION
# ==============================================================================

def plot_dynamic_allocation(history_df, universe_dict):
    """
    Plots historical asset allocation stacked by specific asset classes.
    """
    fig = go.Figure()
    
    ticker_to_class = {}
    for cat, tickers in universe_dict.items():
        for t in tickers:
            ticker_to_class[t] = cat
    ticker_to_class['CASH'] = 'Cash' 
    
    excluded_cols = ['NAV', 'Daily_Ret', 'Benchmark_NAV']
    ticker_cols = [c for c in history_df.columns if c not in excluded_cols]
    alloc_df = pd.DataFrame(index=history_df.index)
    
    classes = list(universe_dict.keys())
    if 'Cash' not in classes: classes.append('Cash')

    for c in classes: alloc_df[c] = 0.0
        
    for t in ticker_cols:
        if t in history_df.columns:
            asset_class = ticker_to_class.get(t, 'Cash') 
            alloc_df[asset_class] = alloc_df[asset_class].fillna(0) + history_df[t].fillna(0)
            
    colors = {'Actions': '#00CC96', 'Bonds': '#636EFA', 'Commodities': '#EF553B', 'Cash': '#808080'}
    
    for col in alloc_df.columns:
        if alloc_df[col].sum() > 0.001: 
            fig.add_trace(go.Scatter(
                x=alloc_df.index, y=alloc_df[col], mode='lines', name=col, stackgroup='one', 
                line=dict(width=0.5), fillcolor=colors.get(col, '#d3d3d3'), 
                marker=dict(color=colors.get(col, '#d3d3d3')), hovertemplate=f'{col}: %{{y:.1%}}<extra></extra>'
            ))

    fig.update_layout(
        title={'text': "<b>Dynamic Asset Allocation</b> (Evolution by Asset Class)", 'y':0.9, 'x':0.5, 'xanchor': 'center'},
        xaxis_title="Date", yaxis_title="Weight", yaxis=dict(tickformat=".0%", range=[0, 1.05]), 
        template="plotly_white", hovermode="x unified", margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig

def plot_asset_rotation_heatmap(history_df, top_n_display=10):
    """
    Visualizes asset rotation by plotting heatmap of monthly weights for top N active tickers.
    """
    excluded_cols = ['NAV', 'Daily_Ret', 'Benchmark_NAV', 'CASH']
    ticker_cols = [c for c in history_df.columns if c not in excluded_cols]
    if not ticker_cols: return go.Figure()

    monthly_weights = history_df[ticker_cols].resample('ME').mean()
    top_tickers = monthly_weights.sum().sort_values(ascending=False).head(top_n_display).index
    plot_data = monthly_weights[top_tickers].T 
    y_labels = [get_asset_name(t) for t in plot_data.index]
    
    fig = go.Figure(data=go.Heatmap(
        z=plot_data.values, x=plot_data.columns, y=y_labels, colorscale='Blues', 
        zmin=0, zmax=0.5, colorbar=dict(title='Weight'),
        hovertemplate='Date: %{x}<br>Asset: %{y}<br>Weight: %{z:.1%}<extra></extra>'
    ))

    fig.update_layout(
        title={'text': "<b>Asset Rotation Heatmap</b> (Top Holdings History)", 'y':0.9, 'x':0.5, 'xanchor': 'center'},
        xaxis_title="Date", yaxis_title="Ticker", template="plotly_white", margin=dict(l=40, r=40, t=80, b=40), height=500 
    )
    return fig

def plot_monthly_contribution(history_df, market_df, universe_dict):
    """
    Plots weighted return attribution for asset classes on a monthly basis.
    """
    asset_returns = market_df.pct_change()
    
    common_index = history_df.index.intersection(asset_returns.index)
    weights_df = history_df.loc[common_index]
    returns_df = asset_returns.loc[common_index]
    
    daily_contrib = pd.DataFrame(index=common_index)
    
    classes = list(universe_dict.keys())
    for cat in classes:
        cat_daily = pd.Series(0.0, index=common_index)
        tickers = [t for t in universe_dict[cat] if t in weights_df.columns and t in returns_df.columns]
        for t in tickers:
            w = weights_df[t].shift(1).fillna(0)
            r = returns_df[t]
            cat_daily += w * r
        daily_contrib[cat] = cat_daily

    monthly_contrib = daily_contrib.resample('ME').sum()
    
    fig = go.Figure()
    colors = {'Actions': '#00CC96', 'Bonds': '#636EFA', 'Commodities': '#EF553B', 'Cash': '#808080'}

    for col in monthly_contrib.columns:
        fig.add_trace(go.Bar(
            x=monthly_contrib.index, y=monthly_contrib[col], name=col,
            marker_color=colors.get(col, None), hovertemplate=f'{col}: %{{y:.2%}}<extra></extra>'
        ))

    total_monthly = monthly_contrib.sum(axis=1)
    fig.add_trace(go.Scatter(
        x=monthly_contrib.index, y=total_monthly, mode='markers+lines', name='Total Net',
        line=dict(color='black', width=1, dash='dot'), marker=dict(symbol='diamond', size=6, color='black'),
        hovertemplate='Total: %{y:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title={'text': "<b>Monthly Performance Attribution</b> (Weighted Contribution)", 'y':0.9, 'x':0.5, 'xanchor': 'center'},
        xaxis_title="Month", yaxis_title="Monthly Contribution", barmode='relative', 
        yaxis=dict(tickformat=".1%"), template="plotly_white", hovermode="x unified", margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig

def plot_allocation_donut(snapshot_series):
    """
    Plots a point-in-time composition of the portfolio.
    """
    excluded_cols = ['NAV', 'Daily_Ret', 'Benchmark_NAV']
    weights = snapshot_series.drop(labels=[c for c in excluded_cols if c in snapshot_series.index])
    weights = weights[weights > 0.01] 
    labels_list = [get_asset_name(t) for t in weights.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels_list, values=weights.values, hole=.4, 
        textinfo='label+percent', hoverinfo='label+percent+value'
    )])
    
    date_str = snapshot_series.name.strftime('%Y-%m-%d') if hasattr(snapshot_series.name, 'strftime') else str(snapshot_series.name)
    
    fig.update_layout(
        title={'text': f"<b>Allocation Snapshot</b><br><span style='font-size:12px'>Date: {date_str}</span>", 'y':0.9, 'x':0.5, 'xanchor': 'center'},
        template="plotly_white", margin=dict(l=40, r=40, t=80, b=40), showlegend=False 
    )
    return fig

def get_holdings_table(snapshot_series, universe_dict):
    """
    Formats the portfolio state into a tabular structure.
    """
    excluded_cols = ['NAV', 'Daily_Ret', 'Benchmark_NAV']
    holdings = []
    
    ticker_to_class = {}
    for cat, tickers in universe_dict.items():
        for t in tickers:
            ticker_to_class[t] = cat
    ticker_to_class['CASH'] = 'Cash/Hedge'

    nav = snapshot_series.get('NAV', 0)

    for ticker in snapshot_series.index:
        if ticker not in excluded_cols:
            weight = snapshot_series[ticker]
            if abs(weight) > 0.001: 
                holdings.append({
                    "Asset": get_asset_name(ticker),
                    "Class": ticker_to_class.get(ticker, "Unknown"),
                    "Weight": weight,
                    "Value ($)": weight * nav
                })
    
    df = pd.DataFrame(holdings)
    if not df.empty: df = df.sort_values(by="Weight", ascending=False)
    return df


# ==============================================================================
# TAB 3: SIGNALS & SELECTION
# ==============================================================================

def calculate_all_signals(market_df, signal_method, lookback=126):
    """
    Generates momentum indicators (Z-Score, MA Distance, or RSI) across the whole asset pool.
    """
    if isinstance(market_df.columns, pd.MultiIndex):
        try: prices = market_df['Adj Close']
        except KeyError: prices = market_df.xs('Close', level=1, axis=1) 
    else: prices = market_df

    signals = pd.DataFrame(index=prices.index, columns=prices.columns)
    
    if signal_method == 'z_score':
        rolling_mean = prices.rolling(window=lookback).mean()
        rolling_std = prices.rolling(window=lookback).std()
        signals = (prices - rolling_mean) / rolling_std
        
    elif signal_method == 'distance_ma':
        ma = prices.rolling(window=lookback).mean()
        signals = (prices - ma) / ma
        
    elif signal_method == 'rsi':
        rsi_window = 14 
        delta = prices.diff()
        gain, loss = delta.where(delta > 0, 0), -delta.where(delta < 0, 0)
        avg_gain, avg_loss = gain.rolling(window=rsi_window).mean(), loss.rolling(window=rsi_window).mean()
        rs = avg_gain / avg_loss
        signals = 100 - (100 / (1 + rs))
        
    return signals.dropna()

def plot_signal_race(signals_df, highlight_assets=None, signal_method="z_score"):
    """
    Plots the full signal history, spotlighting specific selected assets.
    """
    fig = go.Figure()
    if highlight_assets is None: highlight_assets = []
        
    background_assets = [c for c in signals_df.columns if c not in highlight_assets]
    for col in background_assets:
        if signals_df[col].isna().all(): continue
        fig.add_trace(go.Scatter(
            x=signals_df.index, y=signals_df[col], mode='lines', name=get_asset_name(col),
            line=dict(color='#cccccc', width=1.5), opacity=0.6, showlegend=False, hoverinfo='skip' 
        ))

    colors_palette = pc.qualitative.Plotly 
    for i, col in enumerate(highlight_assets):
        if col in signals_df.columns:
            color = colors_palette[i % len(colors_palette)] 
            fig.add_trace(go.Scatter(
                x=signals_df.index, y=signals_df[col], mode='lines', name=get_asset_name(col),
                line=dict(color=color, width=2), opacity=1.0, hovertemplate=f'<b>{col}</b>: %{{y:.2f}}<extra></extra>'
            ))
        
    if signal_method == 'rsi':
        fig.add_hline(y=70, line_width=1, line_dash="dot", line_color="red", opacity=0.5)
        fig.add_hline(y=30, line_width=1, line_dash="dot", line_color="green", opacity=0.5)
        y_title = "RSI (0-100)"
    elif signal_method in ['z_score', 'distance_ma']:
        fig.add_hline(y=0, line_width=1.5, line_color="black", opacity=0.8)
        if signal_method == 'z_score':
            fig.add_hline(y=2, line_width=1, line_dash="dot", line_color="gray", opacity=0.5)
            fig.add_hline(y=-2, line_width=1, line_dash="dot", line_color="gray", opacity=0.5)
            y_title = "Z-Score (Std Dev)"
        else: y_title = "% Distance to MA"

    fig.update_layout(
        title={'text': f"<b>Signal Evolution Race</b> (Spotlight View)", 'y':0.9, 'x':0.5, 'xanchor': 'center'},
        xaxis_title="Date", yaxis_title=y_title, template="plotly_white", hovermode="x unified",
        margin=dict(l=40, r=150, t=80, b=40), legend=dict(orientation="v", y=1, x=1.02, xanchor='left', yanchor='top', font=dict(size=10))
    )
    return fig

def plot_signal_ranking_bar(signals_df, target_date, actual_selections=None):
    """
    Plots horizontal bars representing signal strength for a specific point in time.
    """
    fig = go.Figure()
    try:
        target_ts = pd.Timestamp(target_date)
        if target_ts not in signals_df.index:
            idx_pos = signals_df.index.get_indexer([target_ts], method='pad')[0]
            if idx_pos == -1: return go.Figure()
            actual_date = signals_df.index[idx_pos]
        else: actual_date = target_ts
        
        row_sorted = signals_df.loc[actual_date].sort_values(ascending=True).dropna()
        if row_sorted.empty: return go.Figure()
    except Exception: return go.Figure()

    y_labels = [get_asset_name(t) for t in row_sorted.index]
    colors, status_texts = [], []
    
    for ticker in row_sorted.index:
        val = row_sorted[ticker]
        if actual_selections and ticker in actual_selections:
            colors.append('#00CC96')  
            status_texts.append(f"SELECTED<br>Score: {val:.2f}")
        elif val > 0:
            colors.append('#BDC3C7')  
            status_texts.append(f"SKIPPED")
        else:
            colors.append('#E6B0AA')  
            status_texts.append(f"NEGATIVE<br>Score: {val:.2f}")

    fig.add_trace(go.Bar(
        x=row_sorted.values, y=y_labels, orientation='h', marker_color=colors,
        text=row_sorted.values.round(2), textposition='auto',
        hovertext=status_texts, hovertemplate='<b>%{y}</b><br>%{hovertext}<extra></extra>'
    ))

    dynamic_height = max(250, len(row_sorted) * 30)
    fig.update_layout(
        title=" ", xaxis_title=None, yaxis=dict(autorange=True, tickfont=dict(size=11)), 
        template="plotly_white", margin=dict(l=10, r=10, t=10, b=20), height=dynamic_height, showlegend=False
    )
    return fig

def plot_correlation_matrix(market_df, ranking_df, threshold=0.7, window=60):
    """
    Generates a compact heat-mapped correlation matrix to visualize diversification filters.
    """
    sorted_tickers = ranking_df.index.tolist() if hasattr(ranking_df, 'index') else list(ranking_df)
    excluded = ['NAV', 'Cash', 'Bench', 'Total']
    sorted_tickers = [t for t in sorted_tickers if t not in excluded]
    if len(sorted_tickers) < 2: return None

    prices = pd.DataFrame()
    try:
        if isinstance(market_df.columns, pd.MultiIndex):
            target_col = 'Adj Close' if 'Adj Close' in market_df.columns.get_level_values(0) else 'Close'
            valid_tickers = [t for t in sorted_tickers if t in market_df[target_col].columns]
            prices = market_df[target_col][valid_tickers]
        else:
            valid_tickers = [t for t in sorted_tickers if t in market_df.columns]
            prices = market_df.loc[:, valid_tickers]     
    except: return None
    if prices.empty: return None

    recent_returns = prices.pct_change().tail(window).dropna(how='all').fillna(0)
    corr_matrix = recent_returns.corr()
    mask = np.eye(len(corr_matrix), dtype=bool)
    corr_display = corr_matrix.where(~mask, np.nan) 

    thresh_norm = (threshold + 1) / 2
    colors = [
        [0.0, 'white'],           
        [thresh_norm, 'white'],   
        [thresh_norm, '#EF553B'], 
        [1.0, '#B22222']          
    ]

    fig = px.imshow(
        corr_display, text_auto=".2f", aspect="equal", color_continuous_scale=colors, zmin=-1, zmax=1
    )
    fig.update_layout(
        title=None, coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0), 
        xaxis=dict(side="bottom", showticklabels=True), yaxis=dict(showticklabels=True),
        height=450, plot_bgcolor='rgba(240,240,240, 1)'
    )
    if len(sorted_tickers) > 8: fig.update_xaxes(tickangle=-45)
    
    return fig

def plot_signal_vs_price(market_df, signals_df, ticker):
    """
    Charts both raw price (line) and signal value (area) continuously to validate metric behaviour.
    """
    if ticker not in market_df.columns and isinstance(market_df.columns, pd.MultiIndex):
        try: price_series = market_df['Adj Close'][ticker]
        except: price_series = market_df.xs('Close', level=1, axis=1)[ticker]
    elif ticker in market_df.columns: price_series = market_df[ticker]
    else: return go.Figure()
        
    signal_series = signals_df[ticker]
    full_name = get_asset_name(ticker)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=price_series.index, y=price_series, name=f"{full_name} Price", line=dict(color='black', width=1)), secondary_y=False)
    fig.add_trace(go.Scatter(x=signal_series.index, y=signal_series, name=f"{full_name} Signal", line=dict(color='#00CC96', width=1.5, dash='dot'), fill='tozeroy', fillcolor='rgba(0, 204, 150, 0.1)'), secondary_y=True)

    fig.update_layout(
        title={'text': f"<b>Price vs Signal Analysis</b> ({full_name})", 'y':0.9, 'x':0.5, 'xanchor': 'center'},
        template="plotly_white", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=40, r=40, t=80, b=40)
    )
    fig.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="Signal Value", secondary_y=True, showgrid=False) 

    return fig

# ==============================================================================
# TAB 4: RISK & HEDGE
# ==============================================================================

def plot_hedge_ratio(hedge_series):
    """
    Plots the intensity of the portfolio short over time as a percentage.
    """
    data = hedge_series.copy()
    data.index = pd.to_datetime(data.index)
    data = data.abs().fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, y=data, mode='lines', name='Hedge Level', 
        line=dict(color='#EF553B', width=2), fill='tozeroy', fillcolor='rgba(239, 85, 59, 0.1)'
    ))
    fig.update_layout(
        title={'text': "<b>Hedge Ratio Evolution</b><br><span style='font-size:12px'>Intensity of Portfolio Protection (Absolute %)</span>", 'y':0.9, 'x':0.5, 'xanchor': 'center'},
        xaxis_title="Date", yaxis_title="Hedge Intensity (%)", yaxis=dict(tickformat=".0%", range=[0, 1.1]), 
        template="plotly_white", margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig

def plot_hedge_impact(hedge_series, market_df):
    """
    Computes and plots the cumulative PnL originating strictly from the hedging overlay.
    """
    if isinstance(market_df.columns, pd.MultiIndex): spy_prices = market_df['Adj Close']['SPY']
    else: spy_prices = market_df['SPY'] if 'SPY' in market_df.columns else market_df.iloc[:, 0]
        
    spy_ret = spy_prices.pct_change()
    common_idx = hedge_series.index.intersection(spy_ret.index)
    
    aligned_hedge = hedge_series.loc[common_idx]
    aligned_spy = spy_ret.loc[common_idx]
    
    hedge_contrib = (aligned_hedge.shift(1) * aligned_spy).fillna(0)
    cumulative_impact = hedge_contrib.cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative_impact.index, y=cumulative_impact, mode='lines', name='Hedge PnL',
        line=dict(color='#636EFA', width=2), fill='tozeroy',
    ))

    fig.update_layout(
        title={'text': "<b>Cumulative Hedge Impact</b><br><span style='font-size:12px'>Cost or Gain generated by the Short Position</span>", 'y':0.9, 'x':0.5, 'xanchor': 'center'},
        xaxis_title="Date", yaxis_title="Cumulative Return", yaxis=dict(tickformat=".1%"), template="plotly_white", margin=dict(l=40, r=40, t=80, b=40)
    )
    fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.5, line_dash="dash")
    
    return fig

def plot_rolling_volatility(nav_series, benchmark_series, window=21):
    """
    Calculates rolling historical annualized volatility.
    """
    vol_port = nav_series.pct_change().rolling(window).std() * (252**0.5)
    vol_bench = benchmark_series.pct_change().rolling(window).std() * (252**0.5)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vol_port.index, y=vol_port, name="Strategy Volatility"))
    fig.add_trace(go.Scatter(x=vol_bench.index, y=vol_bench, name="SPY Volatility", line=dict(dash='dot')))
    return fig