import pandas as pd
import numpy as np
import plotly.graph_objects as go

def calculate_kpis(portfolio_nav, benchmark_data, benchmark_label="Benchmark"):
    """
    Calculates Key Performance Indicators (KPIs) for the portfolio and benchmark.
    Uses dynamic label for the benchmark column.
    """
    # 1. Align Dates
    common_index = portfolio_nav.index.intersection(benchmark_data.index)
    port = portfolio_nav.loc[common_index]
    bench = benchmark_data.loc[common_index]
    
    # Calculate Daily Returns
    port_rets = port.pct_change().dropna()
    bench_rets = bench.pct_change().dropna()
    
    def get_metrics(series, rets):
        if series.empty: return 0, 0, 0, 0, 0
        
        # Total Return
        total_ret = (series.iloc[-1] / series.iloc[0]) - 1
        
        # CAGR (Compound Annual Growth Rate)
        days = (series.index[-1] - series.index[0]).days
        years = days / 365.25
        cagr = (series.iloc[-1] / series.iloc[0])**(1/years) - 1 if years > 0 else 0
        
        # Volatility (Annualized)
        vol = rets.std() * np.sqrt(252)
        
        # Sharpe Ratio (assuming Risk Free Rate ~ 0 for simplicity)
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() != 0 else 0
        
        # Max Drawdown
        rolling_max = series.cummax()
        drawdown = (series / rolling_max) - 1
        max_dd = drawdown.min()
        
        return total_ret, cagr, vol, sharpe, max_dd

    # Compute metrics
    p_tot, p_cagr, p_vol, p_sharpe, p_dd = get_metrics(port, port_rets)
    b_tot, b_cagr, b_vol, b_sharpe, b_dd = get_metrics(bench, bench_rets)
    
    # Create KPI DataFrame with dynamic column name
    metrics = {
        'Metric': ['Total Return', 'CAGR', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown'],
        'Strategy': [f"{p_tot:.2%}", f"{p_cagr:.2%}", f"{p_vol:.2%}", f"{p_sharpe:.2f}", f"{p_dd:.2%}"],
        benchmark_label: [f"{b_tot:.2%}", f"{b_cagr:.2%}", f"{b_vol:.2%}", f"{b_sharpe:.2f}", f"{b_dd:.2%}"],
        'Alpha (Diff)': [f"{(p_tot - b_tot):.2%}", f"{(p_cagr - b_cagr):.2%}", f"{(p_vol - b_vol):.2%}", f"{(p_sharpe - b_sharpe):.2f}", f"{(p_dd - b_dd):.2%}"]
    }
    
    return pd.DataFrame(metrics)

def plot_equity_curve(portfolio_nav, benchmark_data, benchmark_ticker="Benchmark"):
    """
    Generates an interactive Plotly chart comparing Strategy vs Benchmark.
    Benchmark is rebased to match the Strategy's initial capital.
    """
    # 1. Align Data
    common_index = portfolio_nav.index.intersection(benchmark_data.index)
    port_series = portfolio_nav.loc[common_index]
    bench_series_raw = benchmark_data.loc[common_index]
    
    # 2. Rebase Benchmark
    initial_capital = port_series.iloc[0]
    initial_bench_price = bench_series_raw.iloc[0]
    scale_factor = initial_capital / initial_bench_price
    bench_series_scaled = bench_series_raw * scale_factor

    # 3. Plot
    fig = go.Figure()

    # Strategy Trace
    fig.add_trace(go.Scatter(
        x=port_series.index, 
        y=port_series,
        mode='lines',
        name='AlphaStream Strategy',
        line=dict(color='#00CC96', width=2),
        hovertemplate='$%{y:,.0f} (Strategy)<extra></extra>'
    ))

    # Benchmark Trace (Dynamic Name)
    fig.add_trace(go.Scatter(
        x=bench_series_scaled.index, 
        y=bench_series_scaled,
        mode='lines',
        name=benchmark_ticker, # Uses the label passed from interface
        line=dict(color='#EF553B', width=2, dash='dot'),
        hovertemplate=f'$%{{y:,.0f}} ({benchmark_ticker})<extra></extra>'
    ))

    # Layout
    fig.update_layout(
        title={
            'text': f"<b>Equity Curve vs {benchmark_ticker}</b> (Rebased)",
            'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1
        ),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig

def plot_returns_distribution(portfolio_nav, benchmark_data, benchmark_label="Benchmark"):
    """
    Plots the distribution (histogram) of daily returns for Strategy vs Benchmark.
    Includes vertical lines for the mean return of each.
    """
    # 1. Data Prep
    common_index = portfolio_nav.index.intersection(benchmark_data.index)
    port_rets = portfolio_nav.loc[common_index].pct_change().dropna()
    bench_rets = benchmark_data.loc[common_index].pct_change().dropna()

    mean_port = port_rets.mean()
    mean_bench = bench_rets.mean()

    fig = go.Figure()

    # Benchmark Histogram (Background)
    fig.add_trace(go.Histogram(
        x=bench_rets,
        name=benchmark_label,
        marker_color='#EF553B',
        opacity=0.5, # Transparent to see overlaps
        nbinsx=50,   # Number of bars
        histnorm='probability' # Normalize to show % frequency
    ))

    # Strategy Histogram (Foreground)
    fig.add_trace(go.Histogram(
        x=port_rets,
        name='AlphaStream Strategy',
        marker_color='#00CC96',
        opacity=0.6,
        nbinsx=50,
        histnorm='probability'
    ))

    # 2. Add Vertical Lines for Means
    # Shorten label for annotation if too long
    short_label = (benchmark_label[:12] + '..') if len(benchmark_label) > 12 else benchmark_label
    
    fig.add_vline(x=mean_bench, line_width=2, line_dash="dot", line_color="#EF553B", 
                  annotation_text=f"{short_label} Avg", annotation_position="top left")
    
    fig.add_vline(x=mean_port, line_width=2, line_dash="dot", line_color="#00CC96", 
                  annotation_text="Strat Avg", annotation_position="top right")

    # Layout
    fig.update_layout(
        title={
            'text': "<b>Daily Returns Distribution</b> (Risk Profile)",
            'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        xaxis_title="Daily Return",
        yaxis_title="Probability (Frequency)",
        template="plotly_white",
        barmode='overlay', # Crucial for overlapping
        legend=dict(orientation="h", y=1.02, x=1),
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis=dict(tickformat=".1%") # Format axis as percentage
    )

    return fig

def plot_drawdown_underwater(portfolio_nav):
    """
    Generates an Underwater Plot showing the depth and duration of drawdowns.
    (Usually focuses only on the Strategy to analyze its specific pain points)
    """
    # 1. Calculate Drawdown
    rolling_max = portfolio_nav.cummax()
    drawdown = (portfolio_nav / rolling_max) - 1

    fig = go.Figure()

    # Area Chart
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        mode='lines',
        name='Drawdown',
        fill='tozeroy', # Fills the area to the X-axis (zero)
        line=dict(color='#d62728', width=1), # Red color
        hovertemplate='Drawdown: %{y:.2%}<extra></extra>'
    ))

    # Layout
    fig.update_layout(
        title={
            'text': "<b>Underwater Plot</b> (Drawdown Severity)",
            'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        yaxis=dict(tickformat=".0%"), # Axis as percentage (0%, -5%, -10%)
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig

def plot_monthly_heatmap(portfolio_nav):
    """
    Generates a Heatmap of monthly returns (Year vs Month).
    Standard institutional view to check for seasonality and consistency.
    """
    # 1. Resample to Monthly Returns
    monthly_ret = portfolio_nav.resample('ME').last().pct_change().dropna()
    
    # 2. Prepare Data for Heatmap
    # Extract Year and Month
    monthly_ret_df = pd.DataFrame(monthly_ret)
    monthly_ret_df['Year'] = monthly_ret_df.index.year
    monthly_ret_df['Month'] = monthly_ret_df.index.strftime('%b') # Jan, Feb...
    monthly_ret_df['Month_int'] = monthly_ret_df.index.month # To sort correctly
    
    # Pivot Table: Rows=Year, Cols=Month
    pivot_table = monthly_ret_df.pivot_table(values=portfolio_nav.name, index='Year', columns=['Month_int', 'Month'])
    
    # Clean up columns (remove the int used for sorting)
    pivot_table.columns = pivot_table.columns.droplevel(0)
    
    # Ensure columns are in order (Jan to Dec)
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot_table = pivot_table.reindex(columns=months_order)
    
    # Reverse Index so current year is at top
    pivot_table = pivot_table.sort_index(ascending=False)

    # 3. Create Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='RdYlGn', # Red to Green
        zmid=0, # Center the color scale at 0%
        text=[[f"{val:.1%}" if not np.isnan(val) else "" for val in row] for row in pivot_table.values],
        texttemplate="%{text}", # Show numbers inside cells
        textfont={"size": 10},
        hoverongaps=False,
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2%}<extra></extra>'
    ))

    # Layout
    fig.update_layout(
        title={
            'text': "<b>Monthly Returns Heatmap</b> (Consistency Check)",
            'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        xaxis_title="Month",
        yaxis_title="Year",
        template="plotly_white",
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    return fig

def plot_rolling_sharpe(portfolio_nav, benchmark_data, window_months=6, benchmark_label="Benchmark"):
    """
    Plots the Rolling Sharpe Ratio over a sliding window (default 6 months).
    Detects if the strategy is deteriorating (Alpha Decay).
    """
    # 1. Data Prep (Daily Returns)
    common_index = portfolio_nav.index.intersection(benchmark_data.index)
    port_rets = portfolio_nav.loc[common_index].pct_change().dropna()
    bench_rets = benchmark_data.loc[common_index].pct_change().dropna()
    
    # Convert months to trading days (approx 21 days/month)
    window = window_months * 21
    
    # 2. Calculate Rolling Sharpe (Annualized)
    # Mean / Std * sqrt(252)
    def get_rolling_sharpe(series):
        return (series.rolling(window).mean() / series.rolling(window).std()) * np.sqrt(252)

    roll_sharpe_port = get_rolling_sharpe(port_rets)
    roll_sharpe_bench = get_rolling_sharpe(bench_rets)
    
    # Drop NaNs created by the window
    roll_sharpe_port = roll_sharpe_port.dropna()
    roll_sharpe_bench = roll_sharpe_bench.dropna()

    # 3. Plot
    fig = go.Figure()

    # Strategy Trace
    fig.add_trace(go.Scatter(
        x=roll_sharpe_port.index,
        y=roll_sharpe_port,
        mode='lines',
        name='Rolling Sharpe (Strategy)',
        line=dict(color='#00CC96', width=2)
    ))

    # Benchmark Trace (Dynamic Name)
    fig.add_trace(go.Scatter(
        x=roll_sharpe_bench.index,
        y=roll_sharpe_bench,
        mode='lines',
        name=f'Rolling Sharpe ({benchmark_label})',
        line=dict(color='#EF553B', width=1.5, dash='dot')
    ))

    # Add Zero Line (Risk Free / Danger Zone)
    fig.add_hline(y=0, line_width=1, line_color="black", line_dash="solid")
    fig.add_hline(y=1, line_width=1, line_color="gray", line_dash="dot", annotation_text="Good (>1)", annotation_position="bottom right")

    # Layout
    fig.update_layout(
        title={
            'text': f"<b>{window_months}-Month Rolling Sharpe Ratio</b> (Stability)",
            'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig