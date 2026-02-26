import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# We import the classes to allow temporary instantiation within plot functions
from src.derivatives.pricing_model import EuropeanOption

# ==============================================================================
# EUROPEAN OPTION ANALYTICS
# ==============================================================================

def plot_payoff_european(option, spot_range):
    """
    Generates an interactive Plotly chart showing the theoretical P&L profile at maturity.
    Overlays Client (Long Option) and Bank (Short Option) perspectives.
    """
    spots = np.linspace(spot_range[0], spot_range[1], 100)
    premium = option.price()
    
    if option.option_type == "call":
        intrinsic_value = np.maximum(spots - option.K, 0)
    else:
        intrinsic_value = np.maximum(option.K - spots, 0)

    pnl_client = intrinsic_value - premium
    pnl_bank = premium - intrinsic_value
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=spots, y=pnl_client, 
        mode='lines', 
        name=f'Client (Long {option.option_type.title()})', 
        line=dict(color='green', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=spots, y=pnl_bank, 
        mode='lines', 
        name=f'Bank (Short {option.option_type.title()})', 
        line=dict(color='red', width=3)
    ))

    fig.add_hline(y=0, line_color="white", line_width=1, opacity=0.5)

    fig.add_vline(
        x=option.K, 
        line_dash="dash", line_color="gray", 
        annotation_text=f"Strike ({option.K:.1f})", annotation_position="top left"
    )

    fig.add_vline(
        x=option.S, 
        line_dash="dot", line_color="cyan", 
        annotation_text=f"Current Spot ({option.S:.1f})", annotation_position="bottom right"
    )

    fig.update_layout(
        title=f" ",
        xaxis_title="Underlying price at maturity",
        yaxis_title="Profit / Loss (€)",
        template="plotly_dark", 
        hovermode="x unified",   
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def plot_price_vs_strike_european(option, current_spot):
    """
    Plots the theoretical option price sensitivity across varying Strike bounds (Moneyness).
    """
    strikes = np.linspace(current_spot * 0.5, current_spot * 1.5, 100)
    prices = []
    
    for k in strikes:
        temp_opt = EuropeanOption(
            S=option.S, K=k, T=option.T, r=option.r, sigma=option.sigma, q=option.q, option_type=option.option_type
        )
        prices.append(temp_opt.price())
        
    current_price = option.price()
    current_k = option.K
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=strikes, y=prices, 
        mode='lines', 
        name='Theoric Prices',
        line=dict(color='royalblue', width=2)))
    
    fig.add_trace(go.Scatter(
        x=[current_k], y=[current_price],
        mode='markers',
        name='Your Selection',
        marker=dict(color='red', size=12, line=dict(color='white', width=2))))
    
    fig.add_vline(x=current_spot, line_dash="dot", line_color="gray", annotation_text="Current Spot")

    fig.update_layout(
        title=" ", 
        xaxis_title="Strike Price",
        yaxis_title="Option Price (€)",
        template="plotly_white",
        height=300,
        margin=dict(l=40, r=20, t=10, b=40),
        hovermode="x unified",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    return fig


def plot_price_vs_vol_european(option, current_vol):
    """Plots the theoretical option price sensitivity to Implied Volatility (Vega impact)."""
    vols = np.linspace(0.05, 0.80, 50)
    prices = []
    
    for v in vols:
        tmp = EuropeanOption(S=option.S, K=option.K, T=option.T, r=option.r, sigma=v, q=option.q, option_type=option.option_type)
        prices.append(tmp.price())
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vols*100, y=prices, mode='lines', name='Price', line=dict(color='orange', width=3)))
    
    curr_price = option.price()
    fig.add_trace(go.Scatter(x=[current_vol*100], y=[curr_price], mode='markers', name='Current Vol', 
                             marker=dict(color='red', size=12, line=dict(color='white', width=2))))
    
    fig.update_layout(
        title=" ", 
        xaxis_title="Volatility (%)",
        yaxis_title="Option Price (€)",
        template="plotly_white",
        height=300,
        margin=dict(l=40, r=20, t=10, b=40),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    return fig


def plot_greeks_profile_european(option):
    """
    Generates structural risk graphs displaying Delta, Gamma, and Vega across the spot domain.
    Used to identify areas of peak convexity and volatility exposure.
    """
    lower_bound = 0.01
    upper_bound = option.K * 2.0
    spots = np.linspace(lower_bound, upper_bound, 100)
    
    deltas, gammas, vegas = [], [], []
    
    current_S = option.S
    
    for s in spots:
        option.S = s
        d = option.delta()
        g = option.gamma()
        v = option.vega_point()
        
        deltas.append(-d)
        gammas.append(-g)
        vegas.append(-v)
        
    option.S = current_S
    
    curr_vals = {
        'Delta': -option.delta(),
        'Gamma': -option.gamma(),
        'Vega': -option.vega_point()
    }

    fig = make_subplots(
        rows=3, cols=1, 
        subplot_titles=("Delta (Δ)", "Gamma (Γ)", "Vega (ν)"),
        shared_xaxes=True,
        vertical_spacing=0.05
    )

    def add_trace_with_markers(row, col, x_data, y_data, name, current_val):
        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=name, 
                                 line=dict(color='#1f77b4', width=2), showlegend=False), 
                      row=row, col=col)
        
        fig.add_vline(x=option.K, line_width=1, line_dash="dash", line_color="gray", row=row, col=col)
        
        if row == 3: 
             fig.add_annotation(x=option.K, y=min(y_data), text="Strike", showarrow=False, yshift=-10, font=dict(size=10, color="gray"), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=[current_S], y=[current_val], mode='markers', 
            marker=dict(color='red', size=8, symbol='circle'),
            name="Current", showlegend=False
        ), row=row, col=col)

    add_trace_with_markers(1, 1, spots, deltas, "Delta", curr_vals['Delta'])
    add_trace_with_markers(2, 1, spots, gammas, "Gamma", curr_vals['Gamma'])
    add_trace_with_markers(3, 1, spots, vegas, "Vega", curr_vals['Vega'])

    fig.update_layout(height=700, title_text="Greeks Structural Profile (Bank View)", margin=dict(t=60, b=20, l=20, r=20))
    fig.update_xaxes(title_text="Spot Price", range=[0, upper_bound], row=3, col=1)
    
    return fig


def plot_risk_profile_european(option, spot_range):
    """
    Displays simultaneous secondary market risks (Gamma & Vega) as a function of the Spot.
    Indicates the most complex hedging regions (Hedging Difficulty View).
    """
    spots = np.linspace(spot_range[0], spot_range[1], 100)
    gammas = []
    vegas = []
    
    for s in spots:
        temp_opt = EuropeanOption(
            S=s, K=option.K, T=option.T, r=option.r, sigma=option.sigma, q=option.q, option_type=option.option_type
        )
        gammas.append(temp_opt.gamma())
        vegas.append(temp_opt.vega_point()) 
        
    current_gamma = option.gamma()
    current_vega = option.vega_point()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=spots, y=gammas, mode='lines', name='Gamma (Convexity)', line=dict(color='crimson', width=3)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=spots, y=vegas, mode='lines', name='Vega (Vol Risk)', line=dict(color='royalblue', width=2, dash='dash')),
        secondary_y=True
    )

    fig.add_trace(
        go.Scatter(x=[option.S], y=[current_gamma], mode='markers', name='Mon Gamma', marker=dict(color='crimson', size=10)),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=[option.S], y=[current_vega], mode='markers', name='Mon Vega', marker=dict(color='royalblue', size=10)),
        secondary_y=True
    )

    fig.update_layout(
        title="Hedging Difficulties: Gamma & Vega Sensitivity",
        xaxis_title="Spot Price (Scenarios)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Gamma", title_font=dict(color="crimson"), tickfont=dict(color="crimson"), secondary_y=False)
    fig.update_yaxes(title_text="Vega", title_font=dict(color="royalblue"), tickfont=dict(color="royalblue"), secondary_y=True)
    
    fig.add_vline(x=option.S, line_dash="dot", line_color="gray", annotation_text="Current Spot")

    return fig


# ==============================================================================
# PHOENIX STRUCTURE ANALYTICS
# ==============================================================================

def plot_payoff_phoenix(struct, spot_range=None):
    """Plots the Phoenix theoretical payout profile at maturity across spot levels."""
    prot_level = struct.protection_barrier 
    cpn_level = struct.coupon_barrier
    
    low_bound = min(struct.S * 0.3, prot_level * 0.8)
    high_bound = struct.S * 1.5
    spots = np.linspace(low_bound, high_bound, 200)
    payoffs = []
    
    for s in spots:
        if s >= cpn_level:
            val = 1.0 + struct.coupon_rate 
        elif s >= prot_level:
            val = 1.0
        else:
            val = s / struct.S
        
        payoffs.append(val * 100) 

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=spots, y=payoffs, 
        mode='lines', 
        name=' ',
        line=dict(color='#00CC96', width=3),
        hovertemplate="Spot: %{x:.2f}<br>Payoff: %{y:.1f}%<extra></extra>"
    ))

    fig.add_vline(x=prot_level, line_dash="dash", line_color="red", 
                  annotation_text=f"Prot: {prot_level:.2f}")
    
    fig.add_vline(x=cpn_level, line_dash="dash", line_color="orange", 
                  annotation_text=f"Cpn: {cpn_level:.2f}", annotation_position="top")

    fig.update_layout(
        title=" ",
        xaxis_title="Spot Price at Maturity",
        yaxis_title="Payoff (% Nominal)",
        template="plotly_white",
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified"
    )
    return fig


def plot_price_vs_strike_phoenix(struct, current_spot):
    """Plots Phoenix price as a function of the Spot level (Moneyness sensitivity)."""
    spots = np.linspace(current_spot * 0.5, current_spot * 1.5, 50)
    prices = []
    
    original_S = struct.S
    
    for s in spots:
        struct.S = s
        prices.append(struct.price())
        
    struct.S = original_S
    current_price = struct.price()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=prices, mode='lines', name='Price', line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=[current_spot], y=[current_price], mode='markers', name='Current Spot', marker=dict(color='red', size=10)))
    
    fig.update_layout(
        title=" ",
        xaxis_title="Spot Price",
        yaxis_title="Phoenix Price",
        template="plotly_white",
        height=300,
        margin=dict(l=40, r=20, t=30, b=40)
    )
    return fig


def plot_price_vs_vol_phoenix(struct, current_vol):
    """Plots Phoenix price sensitivity to implied Volatility (Vega behavior overview)."""
    vols = np.linspace(0.05, 0.60, 30) 
    prices = []
    
    original_sigma = struct.sigma
    
    for v in vols:
        struct.sigma = v
        prices.append(struct.price())
        
    struct.sigma = original_sigma
    current_price = struct.price()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vols*100, y=prices, mode='lines', name='Price', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=[current_vol*100], y=[current_price], mode='markers', name='Current Vol', marker=dict(color='red', size=10)))
    
    fig.update_layout(
        title=" ",
        xaxis_title="Volatility (%)",
        yaxis_title="Phoenix Price",
        template="plotly_white",
        height=300,
        margin=dict(l=40, r=20, t=30, b=40)
    )
    return fig


def plot_phoenix_tunnel(struct):
    """
    Visualizes Monte Carlo paths grouped and color-coded by their final payout scenario.
    Creates boolean masks to identify Autocall (Early Exit), Capital Protection (Maturity), and Loss paths.
    """
    original_N = struct.N
    struct.N = 1000 
    
    paths = struct.generate_paths()
    obs_indices = struct.get_observation_indices()
    
    obs_prices = paths[obs_indices] 
    
    autocall_mask = np.any(obs_prices >= struct.autocall_barrier, axis=0)
    
    final_prices = paths[-1]
    crash_mask = (~autocall_mask) & (final_prices < struct.protection_barrier)
    safe_mask = (~autocall_mask) & (final_prices >= struct.protection_barrier)
    
    fig = go.Figure()
    
    max_lines = 200
    x_axis = np.arange(paths.shape[0])
    
    def add_lines(mask, color, name, opacity):
        indices = np.where(mask)[0]
        if len(indices) == 0: return
        selected = indices[:max_lines]
        
        x_flat = []
        y_flat = []
        for idx in selected:
            x_flat.extend(x_axis)
            x_flat.append(None) 
            y_flat.extend(paths[:, idx])
            y_flat.append(None)
        
        fig.add_trace(go.Scatter(
            x=x_flat, y=y_flat, 
            mode='lines', 
            line=dict(color=color, width=1), 
            opacity=opacity,
            name=name,
            showlegend=True
        ))

    add_lines(autocall_mask, 'green', 'Autocall (Early Exit)', 0.15)
    add_lines(safe_mask, 'gray', 'Maturity (Capital Protected)', 0.4)
    add_lines(crash_mask, 'red', 'Loss (Barrier Hit)', 0.6)
    
    days = paths.shape[0] - 1
    fig.add_hline(y=struct.autocall_barrier, line_dash="dash", line_color="green", annotation_text="Autocall Lvl")
    fig.add_hline(y=struct.protection_barrier, line_dash="dash", line_color="red", annotation_text="Protection Lvl")
    if struct.coupon_barrier != struct.protection_barrier:
        fig.add_hline(y=struct.coupon_barrier, line_dash="dot", line_color="cyan", annotation_text="Coupon Lvl")
        
    for idx in obs_indices:
        fig.add_vline(x=idx, line_width=1, line_color="white", opacity=0.2)
        
    n_auto, n_safe, n_crash = np.sum(autocall_mask), np.sum(safe_mask), np.sum(crash_mask)
    stats_text = (
        f"<b>SCENARIOS (N={struct.N})</b><br>"
        f"<span style='color:green'>Autocall: {n_auto} ({n_auto/struct.N:.1%})</span><br>"
        f"<span style='color:gray'>Mature: {n_safe} ({n_safe/struct.N:.1%})</span><br>"
        f"<span style='color:red'>Loss: {n_crash} ({n_crash/struct.N:.1%})</span>"
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.99, y=0.99,
        text=stats_text,
        showarrow=False,
        align="right",
        bgcolor="rgba(0,0,0,0.8)",
        bordercolor="white",
        borderwidth=1
    )

    fig.update_layout(
        title="Monte Carlo Path Analysis (Tunnel)",
        xaxis_title="Trading Days",
        yaxis_title="Spot Price",
        template="plotly_dark",
        showlegend=True
    )
    
    struct.N = original_N
    return fig


def plot_phoenix_distribution(struct):
    """
    Plots a histogram of the discounted Monte Carlo payoffs.
    Illustrates the Value-at-Risk and statistical distribution of returns.
    """
    payoffs = struct.calculate_payoffs_distribution()
    mean_price = np.mean(payoffs)
    
    payoffs_pct = (payoffs / struct.nominal) * 100
    mean_pct = (mean_price / struct.nominal) * 100
    
    fig = px.histogram(
        x=payoffs_pct, 
        nbins=60, 
        title=f"Payoff Distribution (Fair Value: {mean_pct:.2f}%)",
        color_discrete_sequence=['skyblue']
    )
    
    fig.add_vline(x=mean_pct, line_color="red", line_dash="dash", annotation_text=f"Fair Value")
    fig.add_vline(x=100, line_color="green", line_dash="dot", annotation_text="Initial Cap")

    fig.update_layout(
        xaxis_title="Payoff (% Nominal)",
        yaxis_title="Frequency",
        template="plotly_dark",
        bargap=0.1
    )
    return fig


def plot_mc_noise_distribution(struct):
    """
    Runs multiple Monte Carlo pricing loops with distinct random seeds 
    to evaluate variance, standard deviation, and convergence stability (MC Noise).
    """
    n_experiments = 30 
    prices = []
    
    original_seed = struct.seed
    
    for i in range(n_experiments):
        struct.seed = i 
        prices.append(struct.price())
        
    struct.seed = original_seed
    
    prices = np.array(prices)
    prices_pct = (prices / struct.nominal) * 100
    mean = np.mean(prices_pct)
    std = np.std(prices_pct)
    
    fig = px.histogram(
        x=prices_pct,
        nbins=15,
        title=f"Monte Carlo Convergence Noise (Std Dev: {std:.2f}%)",
        color_discrete_sequence=['gray']
    )
    
    fig.add_vline(x=mean, line_color="red", line_dash="dash", annotation_text=f"Mean: {mean:.2f}%")
    
    fig.add_vrect(
        x0=mean - 1.96*std, x1=mean + 1.96*std,
        fillcolor="yellow", opacity=0.1,
        annotation_text="95% Confidence"
    )

    fig.update_layout(
        xaxis_title="Price Estimate (% Nominal)",
        yaxis_title="Count",
        template="plotly_dark",
        bargap=0.1
    )
    return fig


# ==============================================================================
# BACKTESTER ANALYTICS
# ==============================================================================

def plot_pnl(engine):
    """
    Visualizes the Delta Hedging results over the backtest period.
    Displays Spot vs Delta tracking, Stock Rebalancing actions, and Cumulative P&L.
    """
    if engine.results is None or engine.results.empty: 
        return None
    
    df = engine.results.reset_index()
    date_col = df.columns[0] 
    
    df['Trade'] = df['Shares Held'].diff().fillna(0)
    
    display_trade = df['Trade'].copy()
    
    if len(display_trade) > 0:
        display_trade.iloc[0] = 0.0
        
    last_trade_idx = display_trade.to_numpy().nonzero()[0]
    if len(last_trade_idx) > 0:
         display_trade.iloc[last_trade_idx[-2]] = 0.0

    df['Buy_Qty'] = display_trade.apply(lambda x: x if x > 0 else np.nan)
    df['Sell_Qty'] = display_trade.apply(lambda x: x if x < 0 else np.nan)
    
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        row_heights=[0.35, 0.30, 0.35],
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=(
            "Position vs Spot", 
            "Rebalancing Activity", 
            "Cumulative P&L"
        )
    )

    fig.add_trace(go.Scatter(
        x=df[date_col], y=df['Spot'], 
        name="Spot Price (Left Axis)", 
        line=dict(color='#1f77b4', width=2)
    ), row=1, col=1, secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df['Shares Held'], 
        name="Shares Held / Delta (Right Axis)", 
        line=dict(color='orange', dash='dot', width=2)
    ), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Bar(
        x=df[date_col], y=df['Buy_Qty'], 
        name="Buy Stock (Rebalancing)", 
        marker_color='#2ca02c',
        opacity=0.7
    ), row=2, col=1, secondary_y=False)
    
    fig.add_trace(go.Bar(
        x=df[date_col], y=df['Sell_Qty'], 
        name="Sell Stock (Rebalancing)", 
        marker_color='#d62728',
        opacity=0.7
    ), row=2, col=1, secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df['Spot'], 
        name="Spot Trend (Ref)", 
        line=dict(color='white', width=1, dash='solid'), 
        opacity=0.3,
        showlegend=True
    ), row=2, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df[date_col], y=df['Cumulative P&L'], 
        name="Total P&L", 
        line=dict(color='#00CC96', width=2), 
        fill='tozeroy'
    ), row=3, col=1)

    fig.update_layout(
        height=900, 
        template="plotly_dark", 
        hovermode="x unified", 
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1
        )
    )
    
    fig.update_yaxes(title_text="Spot (€)", color='#1f77b4', row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Qty Shares", color='orange', row=1, col=1, secondary_y=True, showgrid=False)
    
    fig.update_yaxes(title_text="Trade Qty", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Spot Lvl", color='gray', row=2, col=1, secondary_y=True, showgrid=False)
    
    fig.update_yaxes(title_text="P&L (€)", row=3, col=1)
    
    return fig


# ==============================================================================
# DISPATCHERS (To unify the API)
# ==============================================================================

def plot_payoff(product, spot_range=None):
    if type(product).__name__ == "EuropeanOption":
        return plot_payoff_european(product, spot_range)
    return plot_payoff_phoenix(product, spot_range)

def plot_price_vs_strike(product, current_spot):
    if type(product).__name__ == "EuropeanOption":
        return plot_price_vs_strike_european(product, current_spot)
    return plot_price_vs_strike_phoenix(product, current_spot)

def plot_price_vs_vol(product, current_vol):
    if type(product).__name__ == "EuropeanOption":
        return plot_price_vs_vol_european(product, current_vol)
    return plot_price_vs_vol_phoenix(product, current_vol)

def plot_greeks_profile(product):
    if type(product).__name__ == "EuropeanOption":
        return plot_greeks_profile_european(product)
    return None

def plot_risk_profile(product, spot_range):
    if type(product).__name__ == "EuropeanOption":
        return plot_risk_profile_european(product, spot_range)
    return None


# ==============================================================================
# BASE INSTRUMENT ANALYTICS (from FinancialInstrument)
# ==============================================================================

def plot_risk_matrix(product, spot_range_pct=0.10, vol_range_pct=0.05, n_spot_steps=5, n_vol_steps=3):
    """
    Generates Heatmaps with independent X/Y dimensions.
    """
    spot_moves = np.linspace(-spot_range_pct, spot_range_pct, n_spot_steps)
    vol_moves = np.linspace(-vol_range_pct, vol_range_pct, n_vol_steps)

    original_S = product.S
    original_sigma = product.sigma

    base_price = product.price()
    base_greeks = product.greeks() 
    base_delta = base_greeks.get('delta', 0.0)

    z_unhedged = np.zeros((len(vol_moves), len(spot_moves)))
    z_hedged = np.zeros((len(vol_moves), len(spot_moves)))

    for i, v_chg in enumerate(vol_moves):
        for j, s_chg in enumerate(spot_moves):
            product.S = original_S * (1 + s_chg)
            product.sigma = max(0.01, original_sigma + v_chg)

            new_price = product.price()
            pnl_option = -(new_price - base_price) 
            pnl_hedge = base_delta * (product.S - original_S)

            z_unhedged[i, j] = pnl_option
            z_hedged[i, j] = pnl_option + pnl_hedge

    product.S = original_S
    product.sigma = original_sigma

    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("1. Unhedged P&L", "2. Delta-Hedged P&L (Gamma/Vega)"),
        horizontal_spacing=0.15
    )

    x_labels = [f"{m*100:+.1f}%" for m in spot_moves]
    y_labels = [f"{v*100:+.1f}%" for v in vol_moves]

    fig.add_trace(go.Heatmap(
        z=z_unhedged, x=x_labels, y=y_labels,
        colorscale='RdYlGn', zmid=0, 
        showscale=True, 
        colorbar=dict(title="P&L (€)", x=-0.15),
        texttemplate="%{z:.2f}", textfont={"size":10} 
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=z_hedged, x=x_labels, y=y_labels,
        colorscale='RdYlGn', zmid=0, 
        showscale=True,
        texttemplate="%{z:.2f}", textfont={"size":10},
        colorbar=dict(title="P&L (€)", x=1.02)
    ), row=1, col=2)

    fig.update_layout(
        title="Dynamic Risk Matrices",
        xaxis_title="Spot Variation", 
        yaxis_title="Volatility Variation",
        template="plotly_dark",
        height=500
    )
    fig.update_xaxes(title_text="Spot Variation", row=1, col=2)

    return fig

def plot_pnl_attribution(product, spot_move_pct, vol_move_pct, days_passed=0):
    """
    Explains P&L via Taylor Expansion (Delta, Gamma, Vega, Theta).
    """
    original_S, original_sigma, original_T = product.S, product.sigma, product.T
    base_price = product.price()
    greeks = product.greeks()

    dt = days_passed / 365.0
    dS = original_S * spot_move_pct
    pnl_vega = (greeks['vega'] * (vol_move_pct * 100)) * -1 

    pos_sign = -1
    pnl_delta = (greeks['delta'] * dS) * pos_sign
    pnl_gamma = (0.5 * greeks['gamma'] * (dS**2)) * pos_sign
    pnl_theta = (greeks['theta'] * days_passed) * pos_sign

    predicted_pnl = pnl_delta + pnl_gamma + pnl_vega + pnl_theta

    product.S = original_S * (1 + spot_move_pct)
    product.sigma = original_sigma + vol_move_pct
    product.T = max(0.001, original_T - dt)

    new_price = product.price()
    actual_pnl = (new_price - base_price) * pos_sign
    unexplained = actual_pnl - predicted_pnl

    product.S, product.sigma, product.T = original_S, original_sigma, original_T

    categories = ["Delta", "Gamma", "Vega", "Theta", "Unexplained", "Predicted Total", "Actual Total"]
    values = [pnl_delta, pnl_gamma, pnl_vega, pnl_theta, unexplained, predicted_pnl, actual_pnl]

    colors = []
    for i, val in enumerate(values):
        if i >= 5:
            colors.append('#3366CC') 
        else:
            colors.append('#2ECC40' if val >= 0 else '#FF4136') 

    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f} €" for v in values],
        textposition='auto'
    ))

    fig.add_hline(y=0, line_color="white", line_width=1)

    fig.update_layout(
        title=f"P&L Attribution (Spot {spot_move_pct:+.1%}, Vol {vol_move_pct:+.1%}, {days_passed}j)",
        template="plotly_dark",
        yaxis_title="Profit / Loss (€)",
        showlegend=False
    )

    return fig
