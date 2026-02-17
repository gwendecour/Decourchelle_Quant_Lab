import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from src.alpha.universe import get_asset_name
import plotly.express as px
import scipy.stats as stats

def calculate_kpis(portfolio_nav, benchmark_data, benchmark_label="Benchmark"):
    """
    Calculates KPIs including Calmar Ratio.
    """
    # 1. Align Dates
    common_index = portfolio_nav.index.intersection(benchmark_data.index)
    port = portfolio_nav.loc[common_index]
    bench = benchmark_data.loc[common_index]
    
    port_rets = port.pct_change().dropna()
    bench_rets = bench.pct_change().dropna()
    
    def get_metrics(series, rets):
        if series.empty: return 0, 0, 0, 0, 0, 0 # Ajout d'un 0 pour le Calmar
        
        # Total Return
        total_ret = (series.iloc[-1] / series.iloc[0]) - 1
        
        # CAGR
        days = (series.index[-1] - series.index[0]).days
        years = days / 365.25
        cagr = (series.iloc[-1] / series.iloc[0])**(1/years) - 1 if years > 0 else 0
        
        # Volatility
        vol = rets.std() * np.sqrt(252)
        
        # Sharpe
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() != 0 else 0
        
        # Max Drawdown
        rolling_max = series.cummax()
        drawdown = (series / rolling_max) - 1
        max_dd = drawdown.min()

        # Calmar Ratio (CAGR / Abs(MaxDD))
        # Si MaxDD est 0 (impossible mais bon), on met 0
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        return total_ret, cagr, vol, sharpe, max_dd, calmar

    # Compute metrics (on récupère maintenant 6 valeurs)
    p_tot, p_cagr, p_vol, p_sharpe, p_dd, p_calmar = get_metrics(port, port_rets)
    b_tot, b_cagr, b_vol, b_sharpe, b_dd, b_calmar = get_metrics(bench, bench_rets)
    
    # Create KPI DataFrame
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
    Affiche la distribution continue (KDE) des rendements.
    Ajoute une boîte de statistiques pour comparer les moments (Mean, Vol, Skew, Kurt).
    """
    # 1. Data Prep & Nettoyage
    # On aligne les dates pour une comparaison équitable
    common_index = portfolio_nav.index.intersection(benchmark_data.index)
    
    # On calcule les rendements quotidiens
    port_rets = portfolio_nav.loc[common_index].pct_change().dropna()
    bench_rets = benchmark_data.loc[common_index].pct_change().dropna()

    # 2. Calcul des Statistiques (Les 4 moments)
    stats_dict = {
        'Strat': {
            'Mean': port_rets.mean(),
            'Vol': port_rets.std(),
            'Skew': port_rets.skew(),
            'Kurt': port_rets.kurtosis()
        },
        'Bench': {
            'Mean': bench_rets.mean(),
            'Vol': bench_rets.std(),
            'Skew': bench_rets.skew(),
            'Kurt': bench_rets.kurtosis()
        }
    }

    # 3. Préparation KDE (Kernel Density Estimation)
    # On définit l'axe X (Min des deux séries à Max des deux séries avec marge)
    min_x = min(port_rets.min(), bench_rets.min()) - 0.01
    max_x = max(port_rets.max(), bench_rets.max()) + 0.01
    x_range = np.linspace(min_x, max_x, 500) # 500 points pour une courbe lisse

    # Génération des courbes
    kde_strat = stats.gaussian_kde(port_rets)(x_range)
    kde_bench = stats.gaussian_kde(bench_rets)(x_range)

    fig = go.Figure()

    # 4. Tracé Benchmark (Arrière-plan, Orange)
    fig.add_trace(go.Scatter(
        x=x_range, y=kde_bench,
        mode='lines',
        name=f'{benchmark_label} (KDE)',
        line=dict(color='#EF553B', width=2),
        fill='tozeroy', # Remplissage
        fillcolor='rgba(239, 85, 59, 0.2)' # Transparent
    ))

    # 5. Tracé Strategy (Premier plan, Vert)
    fig.add_trace(go.Scatter(
        x=x_range, y=kde_strat,
        mode='lines',
        name='Strategy (KDE)',
        line=dict(color='#00CC96', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 204, 150, 0.3)'
    ))

    # 6. Lignes Verticales (Moyennes)
    # On prend la hauteur max de la courbe pour dimensionner la ligne
    y_max = max(max(kde_strat), max(kde_bench))
    
    fig.add_vline(x=stats_dict['Bench']['Mean'], line_dash="dot", line_color="#EF553B", opacity=0.8)
    fig.add_vline(x=stats_dict['Strat']['Mean'], line_dash="dash", line_color="#00CC96", opacity=1.0)

    # 7. La "Stats Box" (Tableau d'analyse dans le graph)
    # On formate le texte HTML pour l'annotation
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
        xref="paper", yref="paper",
        x=0.98, y=0.98, # Coin haut droit
        xanchor="right", yanchor="top",
        text=stats_text,
        showarrow=False,
        align="left",
        font=dict(family="Courier New, monospace", size=12, color="black"),
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="black",
        borderwidth=1
    )

    # 8. Layout Final
    fig.update_layout(
        title={
            'text': "<b>Return Distribution Analysis (KDE)</b>",
            'y':0.9, 'x':0.05, 'xanchor': 'left'
        },
        xaxis_title="Daily Return",
        yaxis_title="Probability Density",
        template="plotly_white",
        xaxis=dict(
            tickformat=".1%", 
            range=[min_x, max_x],
            zeroline=True, zerolinewidth=1, zerolinecolor='grey'
        ),
        yaxis=dict(showticklabels=False), # On cache l'échelle Y (densité abstraite)
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=20, r=20, t=60, b=20),
        height=450
    )

    return fig

def plot_drawdown_underwater(portfolio_nav, benchmark_data, benchmark_label="Benchmark"):
    """
    Generates an Underwater Plot comparing Strategy vs Benchmark drawdowns.
    Uses EXACTLY the same data alignment logic as plot_equity_curve.
    """
    # 1. Align Data (Même logique que ta fonction plot_equity_curve)
    common_index = portfolio_nav.index.intersection(benchmark_data.index)
    port_series = portfolio_nav.loc[common_index]
    bench_series = benchmark_data.loc[common_index]

    # 2. Calculate Drawdowns
    # Formule : (Prix / Plus_Haut_Historique) - 1
    
    # Pour la Stratégie
    p_rolling_max = port_series.cummax()
    p_drawdown = (port_series / p_rolling_max) - 1

    # Pour le Benchmark
    b_rolling_max = bench_series.cummax()
    b_drawdown = (bench_series / b_rolling_max) - 1

    # 3. Plot
    fig = go.Figure()

    # Trace Stratégie (En PREMIER = Arrière-plan)
    # On met une aire remplie (Area)
    fig.add_trace(go.Scatter(
        x=p_drawdown.index,
        y=p_drawdown,
        mode='lines',
        name='Strategy',
        fill='tozeroy', # Remplit l'espace jusqu'à 0
        line=dict(color='#00CC96', width=1.5), 
        fillcolor='rgba(0, 204, 150, 0.2)', # Vert transparent
        hovertemplate='Strategy: %{y:.2%}<extra></extra>'
    ))

    # Trace Benchmark (En SECOND = Premier plan)
    # On met juste une ligne (pour bien voir la différence)
    fig.add_trace(go.Scatter(
        x=b_drawdown.index,
        y=b_drawdown,
        mode='lines',
        name=f'{benchmark_label}',
        line=dict(color='#EF553B', width=2, dash='dot'), # Rouge pointillé
        hovertemplate=f'{benchmark_label}: %{{y:.2%}}<extra></extra>'
    ))

    # Layout
    fig.update_layout(
        title={
            'text': f"<b>Underwater Plot</b> (Drawdown vs {benchmark_label})",
            'y':0.9, 'x':0.35, 'xanchor': 'center', 'yanchor': 'top'
        },
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        yaxis=dict(tickformat=".0%"), # Format axe Y en pourcentage
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1
        ),
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
        name='Strategy',
        line=dict(color='#00CC96', width=2)
    ))

    # Benchmark Trace (Dynamic Name)
    fig.add_trace(go.Scatter(
        x=roll_sharpe_bench.index,
        y=roll_sharpe_bench,
        mode='lines',
        name=f'{benchmark_label}',
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
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1
        ),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig

def plot_dynamic_allocation(history_df, universe_dict):
    """
    Trace l'évolution de l'allocation d'actifs par classe (Actions, Bonds, Commo, Cash).
    Prend en entrée le DataFrame 'results' qui contient maintenant les colonnes des tickers.
    """
    fig = go.Figure()
    
    # 1. Identifier les classes d'actifs
    # On inverse le dictionnaire universe: {Ticker: Classe}
    ticker_to_class = {}
    for cat, tickers in universe_dict.items():
        for t in tickers:
            ticker_to_class[t] = cat
    ticker_to_class['CASH'] = 'Cash' # Sécurité
    
    # 2. Agréger les poids par classe jour par jour
    # On ne garde que les colonnes qui sont des tickers (pas NAV, Date, etc.)
    excluded_cols = ['NAV', 'Daily_Ret', 'Benchmark_NAV']
    ticker_cols = [c for c in history_df.columns if c not in excluded_cols]
    
    # On crée un DF temporaire pour sommer par classe
    alloc_df = pd.DataFrame(index=history_df.index)
    
    # Initialisation des classes à 0
    # On s'assure que 'Cash' est dans la liste des classes
    classes = list(universe_dict.keys())
    if 'Cash' not in classes:
        classes.append('Cash')

    for c in classes:
        alloc_df[c] = 0.0
        
    # Remplissage
    for t in ticker_cols:
        if t in history_df.columns:
            # On trouve la classe du ticker
            asset_class = ticker_to_class.get(t, 'Cash') # Par défaut Cash si inconnu
            # On ajoute le poids du ticker à sa classe
            alloc_df[asset_class] = alloc_df[asset_class].fillna(0) + history_df[t].fillna(0)
            
    # 3. Tracer le Stacked Area Chart
    # --- CORRECTION ICI (Le Cash est maintenant un vrai Hex Code Gris) ---
    colors = {
        'Actions': '#00CC96',       # Vert
        'Bonds': '#636EFA',         # Bleu
        'Commodities': '#EF553B',   # Rouge/Orange
        'Cash': '#808080'           # Gris (au lieu de #Gray)
    }
    
    for col in alloc_df.columns:
        # On ne trace que si la colonne n'est pas vide (somme > 0)
        if alloc_df[col].sum() > 0.001: # Petite tolérance
            fig.add_trace(go.Scatter(
                x=alloc_df.index,
                y=alloc_df[col],
                mode='lines',
                name=col,
                stackgroup='one', # C'est ça qui fait l'empilement
                line=dict(width=0.5),
                fillcolor=colors.get(col, '#d3d3d3'), # Couleur par défaut gris clair si inconnue
                marker=dict(color=colors.get(col, '#d3d3d3')), # Couleur de la ligne/légende
                hovertemplate=f'{col}: %{{y:.1%}}<extra></extra>'
            ))

    fig.update_layout(
        title={'text': "<b>Dynamic Asset Allocation</b> (Evolution by Asset Class)", 'y':0.9, 'x':0.5, 'xanchor': 'center'},
        xaxis_title="Date",
        yaxis_title="Weight",
        yaxis=dict(tickformat=".0%", range=[0, 1.05]), # Max 100% (ou un peu plus si levier)
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    return fig

def plot_asset_rotation_heatmap(history_df, top_n_display=10):
    """
    Crée une Heatmap montrant l'évolution du poids de chaque actif individuel.
    Permet de visualiser la 'Rotation' du momentum (ex: passage de la Tech à l'Energie).
    """
    # 1. Nettoyage des données
    excluded_cols = ['NAV', 'Daily_Ret', 'Benchmark_NAV', 'CASH']
    # On garde seulement les colonnes tickers présents dans l'historique
    ticker_cols = [c for c in history_df.columns if c not in excluded_cols]
    
    if not ticker_cols:
        return go.Figure()

    # 2. Resampling Mensuel (pour ne pas avoir 1000 barres)
    # On prend la moyenne des poids sur le mois pour lisser
    monthly_weights = history_df[ticker_cols].resample('ME').mean()
    
    # 3. Filtrage : On ne garde que les actifs qui ont eu une importance significative
    # On garde les N actifs avec la plus grosse somme de poids historique
    top_tickers = monthly_weights.sum().sort_values(ascending=False).head(top_n_display).index
    plot_data = monthly_weights[top_tickers].T # Transposé pour avoir Tickers en Y et Dates en X
    
    y_labels = [get_asset_name(t) for t in plot_data.index]
    # 4. Plot
    fig = go.Figure(data=go.Heatmap(
        z=plot_data.values,
        x=plot_data.columns,
        y=y_labels,
        colorscale='Blues', # ou 'Viridis', 'Magma'
        zmin=0, zmax=0.5, # On sature à 50% pour bien voir les variations
        colorbar=dict(title='Weight'),
        hovertemplate='Date: %{x}<br>Asset: %{y}<br>Weight: %{z:.1%}<extra></extra>'
    ))

    fig.update_layout(
        title={'text': "<b>Asset Rotation Heatmap</b> (Top Holdings History)", 'y':0.9, 'x':0.5, 'xanchor': 'center'},
        xaxis_title="Date",
        yaxis_title="Ticker",
        template="plotly_white",
        margin=dict(l=40, r=40, t=80, b=40),
        height=500 # Un peu plus haut pour lire les tickers
    )
    
    return fig


def plot_allocation_donut(snapshot_series):
    """
    Donut chart pour une date donnée (snapshot).
    Prend une pd.Series (une ligne du dataframe results).
    """
    # Nettoyage des colonnes non-tickers
    excluded_cols = ['NAV', 'Daily_Ret', 'Benchmark_NAV']
    
    # On filtre pour ne garder que les tickers présents dans la série
    weights = snapshot_series.drop(labels=[c for c in excluded_cols if c in snapshot_series.index])
    
    # On enlève les poids nuls ou négatifs (short cash) pour le camembert
    weights = weights[weights > 0.01] # Seuil d'affichage 1%
    labels_list = [get_asset_name(t) for t in weights.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels_list,
        values=weights.values,
        hole=.4, # Donut style
        textinfo='label+percent',
        hoverinfo='label+percent+value'
    )])
    
    date_str = snapshot_series.name.strftime('%Y-%m-%d') if hasattr(snapshot_series.name, 'strftime') else str(snapshot_series.name)
    
    fig.update_layout(
        title={
            'text': f"<b>Allocation Snapshot</b><br><span style='font-size:12px'>Date: {date_str}</span>", 
            'y':0.9, 'x':0.5, 'xanchor': 'center'
        },
        template="plotly_white",
        margin=dict(l=40, r=40, t=80, b=40),
        showlegend=False # On a déjà les labels sur le donut
    )
    
    return fig

def get_holdings_table(snapshot_series, universe_dict):
    """
    Retourne un DataFrame propre des positions pour une date donnée.
    """
    excluded_cols = ['NAV', 'Daily_Ret', 'Benchmark_NAV']
    
    # Création de la liste
    holdings = []
    
    # Helper pour trouver la classe
    ticker_to_class = {}
    for cat, tickers in universe_dict.items():
        for t in tickers:
            ticker_to_class[t] = cat
    ticker_to_class['CASH'] = 'Cash/Hedge'

    # NAV du jour pour calculer la valeur notionnelle
    nav = snapshot_series.get('NAV', 0)

    for ticker in snapshot_series.index:
        if ticker not in excluded_cols:
            weight = snapshot_series[ticker]
            # On affiche tout ce qui est significatif (positif ou short significatif)
            if abs(weight) > 0.001: 
                holdings.append({
                    "Asset": get_asset_name(ticker),
                    "Class": ticker_to_class.get(ticker, "Unknown"),
                    "Weight": weight,
                    "Value ($)": weight * nav
                })
    
    df = pd.DataFrame(holdings)
    if not df.empty:
        df = df.sort_values(by="Weight", ascending=False)
    return df

def plot_contribution_chart(weights_df, market_df, universe_dict, hedge_ratio_series=None):
    """
    Trace la contribution cumulée.
    Version corrigée : Force l'affichage du Hedge même si 0 ou mal aligné.
    """
    # 1. Préparation Market Data
    if isinstance(market_df.columns, pd.MultiIndex):
        if 'Adj Close' in market_df.columns:
             market_prices = market_df['Adj Close']
        else:
             market_prices = market_df
    else:
        market_prices = market_df

    asset_returns = market_prices.pct_change().fillna(0)
    
    # 2. Alignement Dates
    # On base l'index commun sur les poids (ce que le backtest a généré)
    common_index = weights_df.index.intersection(asset_returns.index)
    
    # On filtre tout sur cet index commun
    w_df = weights_df.loc[common_index]
    r_df = asset_returns.loc[common_index]
    
    # Shift des poids (J-1)
    w_shifted = w_df.shift(1).fillna(0)
    
    cumulative_contrib = pd.DataFrame(index=common_index)
    
    # --- A. Contributions Longs ---
    classes = list(universe_dict.keys())
    for cat in classes:
        cat_daily_contrib = pd.Series(0.0, index=common_index)
        tickers = [t for t in universe_dict[cat] if t in w_shifted.columns and t in r_df.columns]
        
        for t in tickers:
            contrib = w_shifted[t] * r_df[t]
            cat_daily_contrib = cat_daily_contrib.add(contrib, fill_value=0)
            
        cumulative_contrib[cat] = cat_daily_contrib.cumsum()

    # --- B. Contribution Hedge (CORRECTION ICI) ---
    # 1. On initialise la colonne à 0 par défaut pour qu'elle existe toujours
    cumulative_contrib['Hedge (Short)'] = 0.0

    if hedge_ratio_series is not None:
        try:
            # 2. On force l'alignement sur l'index commun (Reindex est plus sûr que intersection)
            h_series = hedge_ratio_series.reindex(common_index).fillna(0)
            
            # 3. Shift du Ratio (J-1)
            h_shifted = h_series.shift(1).fillna(0)
            
            # 4. Récupération SPY
            if 'SPY' in r_df.columns:
                spy_ret = r_df['SPY']
            else:
                spy_ret = r_df.iloc[:, 0]
                
            # 5. Calcul PnL Hedge
            hedge_daily = h_shifted * (-spy_ret)
            cumulative_contrib['Hedge (Short)'] = hedge_daily.cumsum()
            
        except Exception as e:
            print(f"Error calculating hedge contrib: {e}")
            # En cas d'erreur, ça reste à 0, mais la colonne existe

    # --- C. Contribution Cash ---
    risk_free_rate_daily = 0.04 / 252 
    if 'CASH' in w_shifted.columns:
        cash_weights = w_shifted['CASH']
    else:
        long_weights_sum = w_shifted.sum(axis=1)
        cash_weights = (1.0 - long_weights_sum).clip(lower=0) # Sécurité pour ne pas avoir de cash négatif bizarre

    cumulative_contrib['Cash'] = (cash_weights * risk_free_rate_daily).cumsum()

    # 4. Plot
    fig = go.Figure()
    
    colors = {
        'Actions': '#00CC96',
        'Bonds': '#636EFA',
        'Commodities': '#EF553B',
        'Cash': '#D3D3D3',          # Gris Clair pour le Cash
        'Hedge (Short)': '#AB63FA'  # Violet pour le Hedge
    }

    # Ordre force : Cash en bas, Hedge en haut ou à la fin
    desired_order = ['Cash'] + classes + ['Hedge (Short)']
    
    # On additionne tout pour la ligne de contrôle
    total_model = cumulative_contrib.sum(axis=1)
    
    for col in desired_order:
        # On ne trace que si la colonne existe (ce qui est garanti maintenant pour Hedge et Cash)
        if col in cumulative_contrib.columns:
            # Sécurité : Si tout est à 0 (ex: Hedge désactivé), on l'affiche quand même plat
            final_val = cumulative_contrib[col].iloc[-1]
            
            fig.add_trace(go.Scatter(
                x=cumulative_contrib.index,
                y=cumulative_contrib[col],
                mode='lines',
                name=col,
                line=dict(width=2, color=colors.get(col, 'grey')),
                hovertemplate=f'<b>{col}</b>: %{{y:.2%}}<br>Total Contrib: {final_val:.2%}<extra></extra>'
            ))

    # Ligne Totale de vérification (Pointillés Noirs)
    fig.add_trace(go.Scatter(
        x=total_model.index,
        y=total_model,
        mode='lines',
        name='Total Model Sum',
        line=dict(width=1, color='black', dash='dot'),
        opacity=0.5,
        hoverinfo='skip' 
    ))

    fig.update_layout(
        title={'text': "<b>Performance Attribution</b> (Cumulative Contribution)", 'y':0.9, 'x':0.5, 'xanchor': 'center'},
        xaxis_title="Date",
        yaxis_title="Cumulative Contribution",
        yaxis=dict(tickformat=".1%"),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    return fig

def plot_monthly_contribution(history_df, market_df, universe_dict):
    """
    Trace la contribution mensuelle (Bar Chart) de chaque classe d'actif.
    Permet de voir l'impact tactique : 'Ce mois-ci, les actions m'ont coûté -2%'.
    """
    # 1. Préparation des rendements quotidiens des actifs
    asset_returns = market_df.pct_change()
    
    # 2. Alignement
    common_index = history_df.index.intersection(asset_returns.index)
    weights_df = history_df.loc[common_index]
    returns_df = asset_returns.loc[common_index]
    
    # 3. Calcul des Contributions Quotidiennes (Poids J-1 * Return J)
    daily_contrib = pd.DataFrame(index=common_index)
    
    # Mapping inversé : Ticker -> Classe
    ticker_to_class = {}
    for cat, tickers in universe_dict.items():
        for t in tickers:
            ticker_to_class[t] = cat
    
    classes = list(universe_dict.keys())
    
    for cat in classes:
        # On initialise la série à 0
        cat_daily = pd.Series(0.0, index=common_index)
        
        tickers = [t for t in universe_dict[cat] if t in weights_df.columns and t in returns_df.columns]
        for t in tickers:
            w = weights_df[t].shift(1).fillna(0)
            r = returns_df[t]
            cat_daily += w * r
        
        daily_contrib[cat] = cat_daily

    # 4. Aggrégation Mensuelle (Somme des contributions quotidiennes)
    # On utilise 'ME' (Month End) pour pandas récents, ou 'M' pour anciens
    monthly_contrib = daily_contrib.resample('ME').sum()
    
    # 5. Plot (Bar Chart Empilé)
    fig = go.Figure()
    
    colors = {'Actions': '#00CC96', 'Bonds': '#636EFA', 'Commodities': '#EF553B', 'Cash': '#808080'}

    for col in monthly_contrib.columns:
        fig.add_trace(go.Bar(
            x=monthly_contrib.index,
            y=monthly_contrib[col],
            name=col,
            marker_color=colors.get(col, None),
            hovertemplate=f'{col}: %{{y:.2%}}<extra></extra>'
        ))

    # Ajout de la courbe de Total Net Return pour voir la somme
    total_monthly = monthly_contrib.sum(axis=1)
    fig.add_trace(go.Scatter(
        x=monthly_contrib.index,
        y=total_monthly,
        mode='markers+lines',
        name='Total Net',
        line=dict(color='black', width=1, dash='dot'),
        marker=dict(symbol='diamond', size=6, color='black'),
        hovertemplate='Total: %{y:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title={'text': "<b>Monthly Performance Attribution</b> (Weighted Contribution)", 'y':0.9, 'x':0.5, 'xanchor': 'center'},
        xaxis_title="Month",
        yaxis_title="Monthly Contribution",
        barmode='relative', # Permet d'avoir des barres positives ET négatives empilées
        yaxis=dict(tickformat=".1%"),
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    return fig

# --- SIGNAL ANALYTICS ---

def calculate_all_signals(market_df, signal_method, lookback=126):
    """
    Recalcule les signaux historiques pour tout l'univers.
    Gère Z-Score, Distance MA, et RSI.
    """
    # 1. Extraction des prix (Gère le MultiIndex de yfinance ou un DataFrame simple)
    if isinstance(market_df.columns, pd.MultiIndex):
        # Si on a Open, High, Low, Close... on prend Adj Close
        try:
            prices = market_df['Adj Close']
        except KeyError:
             # Fallback si Adj Close n'existe pas
            prices = market_df.xs('Close', level=1, axis=1) 
    else:
        prices = market_df

    # Initialisation
    signals = pd.DataFrame(index=prices.index, columns=prices.columns)
    
    # 2. Calcul selon la méthode
    if signal_method == 'z_score':
        # (Prix - Moyenne) / Ecart-type
        rolling_mean = prices.rolling(window=lookback).mean()
        rolling_std = prices.rolling(window=lookback).std()
        signals = (prices - rolling_mean) / rolling_std
        
    elif signal_method == 'distance_ma':
        # (Prix - MA) / MA
        ma = prices.rolling(window=lookback).mean()
        signals = (prices - ma) / ma
        
    elif signal_method == 'rsi':
        # RSI 14 jours (Standard)
        # On utilise 14 jours par défaut pour le RSI même si le lookback momentum est long
        rsi_window = 14 
        delta = prices.diff()
        
        # Séparation gains/pertes
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Moyennes mobiles exponentielles (Wilder's Smoothing est le standard RSI)
        # Mais pour simplifier et être rapide : Rolling Mean simple ou EWM
        avg_gain = gain.rolling(window=rsi_window).mean()
        avg_loss = loss.rolling(window=rsi_window).mean()
        
        rs = avg_gain / avg_loss
        signals = 100 - (100 / (1 + rs))
        
    # On nettoie les NaN du début (période de chauffe)
    return signals.dropna()

def plot_signal_race(signals_df, highlight_assets=None, signal_method="z_score"):
    """
    Line chart interactif "Spotlight".
    - highlight_assets: liste des tickers à mettre en valeur.
    - Les autres sont en gris (ghost lines).
    """
    fig = go.Figure()
    
    if highlight_assets is None:
        highlight_assets = []
        
    # 1. TRACER LE FOND (GHOST LINES)
    # On trace d'abord les non-sélectionnés pour qu'ils soient en arrière-plan
    background_assets = [c for c in signals_df.columns if c not in highlight_assets]
    
    for col in background_assets:
        # Petite sécurité si colonne vide
        if signals_df[col].isna().all(): continue
            
        fig.add_trace(go.Scatter(
            x=signals_df.index,
            y=signals_df[col],
            mode='lines',
            name=get_asset_name(col),
            line=dict(color='#cccccc', width=1.5), # Gris clair, très fin
            opacity=0.6, # Très transparent
            showlegend=False, # On ne pollue pas la légende avec le bruit de fond
            hoverinfo='skip' # On désactive le hover sur le fond pour ne pas gêner
        ))

    # 2. TRACER LES SELECTIONS (HIGHLIGHTS)
    # On utilise la palette qualitative de Plotly pour avoir des couleurs distinctes
    colors = pc.qualitative.Plotly 
    
    for i, col in enumerate(highlight_assets):
        if col in signals_df.columns:
            color = colors[i % len(colors)] # Rotation des couleurs
            
            fig.add_trace(go.Scatter(
                x=signals_df.index,
                y=signals_df[col],
                mode='lines',
                name=get_asset_name(col),
                line=dict(color=color, width=2), # Couleur vive, trait épais
                opacity=1.0, # Opaque
                hovertemplate=f'<b>{col}</b>: %{{y:.2f}}<extra></extra>'
            ))
        
    # --- Lignes de Référence (Seuils) ---
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
        else:
            y_title = "% Distance to MA"

    fig.update_layout(
        title={
            'text': f"<b>Signal Evolution Race</b> (Spotlight View)", 
            'y':0.9, 'x':0.5, 'xanchor': 'center'
        },
        xaxis_title="Date",
        yaxis_title=y_title,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=150, t=80, b=40),
        legend=dict(orientation="v", y=1, x=1.02, xanchor='left', yanchor='top', font=dict(size=10)    # Optionnel : réduire un peu la taille si tu as beaucoup d'assets
                    )
    )
    return fig

def plot_signal_ranking_bar(signals_df, target_date, actual_selections=None):
    """
    Crée un Bar Chart horizontal montrant le classement avec les noms complets.
    Couleurs : Vert = Sélectionné, Gris = Non retenu.
    """
    fig = go.Figure()

    try:
        # 1. Gestion des Dates
        target_ts = pd.Timestamp(target_date)
        if target_ts not in signals_df.index:
            idx_pos = signals_df.index.get_indexer([target_ts], method='pad')[0]
            if idx_pos == -1: return go.Figure()
            actual_date = signals_df.index[idx_pos]
        else:
            actual_date = target_ts
        
        # 2. Tri des données
        row_sorted = signals_df.loc[actual_date].sort_values(ascending=True).dropna()
        if row_sorted.empty: return go.Figure()

    except Exception:
        return go.Figure()

    # 3. Préparation des Labels (Noms complets) et Couleurs
    y_labels = [get_asset_name(t) for t in row_sorted.index]
    
    colors = []
    status_texts = []
    
    for ticker in row_sorted.index:
        val = row_sorted[ticker]
        
        # Cas 1 : L'actif est dans le portefeuille final
        if actual_selections and ticker in actual_selections:
            colors.append('#00CC96')  # VERT (Selection)
            status_texts.append(f"SELECTED<br>Score: {val:.2f}")
            
        # Cas 2 : L'actif a un score positif mais a été rejeté
        elif val > 0:
            colors.append('#BDC3C7')  # GRIS CLAIR (Skipped)
            status_texts.append(f"SKIPPED")
            
        # Cas 3 : Score négatif
        else:
            colors.append('#E6B0AA')  # ROUGE PÂLE (Negatif)
            status_texts.append(f"NEGATIVE<br>Score: {val:.2f}")

    # 4. Plot
    fig.add_trace(go.Bar(
        x=row_sorted.values,
        y=y_labels, # <--- UTILISATION DES NOMS COMPLETS ICI
        orientation='h',
        marker_color=colors,
        text=row_sorted.values.round(2),
        textposition='auto',
        hovertext=status_texts,
        hovertemplate='<b>%{y}</b><br>%{hovertext}<extra></extra>'
    ))

    # 5. Layout
    dynamic_height = max(250, len(row_sorted) * 30)
    
    fig.update_layout(
        title=" ",
        xaxis_title=None,
        yaxis=dict(
            autorange=True,
            tickfont=dict(size=11) # Ajuste la taille si les noms sont très longs
        ), 
        template="plotly_white",
        margin=dict(l=10, r=10, t=10, b=20),
        height=dynamic_height,
        showlegend=False
    )
    
    return fig

# --- FONCTION 2 : PLOT CORRELATION MATRIX (Compact) ---
def plot_correlation_matrix(market_df, ranking_df, threshold=0.7, window=60):
    """
    Affiche la corrélation compacte.
    Optimisé pour l'affichage en colonnes multiples (subplots).
    """
    # 1. Récupération des tickers triés
    # Si ranking_df est une Série ou DataFrame, on prend l'index
    sorted_tickers = ranking_df.index.tolist() if hasattr(ranking_df, 'index') else list(ranking_df)
    
    # 2. Nettoyage
    excluded = ['NAV', 'Cash', 'Bench', 'Total']
    sorted_tickers = [t for t in sorted_tickers if t not in excluded]

    if len(sorted_tickers) < 2:
        return None

    # 3. Extraction des Prix
    prices = pd.DataFrame()
    try:
        # Cas MultiIndex
        if isinstance(market_df.columns, pd.MultiIndex):
            target_col = 'Adj Close' if 'Adj Close' in market_df.columns.get_level_values(0) else 'Close'
            valid_tickers = [t for t in sorted_tickers if t in market_df[target_col].columns]
            prices = market_df[target_col][valid_tickers]
        # Cas Simple
        else:
            valid_tickers = [t for t in sorted_tickers if t in market_df.columns]
            prices = market_df.loc[:, valid_tickers]     
    except:
        return None

    if prices.empty: 
        return None

    # 4. Calcul Corrélation
    recent_returns = prices.pct_change().tail(window).dropna(how='all').fillna(0)
    corr_matrix = recent_returns.corr()

    # 5. Gestion des Couleurs (Threshold)
    mask = np.eye(len(corr_matrix), dtype=bool)
    corr_display = corr_matrix.where(~mask, np.nan) # Diagonale NaN

    # Echelle de couleur personnalisée : Blanc partout, Rouge si > Threshold
    # On normalise le threshold entre 0 et 1 (sachant que l'échelle va de -1 à 1)
    # 0 correspond à -1, 0.5 correspond à 0, 1 correspond à 1.
    thresh_norm = (threshold + 1) / 2
    
    colors = [
        [0.0, 'white'],           
        [thresh_norm, 'white'],   # Blanc jusqu'au seuil
        [thresh_norm, '#EF553B'], # Rouge pile au seuil
        [1.0, '#B22222']          # Rouge foncé au max
    ]

    # 6. Heatmap Compact
    fig = px.imshow(
        corr_display,
        text_auto=".2f",
        aspect="equal", # Carré
        color_continuous_scale=colors, 
        zmin=-1, zmax=1
    )
    
    # Layout Compact
    fig.update_layout(
        title=None,
        coloraxis_showscale=False, # Cache la barre de couleur à droite
        margin=dict(l=0, r=0, t=0, b=0), # Zéro marges
        xaxis=dict(side="bottom", showticklabels=True),
        yaxis=dict(showticklabels=True),
        height=450, # Hauteur fixe carrée (approximative)
        plot_bgcolor='rgba(240,240,240, 1)'
    )
    
    # Rotation des labels si beaucoup d'actifs pour lisibilité
    if len(sorted_tickers) > 8:
        fig.update_xaxes(tickangle=-45)
    
    return fig

def plot_signal_vs_price(market_df, signals_df, ticker):
    """
    Dual-Axis Chart: Comparaison directe entre le Prix (Ligne) et le Signal (Aire/Ligne).
    Permet de valider la pertinence du signal pour un actif donné.
    """
    # 1. Extraction des données
    if ticker not in market_df.columns and isinstance(market_df.columns, pd.MultiIndex):
        # Gestion MultiIndex (yfinance)
        try:
            price_series = market_df['Adj Close'][ticker]
        except:
            price_series = market_df.xs('Close', level=1, axis=1)[ticker]
    elif ticker in market_df.columns:
         price_series = market_df[ticker]
    else:
        return go.Figure()
        
    signal_series = signals_df[ticker]
    full_name = get_asset_name(ticker)
    
    # 2. Création du Graphique à double axe
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Trace 1 : Prix (Axe Gauche)
    fig.add_trace(
        go.Scatter(x=price_series.index, y=price_series, name=f"{full_name} Price",
                   line=dict(color='black', width=1)),
        secondary_y=False
    )

    # Trace 2 : Signal (Axe Droit)
    # On utilise une aire remplie pour bien distinguer du prix
    fig.add_trace(
        go.Scatter(x=signal_series.index, y=signal_series, name=f"{full_name} Signal",
                   line=dict(color='#00CC96', width=1.5, dash='dot'),
                   fill='tozeroy', fillcolor='rgba(0, 204, 150, 0.1)'), # Vert transparent
        secondary_y=True
    )

    # 3. Layout
    fig.update_layout(
        title={'text': f"<b>Price vs Signal Analysis</b> ({full_name})", 'y':0.9, 'x':0.5, 'xanchor': 'center'},
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
        
    )
    
    # Titres des axes
    fig.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="Signal Value", secondary_y=True, showgrid=False) # Pas de grille pour le signal pour ne pas surcharger

    return fig


def plot_hedge_ratio(hedge_series):
    """
    Affiche l'évolution du taux de couverture au fil du temps.
    Args:
        hedge_series (pd.Series): Série temporelle du Hedge Ratio.
    """
    # --- NETTOYAGE ET CONVERSION ---
    data = hedge_series.copy()
    
    # 1. On s'assure que les dates sont bien formatées
    data.index = pd.to_datetime(data.index)
    
    # 2. Conversion en valeur ABSOLUE pour l'affichage
    # On veut voir "45% de protection" (barre haute) et non "-45%" (en bas)
    data = data.abs() 
    
    # 3. Nettoyage final (NaN -> 0)
    data = data.fillna(0)
    # -------------------------------

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data,
        mode='lines',
        name='Hedge Level', # Renommé pour plus de clarté
        line=dict(color='#EF553B', width=2),
        fill='tozeroy', 
        fillcolor='rgba(239, 85, 59, 0.1)'
    ))

    fig.update_layout(
        title={'text': "<b>Hedge Ratio Evolution</b><br><span style='font-size:12px'>Intensity of Portfolio Protection (Absolute %)</span>", 'y':0.9, 'x':0.5, 'xanchor': 'center'},
        xaxis_title="Date",
        yaxis_title="Hedge Intensity (%)",
        yaxis=dict(tickformat=".0%", range=[0, 1.1]), # On garde 0-100% car on a mis .abs()
        template="plotly_white",
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig


def plot_monthly_heatmap(nav_series):
    """
    Transforme la NAV en un tableau Rendements Mensuels par Année.
    """
    # Calcul des rendements mensuels
    monthly_ret = nav_series.resample('M').last().pct_change()
    
    # Création d'un DF avec Année en index et Mois en colonnes
    heatmap_data = pd.DataFrame({
        'Year': monthly_ret.index.year,
        'Month': monthly_ret.index.strftime('%b'), # Jan, Feb...
        'Return': monthly_ret.values
    })
    
    # Pivot pour avoir la structure matricielle
    heatmap_matrix = heatmap_data.pivot(index='Year', columns='Month', values='Return')
    
    # Ordonner les mois correctement
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    heatmap_matrix = heatmap_matrix.reindex(columns=months_order)

    fig = px.imshow(
        heatmap_matrix,
        text_auto=".1%",
        aspect="auto",
        color_continuous_scale="RdYlGn", # Rouge vers Vert
        title="Monthly Returns Heatmap",
        origin='lower' # Années récentes en haut ou en bas selon préférence
    )
    return fig


def plot_rolling_volatility(nav_series, benchmark_series, window=21):
    # Volatilité annualisée glissante
    vol_port = nav_series.pct_change().rolling(window).std() * (252**0.5)
    vol_bench = benchmark_series.pct_change().rolling(window).std() * (252**0.5)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vol_port.index, y=vol_port, name="Strategy Volatility"))
    fig.add_trace(go.Scatter(x=vol_bench.index, y=vol_bench, name="SPY Volatility", line=dict(dash='dot')))
    # ... layout standard
    return fig

import scipy.stats as stats
import plotly.graph_objects as go
import numpy as np

def plot_alpha_beta_scatter(nav_series, benchmark_series, view_mode='full'):
    """
    Affiche la régression de la stratégie vs le benchmark.
    view_mode: 'full' (vue d'ensemble) ou 'zoomed' (focus sur l'intercept/alpha).
    """
    # 1. Calcul des rendements (Identique)
    strat_ret = nav_series.pct_change().dropna()
    bench_ret = benchmark_series.pct_change().dropna()
    
    common_idx = strat_ret.index.intersection(bench_ret.index)
    strat_ret = strat_ret.loc[common_idx]
    bench_ret = bench_ret.loc[common_idx]
    
    if len(common_idx) < 30:
        return None

    # 2. Régression (Identique)
    beta, alpha_daily, r_value, p_value, std_err = stats.linregress(bench_ret, strat_ret)
    alpha_annual = (1 + alpha_daily)**252 - 1
    
    # 3. Préparation du Graphique
    x_min, x_max = bench_ret.min(), bench_ret.max()
    # On étend un peu pour que la ligne rouge aille jusqu'au bout
    x_range = np.linspace(x_min - 0.01, x_max + 0.01, 100)
    y_pred = beta * x_range + alpha_daily

    fig = go.Figure()

    # Nuage de points
    fig.add_trace(go.Scatter(
        x=bench_ret,
        y=strat_ret,
        mode='markers',
        name='Daily Returns',
        marker=dict(color='rgba(100, 149, 237, 0.5)', size=6), # Bleu semi-transparent
        hovertemplate='Bench: %{x:.2%}<br>Strat: %{y:.2%}<extra></extra>'
    ))

    # Ligne de Régression
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name=f'Regression (β={beta:.2f})',
        line=dict(color='#FF3131', width=3) # Rouge Vif
    ))

    # Grille Zéro
    fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.5)
    fig.add_vline(x=0, line_width=1, line_color="black", opacity=0.5)

    # --- LOGIQUE D'AFFICHAGE SELON LE MODE ---
    if view_mode == 'zoomed':
        # FOCUS : On zoom sur l'origine
        zoom_range = [-0.002, 0.002]
        
        fig.update_layout(
            title="<b> Alpha Zoom (Intercept Check)</b>",
            xaxis=dict(range=zoom_range, title="Benchmark Return", zeroline=False),
            yaxis=dict(range=zoom_range, title="Strategy Return", zeroline=False),
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            height=400
        )
        
        # Annotation flèche sur l'intercept
        if alpha_daily > 0:
            arrow_col = "#000000"
            txt = "<b>POSITIVE ALPHA</b>"
            ay_offset = 40 # Flèche vers le bas
        else:
            arrow_col = "#EF553B" # Rouge
            txt = "Negative Alpha"
            ay_offset = -40 # Flèche vers le haut
            
        fig.add_annotation(
            x=0, y=alpha_daily,
            xref="x", yref="y",
            text=txt,
            showarrow=True,
            arrowhead=2, arrowsize=1, arrowwidth=2,
            arrowcolor=arrow_col,
            ax=0, ay=ay_offset,
            font=dict(color=arrow_col, size=12)
        )

    else:
        # VUE GLOBALE
        fig.update_layout(
            title="<b> Alpha Generation (Full View)</b>",
            xaxis_title="Benchmark Return",
            yaxis_title="Strategy Return",
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            height=400,
            # La légende reste en haut à gauche
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.8)')
        )
        
        # --- CORRECTION ICI : Déplacement de la boîte de stats ---
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.98, y=0.02,
            xanchor="right", yanchor="bottom",
            text=f"<b>Alpha (Ann): {alpha_annual:+.2%}</b><br>Beta: {beta:.2f}",
            showarrow=False,
            font=dict(size=13, color="black"),
            bgcolor="#DDDDDD", 
            bordercolor="#000000",
            borderwidth=2,
            opacity=0.9
        )
    
    return fig

def plot_hedge_impact(hedge_series, market_df):
    """
    Calcule et affiche le profit/perte (PnL) généré uniquement par la position de hedge.
    Args:
        hedge_series (pd.Series): Série du Hedge Ratio.
        market_df (pd.DataFrame): Données de marché (SPY).
    """
    # 1. Extraction sécurisée du SPY
    # On gère le cas MultiIndex (yfinance standard) ou SingleIndex
    if isinstance(market_df.columns, pd.MultiIndex):
        spy_prices = market_df['Adj Close']['SPY']
    else:
        # Fallback si l'utilisateur a déjà aplati le DF ou si structure différente
        spy_prices = market_df['SPY'] if 'SPY' in market_df.columns else market_df.iloc[:, 0]
        
    spy_ret = spy_prices.pct_change()
    
    # 2. Alignement strict des dates (Intersection)
    common_idx = hedge_series.index.intersection(spy_ret.index)
    
    # On filtre sur les dates communes
    aligned_hedge = hedge_series.loc[common_idx]
    aligned_spy = spy_ret.loc[common_idx]
    
    # 3. PnL du Hedge = Hedge_Ratio(t-1) * -Benchmark_Return(t)
    # Explication : Le hedge décidé hier soir (t-1) subit le rendement du marché d'aujourd'hui (t)
    # Le .shift(1) est CRUCIAL ici.
    hedge_contrib = (aligned_hedge.shift(1) * aligned_spy).fillna(0)
    
    cumulative_impact = hedge_contrib.cumsum()

    fig = go.Figure()
    
    # Zone verte/rouge pour indiquer gain/perte
    fig.add_trace(go.Scatter(
        x=cumulative_impact.index,
        y=cumulative_impact,
        mode='lines',
        name='Hedge PnL',
        line=dict(color='#636EFA', width=2),
        fill='tozeroy',
        # On peut faire un remplissage conditionnel plus tard, mais simple ici :
    ))

    fig.update_layout(
        title={'text': "<b>Cumulative Hedge Impact</b><br><span style='font-size:12px'>Cost or Gain generated by the Short Position</span>", 'y':0.9, 'x':0.5, 'xanchor': 'center'},
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        yaxis=dict(tickformat=".1%"),
        template="plotly_white",
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    # Ligne horizontale à 0 pour repère visuel
    fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.5, line_dash="dash")
    
    return fig