import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.shared.market_data import MarketData
# Les imports suivants ne sont plus strictement nécessaires ici si on passe l'objet option, 
# mais on les garde pour la compatibilité de type si besoin.
from src.derivatives.structured_products import PhoenixStructure
from src.derivatives.pricing_model import EuropeanOption 

class DeltaHedgingEngine:
    def __init__(self, option, market_data, risk_free_rate, dividend_yield, volatility, transaction_cost=0.0, rebalancing_freq="daily"):
        """
        Engine for backtesting delta hedging strategies.
        
        :param option: Instrument object (EuropeanOption or PhoenixStructure) DEJA INSTANCIÉ
        :param market_data: DataFrame containing historical data (must have 'Close')
        :param risk_free_rate: Annual risk-free rate (r)
        :param dividend_yield: Annual dividend yield (q)
        :param volatility: Fixed volatility to use for pricing (sigma)
        :param transaction_cost: Proportional transaction cost (e.g. 0.001 for 0.1%)
        """
        self.option = option
        
        # --- GESTION ROBUSTE DES DONNEES (DataFrame vs Series) ---
        if isinstance(market_data, pd.DataFrame):
            if 'Close' in market_data.columns:
                self.spot_series = market_data['Close']
            else:
                # Fallback: on prend la première colonne si pas de 'Close'
                self.spot_series = market_data.iloc[:, 0]
        elif isinstance(market_data, pd.Series):
            self.spot_series = market_data
        else:
            raise ValueError("market_data must be a pandas DataFrame or Series")
            
        self.r = risk_free_rate
        self.q = dividend_yield
        self.sigma = volatility
        self.tc = transaction_cost
        self.dt = 1/252.0 # Hypothèse journalière
        
        # Résultats
        self.history = []
        self.results = None

    def run_backtest(self):
        spots = self.spot_series.values
        dates = self.spot_series.index
        n_days = len(spots)
        
        # --- INIT PARAMS ---
        contract_maturity = self.option.T 
        
        # Paramètres Phoenix
        is_phoenix = hasattr(self.option, 'autocall_barrier')
        obs_freq = getattr(self.option, 'obs_frequency', 4) 
        days_between_obs = int(252 / obs_freq) if obs_freq > 0 else 99999
        
        ac_barrier = getattr(self.option, 'autocall_barrier', 999999)
        cpn_barrier = getattr(self.option, 'coupon_barrier', 0)
        nominal = getattr(self.option, 'nominal', spots[0])
        coupon_amount = nominal * getattr(self.option, 'coupon_rate', 0) / obs_freq if obs_freq > 0 else 0

        # Containers
        portfolio_values = np.full(n_days, np.nan)
        cash_account = np.zeros(n_days)
        shares_held = np.zeros(n_days)
        deltas = np.zeros(n_days)
        option_prices = np.zeros(n_days)
        
        # --- T=0 (LANCEMENT) ---
        s0 = spots[0]
        self.option.S = s0
        self.option.T = contract_maturity
        self.option.sigma = self.sigma
        
        if hasattr(self.option, 'calculate_delta_quick'):
             init_delta = self.option.calculate_delta_quick(n_sims=5000)
             init_price = self.option.price()
        else:
             init_price = self.option.price()
             init_delta = self.option.delta()
        
        # LOGIQUE DECOMPOSÉE
        initial_premium = init_price
        initial_hedge_cost = (init_delta * s0)
        initial_fees = abs(initial_hedge_cost) * self.tc
        
        # Cash = Prime - Coût Hedge - Frais
        initial_cash = initial_premium - initial_hedge_cost - initial_fees
        
        cash_account[0] = initial_cash
        shares_held[0] = init_delta
        deltas[0] = init_delta
        option_prices[0] = init_price
        portfolio_values[0] = initial_cash + (init_delta * s0) - init_price 
        
        # --- VARIABLES DE SUIVI ---
        product_alive = True
        final_idx = 0
        status = "Running"
        coupons_paid_count = 0
        final_date = dates[-1]
        
        # Cumul des flux pour l'analyse finale
        # On ne compte pas le hedge initial ici, on le rajoutera au "Trading P&L" global
        total_payouts_paid = 0.0 
        total_trans_costs = initial_fees

        for i in range(1, n_days):
            if not product_alive:
                # Freeze
                cash_account[i] = cash_account[i-1]
                portfolio_values[i] = portfolio_values[i-1]
                shares_held[i] = 0
                deltas[i] = 0
                option_prices[i] = 0
                continue 
            
            final_idx = i
            s_curr = spots[i]
            time_passed_years = i / 252.0
            t_remain = contract_maturity - time_passed_years
            
            # --- CHECK MATURITÉ ---
            if t_remain <= 0:
                product_alive = False
                status = "Matured"
                final_date = dates[i]
                
                # A. PAYOFF FINAL
                final_payoff = 0.0
                if is_phoenix:
                    if s_curr >= cpn_barrier: final_payoff += coupon_amount
                    prot_lvl = getattr(self.option, 'protection_barrier', 0)
                    if s_curr >= prot_lvl: final_payoff += nominal
                    else: final_payoff += nominal * (s_curr / s0)
                else:
                    k = getattr(self.option, 'K', 0)
                    otype = getattr(self.option, 'option_type', 'Call')
                    final_payoff = max(s_curr - k, 0) if otype == 'Call' else max(k - s_curr, 0)

                total_payouts_paid += final_payoff

                # B. LIQUIDATION HEDGE
                cash_from_hedge = shares_held[i-1] * s_curr
                fees = abs(cash_from_hedge) * self.tc
                total_trans_costs += fees
                
                # C. CASH FINAL
                final_cash = cash_account[i-1] + cash_from_hedge - fees - final_payoff
                
                cash_account[i] = final_cash
                portfolio_values[i] = final_cash
                shares_held[i] = 0
                deltas[i] = 0
                option_prices[i] = final_payoff
                continue

            # --- VIE DU PRODUIT ---
            is_observation = (i % days_between_obs == 0)
            payout_flow = 0.0
            just_autocalled = False
            
            if is_phoenix and is_observation:
                if s_curr >= ac_barrier:
                    payout_flow = nominal + coupon_amount
                    just_autocalled = True
                    product_alive = False
                    status = "Autocalled"
                    final_date = dates[i]
                    coupons_paid_count += 1
                elif s_curr >= cpn_barrier:
                    payout_flow = coupon_amount
                    coupons_paid_count += 1
            
            total_payouts_paid += payout_flow

            # Pricing
            self.option.S = s_curr
            self.option.T = max(t_remain, 0.0001)
            
            if just_autocalled:
                curr_price = 0.0
                curr_delta = 0.0
            else:
                if hasattr(self.option, 'calculate_delta_quick'):
                     curr_delta = self.option.calculate_delta_quick(n_sims=2000)
                     curr_price = self.option.price()
                else:
                     curr_price = self.option.price()
                     curr_delta = self.option.delta()
            
            # Cash Flow
            prev_cash = cash_account[i-1]
            interest = prev_cash * (np.exp(self.r * self.dt) - 1)
            
            cash_after_payout = prev_cash + interest - payout_flow
            
            # Rebalancement
            prev_shares = shares_held[i-1]
            shares_target = curr_delta
            shares_trade = shares_target - prev_shares
            
            trade_cash_impact = shares_trade * s_curr
            trade_fees = abs(trade_cash_impact) * self.tc
            total_trans_costs += trade_fees
            
            new_cash = cash_after_payout - trade_cash_impact - trade_fees
            
            cash_account[i] = new_cash
            shares_held[i] = shares_target
            deltas[i] = shares_target
            option_prices[i] = curr_price
            portfolio_values[i] = new_cash + (shares_target * s_curr) - curr_price

        # --- RESULTS ---
        self.results = pd.DataFrame({
            'Spot': spots,
            'Option Price': option_prices,
            'Delta': deltas,
            'Shares Held': shares_held,
            'Cash': cash_account,
            'Cumulative P&L': portfolio_values
        }, index=dates).iloc[:final_idx+1]
        
        # --- ANALYSE DECOMPOSITION ---
        total_pnl = self.results['Cumulative P&L'].iloc[-1]
        
        # Trading P&L (Le résultat pur de l'activité d'achat/vente d'actions)
        # On peut le déduire par P&L = Premium - Payouts - Costs + Trading_P&L
        # Donc Trading_P&L = Total_P&L - Premium + Payouts + Costs
        
        trading_pnl_gross = total_pnl - initial_premium + total_payouts_paid + total_trans_costs
        
        if len(self.results) > 1:
            log_returns = np.log(self.results['Spot'] / self.results['Spot'].shift(1)).dropna()
            realized_vol = log_returns.std() * np.sqrt(252)
        else:
            realized_vol = 0.0
            
        pnl_diff = np.diff(self.results['Cumulative P&L'])
        std_hedge_error = np.std(pnl_diff) if len(pnl_diff) > 0 else 0.0
        
        duration_months = (final_idx / 21.0)

        metrics = {
            'Total P&L': total_pnl,
            'Hedge Error Std': std_hedge_error,
            'Total Transaction Costs': total_trans_costs,
            'Realized Volatility': realized_vol,
            'Pricing Volatility': self.sigma,
            'Option Premium': initial_premium,
            'Total Payouts': total_payouts_paid,   # Somme de tout ce qu'on a payé au client
            'Trading P&L (Gross)': trading_pnl_gross, # Ce qu'on a gagné en spéculant sur le Gamma (avant frais)
            'Status': status,
            'Duration (Months)': duration_months,
            'Coupons Paid': coupons_paid_count,
            'Final Date': final_date.strftime("%Y-%m-%d")
        }
        
        return self.results, metrics

    def _calculate_greeks_at_date(self, ctx):
        p = ctx['params']
        s = ctx['spot']
        t = ctx['T']
        vol = ctx['vol']
        prod_type = ctx['type']

        if prod_type in ['call', 'put']:
            opt = EuropeanOption(
                S=s, K=ctx['strike'], T=t, 
                r=p.get('r', 0.05), sigma=vol, q=p.get('q', 0.0), 
                option_type=prod_type
            )
            return opt.price(), opt.delta(), opt.gamma(), opt.vega_point(), opt.daily_theta()

        elif prod_type == 'phoenix':
            def pricing_kernel(spot_val, vol_val, time_val):
                safe_s = spot_val if spot_val > 0 else 1.0
                rel_auto = ctx['auto'] / safe_s
                rel_prot = ctx['prot'] / safe_s
                rel_coup = ctx['coup'] / safe_s
                
                phx = PhoenixStructure(
                    S=spot_val, T=max(time_val, 0.001), 
                    r=p.get('r', 0.05), sigma=vol_val, q=p.get('q', 0.0),
                    coupon_rate=p.get('coupon_rate', 0.08),
                    autocall_barrier=rel_auto,
                    protection_barrier=rel_prot,
                    coupon_barrier=rel_coup, 
                    obs_frequency=p.get('obs_frequency', 4), 
                    num_simulations=2000, 
                    seed=42
                )
                return phx.price()

            # Différences Finies
            eps = s * 0.01
            p_base = pricing_kernel(s, vol, t)
            p_up   = pricing_kernel(s + eps, vol, t)
            p_down = pricing_kernel(s - eps, vol, t)
            
            delta = (p_up - p_down) / (2 * eps)
            gamma = (p_up - 2*p_base + p_down) / (eps**2)
            
            p_vol_up = pricing_kernel(s, vol + 0.01, t)
            vega = (p_vol_up - p_base) 
            
            dt = 1/252.0
            p_tomorrow = pricing_kernel(s, vol, t - dt)
            theta = p_tomorrow - p_base 

            return p_base, delta, gamma, vega, theta
            
        else:
            raise ValueError(f"Unknown product type: {prod_type}")

    # ==========================================================================
    # MODIFICATIONS : VERSIONS PLOTLY (INTERACTIVES)
    # ==========================================================================
    
    def plot_pnl(self):
        """
        Visualise le Backtest avec des légendes claires et des couleurs explicites.
        """
        if self.results is None or self.results.empty: 
            return None
        
        df = self.results.reset_index()
        date_col = df.columns[0] 
        
        # Calcul du volume de trade quotidien
        df['Trade'] = df['Shares Held'].diff().fillna(0)
        
        # SEPARATION ACHAT / VENTE (Pour avoir 2 légendes distinctes !)
        # On met np.nan au lieu de 0 pour ne pas afficher de barres vides
        df['Buy_Qty'] = df['Trade'].apply(lambda x: x if x > 0 else np.nan)
        df['Sell_Qty'] = df['Trade'].apply(lambda x: x if x < 0 else np.nan)
        
        # Création de 3 lignes avec axes secondaires activés
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

        # --- GRAPH 1 : SPOT & POSITION ---
        # A. Le Spot (Prix de l'action) - Axe Gauche (Bleu)
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df['Spot'], 
            name="Spot Price (Left Axis)", 
            line=dict(color='#1f77b4', width=2)
        ), row=1, col=1, secondary_y=False)
        
        # B. La Position (Nombre d'actions détenues) - Axe Droit (Orange)
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df['Shares Held'], 
            name="Shares Held / Delta (Right Axis)", 
            line=dict(color='orange', dash='dot', width=2)
        ), row=1, col=1, secondary_y=True)

        # --- GRAPH 2 : TRADING ACTIVITY (GAMMA) ---
        # C. Achats (Barres Vertes)
        fig.add_trace(go.Bar(
            x=df[date_col], y=df['Buy_Qty'], 
            name="Buy Stock (Rebalancing)", 
            marker_color='#2ca02c', # Vert
            opacity=0.7
        ), row=2, col=1, secondary_y=False)
        
        # D. Ventes (Barres Rouges)
        fig.add_trace(go.Bar(
            x=df[date_col], y=df['Sell_Qty'], 
            name="Sell Stock (Rebalancing)", 
            marker_color='#d62728', # Rouge
            opacity=0.7
        ), row=2, col=1, secondary_y=False)
        
        # E. Spot "Fantôme" (Ligne grise fine) pour le contexte
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df['Spot'], 
            name="Spot Trend (Ref)", 
            line=dict(color='white', width=1, dash='solid'), 
            opacity=0.3,
            showlegend=True # On le garde dans la légende pour info
        ), row=2, col=1, secondary_y=True)

        # --- GRAPH 3 : P&L ---
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df['Cumulative P&L'], 
            name="Total P&L", 
            line=dict(color='#00CC96', width=2), 
            fill='tozeroy'
        ), row=3, col=1)

        # --- LAYOUT & AXES ---
        # showlegend=True est CRUCIAL ici !
        fig.update_layout(
            height=900, 
            template="plotly_dark", 
            hovermode="x unified", 
            showlegend=True,
            legend=dict(
                orientation="h", # Légende horizontale
                yanchor="bottom", y=1.02, # Juste au-dessus du graphe 1
                xanchor="right", x=1
            )
        )
        
        # Labels Axes
        fig.update_yaxes(title_text="Spot (€)", color='#1f77b4', row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Qty Shares", color='orange', row=1, col=1, secondary_y=True, showgrid=False)
        
        fig.update_yaxes(title_text="Trade Qty", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Spot Lvl", color='gray', row=2, col=1, secondary_y=True, showgrid=False)
        
        fig.update_yaxes(title_text="P&L (€)", row=3, col=1)
        
        return fig

    def plot_attribution(self):
        """
        Generates a dashboard to explain P&L sources (Plotly Version).
        Graph 1: Daily Actual vs Predicted (avec zone d'erreur)
        Graph 2: Cumulative P&L Drivers (Delta, Gamma, Theta...)
        """
        if not hasattr(self, 'attribution_history') or not self.attribution_history:
            return None

        df = pd.DataFrame(self.attribution_history)
        df['Date'] = pd.to_datetime(df['Date'])
        
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Model Accuracy: Actual vs Predicted Daily P&L", "Cumulative P&L Attribution")
        )

        # --- GRAPH 1 : DAILY ACCURACY ---
        # Predicted (Bleu pointillé)
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Predicted_PnL'],
            name="Predicted (Greeks)",
            line=dict(color='blue', dash='dash')
        ), row=1, col=1)

        # Actual (Noir)
        # Astuce Plotly pour le "Fill Between" : On remplit vers la trace précédente ('tonexty')
        # Pour simuler l'erreur rouge, on triche un peu visuellement ou on affiche les deux lignes
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Actual_PnL'],
            name="Actual Product Change",
            line=dict(color='white'), # Blanc sur fond noir ressort mieux que noir
            fill='tonexty', # Remplit l'espace entre cette ligne et la précédente (Predicted)
            fillcolor='rgba(255, 0, 0, 0.2)' # Rouge transparent pour l'erreur
        ), row=1, col=1)

        # --- GRAPH 2 : CUMULATIVE DRIVERS ---
        cols = ['Delta_PnL', 'Gamma_PnL', 'Theta_PnL', 'Unexplained']
        df_cum = df.set_index('Date')[cols].cumsum().reset_index()

        # Delta P&L (Violet)
        fig.add_trace(go.Scatter(
            x=df_cum['Date'], y=df_cum['Delta_PnL'],
            name="Delta P&L", line=dict(color='purple')
        ), row=2, col=1)

        # Gamma P&L (Orange)
        fig.add_trace(go.Scatter(
            x=df_cum['Date'], y=df_cum['Gamma_PnL'],
            name="Gamma P&L", line=dict(color='orange')
        ), row=2, col=1)

        # Theta P&L (Vert)
        fig.add_trace(go.Scatter(
            x=df_cum['Date'], y=df_cum['Theta_PnL'],
            name="Theta P&L", line=dict(color='green')
        ), row=2, col=1)

        # Unexplained (Gris pointillé)
        fig.add_trace(go.Scatter(
            x=df_cum['Date'], y=df_cum['Unexplained'],
            name="Unexplained", line=dict(color='gray', dash='dot')
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=700,
            hovermode="x unified"
        )
        
        fig.update_yaxes(title_text="Daily Change (€)", row=1, col=1)
        fig.update_yaxes(title_text="Total P&L (€)", row=2, col=1)

        return fig