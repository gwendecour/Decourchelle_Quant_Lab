import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.shared.market_data import MarketData
from src.derivatives.structured_products import PhoenixStructure
from src.derivatives.pricing_model import EuropeanOption

class DeltaHedgingEngine:
    def __init__(self, option, market_data, risk_free_rate, dividend_yield, volatility, transaction_cost=0.0, rebalancing_freq="daily"):
        """
        Initializes a Delta-Neutral hedging backtester.
        Replicates the option payoff by trading the underlying asset dynamically (Delta Hedging).
        """
        self.option = option
        
        if isinstance(market_data, pd.DataFrame):
            if 'Close' in market_data.columns:
                self.spot_series = market_data['Close']
            else:
                self.spot_series = market_data.iloc[:, 0]
        elif isinstance(market_data, pd.Series):
            self.spot_series = market_data
        else:
            raise ValueError("market_data must be a pandas DataFrame or Series")
            
        self.r = risk_free_rate
        self.q = dividend_yield
        self.sigma = volatility
        self.tc = transaction_cost
        self.dt = 1/252.0 
        
        self.history = []
        self.results = None

    # ==========================================================================
    # CORE HEDGING ENGINE
    # ==========================================================================

    def run_backtest(self):
        """
        Executes the delta hedging simulation over historical market data.
        Maintains a self-financing portfolio tracking the option's value changes.
        """
        spots = self.spot_series.values
        dates = self.spot_series.index
        n_days = len(spots)
        
        contract_maturity = self.option.T 
        
        is_phoenix = hasattr(self.option, 'autocall_barrier')
        obs_freq = getattr(self.option, 'obs_frequency', 4)
        days_between_obs = int(252 / obs_freq) if obs_freq > 0 else 99999
        
        ac_barrier = getattr(self.option, 'autocall_barrier', 999999)
        cpn_barrier = getattr(self.option, 'coupon_barrier', 0)
        nominal = getattr(self.option, 'nominal', spots[0])
        coupon_amount = nominal * getattr(self.option, 'coupon_rate', 0) / obs_freq if obs_freq > 0 else 0

        portfolio_values = np.full(n_days, np.nan)
        cash_account = np.zeros(n_days)
        shares_held = np.zeros(n_days)
        deltas = np.zeros(n_days)
        option_prices = np.zeros(n_days)
        
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
        
        initial_premium = init_price
        initial_hedge_cost = (init_delta * s0)
        initial_fees = abs(initial_hedge_cost) * self.tc
        
        initial_cash = initial_premium - initial_hedge_cost - initial_fees
        
        cash_account[0] = initial_cash
        shares_held[0] = init_delta
        deltas[0] = init_delta
        option_prices[0] = init_price
        portfolio_values[0] = initial_cash + (init_delta * s0) - init_price 
        
        product_alive = True
        final_idx = 0
        status = "Running"
        coupons_paid_count = 0
        final_date = dates[-1]
        
        total_payouts_paid = 0.0 
        total_trans_costs = initial_fees

        for i in range(1, n_days):
            if not product_alive:
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
            
            if t_remain <= 0:
                product_alive = False
                status = "Matured"
                final_date = dates[i]
                
                engine_payoff = 0.0
                
                if is_phoenix:
                    if s_curr >= cpn_barrier: engine_payoff += coupon_amount
                    prot_lvl = getattr(self.option, 'protection_barrier', 0)
                    if s_curr >= prot_lvl: engine_payoff += nominal
                    else: engine_payoff += nominal * (s_curr / s0)
                else:
                    engine_payoff = 0.0 

                total_payouts_paid += engine_payoff

                cash_from_hedge = shares_held[i-1] * s_curr
                fees = abs(cash_from_hedge) * self.tc
                total_trans_costs += fees
                
                final_cash = cash_account[i-1] + cash_from_hedge - fees - engine_payoff
                
                cash_account[i] = final_cash
                portfolio_values[i] = final_cash
                shares_held[i] = 0
                deltas[i] = 0
                option_prices[i] = engine_payoff
                continue

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
            
            prev_cash = cash_account[i-1]
            interest = prev_cash * (np.exp(self.r * self.dt) - 1)
            cash_after_payout = prev_cash + interest - payout_flow
            
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

        engine_pnl = portfolio_values[final_idx] 
        
        if len(self.results) > 1 if self.results is not None else len(spots) > 1:
            log_returns = np.log(spots[:final_idx+1] / np.roll(spots[:final_idx+1], 1))[1:]
            realized_vol = np.std(log_returns) * np.sqrt(252)
        else:
            realized_vol = 0.0
            
        final_spot_val = spots[final_idx]
        
        self.results = pd.DataFrame({
            'Spot': spots,
            'Option Price': option_prices,
            'Delta': deltas,
            'Shares Held': shares_held,
            'Cash': cash_account,
            'Cumulative P&L': portfolio_values
        }, index=dates).iloc[:final_idx+1]

        metrics = {
            'Engine P&L': engine_pnl,
            'Total Transaction Costs': total_trans_costs,
            'Realized Volatility': realized_vol,
            'Pricing Volatility': self.sigma,
            'Option Premium': initial_premium,
            'Phoenix Payouts Included': total_payouts_paid, 
            'Status': status,
            'Duration (Months)': (final_idx / 21.0),
            'Coupons Paid': coupons_paid_count,
            'Final Date': final_date.strftime("%Y-%m-%d"),
            'Final Spot': final_spot_val
        }
        
        return self.results, metrics

    def _calculate_greeks_at_date(self, ctx):
        """
        Internal utility: Values the relevant instrument under specific market condition contexts.
        Used primarily for the P&L Attribution process (Taylor series approximations).
        """
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
    # TAB 3 VISUALIZATIONS: BACKTEST RESULTS
    # ==========================================================================
