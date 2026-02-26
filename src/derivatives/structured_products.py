import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
from src.derivatives.monte_carlo import MonteCarloEngine
from src.derivatives.instruments import FinancialInstrument
import plotly.express as px

class PhoenixStructure(MonteCarloEngine, FinancialInstrument):
    
    def __init__(self, **kwargs):
        """
        Initializes a Phoenix Autocall structure and the underlying Monte Carlo engine.
        Converts percentage barriers into absolute spot levels.
        """
        S = float(kwargs.get('S'))
        self.nominal = S
        self.coupon_rate = kwargs.get('coupon_rate')
        
        self.autocall_barrier = S * kwargs.get('autocall_barrier')
        self.protection_barrier = S * kwargs.get('protection_barrier')
        self.coupon_barrier = S * kwargs.get('coupon_barrier')
        
        self.obs_frequency = kwargs.get('obs_frequency', 4)
        
        maturity = float(kwargs.get('T'))
        
        steps = max(int(252 * maturity), 1)
        self.steps = steps
        
        num_simulations = kwargs.get('num_simulations', 10000)
        self.num_simulations = num_simulations

        MonteCarloEngine.__init__(self, 
            S=S, K=S, T=maturity, 
            r=kwargs.get('r'), 
            sigma=kwargs.get('sigma'), 
            q=kwargs.get('q', 0.0), 
            num_simulations=num_simulations, 
            num_steps=steps, 
            seed=kwargs.get('seed')
        )
        
        FinancialInstrument.__init__(self, **kwargs)

    # ==========================================================================
    # CORE PRICING (MONTE CARLO)
    # ==========================================================================

    def get_observation_indices(self):
        """Returns the array indices corresponding to coupon observation dates."""
        step_size = int(252 / self.obs_frequency)
        indices = np.arange(step_size, self.steps + 1, step_size, dtype=int)
        return indices

    def calculate_payoffs_distribution(self):
        """
        Computes the payoff for each Monte Carlo path based on barrier conditions.
        Discounting is applied based on the time the cash flow occurs (early exit or maturity).
        """
        paths = self.generate_paths() 
        payoffs = np.zeros(self.N)
        active_paths = np.ones(self.N, dtype=bool)
        indices = self.get_observation_indices()
        
        coupon_amt = self.nominal * self.coupon_rate * (1.0/self.obs_frequency)
        
        for i, idx in enumerate(indices):
            if idx >= len(paths): break
            current_prices = paths[idx]
            
            did_autocall = (current_prices >= self.autocall_barrier) & active_paths
            did_just_coupon = (current_prices >= self.coupon_barrier) & (current_prices < self.autocall_barrier) & active_paths
            
            time_fraction = idx / 252.0
            df = np.exp(-self.r * time_fraction)
            
            payoffs[did_just_coupon] += coupon_amt * df
            payoffs[did_autocall] += (self.nominal + coupon_amt) * df
            
            active_paths[did_autocall] = False
            if not np.any(active_paths): break
    
        if np.any(active_paths):
            final_prices = paths[-1]
            survivors = active_paths
            df_final = np.exp(-self.r * self.T)
            
            safe_mask = survivors & (final_prices >= self.protection_barrier)
            payoffs[safe_mask] += self.nominal * df_final
            
            crash_mask = survivors & (final_prices < self.protection_barrier)
            payoffs[crash_mask] += final_prices[crash_mask] * df_final

        return payoffs

    def price(self):
        """Returns the fair value as the mean of the discounted simulated payoffs."""
        payoffs = self.calculate_payoffs_distribution()
        return np.mean(payoffs)

    # ==========================================================================
    # SENSITIVITIES & RISK (FINITE DIFFERENCES)
    # ==========================================================================

    def calculate_delta_quick(self, n_sims=2000):
        """
        Approximates Delta via Finite Differences (Bump & Revalue).
        Reduces default simulations count to maintain UI responsiveness.
        """
        original_N = self.N
        original_S = self.S
        original_seed = self.seed if self.seed is not None else 42
        
        self.N = n_sims
        epsilon = self.S * 0.01
        
        self.S = original_S + epsilon
        self.seed = original_seed
        price_up = self.price()
        
        self.S = original_S - epsilon
        self.seed = original_seed
        price_down = self.price()
        
        self.S = original_S
        self.N = original_N
        self.seed = original_seed
        
        delta = (price_up - price_down) / (2 * epsilon)
        
        return delta

    def greeks(self):
        """
        Computes Delta, Gamma, and Vega using Finite Differences (Bump & Revalue method).
        Spot is bumped by 1%, Volatility by 1 absolute point.
        """
        original_seed = self.seed if self.seed else 42
        self.seed = original_seed
        base_price = self.price()
        
        epsilon = self.S * 0.01 
        orig_S = self.S
        
        self.S = orig_S + epsilon
        self.seed = original_seed
        p_up = self.price()
        
        self.S = orig_S - epsilon
        self.seed = original_seed
        p_down = self.price()
        
        self.S = orig_S 
        
        delta = (p_up - p_down) / (2 * epsilon)
        gamma = (p_up - 2 * base_price + p_down) / (epsilon**2)
        
        orig_sigma = self.sigma
        self.sigma = orig_sigma + 0.01
        self.seed = original_seed
        p_vol_up = self.price()
        self.sigma = orig_sigma
        
        vega = p_vol_up - base_price
        
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": 0.0, "rho": 0.0}

    def compute_scenario_matrices(self, spot_range_pct, vol_range_abs, n_spot, n_vol, matrix_sims=1000):
        """
        Computes Hedged and Unhedged P&L matrices across Spot and Volatility shifts.
        Limits N simulations (matrix_sims) to ensure realistic computation times.
        """
        original_N = self.N
        original_S = self.S
        original_sigma = self.sigma
        original_seed = self.seed if self.seed is not None else 42
        
        self.N = matrix_sims
        
        self.seed = original_seed
        initial_price = self.price()
        
        initial_delta = self.calculate_delta_quick(n_sims=matrix_sims)

        spot_moves = np.linspace(-spot_range_pct, spot_range_pct, int(n_spot))
        vol_moves = np.linspace(-vol_range_abs, vol_range_abs, int(n_vol))

        matrix_unhedged = np.zeros((len(vol_moves), len(spot_moves)))
        matrix_hedged = np.zeros((len(vol_moves), len(spot_moves)))

        for i, v_chg in enumerate(vol_moves):
            for j, s_chg in enumerate(spot_moves):
                
                self.S = original_S * (1 + s_chg)
                self.sigma = max(0.01, original_sigma + v_chg) 
                self.seed = original_seed 
                
                new_price = self.price()

                pnl_opt = initial_price - new_price
                
                pnl_shares = initial_delta * (self.S - original_S)
                
                matrix_unhedged[i, j] = pnl_opt
                matrix_hedged[i, j] = pnl_opt + pnl_shares

        self.N = original_N
        self.S = original_S
        self.sigma = original_sigma
        self.seed = original_seed

        return matrix_unhedged, matrix_hedged, spot_moves, vol_moves

    # ==========================================================================
    # TAB 1 VISUALIZATIONS: PRICING & PAYOFF
    # ==========================================================================

    def plot_payoff(self, spot_range=None):
        """Plots the Phoenix theoretical payout profile at maturity across spot levels."""
        prot_level = self.protection_barrier 
        cpn_level = self.coupon_barrier
        
        low_bound = min(self.S * 0.3, prot_level * 0.8)
        high_bound = self.S * 1.5
        spots = np.linspace(low_bound, high_bound, 200)
        payoffs = []
        
        for s in spots:
            if s >= cpn_level:
                val = 1.0 + self.coupon_rate 
            elif s >= prot_level:
                val = 1.0
            else:
                val = s / self.S
            
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

    def plot_price_vs_strike(self, current_spot):
        """Plots Phoenix price as a function of the Spot level (Moneyness sensitivity)."""
        spots = np.linspace(current_spot * 0.5, current_spot * 1.5, 50)
        prices = []
        
        original_S = self.S
        
        for s in spots:
            self.S = s
            prices.append(self.price())
            
        self.S = original_S
        current_price = self.price()
        
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

    def plot_price_vs_vol(self, current_vol):
        """Plots Phoenix price sensitivity to implied Volatility (Vega behavior overview)."""
        vols = np.linspace(0.05, 0.60, 30) 
        prices = []
        
        original_sigma = self.sigma
        
        for v in vols:
            self.sigma = v
            prices.append(self.price())
            
        self.sigma = original_sigma
        current_price = self.price()
        
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
        
    # ==========================================================================
    # TAB 2 VISUALIZATIONS: RISK & MONTE CARLO ANALYSIS
    # ==========================================================================

    def plot_phoenix_tunnel(self):
        """
        Visualizes Monte Carlo paths grouped and color-coded by their final payout scenario.
        Creates boolean masks to identify Autocall (Early Exit), Capital Protection (Maturity), and Loss paths.
        """
        original_N = self.N
        self.N = 1000 
        
        paths = self.generate_paths()
        obs_indices = self.get_observation_indices()
        
        obs_prices = paths[obs_indices] 
        
        autocall_mask = np.any(obs_prices >= self.autocall_barrier, axis=0)
        
        final_prices = paths[-1]
        crash_mask = (~autocall_mask) & (final_prices < self.protection_barrier)
        safe_mask = (~autocall_mask) & (final_prices >= self.protection_barrier)
        
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
        fig.add_hline(y=self.autocall_barrier, line_dash="dash", line_color="green", annotation_text="Autocall Lvl")
        fig.add_hline(y=self.protection_barrier, line_dash="dash", line_color="red", annotation_text="Protection Lvl")
        if self.coupon_barrier != self.protection_barrier:
            fig.add_hline(y=self.coupon_barrier, line_dash="dot", line_color="cyan", annotation_text="Coupon Lvl")
            
        for idx in obs_indices:
            fig.add_vline(x=idx, line_width=1, line_color="white", opacity=0.2)
            
        n_auto, n_safe, n_crash = np.sum(autocall_mask), np.sum(safe_mask), np.sum(crash_mask)
        stats_text = (
            f"<b>SCENARIOS (N={self.N})</b><br>"
            f"<span style='color:green'>Autocall: {n_auto} ({n_auto/self.N:.1%})</span><br>"
            f"<span style='color:gray'>Mature: {n_safe} ({n_safe/self.N:.1%})</span><br>"
            f"<span style='color:red'>Loss: {n_crash} ({n_crash/self.N:.1%})</span>"
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
        
        self.N = original_N
        return fig

    def plot_phoenix_distribution(self):
        """
        Plots a histogram of the discounted Monte Carlo payoffs.
        Illustrates the Value-at-Risk and statistical distribution of returns.
        """
        payoffs = self.calculate_payoffs_distribution()
        mean_price = np.mean(payoffs)
        
        payoffs_pct = (payoffs / self.nominal) * 100
        mean_pct = (mean_price / self.nominal) * 100
        
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

    def plot_mc_noise_distribution(self):
        """
        Runs multiple Monte Carlo pricing loops with distinct random seeds 
        to evaluate variance, standard deviation, and convergence stability (MC Noise).
        """
        n_experiments = 30 
        prices = []
        
        original_seed = self.seed
        
        for i in range(n_experiments):
            self.seed = i 
            prices.append(self.price())
            
        self.seed = original_seed
        
        prices = np.array(prices)
        prices_pct = (prices / self.nominal) * 100
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