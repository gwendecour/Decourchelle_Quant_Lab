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

