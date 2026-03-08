import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
from src.derivatives.monte_carlo import MonteCarloEngine
from src.derivatives.instruments import FinancialInstrument
from src.derivatives.numerical_greeks import NumericalGreeksEngine
import plotly.express as px

class PhoenixStructure(MonteCarloEngine, NumericalGreeksEngine, FinancialInstrument):
    
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

class BarrierOption(MonteCarloEngine, NumericalGreeksEngine, FinancialInstrument):
    def __init__(self, **kwargs):
        S = float(kwargs.get('S'))
        self.nominal = S
        self.coupon_rate = kwargs.get('coupon_rate')

        self.knock_type = kwargs.get('knock_type')
        self.direction = kwargs.get('direction')
        self.option_type = kwargs.get('option_type')
        self.K = kwargs.get('K') 
        self.barrier = S * kwargs.get('barrier')
        maturity = float(kwargs.get('T'))
        
        self.window_start = float(kwargs.get('window_start', 0.0))
        self.window_end = float(kwargs.get('window_end', maturity))
        
        r = kwargs.get('r')

        steps = max(int(252 * maturity), 1)
        self.steps = steps

        num_simulations = kwargs.get('num_simulations', 10000)
        self.num_simulations = num_simulations

        MonteCarloEngine.__init__(self,S=S, K=self.K, T=maturity, r=r, sigma=kwargs.get('sigma'), q=kwargs.get('q', 0.0), num_simulations=num_simulations, num_steps=steps, seed=kwargs.get('seed'))
        FinancialInstrument.__init__(self, **kwargs)    

        pass

    # ==========================================================================
    # CORE PRICING (MONTE CARLO)
    # ==========================================================================
    def calculate_payoffs_distribution(self):
        paths = self.generate_paths()
        
        if self.option_type == "call":
            vanilla_payoffs = np.maximum(paths[-1] - self.K, 0)
        elif self.option_type == "put":
            vanilla_payoffs = np.maximum(self.K - paths[-1], 0)
        elif self.option_type in ["one touch", "no touch"]:
            vanilla_payoffs = np.full(self.N, self.nominal)
            
        start_idx = int((self.window_start / self.T) * self.steps) if self.T > 0 else 0
        end_idx = int((self.window_end / self.T) * self.steps) + 1 if self.T > 0 else len(paths)
        window_paths = paths[start_idx:end_idx]
            
        if self.direction == "up":
            touched_barrier = np.max(window_paths, axis=0) >= self.barrier
        elif self.direction == "down":
            touched_barrier = np.min(window_paths, axis=0) <= self.barrier
        
        payoffs = np.zeros(self.N) 
        
        if self.option_type == "one touch":
            payoffs[touched_barrier] = self.nominal
        elif self.option_type == "no touch":
            not_touched = ~touched_barrier 
            payoffs[not_touched] = self.nominal
        elif self.knock_type == "in":
            payoffs[touched_barrier] = vanilla_payoffs[touched_barrier]
        elif self.knock_type == "out":
            not_touched = ~touched_barrier 
            payoffs[not_touched] = vanilla_payoffs[not_touched]
            
        return payoffs 
    def price(self):
        payoffs = self.calculate_payoffs_distribution()
        discount_factor = np.exp(-self.r * self.T)
        return np.mean(payoffs) * discount_factor