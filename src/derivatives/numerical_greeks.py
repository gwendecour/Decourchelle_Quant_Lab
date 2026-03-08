import numpy as np

class NumericalGreeksEngine:
    """
    A base mixin class that provides numerical estimation of Greeks 
    for any generic financial instrument using Finite Differences (Bump & Revalue).
    """

    def calculate_delta_quick(self, n_sims=2000):
        """
        Approximates Delta via Finite Differences.
        Temporarily reduces the number of simulations to maintain UI responsiveness.
        """
        original_N = getattr(self, 'N', getattr(self, 'num_simulations', 10000))
        original_S = self.S
        original_seed = getattr(self, 'seed', 42) if getattr(self, 'seed', None) is not None else 42
        
        self.N = n_sims
        self.num_simulations = n_sims
        epsilon = original_S * 0.01
        
        # Up scenario
        self.S = original_S + epsilon
        self.seed = original_seed
        price_up = self.price()
        
        # Down scenario
        self.S = original_S - epsilon
        self.seed = original_seed
        price_down = self.price()
        
        # Restore state
        self.S = original_S
        self.N = original_N
        self.num_simulations = original_N
        self.seed = original_seed
        
        delta = (price_up - price_down) / (2 * epsilon)
        return delta

    def greeks(self):
        """
        Computes Delta, Gamma, and Vega using Finite Differences (Central/Forward).
        Spot is bumped by 1%, Volatility by 1 absolute point.
        """
        original_seed = getattr(self, 'seed', 42) if getattr(self, 'seed', None) is not None else 42
        self.seed = original_seed
        
        base_price = self.price()
        
        epsilon = self.S * 0.01 
        orig_S = self.S
        
        # Delta & Gamma
        self.S = orig_S + epsilon
        self.seed = original_seed
        p_up = self.price()
        
        self.S = orig_S - epsilon
        self.seed = original_seed
        p_down = self.price()
        
        self.S = orig_S 
        
        delta = (p_up - p_down) / (2 * epsilon)
        gamma = (p_up - 2 * base_price + p_down) / (epsilon**2)
        
        # Vega
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
        original_N = getattr(self, 'N', getattr(self, 'num_simulations', 10000))
        original_S = self.S
        original_sigma = self.sigma
        original_seed = getattr(self, 'seed', 42) if getattr(self, 'seed', None) is not None else 42
        
        self.N = matrix_sims
        self.num_simulations = matrix_sims
        
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

        # Restore state
        self.N = original_N
        self.num_simulations = original_N
        self.S = original_S
        self.sigma = original_sigma
        self.seed = original_seed

        return matrix_unhedged, matrix_hedged, spot_moves, vol_moves
