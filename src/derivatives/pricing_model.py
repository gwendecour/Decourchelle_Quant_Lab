import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from src.derivatives.instruments import FinancialInstrument

class EuropeanOption(FinancialInstrument):
    
    def __init__(self, **kwargs):
        """
        Initializes a Vanilla European Option.
        """
        super().__init__(**kwargs)
        
        self.S = float(kwargs.get('S'))
        self.K = float(kwargs.get('K'))
        self.T = float(kwargs.get('T'))
        self.r = float(kwargs.get('r'))
        self.sigma = float(kwargs.get('sigma'))
        self.q = float(kwargs.get('q', 0.0)) 
        self.option_type = kwargs.get('option_type', 'call').lower()

    # ==========================================================================
    # CORE PRICING (BLACK-SCHOLES-MERTON)
    # ==========================================================================

    def _d1(self):
        """Calculates the d1 probability factor from the Black-Scholes formula."""
        return (np.log(self.S/self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def _d2(self):
        """Calculates the d2 probability factor from the Black-Scholes formula."""
        return self._d1() - self.sigma * np.sqrt(self.T)

    def price(self):
        """
        Calculates the theoretical fair value using the closed-form Black-Scholes-Merton equation.
        Incorporates continuous dividend yield (q) discounting.
        """
        d1 = self._d1()
        d2 = self._d2()
        
        if self.option_type == "call":
            return self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)
        
    # ==========================================================================
    # SENSITIVITIES & RISK (ANALYTICAL GREEKS)
    # ==========================================================================

    def greeks(self):
        """Returns the complete set of primary option sensitivities."""
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega": self.vega_point(),
            "theta": self.daily_theta(),
            "rho": self.rho_point()
        }

    def delta(self):
        """Rate of change of option price with respect to the underlying's price."""
        if self.option_type == "call":
            return np.exp(-self.q * self.T) * norm.cdf(self._d1())
        else:
            return -np.exp(-self.q * self.T) * norm.cdf(-self._d1())

    def gamma(self):
        """Rate of change of Delta (Convexity). Identical for Calls and Puts."""
        return np.exp(-self.q * self.T) * norm.pdf(self._d1()) / (self.S * self.sigma * np.sqrt(self.T))

    def vega_point(self):
        """Sensitivity to implied volatility. Scaled to represent a 1% absolute shift."""
        return (self.S * np.exp(-self.q * self.T) * norm.pdf(self._d1()) * np.sqrt(self.T)) / 100

    def daily_theta(self):
        """Time decay of the option's value. Scaled to represent a 1-day passage of time."""
        d1 = self._d1()
        d2 = self._d2()
        common = -(self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        
        if self.option_type == "call":
            theta = common - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2) + self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(d1)
        else:
            theta = common + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)
            
        return theta / 365

    def rho_point(self):
        """Sensitivity to the risk-free interest rate. Scaled to represent a 1% absolute shift."""
        d2 = self._d2()
        if self.option_type == "call":
            return (self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)) / 100
        else:
            return -(self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)) / 100

    def compute_scenario_matrices(self, spot_range_pct, vol_range_abs, n_spot, n_vol, matrix_sims=None):
        """
        Calculates Hedged and Unhedged P&L matrices across Spot and Volatility shifts.
        Uses exact analytical pricing for performance.
        """
        initial_price = self.price()
        initial_delta = self.delta() 

        spot_moves = np.linspace(-spot_range_pct, spot_range_pct, int(n_spot))
        vol_moves = np.linspace(-vol_range_abs, vol_range_abs, int(n_vol))

        matrix_unhedged = np.zeros((len(vol_moves), len(spot_moves)))
        matrix_hedged = np.zeros((len(vol_moves), len(spot_moves)))

        for i, v_chg in enumerate(vol_moves):
            for j, s_chg in enumerate(spot_moves):
                
                new_S = self.S * (1 + s_chg)
                new_vol = self.sigma + v_chg
                
                if new_vol < 0.001: new_vol = 0.001

                scenario_opt = EuropeanOption(
                    S=new_S, K=self.K, T=self.T, r=self.r, sigma=new_vol, q=self.q, option_type=self.option_type
                )
                new_price = scenario_opt.price()

                pnl_opt = -(new_price - initial_price)
                pnl_shares = initial_delta * (new_S - self.S)
                
                matrix_unhedged[i, j] = pnl_opt
                matrix_hedged[i, j] = pnl_opt + pnl_shares

        return matrix_unhedged, matrix_hedged, spot_moves, vol_moves

