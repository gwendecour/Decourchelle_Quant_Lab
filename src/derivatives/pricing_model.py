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

    # ==========================================================================
    # TAB 1 VISUALIZATIONS: PRICING & PAYOFF
    # ==========================================================================

    def plot_payoff(self, spot_range):
        """
        Generates an interactive Plotly chart showing the theoretical P&L profile at maturity.
        Overlays Client (Long Option) and Bank (Short Option) perspectives.
        """
        spots = np.linspace(spot_range[0], spot_range[1], 100)
        premium = self.price()
        
        if self.option_type == "call":
            intrinsic_value = np.maximum(spots - self.K, 0)
        else:
            intrinsic_value = np.maximum(self.K - spots, 0)

        pnl_client = intrinsic_value - premium
        pnl_bank = premium - intrinsic_value
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=spots, y=pnl_client, 
            mode='lines', 
            name=f'Client (Long {self.option_type.title()})', 
            line=dict(color='green', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=spots, y=pnl_bank, 
            mode='lines', 
            name=f'Bank (Short {self.option_type.title()})', 
            line=dict(color='red', width=3)
        ))

        fig.add_hline(y=0, line_color="white", line_width=1, opacity=0.5)

        fig.add_vline(
            x=self.K, 
            line_dash="dash", line_color="gray", 
            annotation_text=f"Strike ({self.K:.1f})", annotation_position="top left"
        )

        fig.add_vline(
            x=self.S, 
            line_dash="dot", line_color="cyan", 
            annotation_text=f"Current Spot ({self.S:.1f})", annotation_position="bottom right"
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
    
    def plot_price_vs_strike(self, current_spot):
        """
        Plots the theoretical option price sensitivity across varying Strike bounds (Moneyness).
        """
        strikes = np.linspace(current_spot * 0.5, current_spot * 1.5, 100)
        prices = []
        
        for k in strikes:
            temp_opt = EuropeanOption(
                S=self.S, K=k, T=self.T, r=self.r, sigma=self.sigma, q=self.q, option_type=self.option_type
            )
            prices.append(temp_opt.price())
            
        current_price = self.price()
        current_k = self.K
        
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

    def plot_price_vs_vol(self, current_vol):
        """Plots the theoretical option price sensitivity to Implied Volatility (Vega impact)."""
        vols = np.linspace(0.05, 0.80, 50)
        prices = []
        
        for v in vols:
            tmp = EuropeanOption(S=self.S, K=self.K, T=self.T, r=self.r, sigma=v, q=self.q, option_type=self.option_type)
            prices.append(tmp.price())
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vols*100, y=prices, mode='lines', name='Price', line=dict(color='orange', width=3)))
        
        curr_price = self.price()
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

    # ==========================================================================
    # TAB 2 VISUALIZATIONS: RISK & HEATMAPS
    # ==========================================================================

    def plot_greeks_profile(self):
        """
        Generates structural risk graphs displaying Delta, Gamma, and Vega across the spot domain.
        Used to identify areas of peak convexity and volatility exposure.
        """
        lower_bound = 0.01
        upper_bound = self.K * 2.0
        spots = np.linspace(lower_bound, upper_bound, 100)
        
        deltas, gammas, vegas = [], [], []
        
        current_S = self.S
        
        for s in spots:
            self.S = s
            d = self.delta()
            g = self.gamma()
            v = self.vega_point()
            
            deltas.append(-d)
            gammas.append(-g)
            vegas.append(-v)
            
        self.S = current_S
        
        curr_vals = {
            'Delta': -self.delta(),
            'Gamma': -self.gamma(),
            'Vega': -self.vega_point()
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
            
            fig.add_vline(x=self.K, line_width=1, line_dash="dash", line_color="gray", row=row, col=col)
            
            if row == 3: 
                 fig.add_annotation(x=self.K, y=min(y_data), text="Strike", showarrow=False, yshift=-10, font=dict(size=10, color="gray"), row=row, col=col)

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
    
    def plot_risk_profile(self, spot_range):
        """
        Displays simultaneous secondary market risks (Gamma & Vega) as a function of the Spot.
        Indicates the most complex hedging regions (Hedging Difficulty View).
        """
        spots = np.linspace(spot_range[0], spot_range[1], 100)
        gammas = []
        vegas = []
        
        for s in spots:
            temp_opt = EuropeanOption(
                S=s, K=self.K, T=self.T, r=self.r, sigma=self.sigma, q=self.q, option_type=self.option_type
            )
            gammas.append(temp_opt.gamma())
            vegas.append(temp_opt.vega_point()) 
            
        current_gamma = self.gamma()
        current_vega = self.vega_point()

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
            go.Scatter(x=[self.S], y=[current_gamma], mode='markers', name='Mon Gamma', marker=dict(color='crimson', size=10)),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=[self.S], y=[current_vega], mode='markers', name='Mon Vega', marker=dict(color='royalblue', size=10)),
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
        
        fig.add_vline(x=self.S, line_dash="dot", line_color="gray", annotation_text="Current Spot")

        return fig