from abc import ABC, abstractmethod
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

class FinancialInstrument(ABC):
    """
    Abstract base class. 
    Forces all options (Call, Phoenix...) to implement the same methods.
    """
    def __init__(self, **params):
        self.params = params
        
    @abstractmethod
    def price(self) -> float:
        pass

    @abstractmethod
    def greeks(self) -> dict:
        """Returns a dictionary e.g.: {'delta': 0.5, 'gamma': 0.02, ...}"""
        pass

    @abstractmethod
    def plot_payoff(self, spot_range) -> go.Figure:
        pass

    def plot_risk_matrix(self, spot_range_pct=0.10, vol_range_pct=0.05, n_spot_steps=5, n_vol_steps=3):
        """
        Generates Heatmaps with independent X/Y dimensions.
        """
        spot_moves = np.linspace(-spot_range_pct, spot_range_pct, n_spot_steps)
        vol_moves = np.linspace(-vol_range_pct, vol_range_pct, n_vol_steps)
        
        original_S = self.S
        original_sigma = self.sigma
        
        base_price = self.price()
        base_greeks = self.greeks() 
        base_delta = base_greeks.get('delta', 0.0)
        
        z_unhedged = np.zeros((len(vol_moves), len(spot_moves)))
        z_hedged = np.zeros((len(vol_moves), len(spot_moves)))
        
        for i, v_chg in enumerate(vol_moves):
            for j, s_chg in enumerate(spot_moves):
                self.S = original_S * (1 + s_chg)
                self.sigma = max(0.01, original_sigma + v_chg)
                
                new_price = self.price()
                pnl_option = -(new_price - base_price) 
                pnl_hedge = base_delta * (self.S - original_S)
                
                z_unhedged[i, j] = pnl_option
                z_hedged[i, j] = pnl_option + pnl_hedge

        self.S = original_S
        self.sigma = original_sigma
        
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
    
    def plot_pnl_attribution(self, spot_move_pct, vol_move_pct, days_passed=0):
        """
        Explains P&L via Taylor Expansion (Delta, Gamma, Vega, Theta).
        """
        original_S, original_sigma, original_T = self.S, self.sigma, self.T
        base_price = self.price()
        greeks = self.greeks()
        
        dt = days_passed / 365.0
        dS = original_S * spot_move_pct
        pnl_vega = (greeks['vega'] * (vol_move_pct * 100)) * -1 
        
        pos_sign = -1
        pnl_delta = (greeks['delta'] * dS) * pos_sign
        pnl_gamma = (0.5 * greeks['gamma'] * (dS**2)) * pos_sign
        pnl_theta = (greeks['theta'] * days_passed) * pos_sign
        
        predicted_pnl = pnl_delta + pnl_gamma + pnl_vega + pnl_theta
        
        self.S = original_S * (1 + spot_move_pct)
        self.sigma = original_sigma + vol_move_pct
        self.T = max(0.001, original_T - dt)
        
        new_price = self.price()
        actual_pnl = (new_price - base_price) * pos_sign
        unexplained = actual_pnl - predicted_pnl
        
        self.S, self.sigma, self.T = original_S, original_sigma, original_T
        
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

class InstrumentFactory:
    """
    Factory class that creates the correct object based on user choice.
    """
    @staticmethod
    def create_instrument(instrument_type, **kwargs):
        from src.derivatives.pricing_model import EuropeanOption
        from src.derivatives.structured_products import PhoenixStructure
        
        if instrument_type in ["Call", "Put"]:
            return EuropeanOption(option_type=instrument_type.lower(), **kwargs)
        
        elif instrument_type == "Phoenix Autocall":
            return PhoenixStructure(**kwargs)
        
        else:
            raise ValueError(f"Unknown instrument: {instrument_type}")