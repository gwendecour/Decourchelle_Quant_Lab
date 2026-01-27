import pandas as pd
import numpy as np
import logging

# On configure le logger pour voir les messages dans la console
logger = logging.getLogger("PortfolioConstructor")

class PortfolioConstructor: # <--- Vérifie que cette ligne est bien là !
    def __init__(self, data, universe):
        self.data = data
        self.returns = data.pct_change().dropna()
        self.universe = universe

    def get_diversified_top_n(self, z_scores, category, top_n=3, corr_threshold=0.7):
        universe_tickers = self.universe.get(category, [])
        
        if len(universe_tickers) < top_n:
            top_n = len(universe_tickers)

        cat_scores = z_scores[universe_tickers]
        sorted_tickers = cat_scores[cat_scores > 0].sort_values(ascending=False)
        
        if sorted_tickers.empty:
            return []

        selected = []
        for ticker in sorted_tickers.index:
            if len(selected) >= top_n:
                break
            
            if len(selected) == 0:
                selected.append(ticker)
                continue
            
            is_too_correlated = False
            for s_ticker in selected:
                correlation = self.returns[ticker].corr(self.returns[s_ticker])
                if correlation > corr_threshold:
                    is_too_correlated = True
                    break
            
            if not is_too_correlated:
                selected.append(ticker)
        
        return selected

    def compute_weights(self, selected_assets_dict, vol_window=60):
        class_vols = {}
        final_weights = {}
        
        for category, tickers in selected_assets_dict.items():
            if not tickers:
                continue
            class_returns = self.returns[tickers].mean(axis=1)
            class_vols[category] = class_returns.rolling(window=vol_window).std().iloc[-1] * np.sqrt(252)
        
        valid_classes = {k: v for k, v in class_vols.items() if not np.isnan(v) and v > 0}
        
        if not valid_classes:
            return {"CASH": 1.0}
        
        inv_vols = {k: 1/v for k, v in valid_classes.items()}
        total_inv_vol = sum(inv_vols.values())
        class_weights = {k: v / total_inv_vol for k, v in inv_vols.items()}
        
        for category, weight in class_weights.items():
            tickers = selected_assets_dict[category]
            n = len(tickers)
            for t in tickers:
                final_weights[t] = weight / n
                
        return final_weights