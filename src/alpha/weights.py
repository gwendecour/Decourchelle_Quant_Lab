import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("PortfolioConstructor")

class PortfolioConstructor:
    def __init__(self, data, universe):
        """
        Initializes the allocation engine.
        """
        self.data = data
        self.returns = data.pct_change().dropna()
        self.universe = universe

    def get_diversified_top_n(self, z_scores, category, top_n=3, corr_threshold=0.7, corr_lookback=60):
        """
        Selects top-performing assets while enforcing diversification.
        Assets exceeding the correlation threshold are skipped.
        Returns empty if no assets exhibit positive momentum (Bear Market mode).
        """
        universe_tickers = self.universe.get(category, [])
        valid_tickers = [t for t in universe_tickers if t in z_scores.index and t in self.returns.columns]
        
        if not valid_tickers: return []

        cat_scores = z_scores[valid_tickers]
        positive_scores = cat_scores[cat_scores > 0]
        
        if positive_scores.empty: return []

        sorted_candidates = positive_scores.sort_values(ascending=False)
        selected = []
        
        recent_returns = self.returns.tail(corr_lookback) if len(self.returns) > corr_lookback else self.returns
        
        for candidate in sorted_candidates.index:
            if len(selected) >= top_n: break
            
            if not selected:
                selected.append(candidate)
                continue
            
            is_too_correlated = False
            for existing in selected:
                try:
                    corr = recent_returns[candidate].corr(recent_returns[existing])
                    corr = 0 if pd.isna(corr) else corr
                        
                    if corr > corr_threshold:
                        is_too_correlated = True
                        break
                except: continue 
            
            if not is_too_correlated: selected.append(candidate)
        
        return selected

    def compute_weights(self, selected_assets_dict, vol_window=60):
        """
        Computes portfolio weights using Inverse Volatility (Risk Parity) across asset classes.
        Weights are then distributed equally among selected assets within each class.
        Defaults to 100% Cash if no assets are selected.
        """
        class_vols = {}
        final_weights = {}
        has_assets = False
        
        for category, tickers in selected_assets_dict.items():
            if not tickers: continue
            has_assets = True
            
            if len(tickers) == 1: class_ret = self.returns[tickers[0]]
            else: class_ret = self.returns[tickers].mean(axis=1)
            
            vol = class_ret.rolling(window=vol_window).std().iloc[-1] * np.sqrt(252)
            vol = 0.15 if pd.isna(vol) or vol == 0 else vol
            class_vols[category] = vol
            
        if not has_assets or not class_vols:
            return {'CASH': 1.0}
            
        inv_vols = {k: 1.0 / v for k, v in class_vols.items()}
        sum_inv_vol = sum(inv_vols.values())
        
        if sum_inv_vol == 0: return {'CASH': 1.0}

        class_weights = {k: v / sum_inv_vol for k, v in inv_vols.items()}
        
        for category, weight in class_weights.items():
            tickers = selected_assets_dict[category]
            if tickers:
                w_per_asset = weight / len(tickers)
                for t in tickers: final_weights[t] = w_per_asset
                    
        return final_weights