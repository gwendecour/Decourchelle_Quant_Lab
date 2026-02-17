import pandas as pd
import numpy as np
import logging

# Configure logger
logger = logging.getLogger("PortfolioConstructor")

class PortfolioConstructor:
    def __init__(self, data, universe):
        self.data = data
        self.returns = data.pct_change().dropna()
        self.universe = universe

    # --- FIX: ADDED corr_lookback ARGUMENT HERE ---
    def get_diversified_top_n(self, z_scores, category, top_n=3, corr_threshold=0.7, corr_lookback=60):
        """
        Selects top assets based on momentum, filtering out highly correlated ones.
        Returns an empty list (Cash) if all assets have negative momentum.
        """
        universe_tickers = self.universe.get(category, [])
        
        # 1. Strict Intersection: Tickers existing in scores AND prices
        valid_tickers = [t for t in universe_tickers if t in z_scores.index and t in self.returns.columns]
        
        if not valid_tickers:
            return []

        # 2. Positive Momentum Filtering
        # We don't want to catch a falling knife, even if it's the "best" one.
        cat_scores = z_scores[valid_tickers]
        positive_scores = cat_scores[cat_scores > 0]
        
        # If the whole sector is red (Bear Market), stay Cash
        if positive_scores.empty:
            return []

        # 3. Sort
        sorted_candidates = positive_scores.sort_values(ascending=False)
        
        # 4. Greedy Selection with Decorrelation
        selected = []
        
        # --- FIX: USE THE PASSED corr_lookback ARGUMENT ---
        if len(self.returns) > corr_lookback:
            recent_returns = self.returns.tail(corr_lookback) 
        else:
            recent_returns = self.returns
        
        for candidate in sorted_candidates.index:
            if len(selected) >= top_n:
                break
            
            # First one is always selected
            if not selected:
                selected.append(candidate)
                continue
            
            # Correlation Check
            is_too_correlated = False
            for existing in selected:
                try:
                    corr = recent_returns[candidate].corr(recent_returns[existing])
                    if pd.isna(corr): 
                        corr = 0 # Assume 0 if not enough data
                        
                    if corr > corr_threshold:
                        is_too_correlated = True
                        break
                except:
                    continue # Ignore calc errors
            
            if not is_too_correlated:
                selected.append(candidate)
        
        return selected

    def compute_weights(self, selected_assets_dict, vol_window=60):
        """
        Calculates weights. Handles the "All Cash" case.
        """
        class_vols = {}
        final_weights = {}
        has_assets = False
        
        # 1. Volatility per Class
        for category, tickers in selected_assets_dict.items():
            if not tickers:
                continue
                
            has_assets = True
            
            # Synthetic index for the class
            if len(tickers) == 1:
                class_ret = self.returns[tickers[0]]
            else:
                class_ret = self.returns[tickers].mean(axis=1)
            
            # Volatility
            vol = class_ret.rolling(window=vol_window).std().iloc[-1] * np.sqrt(252)
            
            # Safety: If Vol is NaN or 0, use default
            if pd.isna(vol) or vol == 0:
                vol = 0.15 
                
            class_vols[category] = vol
            
        # --- FIX CRITICAL: 100% CASH CASE ---
        if not has_assets or not class_vols:
            return {'CASH': 1.0}
            
        # 2. Risk Parity between classes (Inverse Volatility)
        inv_vols = {k: 1.0 / v for k, v in class_vols.items()}
        sum_inv_vol = sum(inv_vols.values())
        
        if sum_inv_vol == 0:
             return {'CASH': 1.0}

        class_weights = {k: v / sum_inv_vol for k, v in inv_vols.items()}
        
        # 3. Equal Weight within class
        for category, weight in class_weights.items():
            tickers = selected_assets_dict[category]
            if tickers:
                w_per_asset = weight / len(tickers)
                for t in tickers:
                    final_weights[t] = w_per_asset
                    
        return final_weights
        