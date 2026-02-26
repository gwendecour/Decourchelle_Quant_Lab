import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("PortfolioConstructor")

class MomentumSignals:
    def __init__(self, data):
        """
        Initializes the signal generator with historical pricing data.
        """
        self.data = data

    def get_z_score_momentum(self, universe, lookback=100, skip_recent=21):
        """
        Calculates momentum Z-scores and groups results by asset class.
        Returns a dictionary mapping asset classes to Series of Z-scores.
        """
        returns = (self.data.shift(skip_recent) / self.data.shift(lookback)) - 1
        vol = self.data.pct_change().rolling(window=21).std() * np.sqrt(252)
        all_scores = returns / vol
        last_scores = all_scores.iloc[-1]
    
        silo_scores = {}
        for category, tickers in universe.items():
            valid_tickers = [t for t in tickers if t in last_scores.index]
            silo_scores[category] = last_scores[valid_tickers].sort_values(ascending=False)
        
        return silo_scores

    def get_distance_ma(self, window=200):
        """
        Calculates the percentage distance from the specified Moving Average.
        """
        ma = self.data.rolling(window=window).mean()
        scores = (self.data / ma) - 1
        return scores.iloc[-1]

    def get_rsi(self, window=14):
        """
        Calculates the Relative Strength Index (RSI).
        """
        delta = self.data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

def select_top_assets(scores, universe, top_n=3):
    """
    Selects the top N assets per category based on momentum scores.
    """
    final_selection = {}
    
    for category, tickers in universe.items():
        cat_scores = scores[tickers].sort_values(ascending=False)
        selected = cat_scores.head(top_n).index.tolist()
        final_selection[category] = selected
        
    return final_selection