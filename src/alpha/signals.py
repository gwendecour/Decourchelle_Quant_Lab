import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("PortfolioConstructor")

class MomentumSignals:
    def __init__(self, data):
        self.data = data # Prix de clôture ajustés

    # Dans signals.py, modifie l'appel du Z-score
    def get_z_score_momentum(self, universe, lookback=100, skip_recent=21):
        """
        Calcule les Z-scores et les sépare par classe d'actif.
    
        Returns:
            dict: { 'Actions': Series_Scores, 'Bonds': Series_Scores, ... }
        """
        # 1. Calcul global (vectorisé pour la performance)
        returns = (self.data.shift(skip_recent) / self.data.shift(lookback)) - 1
        vol = self.data.pct_change().rolling(window=21).std() * np.sqrt(252)
        all_scores = returns / vol
        last_scores = all_scores.iloc[-1]
    
        # 2. Séparation par Silos
        silo_scores = {}
        for category, tickers in universe.items():
            # On ne garde que les tickers présents dans les colonnes de data
            valid_tickers = [t for t in tickers if t in last_scores.index]
            silo_scores[category] = last_scores[valid_tickers].sort_values(ascending=False)
        
        return silo_scores

    def get_distance_ma(self, window=200):
        """Méthode 2: Distance par rapport à la Moyenne Mobile 200j"""
        ma = self.data.rolling(window=window).mean()
        scores = (self.data / ma) - 1
        return scores.iloc[-1]

    def get_rsi(self, window=14):
        """Méthode 3: RSI (Relative Strength Index)"""
        delta = self.data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

def select_top_assets(scores, universe, top_n=3):
    """
    Logique de sélection : Top N par classe d'actif.
    """
    final_selection = {}
    
    for category, tickers in universe.items():
        # On récupère les scores pour les tickers de la catégorie
        cat_scores = scores[tickers].sort_values(ascending=False)
        
        # Filtre de tendance positive (Score > 0 ou RSI > 50 selon la méthode)
        # Ici on prend juste les N meilleurs
        selected = cat_scores.head(top_n).index.tolist()
        final_selection[category] = selected
        
    return final_selection