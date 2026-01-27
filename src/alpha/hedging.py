import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("BetaHedgeManager")

class BetaHedgeManager:
    def __init__(self, data, benchmark_ticker='SPY'):
        self.data = data
        self.returns = data.pct_change().dropna()
        self.benchmark_ticker = benchmark_ticker
        
        if benchmark_ticker not in self.returns.columns:
            logger.error(f"Benchmark {benchmark_ticker} manquant dans les données !")

    def calculate_rolling_beta(self, ticker, window=60):
        """
        Calcule le bêta historique d'un actif par rapport au benchmark.
        """
        try:
            asset_rets = self.returns[ticker]
            bench_rets = self.returns[self.benchmark_ticker]
            
            # Calcul de la covariance et de la variance sur fenêtre glissante
            covariance = asset_rets.rolling(window=window).cov(bench_rets)
            variance = bench_rets.rolling(window=window).var()
            
            beta = covariance / variance
            return beta.iloc[-1]
        except Exception as e:
            logger.warning(f"Impossible de calculer le bêta pour {ticker}: {e}")
            return 1.0 # Valeur par défaut prudente

    def get_hedge_ratio(self, weights_dict, beta_window=60):
        """
        Calcule le bêta global du portefeuille et le ratio de couverture.
        
        weights_dict: { 'AAPL': 0.15, 'TLT': 0.20, ... }
        """
        portfolio_beta = 0.0

        for ticker, weight in weights_dict.items():
            if ticker == "CASH":
                portfolio_beta += weight * 0.0 # Le cash ne bouge pas avec le marché
            continue
        
        for ticker, weight in weights_dict.items():
            if ticker == self.benchmark_ticker:
                # Si le benchmark est déjà dans le long, son bêta est 1
                portfolio_beta += weight * 1.0
                continue
                
            asset_beta = self.calculate_rolling_beta(ticker, window=beta_window)
            portfolio_beta += weight * asset_beta
            
        # Le Hedge Ratio est l'opposé du Bêta pour annuler l'exposition
        # Si Beta Portefeuille = 0.8, on doit shorter 0.8 de SPY
        hedge_ratio = -portfolio_beta
        
        logger.info(f"Bêta du portefeuille calculé : {portfolio_beta:.2f}. Hedge Ratio : {hedge_ratio:.2f}")
        
        return hedge_ratio, portfolio_beta