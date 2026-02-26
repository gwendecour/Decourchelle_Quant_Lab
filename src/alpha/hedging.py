import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("BetaHedgeManager")

class BetaHedgeManager:
    def __init__(self, data, benchmark_ticker='SPY'):
        """
        Initializes the dynamic hedging engine to calculate and offset portfolio Beta.
        """
        self.data = data
        self.returns = data.pct_change().dropna()
        self.benchmark_ticker = benchmark_ticker
        
        if benchmark_ticker not in self.returns.columns:
            logger.error(f"Benchmark {benchmark_ticker} missing from data!")

    def calculate_rolling_beta(self, ticker, window=60):
        """
        Calculates the historical rolling beta of an asset relative to the benchmark.
        """
        try:
            asset_rets = self.returns[ticker]
            bench_rets = self.returns[self.benchmark_ticker]
            
            covariance = asset_rets.rolling(window=window).cov(bench_rets)
            variance = bench_rets.rolling(window=window).var()
            
            beta = covariance / variance
            return beta.iloc[-1]
        except Exception as e:
            logger.warning(f"Failed to calculate beta for {ticker}: {e}")
            return 1.0 

    def get_hedge_ratio(self, weights_dict, beta_window=60):
        """
        Determines the overall portfolio Beta and the required hedge ratio.
        The hedge ratio is the negative reciprocal value needed to achieve Beta neutrality.
        """
        portfolio_beta = 0.0

        for ticker, weight in weights_dict.items():
            if ticker == "CASH":
                portfolio_beta += weight * 0.0 
                continue
            elif ticker == self.benchmark_ticker:
                portfolio_beta += weight * 1.0
                continue
                
            asset_beta = self.calculate_rolling_beta(ticker, window=beta_window)
            portfolio_beta += weight * asset_beta
            
        hedge_ratio = -portfolio_beta
        
        logger.info(f"Calculated Portfolio Beta: {portfolio_beta:.2f}. Hedge Ratio: {hedge_ratio:.2f}")
        
        return hedge_ratio, portfolio_beta