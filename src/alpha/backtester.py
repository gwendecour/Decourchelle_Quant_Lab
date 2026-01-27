import pandas as pd
import numpy as np
import logging
from src.alpha.signals import MomentumSignals
from src.alpha.weights import PortfolioConstructor
from src.alpha.hedging import BetaHedgeManager

logger = logging.getLogger("BacktestEngine")

class BacktestEngine:
    def __init__(self, data, universe, initial_capital=100000):
        self.data = data
        self.returns = data.pct_change().dropna()
        self.universe = universe
        self.initial_capital = initial_capital

    def run(self, start_date, freq='ME', signal_method='z_score', top_n=2, hedge_on=True, lookback=126,  corr_threshold=0.6):
        """
        Runs the backtest simulation.
        
        Args:
            start_date (datetime): Start date of the simulation.
            freq (str): Rebalancing frequency ('ME', 'W-FRI', 'QE').
            signal_method (str): Method for signal generation ('z_score', 'rsi', 'distance_ma').
            top_n (int): Number of assets to select per category.
            corr_threshold (float): Correlation threshold for diversification.
            lookback (int): Lookback period for momentum calculation (default 126).
            hedge_on (bool): Whether to apply beta hedging (default True).
        """
        # Dates de décision selon la fréquence choisie
        # Use 'ME' (Month End) instead of 'M' which is deprecated in newer pandas versions
        rebalance_dates = self.data[start_date:].resample(freq).last().index
        
        history = []
        current_nav = self.initial_capital
        
        # Initialisation
        weights = {}
        hedge_ratio = 0.0

        for i in range(len(rebalance_dates) - 1):
            t0 = rebalance_dates[i]
            t1 = rebalance_dates[i+1]
            past_data = self.data.loc[:t0]

            # --- 1. SELECTION DES ACTIFS (Rééquilibrage selon freq) ---
            sig_gen = MomentumSignals(past_data)
            
            # Appel dynamique de la méthode (Z-score, RSI, etc.)
            scores_by_cat = {}
            if signal_method == 'z_score':
                # Pass lookback to the signal generator
                scores_by_cat = sig_gen.get_z_score_momentum(self.universe, lookback=lookback)
            elif signal_method == 'rsi':
                # RSI typically uses a fixed shorter window (e.g., 14), but logic is similar
                rsi_values = sig_gen.get_rsi()
                scores_by_cat = {cat: rsi_values[tickers] for cat, tickers in self.universe.items() if all(t in rsi_values.index for t in tickers)}
            elif signal_method == 'distance_ma':
                 # Assuming you implement get_distance_ma in signals.py
                 ma_scores = sig_gen.get_distance_ma(window=lookback)
                 scores_by_cat = {cat: ma_scores[tickers] for cat, tickers in self.universe.items() if all(t in ma_scores.index for t in tickers)}

            constructor = PortfolioConstructor(past_data, self.universe)
            
            # Safety check: ensure scores exist before calling constructor
            if scores_by_cat:
                selected_assets = {cat: constructor.get_diversified_top_n(scores, cat, top_n, corr_threshold) 
                                   for cat, scores in scores_by_cat.items()}
                weights = constructor.compute_weights(selected_assets)
            else:
                 weights = {"CASH": 1.0}

            # --- 2. CALCUL DU HEDGE (Beta Neutral) ---
            if hedge_on:
                hedge_mgr = BetaHedgeManager(past_data)
                hedge_ratio, _ = hedge_mgr.get_hedge_ratio(weights)
            else:
                hedge_ratio = 0.0

            # --- 3. CALCUL DE LA PERFORMANCE ---
            # On boucle jour par jour entre t0 et t1
            # iloc[1:] avoids recalculating t0
            daily_period_data = self.returns.loc[t0:t1].iloc[1:] 
            
            for date, daily_rets in daily_period_data.iterrows():
                # Performance du Long
                long_ret = sum(daily_rets[t] * w for t, w in weights.items() if t in daily_rets)
                
                # Performance du Short (Hedge)
                # If market (SPY) goes up, short position loses value
                short_ret = hedge_ratio * daily_rets.get('SPY', 0.0)
                
                # Update NAV
                current_nav *= (1 + long_ret + short_ret)
                history.append({'Date': date, 'NAV': current_nav, 'Daily_Ret': long_ret + short_ret})

        if not history:
             return pd.DataFrame()
             
        return pd.DataFrame(history).set_index('Date')

    def run_risk_parity_benchmark(self, start_date, lookback=126):
        """
        Simule un Benchmark 'Risk Parity' (Equal Risk Contribution par classe).
        Méthode : Inverse Volatility Weighting entre les classes.
        Allocation intra-classe : Equal Weight.
        """
        rebalance_dates = self.data[start_date:].resample('ME').last().index
        history = []
        current_nav = self.initial_capital
        
        # Sécurité : quelles classes sont disponibles ?
        active_categories = [cat for cat in self.universe.keys() if self.universe[cat]]
        
        for i in range(len(rebalance_dates) - 1):
            t0 = rebalance_dates[i]
            t1 = rebalance_dates[i+1]
            
            # 1. Calcul des Volatilités par Classe (sur le passé t0)
            # On a besoin d'une fenêtre passée pour estimer le risque
            past_data = self.returns.loc[:t0].tail(lookback)
            
            class_volatilities = {}
            
            for cat in active_categories:
                tickers = [t for t in self.universe[cat] if t in past_data.columns]
                if not tickers:
                    continue
                # On simule un indice équiréparti pour la classe pour mesurer sa vol globale
                # (Car la vol d'un panier < somme des vols individuelles grâce à la diversification interne)
                cat_index_returns = past_data[tickers].mean(axis=1)
                class_volatilities[cat] = cat_index_returns.std()
            
            # 2. Calcul des Poids par Classe (Inverse Volatility)
            # Formule : w_i = (1/vol_i) / Sum(1/vol_j)
            inv_vols = {cat: 1.0/vol if vol > 0 else 0 for cat, vol in class_volatilities.items()}
            sum_inv_vol = sum(inv_vols.values())
            
            class_weights = {}
            if sum_inv_vol > 0:
                class_weights = {cat: val/sum_inv_vol for cat, val in inv_vols.items()}
            else:
                # Fallback si pas de données : Equal Weight
                class_weights = {cat: 1.0/len(active_categories) for cat in active_categories}

            # 3. Distribution aux actifs individuels
            final_weights = {}
            for cat, weight in class_weights.items():
                tickers = [t for t in self.universe[cat] if t in self.data.columns]
                if not tickers: continue
                weight_per_asset = weight / len(tickers)
                for t in tickers:
                    final_weights[t] = weight_per_asset
            
            # 4. Calcul de la perf sur le mois suivant (t0 à t1)
            daily_period_data = self.returns.loc[t0:t1].iloc[1:]
            
            for date, daily_rets in daily_period_data.iterrows():
                day_ret = sum(daily_rets.get(t, 0.0) * w for t, w in final_weights.items())
                current_nav *= (1 + day_ret)
                history.append({'Date': date, 'Benchmark_NAV': current_nav})
                
        if not history:
             return pd.DataFrame()

        return pd.DataFrame(history).set_index('Date')['Benchmark_NAV']