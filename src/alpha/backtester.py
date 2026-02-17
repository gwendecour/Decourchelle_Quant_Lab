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

    # 1. AJOUT DE L'ARGUMENT corr_lookback=60 DANS LA SIGNATURE
    def run(self, start_date, freq='ME', signal_method='z_score', top_n=2, hedge_on=True, lookback=126, corr_threshold=0.6, corr_lookback=60):
        """
        Runs the backtest simulation.
        """
        # Dates de décision
        rebalance_dates = self.data[start_date:].resample(freq).last().index
        
        history = []
        current_nav = self.initial_capital
        
        self.selections_history = {} 
        
        # Initialisation
        weights = {}
        hedge_ratio = 0.0

        for i in range(len(rebalance_dates) - 1):
            t0 = rebalance_dates[i]
            t1 = rebalance_dates[i+1]
            past_data = self.data.loc[:t0]

            # --- 1. SELECTION DES ACTIFS ---
            sig_gen = MomentumSignals(past_data)
            
            # Calcul des scores
            scores_by_cat = {}
            if signal_method == 'z_score':
                scores_by_cat = sig_gen.get_z_score_momentum(self.universe, lookback=lookback)
            elif signal_method == 'rsi':
                rsi_values = sig_gen.get_rsi()
                scores_by_cat = {cat: rsi_values[tickers] for cat, tickers in self.universe.items() if all(t in rsi_values.index for t in tickers)}
            elif signal_method == 'distance_ma':
                 ma_scores = sig_gen.get_distance_ma(window=lookback)
                 scores_by_cat = {cat: ma_scores[tickers] for cat, tickers in self.universe.items() if all(t in ma_scores.index for t in tickers)}

            constructor = PortfolioConstructor(past_data, self.universe)
            
            # Allocation
            if scores_by_cat:
                # 2. TRANSMISSION DE L'ARGUMENT AU CONSTRUCTEUR
                selected_assets = {
                    cat: constructor.get_diversified_top_n(
                        scores, 
                        cat, 
                        top_n, 
                        corr_threshold, 
                        corr_lookback=corr_lookback # <--- C'EST ICI QU'ON PASSE LA VARIABLE
                    ) 
                    for cat, scores in scores_by_cat.items()
                }
                weights = constructor.compute_weights(selected_assets)
            else:
                 weights = {"CASH": 1.0}

            # ... (Le reste de la fonction reste identique) ...
            
            # Enregistrement Vérité Terrain
            selected_tickers_list = [t for t, w in weights.items() if w > 0 and t != 'CASH']
            self.selections_history[t0] = selected_tickers_list

            # Calcul Hedge
            if hedge_on:
                hedge_mgr = BetaHedgeManager(past_data)
                hedge_ratio, _ = hedge_mgr.get_hedge_ratio(weights)
            else:
                hedge_ratio = 0.0

            # Calcul Perf & Drift
            current_positions = {}
            for t, w in weights.items():
                current_positions[t] = current_nav * w
            
            if 'CASH' not in current_positions:
                current_positions['CASH'] = 0.0
            
            daily_period_data = self.returns.loc[t0:t1].iloc[1:] 
            
            for date, daily_rets in daily_period_data.iterrows():
                for t in list(current_positions.keys()):
                    if t != 'CASH':
                        ret = daily_rets.get(t, 0.0)
                        current_positions[t] *= (1 + ret)
                
                spy_ret = daily_rets.get('SPY', 0.0)
                short_pnl = (current_nav * hedge_ratio) * (-spy_ret)
                current_positions['CASH'] += short_pnl
                
                current_nav = sum(current_positions.values())
                
                snapshot = {'Date': date, 'NAV': current_nav, 'Hedge Ratio': hedge_ratio}
                
                if current_nav > 0:
                    for t, amount in current_positions.items():
                        snapshot[t] = amount / current_nav
                else:
                    for t in current_positions:
                        snapshot[t] = 0.0
                
                history.append(snapshot)

        if not history:
             return {}
             
        full_hist_df = pd.DataFrame(history).set_index('Date')
        
        return {'NAV': full_hist_df['NAV'], 'Hedge Ratio': full_hist_df['Hedge Ratio'], 'Weights': full_hist_df.drop(columns=['NAV', 'Hedge Ratio']), 'Selections': self.selections_history}
    

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