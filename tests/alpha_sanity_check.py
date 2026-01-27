import sys
import os
# Ajoute le chemin racine pour permettre les imports depuis src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.shared.market_data import MarketData
from src.alpha.signals import MomentumSignals
from src.alpha.weights import PortfolioConstructor
from src.alpha.hedging import BetaHedgeManager

def run_test():
    print("=== DÉBUT DU SANITY CHECK ===")
    
    # 1. Univers de test
    universe = {
        'Actions': ['SPY', 'QQQ', 'XLE'],
        'Bonds': ['TLT', 'LQD'],
        'Commodities': ['GLD', 'USO']
    }
    all_tickers = [t for sublist in universe.values() for t in sublist] + ['SPY']

    # 2. Data
    print("\n[1/4] Téléchargement des données...")
    data, meta = MarketData.get_clean_multiticker_data(all_tickers, "2024-06-01", "2025-01-01")
    # --- AJOUTE CETTE SÉCURITÉ ICI ---
    if data is None or meta is None:
        print("STOP : Impossible de continuer le test car les données sont invalides.")
        return 
    # ---------------------------------
    print(f"Qualité : {meta['global_ffill_rate']:.2f}% de remplissage.")

    # [2/4] Calcul des Signaux
    sig_gen = MomentumSignals(data)
    # On récupère maintenant un dictionnaire
    scores_by_cat = sig_gen.get_z_score_momentum(universe, lookback=100)

    for cat, scores in scores_by_cat.items():
        print(f"\n--- Podium {cat} ---")
        print(scores)

    # 4. Sélection et Weights
    print("\n[3/4] Sélection des champions (Diversified)...")
    constructor = PortfolioConstructor(data, universe)
    
    selected_assets = {}
    for cat, scores in scores_by_cat.items():
        selected = constructor.get_diversified_top_n(scores, cat, top_n=2, corr_threshold=0.6)
        selected_assets[cat] = selected
    
    weights = constructor.compute_weights(selected_assets)
    print(f"Actifs sélectionnés : {selected_assets}")
    print(f"Poids : {weights}")

    # 5. Hedge
    print("\n[4/4] Calcul du Bêta Neutral...")
    hedge_mgr = BetaHedgeManager(data)
    hedge_ratio, p_beta = hedge_mgr.get_hedge_ratio(weights)
    
    print("-" * 30)
    print(f"RÉSULTAT : Portefeuille Bêta = {p_beta:.2f}")
    print(f"ACTION : Shorter {abs(hedge_ratio)*100:.1f}% de la valeur du portefeuille en SPY")
    print("-" * 30)

if __name__ == "__main__":
    run_test()