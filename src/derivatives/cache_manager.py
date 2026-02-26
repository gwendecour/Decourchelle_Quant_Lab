import streamlit as st
import threading
from src.derivatives.structured_products import PhoenixStructure
from src.derivatives.pricing_model import EuropeanOption
from src.derivatives.backtester import DeltaHedgingEngine

@st.cache_data(show_spinner=False)
def get_cached_scenario_matrices(p_type, S, K, T, r, sigma, q, 
                                 autocall_pct, barrier_pct, coupon_barrier_pct, coupon_rate, mc_prec,
                                 hm_spot_rng, hm_vol_rng, n_g):
    if p_type == "Phoenix":
        prod = PhoenixStructure(S=S, T=T, r=r, sigma=sigma, q=q,
                                autocall_barrier=autocall_pct,
                                protection_barrier=barrier_pct,
                                coupon_barrier=coupon_barrier_pct,
                                coupon_rate=coupon_rate, obs_frequency=4, num_simulations=mc_prec)
        return prod.compute_scenario_matrices(spot_range_pct=hm_spot_rng, vol_range_abs=hm_vol_rng, n_spot=n_g, n_vol=n_g, matrix_sims=mc_prec)
    else:
        prod = EuropeanOption(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=p_type)
        return prod.compute_scenario_matrices(spot_range_pct=hm_spot_rng, vol_range_abs=hm_vol_rng, n_spot=n_g, n_vol=n_g)

@st.cache_data(show_spinner=False)
def get_cached_backtest(p_type, init_spot, bt_maturity, r, sold_vol, q, 
                        bt_autocall, bt_protection, bt_coupon_bar, bt_coupon_rate,
                        bt_strike_pct, transaction_cost_pct, hist_data):
    if p_type == "Phoenix":
        opt_hedge = PhoenixStructure(
            S=init_spot, T=bt_maturity, r=r, sigma=sold_vol, q=q,
            autocall_barrier=bt_autocall, protection_barrier=bt_protection,   
            coupon_barrier=bt_coupon_bar, coupon_rate=bt_coupon_rate,         
            obs_frequency=4, num_simulations=1000  # Limited slightly for backtest speed integration
        )
    else:
        strike_bt = init_spot * bt_strike_pct
        is_call = "Call" in p_type
        opt_hedge = EuropeanOption(
            S=init_spot, K=strike_bt, T=bt_maturity, r=r, sigma=sold_vol, q=q, 
            option_type="Call" if is_call else "Put"
        )
    
    hedging_engine = DeltaHedgingEngine(
        option=opt_hedge, market_data=hist_data,
        risk_free_rate=r, dividend_yield=q, volatility=sold_vol,
        transaction_cost=transaction_cost_pct
    )
    return hedging_engine.run_backtest()

def launch_background_prewarming(kwargs_matrix, kwargs_backtest=None):
    """
    Spawns a background thread to pre-warm the st.cache_data for heavy functions,
    ensuring a 0-second wait when the user switches tabs.
    """
    def worker():
        get_cached_scenario_matrices(**kwargs_matrix)
        if kwargs_backtest is not None:
             get_cached_backtest(**kwargs_backtest)
            
    thread = threading.Thread(target=worker, daemon=True)
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx
        add_script_run_ctx(thread)
    except Exception:
        pass
    thread.start()
