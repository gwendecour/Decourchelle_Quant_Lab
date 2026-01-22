import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # <-- AJOUTÉ : Corrige le NameError
import datetime

# --- IMPORTS MODULES ---
from src.shared.market_data import MarketData
from src.derivatives.instruments import InstrumentFactory
from src.derivatives.pricing_model import EuropeanOption
from src.derivatives.structured_products import PhoenixStructure
from src.derivatives.backtester import DeltaHedgingEngine
from src.shared.ui import render_header

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="Pricing Engine", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>

        .block-container {
            padding-top: 2rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# CSS pour l'alignement des inputs/sliders
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
        .stButton button {height: 2.2rem; font-size: 0.8rem;}
        .stNumberInput input {height: 2rem;}
    </style>
""", unsafe_allow_html=True)

render_header()

# ==============================================================================
# 1. STATE & UTILS
# ==============================================================================
defaults = {
    'custom_spot': 100.0, 'custom_vol': 0.20, 'custom_rate': 0.04, 'custom_div': 0.00,
    'ticker_ref': None, 'global_product_type': None, 
    'strike_pct': 100.0, 'barrier_pct': 0.60, 'coupon_barrier_pct': 0.70, 
    'autocall_pct': 1.00, 'coupon_rate': 0.08, 'maturity': 1.0,
    'market_spot': None, 'market_vol': None, 'market_rate': None, 'market_div': None
}
for key, val in defaults.items():
    if key not in st.session_state: st.session_state[key] = val

# On initialise les clés de l'onglet Greeks pour qu'elles existent même si on est sur l'onglet 1
if 'gk_spot' not in st.session_state: st.session_state['gk_spot'] = st.session_state['custom_spot']
if 'gk_vol' not in st.session_state: st.session_state['gk_vol'] = st.session_state['custom_vol']
if 'gk_rate' not in st.session_state: st.session_state['gk_rate'] = st.session_state['custom_rate']

# --- HELPER POUR METTRE A JOUR TOUS LES WIDGETS ---
def update_all_widget_keys(spot=None, vol=None, rate=None, div=None, strike=None, maturity=None):
    """Met à jour les variables d'état et force le rafraîchissement visuel des widgets"""
    if spot is not None:
        val = float(spot)
        # Tab 1 & Global
        st.session_state.custom_spot = val
        st.session_state['sl_custom_spot'] = val
        st.session_state['num_custom_spot'] = val
        # st.session_state['gk_spot'] = val  <-- INUTILE MAINTENANT (On utilise sim_spot_val)
        
        # --- TAB 2 (GREEKS) ---
        # 1. Update Fixed Strike
        st.session_state['fix_strike_input'] = val
        
        # 2. Update Sim Spot (Interne + Widgets)
        st.session_state['sim_spot_val'] = val
        st.session_state['gk_slider_spot'] = val # Force le slider visuel
        st.session_state['gk_box_spot'] = val    # Force la box visuelle

    if vol is not None:
        val = float(vol)
        # Tab 1 & Global
        st.session_state.custom_vol = val
        st.session_state['sl_custom_vol'] = val
        st.session_state['num_custom_vol'] = val
        # st.session_state['gk_vol'] = val <-- INUTILE
        
        # Tab 2
        st.session_state['gk_vol_slider'] = val * 100.0

    if rate is not None:
        st.session_state.custom_rate = float(rate)
        st.session_state['sl_custom_rate'] = float(rate)
        st.session_state['num_custom_rate'] = float(rate)
    
    if div is not None:
        st.session_state.custom_div = float(div)
        st.session_state['sl_custom_div'] = float(div)
        st.session_state['num_custom_div'] = float(div)
        
    if strike is not None:
        st.session_state.strike_pct = float(strike)
        if 'sl_strike_pct' in st.session_state: st.session_state['sl_strike_pct'] = float(strike)
        if 'num_strike_pct' in st.session_state: st.session_state['num_strike_pct'] = float(strike)

    if maturity is not None:
        new_mat = max(0.01, float(maturity))
        st.session_state.maturity = new_mat
        if 'num_maturity_pricing' in st.session_state: st.session_state['num_maturity_pricing'] = new_mat
        if 'num_maturity_greeks' in st.session_state: st.session_state['num_maturity_greeks'] = new_mat

def sync_sim_to_strike():
    """Aligne le Spot Simulé sur la valeur du Strike Fixe"""
    new_val = st.session_state.fix_strike_input
    st.session_state.sim_spot_val = new_val
    st.session_state.slider_sim_spot = new_val
    st.session_state.box_sim_spot = new_val

# --- Fonctions de synchro Slider <-> Input (TAB 1) ---
def sync_input(master_key, changed_widget_key):
    """
    Synchronise Slider <-> Number Input <-> Master Value
    Et passe le ticker en CUSTOM si on touche aux valeurs.
    """
    # 1. On récupère la nouvelle valeur de celui qui a bougé
    new_value = st.session_state[changed_widget_key]
    
    # 2. On met à jour la clé MAÎTRE (utilisée pour les calculs)
    st.session_state[master_key] = new_value
    
    # 3. On met à jour le FRÈRE JUMEAU (pour l'affichage)
    if "sl_" in changed_widget_key:
        # C'est le Slider qui a bougé -> On force la Box
        st.session_state[f"num_{master_key}"] = new_value
    else:
        # C'est la Box qui a bougé -> On force le Slider
        st.session_state[f"sl_{master_key}"] = new_value
        
    # 4. On désélectionne le Ticker (Passage en mode manuel)
    st.session_state.ticker_input = "CUSTOM"

def make_input_group(label, key_base, min_v, max_v, step, format_str="%.2f"):
    """Crée un slider et un input box synchronisés"""
    
    # --- CRUCIAL : FORCER LA SYNCHRO AVANT AFFICHAGE ---
    # Si on vient de faire un Reset, key_base a changé, mais pas sl_... ni num_...
    # On force donc les widgets à s'aligner sur la valeur maître actuelle.
    current_master_val = float(st.session_state[key_base])
    st.session_state[f"sl_{key_base}"] = current_master_val
    st.session_state[f"num_{key_base}"] = current_master_val

    col_s, col_i = st.columns([3, 1])
    
    with col_s:
        # Slider
        st.slider(
            label, min_v, max_v, step=step, 
            key=f"sl_{key_base}", 
            on_change=sync_input, 
            args=(key_base, f"sl_{key_base}"), # Arguments passés au callback
            label_visibility="visible"
        )
        
    with col_i:
        # Box
        st.number_input(
            "", min_v, max_v, step=step, format=format_str, 
            key=f"num_{key_base}",
            on_change=sync_input, 
            args=(key_base, f"num_{key_base}"), 
            label_visibility="hidden"
        )

# --- Fonction de synchro TAB 2 (Greeks) vers le reste ---
def sync_from_greeks_tab(): 
    # On met à jour la Source de Vérité
    st.session_state.custom_spot = st.session_state.gk_spot
    st.session_state.custom_vol = st.session_state.gk_vol
    st.session_state.custom_rate = st.session_state.gk_rate
    
    # On propage vers les widgets du Tab 1
    st.session_state['sl_custom_spot'] = st.session_state.gk_spot
    st.session_state['num_custom_spot'] = st.session_state.gk_spot
    st.session_state['sl_custom_vol'] = st.session_state.gk_vol
    st.session_state['num_custom_vol'] = st.session_state.gk_vol
    st.session_state['sl_custom_rate'] = st.session_state.gk_rate
    st.session_state['num_custom_rate'] = st.session_state.gk_rate

def reset_phoenix_props():
    """Réinitialise uniquement les propriétés du Phoenix (Barrières, Coupon)"""
    st.session_state.coupon_rate = 0.08
    st.session_state.autocall_pct = 1.00
    st.session_state.coupon_barrier_pct = 0.70
    st.session_state.barrier_pct = 0.60
    # On ne touche PAS aux données de marché ici


# --- Callbacks ---
def switch_to_custom_market():
    """
    Appelé quand on modifie manuellement Spot/Vol/Rate dans l'onglet Pricing.
    1. Bascule le Ticker sur 'CUSTOM'.
    2. Pousse IMMÉDIATEMENT les nouvelles valeurs vers la simulation (Greeks).
    """
    # 1. Passage visuel en Custom
    st.session_state.ticker_input = "CUSTOM" 
    
    # 2. Synchronisation du Spot (Tab 1 -> Tab 2)
    new_spot = float(st.session_state.custom_spot)
    st.session_state.sim_spot_val = new_spot
    st.session_state.gk_slider_spot = new_spot
    st.session_state.gk_box_spot = new_spot
    
    # 3. Synchronisation de la Volatilité (Tab 1 -> Tab 2)
    # Attention aux échelles : Tab 1 est en décimal (0.20), Tab 2 slider est en % (20.0)
    new_vol = float(st.session_state.custom_vol)
    st.session_state.gk_vol_slider = new_vol * 100.0

def update_market_data():
    ticker = st.session_state.ticker_input
    try:
        spot = MarketData.get_spot(ticker)
        vol = MarketData.get_volatility(ticker, "1y")
        rate = MarketData.get_risk_free_rate() or 0.04
        div = MarketData.get_dividend_yield(ticker) or 0.0
        
        # Sauvegarde des données marché brutes pour le Reset
        st.session_state.market_spot = float(spot)
        st.session_state.market_vol = float(vol)
        st.session_state.market_rate = float(rate)
        st.session_state.market_div = float(div)
        
        # Mise à jour des curseurs
        update_all_widget_keys(float(spot), float(vol), float(rate), float(div), strike=100.0)
        
    except Exception as e:
        st.error(f"Error: {e}")

def set_pricing_scenario(scenario_type):
    """
    Logique Robuste : 
    1. Récupère TOUJOURS les données de marché brutes (Reference).
    2. Applique le choc sur cette référence.
    Cela empêche l'accumulation des modifications (ex: cliquer 2x sur OTM ne baisse pas 2x la vol).
    """
    # 1. RECUPERATION REFERENCE (Market Data ou Défaut 100/20%)
    ref_spot = st.session_state.get('market_spot')
    ref_vol = st.session_state.get('market_vol')
    ref_rate = st.session_state.get('market_rate')
    ref_div = st.session_state.get('market_div')
    
    # Fallback si pas de données chargées
    if ref_spot is None: ref_spot = 100.0
    if ref_vol is None: ref_vol = 0.20
    if ref_rate is None: ref_rate = 0.04
    if ref_div is None: ref_div = 0.00
    
    # Par défaut, on garde la maturité actuelle (sauf pour Time Bleed)
    current_mat = st.session_state.get('maturity', 1.0) 

    # 2. APPLICATION DU SCENARIO
    p_type = st.session_state.global_product_type
    
    # --- SCENARIOS STRUCTURE (Strike Change) ---
    if scenario_type == "ATM":
        # Strike = Spot Ref, Vol = Vol Ref
        update_all_widget_keys(spot=ref_spot, vol=ref_vol, strike=100.0)
        
    elif scenario_type == "ITM":
        # Strike change (80% ou 120%), Spot/Vol = Ref
        new_strike_pct = 80.0 if "Call" in p_type else 120.0
        update_all_widget_keys(spot=ref_spot, vol=ref_vol, strike=new_strike_pct)
        
    elif scenario_type == "OTM":
        # Strike change, Vol baisse (Skew)
        new_strike_pct = 120.0 if "Call" in p_type else 80.0
        skewed_vol = max(ref_vol * 0.9, 0.05) # -10% sur la vol ref
        update_all_widget_keys(spot=ref_spot, vol=skewed_vol, strike=new_strike_pct)

    elif scenario_type == "Reset":
        if ref_spot is not None:
            update_all_widget_keys(spot=ref_spot, vol=ref_vol, rate=ref_rate, div=ref_div, strike=100.0)
        else:
            # Pas de données marché, reset par défaut
            update_all_widget_keys(spot=100.0, vol=0.20, rate=0.04, div=0.00, strike=100.0)

def set_greeks_scenario(scenario_type):
    """
    Gère les scénarios de l'onglet 2 (Stress Test Simulation).
    """

    ref_spot = st.session_state.get('fix_strike_input', st.session_state.custom_spot)
    ref_vol = st.session_state.get('market_vol')
    current_mat = st.session_state.get('gk_fix_mat', 1.0) # Récupère la maturité actuelle

    # Références
    if ref_spot is None: ref_spot = 100.0
    if ref_vol is None: ref_vol = 0.20

    if scenario_type == "Crash":
        new_spot = ref_spot * 0.85
        new_vol_pct = min((ref_vol + 0.20) * 100, 100.0)
        st.session_state.gk_slider_spot = new_spot
        st.session_state.gk_box_spot = new_spot
        st.session_state.gk_vol_slider = new_vol_pct

    elif scenario_type == "Rally":
        new_spot = ref_spot * 1.10
        new_vol_pct = max((ref_vol - 0.05) * 100, 1.0)
        st.session_state.gk_slider_spot = new_spot
        st.session_state.gk_box_spot = new_spot
        st.session_state.gk_vol_slider = new_vol_pct

    elif scenario_type == "TimeBleed":
        # On réduit la maturité de 1 mois, sans toucher au spot/vol
        new_mat = max(0.01, current_mat - (1/12))
        st.session_state.gk_fix_mat = new_mat

    elif scenario_type == "Reset":
        st.session_state.gk_slider_spot = ref_spot
        st.session_state.gk_box_spot = ref_spot
        
        real_mkt_vol = st.session_state.get('market_vol')
        if real_mkt_vol is not None:
            st.session_state.gk_vol_slider = float(real_mkt_vol * 100)
        else:
            st.session_state.gk_vol_slider = 20.0
            
        # --- LIGNE A AJOUTER ICI ---
        st.session_state.force_pnl_zero = True

# ==============================================================================
# 2. HEADER
# ==============================================================================
TICKERS = {
    "GLE.PA": "SocGen (Bank)",
    "BNP.PA": "BNP Paribas (Bank)",
    "MC.PA": "LVMH (Luxury)",
    "TTE.PA": "TotalEnergies (Energy)",
    "SAN.PA": "Sanofi (Health)",
    "AIR.PA": "Airbus (Indus)",
    "CAP.PA": "Capgemini (Tech)",
    "CUSTOM": "User Defined Data"
}

with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1.5, 1.5, 1.5, 4])
    
    with c1:
        selected_ticker = st.selectbox(
            "Ticker", 
            options=list(TICKERS.keys()), 
            index=None, 
            placeholder="Select Ticker...", 
            format_func=lambda x: f"{x} - {TICKERS[x]}", 
            key="ticker_input",
            label_visibility="collapsed"
        )
        
    with c2:
        selected_product = st.selectbox(
            "Product", 
            options=["Call", "Put", "Phoenix"], 
            index=None, 
            placeholder="Select Product...", 
            key="global_product_type",
            label_visibility="collapsed"
        )
        
    with c3:
        btn_disabled = (selected_ticker is None)
        st.button("Load Market Data", on_click=update_market_data, disabled=btn_disabled, use_container_width=True)
        
    with c4:
        # On vérifie si les données de marché ont été chargées (market_spot n'est plus None)
        if st.session_state.get('market_spot') is not None:
            s = st.session_state.custom_spot
            v = st.session_state.custom_vol
            r = st.session_state.custom_rate
            d = st.session_state.custom_div
            
            # On affiche uniquement si c'est chargé
            display_text = f"Spot: <b style='color:black'>{s:.2f}</b> | Vol: <b style='color:black'>{v:.1%}</b> | r: {r:.1%} | q: {d:.1%}"
            
            st.markdown(f"<div style='text-align:right; padding-top:5px; font-family:monospace; color:gray;'>"
                        f"{display_text}</div>", unsafe_allow_html=True)
        else:
            # Si pas chargé, on ne fait RIEN (vide)
            pass

if not selected_ticker or not selected_product or st.session_state.get('market_spot') is None:
    
    # Message guidant l'utilisateur selon ce qui manque
    if not selected_ticker or not selected_product:
        st.markdown("**Please select a Ticker AND a Product above.**")
    else:
        # Si Ticker/Product sont là mais pas les données
        st.markdown("**Please click 'Load Market Data' to initialize the Pricing Engine.**")
        
    st.stop()

# ==============================================================================
# 3. TABS
# ==============================================================================
tab_pricing, tab_greeks, tab_backtest = st.tabs(["Pricing & Payoff", "Greeks & Heatmaps", "Delta Hedging"])

S, sigma = st.session_state.custom_spot, st.session_state.custom_vol
r, q = st.session_state.custom_rate, st.session_state.custom_div
p_type = st.session_state.global_product_type

# --- TAB 1: PRICING ---
with tab_pricing:
    layout_col1, layout_col2, layout_col3 = st.columns([1.2, 1, 2], gap="medium")

    # --- INPUTS MARKET ---
    with layout_col1:
        st.markdown("### Market")
        make_input_group("Spot ($)", "custom_spot", 10.0, 700.0, 0.5)
        make_input_group("Vol (σ)", "custom_vol", 0.01, 1.00, 0.005)
        make_input_group("Rate (r)", "custom_rate", 0.00, 0.20, 0.001, "%.3f")
        make_input_group("Div (q)", "custom_div", 0.00, 0.20, 0.001, "%.3f")

    # --- INPUTS PRODUCT ---
    with layout_col2:
        st.markdown(f"### {p_type}")
        maturity = st.number_input("Maturity (Years)",value=float(st.session_state.get("maturity", 1.0)),min_value=0.1, step=0.1, key="maturity")

        if p_type == "Phoenix":
            # ... (Les sliders cpn, auto, c_bar, p_bar restent ici au dessus) ...
            cpn = st.slider("Cpn", 0.0, 0.20, st.session_state.get('coupon_rate', 0.08), 0.005)
            st.session_state.coupon_rate = cpn
            auto = st.slider("Autocall (%)", 80, 120, int(st.session_state.get('autocall_pct', 1.0)*100), 5)/100
            st.session_state.autocall_pct = auto
            c_bar = st.slider("Cpn Barr (%)", 40, 90, int(st.session_state.get('coupon_barrier_pct', 0.7)*100), 5)/100
            st.session_state.coupon_barrier_pct = c_bar
            p_bar = st.slider("Prot Barr (%)", 30, 80, int(st.session_state.get('barrier_pct', 0.6)*100), 5)/100
            st.session_state.barrier_pct = p_bar
            n_sims = st.selectbox("Sims", [2000, 5000, 10000], index=0)
            
            st.write("")
            st.markdown("### Resets")
            
            # --- VOS DEUX BOUTONS SEPARES ICI ---
            r1, r2 = st.columns(2)
            with r1:
                # Reset 1 : Marché uniquement
                st.button("Reset Market Values", on_click=set_pricing_scenario, args=("Reset",), use_container_width=True, help="Remet Spot/Vol/Rate aux données fetchées")
            with r2:
                # Reset 2 : Propriétés Phoenix uniquement
                st.button("Reset Properties", on_click=reset_phoenix_props, use_container_width=True, help="Remet Barrières et Coupon par défaut")

        elif p_type in ["Call", "Put"]:
            make_input_group("Moneyness (%)", "strike_pct", 50.0, 150.0, 1.0)
            strike_price = S * (st.session_state.strike_pct / 100.0)
            st.caption(f"Strike: **{strike_price:.2f} €**")
            n_sims = 0

            # --- CAS VANILLA : Structure + Market ---
            
            # Ligne 1 : Structure (Moneyness)
            st.caption("1. Structure / Moneyness")
            b1, b2, b3, b4 = st.columns(4, gap="small")
            with b1: 
                st.button("Reset", on_click=set_pricing_scenario, args=("Reset",), use_container_width=True)
            with b2: 
                st.button("ATM", on_click=set_pricing_scenario, args=("ATM",), use_container_width=True, help="Strike = Spot")
            with b3: 
                st.button("ITM", on_click=set_pricing_scenario, args=("ITM",), use_container_width=True, help="In The Money")
            with b4: 
                st.button("OTM", on_click=set_pricing_scenario, args=("OTM",), use_container_width=True, help="Out The Money")

        else:
            st.error(f"Produit inconnu : {p_type}")

            
    # --- OUTPUT ---
    with layout_col3:
        st.markdown("### Analysis")
        
        # Instanciation
        if p_type == "Phoenix":
            product = PhoenixStructure(
                S=S, T=maturity, r=r, sigma=sigma, q=q,
                autocall_barrier=st.session_state.autocall_pct, 
                protection_barrier=st.session_state.barrier_pct,
                coupon_barrier=st.session_state.coupon_barrier_pct, 
                coupon_rate=st.session_state.coupon_rate,
                obs_frequency=4, num_simulations=n_sims
            )
            price = product.price()
            fig_main = product.plot_payoff(spot_range=[S*0.5, S*1.5])
            metric_lbl, metric_val = "Barrier", f"{S*st.session_state.barrier_pct:.2f} €"
        else:
            strike_val = S * (st.session_state.strike_pct / 100.0)
            opt_type = "Call" if "Call" in p_type else "Put"
            product = EuropeanOption(S=S, K=strike_val, T=maturity, r=r, sigma=sigma, q=q, option_type=opt_type)
            price = product.price()
            fig_main = product.plot_payoff(spot_range=[S*0.6, S*1.4])
            metric_lbl, metric_val = "Moneyness", f"{(S/strike_val)*100:.1f}%"

        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("Price", f"{price:.2f} €")
        k2.metric("% Nominal", f"{(price/S)*100:.2f} %")
        k3.metric(metric_lbl, metric_val)

        # Graphique Payoff
        st.plotly_chart(fig_main, use_container_width=True)

        if p_type == "Phoenix":
            # Récupération des seuils pour l'affichage
            p_lvl = S * st.session_state.barrier_pct
            c_lvl = S * st.session_state.coupon_barrier_pct
            a_lvl = S * st.session_state.autocall_pct
            
            st.markdown(f"""
            **Phoenix Payoff Zones (at Maturity):**
            
            **Downside Risk (< {p_lvl:.2f} €):** Below the **Protection Barrier**, the capital protection disappears. You are fully exposed to the stock's fall (1:1 loss like holding the stock).
            
            **Coupon Zone ({c_lvl:.2f} € - {a_lvl:.2f} €):** Between the Coupon Barrier and Autocall level. You recover your **100% Capital** + Potential Coupons (Memory effect usually applies).
            
            **Autocall / Cap (> {a_lvl:.2f} €):** Above the Autocall level. You get **100% Capital + Coupon**. The performance is capped (you don't benefit from the stock's rise beyond the coupon).
            """)
            
        elif p_type == "Call":
             k_val = S * (st.session_state.strike_pct / 100.0)
             st.markdown(f"**Call Payoff:** Client profit if Spot > Strike (**{k_val:.2f} €**). It's the opposite for the bank Short Call")
             
        elif p_type == "Put":
             k_val = S * (st.session_state.strike_pct / 100.0)
             st.markdown(f"**Put Payoff:** Client profit if Spot < Strike (**{k_val:.2f} €**). It's the opposite for the bank Short Put")
        
    # --- LIGNE 2 : ANALYSE SENSIBILITÉ ---
    st.divider()
    
    # Titre de la section
    st.markdown("### Sensitivity Analysis")
    
    graph_col1, graph_col2 = st.columns(2, gap="medium")

    with graph_col1:
        # Titre propre en anglais, sans icône, sans répétition
        st.markdown("**Price Sensitivity to Strike**")
        with st.spinner("Computing..."):
            fig_struct = product.plot_price_vs_strike(current_spot=S)
            # use_container_width=True permet d'occuper toute la colonne
            st.plotly_chart(fig_struct, use_container_width=True, config={'displayModeBar': False})

            if p_type == "Call":
                note = "**Trend:** Decreasing. Higher strike decreases probability of exercise."
            elif p_type == "Put":
                note = "**Trend:** Increasing. Higher strike increases probability of exercise."
            elif p_type == "Phoenix":
                note = "**Trend:** Sharp rise below Protection Barrier, steady in Coupon Zone, flat above Autocall."
            st.caption(note)
            
    with graph_col2:
        # Titre propre en anglais
        st.markdown("**Price Sensitivity to Volatility**")
        with st.spinner("Computing..."):
            fig_vol = product.plot_price_vs_vol(current_vol=sigma)
            st.plotly_chart(fig_vol, use_container_width=True, config={'displayModeBar': False})
            if p_type in ["Call", "Put"]:
                note_vol = "**Trend:** Positive Vega. Long options benefit from higher uncertainty/volatility."
            elif p_type == "Phoenix":
                note_vol = "**Trend:** Negative Vega. Higher volatility increases the risk of hitting the downside barrier, lowering the price."
            st.caption(note_vol)



# --- TAB 2: GREEKS & HEATMAPS ---
with tab_greeks:
    st.subheader("Greeks Sensitivity Analysis")

    # Layout: 2 Colonnes
    col_params, col_metrics = st.columns([1.3, 1], gap="large")

    # ==========================================================================
    # COLONNE GAUCHE : PARAMETRES & SIMULATION
    # ==========================================================================
    with col_params:
        
        # --- 1. CONTRACT SETUP (FIXED) ---
        st.markdown("**Contract Setup (Fixed)**")
        
        c_def1, c_def2 = st.columns(2)
        with c_def1:
            fixed_maturity = st.number_input("Maturity (Years)", 0.01, 10.0, 1.0, 0.1, key="gk_fix_mat")
        
        with c_def2:
            if p_type == "Phoenix":
                st.markdown(f"**Ref Spot:** {S:.2f} €")
                ref_value = S
            else:
                # Callback: Si on change le strike fixe, on réaligne la simulation
                def sync_sim_to_strike():
                    val = st.session_state.fix_strike_input
                    st.session_state.sim_spot_val = val
                    st.session_state.gk_slider_spot = val
                    st.session_state.gk_box_spot = val

                # Valeur par défaut robustes pour éviter le warning jaune
                def_k = float(st.session_state.get('fix_strike_input', S))
                
                fixed_strike = st.number_input("Strike (€)", value=def_k, step=1.0, format="%.2f", 
                                               key="fix_strike_input", 
                                               on_change=sync_sim_to_strike)
                ref_value = fixed_strike

        st.divider()

        # --- 2. MARKET SIMULATION (VARIABLE) ---
        st.markdown("**Market Simulation**")
        
        # A. Initialisation Robuste (AJOUT DE LA VOLATILITE ICI)
        if 'sim_spot_val' not in st.session_state: 
            st.session_state.sim_spot_val = float(S)
        
        # Initialisation Volatilité (Nécessaire car on retire 'value=' plus bas)
        if 'gk_vol_slider' not in st.session_state:
            st.session_state.gk_vol_slider = float(sigma * 100)

        # On s'assure aussi que les clés spécifiques aux widgets existent
        if 'gk_slider_spot' not in st.session_state:
            st.session_state.gk_slider_spot = st.session_state.sim_spot_val
        if 'gk_box_spot' not in st.session_state:
            st.session_state.gk_box_spot = st.session_state.sim_spot_val

        # B. Callbacks de Synchronisation CROISÉE (Box <-> Slider)
        def update_slider():
            val = st.session_state.gk_slider_spot
            st.session_state.sim_spot_val = val
            st.session_state.gk_box_spot = val # Force la box

        def update_box():
            val = st.session_state.gk_box_spot
            st.session_state.sim_spot_val = val
            st.session_state.gk_slider_spot = val # Force le slider

        # C. Définition de la plage
        max_spot = float(ref_value * 2.0) if ref_value > 0 else 100.0
        
        # D. Affichage Slider / Box
        c_sim1, c_sim2 = st.columns([3, 1])
        with c_sim1:
            # CORRECTION : Suppression de 'value='. La 'key' suffit.
            st.slider("Spot Range", 0.0, max_spot, key="gk_slider_spot", 
                      on_change=update_slider, label_visibility="collapsed")
        with c_sim2:
            st.number_input("Spot", 0.0, max_spot, key="gk_box_spot", 
                            on_change=update_box, label_visibility="collapsed")
        
        # E. Variable DYNAMIQUE pour le calcul
        dyn_spot = st.session_state.sim_spot_val 
        
        # Feedback visuel (% move)
        pct_move = (dyn_spot / ref_value - 1) * 100 if ref_value > 0 else 0
        st.caption(f"Simulated Spot: **{dyn_spot:.2f} €** ({pct_move:+.2f}%)")

        # F. Volatilité
        st.write("")
        # CORRECTION : Suppression de 'value='. La 'key' gère tout.
        st.slider("Volatility (%)", 1.0, 100.0, key="gk_vol_slider")
        
        # CORRECTION : On lit DIRECTEMENT le State pour être sûr d'avoir la valeur post-Reset
        dyn_vol = st.session_state.gk_vol_slider / 100.0

        st.divider()
        
        # G. Boutons Scénarios (Tab 2 Only)
        st.caption("Quick Scenarios")
        b1, b2, b3, b4 = st.columns(4)
        
        with b1: st.button("Crash", on_click=set_greeks_scenario, args=("Crash",), use_container_width=True, help="**Market Crash:**\n- Spot: -15%\n- Volatility: +20 pts (Fear spike)\n\nSimulates a sudden market drop panic.")
        with b2: st.button("Rally", on_click=set_greeks_scenario, args=("Rally",), use_container_width=True, help="**Bull Rally:**\n- Spot: +10%\n- Volatility: -5 pts (Calm)\n\nSimulates a steady market rise.")
        with b3: st.button("Bleed", on_click=set_greeks_scenario, args=("TimeBleed",), use_container_width=True, help="**Time Decay:**\n- Maturity: -1 Month\n- Spot/Vol: Unchanged\n\nIsolates the effect of Theta (Time passing).")
        with b4: st.button("Reset", on_click=set_greeks_scenario, args=("Reset",), use_container_width=True, help="**Reset:**\nReverts all parameters (Spot, Vol, Time) to the initial Market Data values.")
    # ==========================================================================
    # COLONNE DROITE : METRICS & P&L
    # ==========================================================================
    with col_metrics:
        st.markdown("#### Greeks (Bank View)")
        
        # 1. PRICING AU SPOT DYNAMIQUE (dyn_spot)
        if p_type == "Phoenix":
            prod_gk = PhoenixStructure(
                S=dyn_spot,       # Spot du Slider
                T=fixed_maturity, 
                r=r, 
                sigma=dyn_vol,    # Vol du Slider
                q=q,
                autocall_barrier=st.session_state.autocall_pct,
                protection_barrier=st.session_state.barrier_pct,
                coupon_barrier=st.session_state.coupon_barrier_pct,
                coupon_rate=st.session_state.coupon_rate, 
                obs_frequency=4, 
                num_simulations=n_sims
            )
            
            # Delta Live
            with st.spinner("Calc Delta..."):
                client_delta = prod_gk.calculate_delta_quick(n_sims=2000)
            
            greeks = {'delta': -client_delta, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}

            # Full Greeks on demand
            if st.button("Compute Full Greeks"):
                c_greeks = prod_gk.greeks()
                greeks = {k: -v for k, v in c_greeks.items()}

        else:
            # Cas Vanilla
            prod_gk = EuropeanOption(S=dyn_spot, K=fixed_strike, T=fixed_maturity, r=r, sigma=dyn_vol, q=q, option_type=p_type)
            cg = prod_gk.greeks()
            greeks = {k: -v for k, v in cg.items()}

        # 2. AFFICHAGE GRECS
        m1, m2 = st.columns(2)
        m1.metric("Delta (Δ)", f"{greeks.get('delta',0):.4f}")
        m1.metric("Gamma (Γ)", f"{greeks.get('gamma',0):.4f}")
        m2.metric("Vega (ν)", f"{greeks.get('vega',0):.4f}")
        m2.metric("Theta (Θ)", f"{greeks.get('theta',0):.4f}")

        # 3. P&L DECOMPOSITION
        st.markdown("#### P&L Attribution")
        
        # Différentiels
        d_spot = dyn_spot - ref_value
        d_vol = dyn_vol - sigma
        
        # Taylor
        pnl_delta = greeks.get('delta', 0) * d_spot
        pnl_gamma = 0.5 * greeks.get('gamma', 0) * (d_spot ** 2)
        pnl_vega = greeks.get('vega', 0) * (d_vol * 100) # Assuming Vega per 1%
        
        taylor_pnl = pnl_delta + pnl_gamma + pnl_vega
        
        # Real P&L (Repricing)
        if p_type == "Phoenix":
            prod_ref = PhoenixStructure(S=ref_value, T=fixed_maturity, r=r, sigma=sigma, q=q,
                                      autocall_barrier=st.session_state.autocall_pct,
                                      protection_barrier=st.session_state.barrier_pct,
                                      coupon_barrier=st.session_state.coupon_barrier_pct,
                                      coupon_rate=st.session_state.coupon_rate, obs_frequency=4, num_simulations=2000)
        else:
            prod_ref = EuropeanOption(S=ref_value, K=fixed_strike, T=fixed_maturity, r=r, sigma=sigma, q=q, option_type=p_type)

        real_pnl = - (prod_gk.price() - prod_ref.price())

        # Si le bouton Reset vient d'être cliqué, on force le nettoyage visuel
        if st.session_state.get('force_pnl_zero', False):
            real_pnl = 0.0
            taylor_pnl = 0.0
            pnl_delta = 0.0
            pnl_gamma = 0.0
            pnl_vega = 0.0
            # On éteint le drapeau pour que les prochains mouvements recalculent normalement
            st.session_state.force_pnl_zero = False

        # Affichage P&L
        c_pnl1, c_pnl2 = st.columns(2)
        color = "normal" if real_pnl >= 0 else "inverse"
        c_pnl1.metric("ACTUAL P&L", f"{real_pnl:+.2f} €", delta_color=color)
        c_pnl2.metric("Taylor Est.", f"{taylor_pnl:+.2f} €", delta=f"{taylor_pnl-real_pnl:.2f} err", delta_color="off")

        cols = st.columns(3)
        cols[0].metric("Delta P&L", f"{pnl_delta:+.2f}")
        cols[1].metric("Gamma P&L", f"{pnl_gamma:+.2f}")
        cols[2].metric("Vega P&L", f"{pnl_vega:+.2f}")

        # ... (Calcul des Greeks et du P&L Attribution fait juste avant) ...
    
        # --- LOGIQUE D'EXPLICATION DYNAMIQUE ---
        st.subheader("P&L Attribution Analysis")
    
        # 1. On détecte le mouvement simulé
        spot_move = st.session_state.sim_spot_val - S # S = Spot initial
        vol_move = (st.session_state.gk_vol_slider/100.0) - sigma # sigma = Vol initiale
    
        explanation = []
    
        # 2. Analyse par Produit (Vue BANQUE / VENDEUR)
        if p_type == "Call":
            role = "Short Call"
            explanation.append(f"**Position:** You are **{role}** (Bank View). You are Short Delta, Short Gamma, Short Vega, Long Theta.")
        
            # Delta/Gamma Analysis
            if spot_move > 0:
                explanation.append(f"**Spot (+):** Market went UP. Being Short Delta, you **lost money** on Delta.")
                explanation.append(f"**Gamma Impact:** As Spot rose, your negative Delta became even more negative (Short Gamma). **Losses accelerated**.")
            elif spot_move < 0:
                explanation.append(f"**Spot (-):** Market went DOWN. Being Short Delta, you **made money** on Delta.")
                explanation.append(f"**Gamma Impact:** Short Gamma worked in your favor here (cushioning losses or accelerating gains).")
            
            # Vega Analysis
            if vol_move > 0:
                explanation.append(f"**Vol (+):** Implied Vol rose. Being Short Vega, the option price increased, so you **lost money** (Mark-to-Market).")
            elif vol_move < 0:
                explanation.append(f"**Vol (-):** Vol dropped. Being Short Vega, you **gained money**.")

        elif p_type == "Put":
            role = "Short Put"
            explanation.append(f"**Position:** You are **{role}**. You are Long Delta (Bullish), Short Gamma, Short Vega, Long Theta.")
        
            # Delta Analysis
            if spot_move > 0:
                explanation.append(f"**Spot (+):** Market went UP. Being Long Delta, you **made money** (Put value dropped).")
            elif spot_move < 0:
                explanation.append(f"**Spot (-):** Market went DOWN. Being Long Delta, you **lost money**.")
            
            # Vega Analysis (Idem Call)
            if vol_move > 0:
                explanation.append(f"**Vol (+):** Vol rose. Short Vega -> **Loss**.")
            elif vol_move < 0:
                explanation.append(f"**Vol (-):** Vol dropped. Short Vega -> **Gain**.")

        elif p_type == "Phoenix":
            role = "Short Phoenix (Issuer)"
            explanation.append(f"**Position:** You are **{role}**. Generally Long Vega (unlike vanilla), Long Theta, and Mixed Delta/Gamma.")
        
            # Vega Specificity for Phoenix
            if vol_move > 0:
                explanation.append(f"**Vol (+):** Uniquely here, Vol rising often helps the Issuer (Long Vega). Higher Vol increases the probability of hitting the downside barrier (Knock-In), lowering the product's value (your liability). -> **Gain**.")
            elif vol_move < 0:
                explanation.append(f"**Vol (-):** Vol dropping makes the product safer for the client. Its value rises. -> **Loss**.")
            
            # Delta/Spot
            if spot_move < 0:
                explanation.append(f"**Spot (-):** Market drop. The product gets closer to the risk barrier. Its value drops heavily. You **gain**.")
            elif spot_move > 0:
                explanation.append(f"**Spot (+):** Market rise. The product gets closer to Autocall (paying 100% + Cpn). Its value rises towards Par. You **lose** (or gain less).")

        
        # Affichage propre
        st.markdown("\n\n".join(explanation))    

    # ==========================================================================
    # PARTIE 3 : HEATMAPS (RECUPEREE DU CODE PRECEDENT)
    # ==========================================================================
    st.divider()
    st.subheader("Risk Heatmaps (Scenario Analysis)")
    
    # Contrôles spécifiques aux Heatmaps
    hm_c1, hm_c2, hm_c3 = st.columns(3)
    with hm_c1: 
        hm_spot_rng = st.slider("Matrix Spot Range (%)", 5, 50, 15, 5) / 100
    with hm_c2: 
        # Slider Vol SANS KEY ou avec key unique pour éviter le conflit
        hm_vol_rng = st.slider("Matrix Vol Range (pts)", 5, 50, 10, 5) / 100
    with hm_c3:
        hm_mode = st.radio("View Mode", ["2D Matrix", "3D Surface"], horizontal=True)
        if p_type == "Phoenix":
            mc_prec = st.select_slider("MC Precision", [500, 1000, 2000], value=1000)
        else:
            mc_prec = 0

    if hm_mode == "2D Matrix":
        with st.spinner("Computing Scenarios..."):
            mat_u, mat_h, x_m, y_m = prod_gk.compute_scenario_matrices(
                spot_range_pct=hm_spot_rng, vol_range_abs=hm_vol_rng, n_spot=5, n_vol=5, matrix_sims=mc_prec
            )
        
        x_lab = [f"{x:+.0%}" for x in x_m]
        y_lab = [f"{y:+.0%}" for y in y_m]
        
        fig_hm = make_subplots(rows=1, cols=2, subplot_titles=("Unhedged P&L", "Delta-Hedged P&L"))
        fig_hm.add_trace(go.Heatmap(z=mat_u, x=x_lab, y=y_lab, colorscale='RdYlGn', zmid=0, text=np.round(mat_u, 2), texttemplate="%{text}", showscale=False), row=1, col=1)
        fig_hm.add_trace(go.Heatmap(z=mat_h, x=x_lab, y=y_lab, colorscale='RdYlGn', zmid=0, text=np.round(mat_h, 2), texttemplate="%{text}", showscale=True), row=1, col=2)
        fig_hm.update_layout(height=400, margin=dict(t=50, b=50))
        fig_hm.update_xaxes(title_text="Spot Variation (%)")
        fig_hm.update_yaxes(title_text="Vol Variation (pts %)")
        st.plotly_chart(fig_hm, use_container_width=True)
    
    else:
        with st.spinner("Generating Surface..."):
            n_g = 15 if p_type != "Phoenix" else 9
            mat_u, _, x_m, y_m = prod_gk.compute_scenario_matrices(
                spot_range_pct=hm_spot_rng, vol_range_abs=hm_vol_rng, n_spot=n_g, n_vol=n_g, matrix_sims=mc_prec
            )
        
        X_pct, Y_pct = np.meshgrid(x_m * 100, y_m * 100)
        fig_3d = go.Figure(data=[go.Surface(z=mat_u, x=X_pct, y=Y_pct, colorscale='Viridis', opacity=0.9)])
        fig_3d.update_layout(title="P&L Surface", scene=dict(xaxis_title='Spot Move (%)', yaxis_title='Vol Move (pts %)', zaxis_title='P&L (€)'), height=400)
        st.plotly_chart(fig_3d, use_container_width=True)
    

    st.divider()
    st.subheader("Structural Analysis: Greeks vs Spot")

    if p_type in ["Call", "Put"]:
        # Plus de slider ici, c'est automatique (0 à 200% du Strike)
        with st.spinner("Computing Greeks Profile..."):
            # L'objet prod_gk contient déjà le dyn_spot (le point rouge) 
            # et le fixed_strike (la ligne pointillée)
            fig_structure = prod_gk.plot_greeks_profile()
            st.plotly_chart(fig_structure, use_container_width=True)
            
    elif p_type == "Phoenix":
        st.info("Structural Analysis graphs are disabled for Phoenix (Computationally too heavy).")

# ==============================================================================
# TAB 3: BACKTEST
# ==============================================================================

with tab_backtest:
    st.subheader("Dynamic Hedging Simulation")
    
    with st.container(border=True):
        # ----------------------------------------------------------------------
        # A. PARAMÈTRES DU PRODUIT (Maturité, Strike/Barrières)
        # ----------------------------------------------------------------------
        st.markdown(f"#### {p_type} Configuration")
        
        # Initialisation des variables
        bt_autocall, bt_coupon_bar, bt_protection, bt_coupon_rate = 0, 0, 0, 0
        bt_strike_pct = 1.0
        
        # Variable pour la maturité du produit (Indépendante de la durée du backtest)
        bt_maturity = 1.0 

        if p_type == "Phoenix":
            # --- CONFIG PHOENIX (5 Colonnes maintenant) ---
            # On ajoute la Maturité ici car c'est structurel
            phx_c1, phx_c2, phx_c3, phx_c4, phx_c5 = st.columns(5)
            
            with phx_c1:
                val_ac = st.number_input("Autocall Barrier (%)", value=100.0, step=5.0, key="bt_ac_input")
                bt_autocall = val_ac / 100.0
            with phx_c2:
                val_cb = st.number_input("Coupon Barrier (%)", value=60.0, step=5.0, key="bt_cb_input")
                bt_coupon_bar = val_cb / 100.0
            with phx_c3:
                val_pb = st.number_input("Protection Barrier (%)", value=60.0, step=5.0, key="bt_pb_input")
                bt_protection = val_pb / 100.0
            with phx_c4:
                val_cr = st.number_input("Annual Coupon (%)", value=8.0, step=0.5, key="bt_cr_input")
                bt_coupon_rate = val_cr / 100.0
            with phx_c5:
                # Maturité Phoenix Standard = 5 ans
                bt_maturity = st.number_input("Maturity (Years)", value=5.0, step=1.0, min_value=0.5, key="bt_mat_phx")
                
        else:
            # --- CONFIG CALL / PUT ---
            vanilla_c1, vanilla_c2, vanilla_c3 = st.columns([1, 1, 2])
            with vanilla_c1:
                bt_strike_pct = st.number_input("Strike % Init Spot", value=1.0, step=0.05, key="bt_strike_input")
            with vanilla_c2:
                # Maturité Option Vanilla Standard = 1 an (ou 3 mois = 0.25)
                bt_maturity = st.number_input("Maturity (Years)", value=1.0, step=0.25, min_value=0.1, key="bt_mat_vanilla")
            with vanilla_c3:
                st.markdown("Define the contract specifics (Strike & Maturity).")

        # ----------------------------------------------------------------------
        # B. PARAMÈTRES DE SIMULATION (Marché & Période)
        # ----------------------------------------------------------------------
        st.markdown("#### Market & Execution Settings")
        
        sim_c1, sim_c2, sim_c3 = st.columns(3)
        
        with sim_c1:
            rebal_freq = st.selectbox("Rebalancing Freq", ["Daily", "Weekly"], index=0, key="bt_freq_input")
            
        with sim_c2:
            tc_val = st.number_input("Transaction Cost (%)", value=0.10, step=0.05, format="%.2f", key="bt_tc_input")
            transaction_cost_pct = tc_val / 100.0
            
        with sim_c3:
            period_choice = st.selectbox(
                "Historical Period", 
                ["Last 3 Months", "Last 6 Months", "Last 1 Year", "Last 2 Years", "YTD"],
                index=2, 
                key="bt_period_input"
            )
            
            today = datetime.date.today()
            if period_choice == "Last 3 Months": start_d_calc = today - datetime.timedelta(days=90)
            elif period_choice == "Last 6 Months": start_d_calc = today - datetime.timedelta(days=180)
            elif period_choice == "Last 1 Year": start_d_calc = today - datetime.timedelta(days=365)
            elif period_choice == "Last 2 Years": start_d_calc = today - datetime.timedelta(days=730)
            else: start_d_calc = datetime.date(today.year, 1, 1)
            
            st.caption(f"{start_d_calc} ➝ {today}")
            date_range = (start_d_calc, today)

    # --------------------------------------------------------------------------
    # C. BOUTON D'EXÉCUTION
    # --------------------------------------------------------------------------
    if st.button("Run Backtest", type="primary", use_container_width=True):
        start_d, end_d = date_range
        lookback_start = start_d - datetime.timedelta(days=365)
        
        with st.spinner("1/3 Calibrating Historical Volatility..."):
            try:
                # --- 1. CALIBRATION ---
                md_calib = MarketData()
                df_calib = md_calib.get_historical_data(st.session_state.ticker_input, lookback_start.strftime("%Y-%m-%d"), start_d.strftime("%Y-%m-%d"))
                
                sold_vol = sigma 
                if df_calib is not None and not df_calib.empty:
                    log_rets = np.log(df_calib['Close'] / df_calib['Close'].shift(1)).dropna()
                    sold_vol = log_rets.std() * np.sqrt(252)
                    st.toast(f"Calibration Done: Sold Volatility = {sold_vol:.2%}")
                
                # --- 2. DONNÉES BACKTEST ---
                md_bt = MarketData()
                hist_data = md_bt.get_historical_data(st.session_state.ticker_input, start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d"))
                
                if hist_data is None or hist_data.empty:
                    st.error("No data found for the simulation period.")
                else:
                    init_spot = hist_data['Close'].iloc[0]
                    
                    # NOTE IMPORTANTE : 
                    # On utilise maintenant 'bt_maturity' (choisi par l'utilisateur) 
                    # au lieu de calculer la durée du backtest.

                    # --- 3. INSTANCIATION ---
                    if p_type == "Phoenix":
                        opt_hedge = PhoenixStructure(
                            S=init_spot, 
                            T=bt_maturity,  # <--- Utilisation de la maturité choisie (ex: 5.0)
                            r=r, 
                            sigma=sold_vol, 
                            q=q,
                            autocall_barrier=bt_autocall,       
                            protection_barrier=bt_protection,   
                            coupon_barrier=bt_coupon_bar,       
                            coupon_rate=bt_coupon_rate,         
                            obs_frequency=4, # Trimestriel
                            num_simulations=2000
                        )
                    else:
                        strike_bt = init_spot * bt_strike_pct
                        is_call = "Call" in p_type
                        opt_hedge = EuropeanOption(
                            S=init_spot, 
                            K=strike_bt, 
                            T=bt_maturity, # <--- Utilisation de la maturité choisie (ex: 1.0)
                            r=r, 
                            sigma=sold_vol, 
                            q=q, 
                            option_type="Call" if is_call else "Put"
                        )

                    # --- 4. LANCEMENT MOTEUR ---
                    hedging_engine = DeltaHedgingEngine(
                        option=opt_hedge, 
                        market_data=hist_data,
                        risk_free_rate=r, 
                        dividend_yield=q, 
                        volatility=sold_vol,
                        transaction_cost=transaction_cost_pct
                    )

                    res, met = hedging_engine.run_backtest()
                    
                    # Spread de Volatilité
                    vol_spread = met['Pricing Volatility'] - met['Realized Volatility']
                    is_winner = vol_spread > 0
                    
                    # Status Phoenix (si applicable)
                    if p_type == "Phoenix" and 'Status' in met:
                         st.info(f"**Product Status:** {met['Status']} on {met['Final Date']} "
                                f"after {met['Duration (Months)']:.1f} months. "
                                f"({met['Coupons Paid']} coupons paid)")

                    st.divider()
                    
                    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
                    kpi1.metric("Initial Premium", f"{met['Option Premium']:.2f} €", help="Cash reçu à la vente (t=0)")
                    kpi2.metric("Sold Volatility", f"{met['Pricing Volatility']:.2%}", help="Volatilité estimée à la vente (N-1)")
                    kpi3.metric("Realized Volatility", f"{met['Realized Volatility']:.2%}", help="Volatilité subie pendant le backtest")
                    kpi4.metric("Vol Spread", f"{vol_spread*100:+.2f} pts", delta_color="normal" if is_winner else "inverse")
                    kpi5.metric("Avg Delta", f"{res['Delta'].abs().mean():.2f}")
                    kpi6.metric("Hedge Error", f"{met['Hedge Error Std']:.2f}")
                    
                    c_pnl1, c_pnl2, c_pnl3, c_pnl4, c_pnl5 = st.columns(5)
                    
                    c_pnl1.metric("1. Premium Received", f"+{met['Option Premium']:.2f} €", help="Argent reçu à la vente (J0)")
                    c_pnl2.metric("2. Trading P&L", f"{met['Trading P&L (Gross)']:.2f} €", help="Gain/Perte brut sur les actions (Gamma Trading)")
                    c_pnl3.metric("3. Payouts Paid", f"-{met['Total Payouts']:.2f} €", help="Coupons + Remboursement versés au client")
                    c_pnl4.metric("4. Trans. Costs", f"-{met['Total Transaction Costs']:.2f} €", help="Frais de courtage cumulés")
                    
                    # P&L NET
                    net_color = "normal" if met['Total P&L'] >= 0 else "inverse"
                    c_pnl5.metric("= NET P&L", f"{met['Total P&L']:.2f} €", delta=met['Total P&L'], delta_color=net_color)
                    
                    st.caption("Equation: Net P&L = Premium + Trading P&L - Payouts - Costs")
                    

                    t1, t2 = st.tabs(["Analysis Dashboard", "Delta History"])
                    with t1:
                        fig_bt = hedging_engine.plot_pnl()
                        if fig_bt: st.plotly_chart(fig_bt, use_container_width=True, key="chart_pnl_unique")
                        else: st.warning("No data.")
                    with t2:
                        fig_d = go.Figure(go.Scatter(x=res.index, y=res['Delta'], fill='tozeroy', name='Delta', line=dict(color='purple')))
                        fig_d.update_layout(title="Hedge Ratio Evolution", template="plotly_dark", height=400)
                        st.plotly_chart(fig_d, use_container_width=True, key="chart_delta_unique")

            except Exception as e:
                st.error(f"Backtest Error: {str(e)}")