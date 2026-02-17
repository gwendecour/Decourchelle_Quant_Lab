
# --- 1. MASTER DICTIONARY : Tickers & Descriptions ---
# C'est la source de vérité pour les noms affichés dans les graphiques.
ASSET_DESCRIPTIONS = {
    # --- ACTIONS US ---
    'SPY': 'S&P 500 (US Large Cap)',
    'QQQ': 'Nasdaq 100 (Tech)',
    'IWM': 'Russell 2000 (Small Cap)',
    'XLE': 'Energy (Pétrole/Gaz)',
    'XLF': 'Financials (Banques)',
    'XLK': 'Technology Sector',
    'XLV': 'Healthcare (Santé)',
    'XLY': 'Conso. Discrétionnaire',
    'XLP': 'Conso. de Base',
    'XLU': 'Utilities',
    'XLI': 'Industrials',
    'XLB': 'Materials',

    # --- ACTIONS INTERNATIONAL ---
    'EFA': 'Dev. Markets (ex-US)',
    'EEM': 'Emerging Markets',
    'VGK': 'Europe Stocks',
    'EWJ': 'Japan Stocks',
    'MCHI': 'China Stocks',
    'INDA': 'India Stocks',

    # --- BONDS (OBLIGATIONS) ---
    'TLT': 'US Treasury 20y+ (Long)',
    'IEF': 'US Treasury 7-10y (Mid)',
    'SHY': 'US Treasury 1-3y (Short)',
    'LQD': 'Corp Bonds (Inv. Grade)',
    'HYG': 'Junk Bonds (High Yield)',
    'BNDX': 'Intl Bonds (Hedged)',
    'EMB': 'Emerging Bonds',
    'TIP': 'TIPS (Inflation Protected)',
    'AGG': 'US Aggregate Bond',
    'MUB': 'Municipal Bonds',

    # --- COMMODITIES (MATIÈRES PREMIÈRES) ---
    'GLD': 'Gold (Or)',
    'SLV': 'Silver (Argent)',
    'USO': 'Oil (Pétrole WTI)',
    'DBA': 'Agriculture',
    'DBC': 'Commodities Index',
    'UNG': 'Natural Gas',
    'COPX': 'Copper (Cuivre)',
    'PALL': 'Palladium'
}

# --- 2. ASSET POOLS : Les listes brutes ---
ASSET_POOLS = {
    'Actions_US': ['SPY', 'QQQ', 'IWM', 'XLE', 'XLF', 'XLK', 'XLV', 'XLY', 'XLP', 'XLU', 'XLI', 'XLB'],
    'Actions_Intl': ['EFA', 'EEM', 'VGK', 'EWJ', 'MCHI', 'INDA'],
    'Bonds': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'BNDX', 'EMB', 'TIP', 'AGG', 'MUB'],
    'Commodities': ['GLD', 'SLV', 'USO', 'DBA', 'DBC', 'UNG', 'COPX', 'PALL']
}

# --- 3. FONCTIONS ---

def get_asset_name(ticker):
    """
    Retourne le nom lisible pour un ticker donné.
    Utilisé par les graphiques pour l'affichage.
    """
    name = ASSET_DESCRIPTIONS.get(ticker)
    if name:
        return f"{ticker} | {name}"
    return ticker

def get_universe(preset_name="Standard (12)"):
    """
    Renvoie le dictionnaire {Catégorie: [Tickers]} selon le preset choisi.
    """
    
    # PRESET 1 : STANDARD (Équilibré et Rapide)
    # Idéal pour le debug et les tests rapides.
    if preset_name == "Standard (12)":
        return {
            'Actions': ['SPY', 'QQQ', 'XLE', 'XLK'],
            'Bonds': ['TLT', 'IEF', 'LQD', 'HYG'],
            'Commodities': ['GLD', 'SLV', 'USO', 'DBA']
        }
    
    # PRESET 2 : LARGE (Diversifié)
    # Pour un backtest robuste avec plus d'options pour l'algo.
    elif preset_name == "Large (24)":
        return {
            # On prend les 8 premiers ETFs US (SPY...XLY)
            'Actions': ASSET_POOLS['Actions_US'][:8], 
            # On prend les 8 premiers Bonds (TLT...TIP)
            'Bonds': ASSET_POOLS['Bonds'][:8],       
            # On prend toutes les commodities
            'Commodities': ASSET_POOLS['Commodities'] 
        }

    # PRESET 3 : NO COMMODITIES (Classique 60/40 dynamique)
    # Pour voir si la stratégie marche sans l'Or et le Pétrole.
    elif preset_name == "No Commodities":
        return {
            'Actions': ['SPY', 'QQQ', 'IWM', 'XLE', 'XLF', 'XLK'],
            'Bonds': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'TIP']
        }
        
    # PRESET 4 : GLOBAL MACRO (Le Total)
    # Inclut l'international
    elif preset_name == "Global Macro (Max)":
        return {
            'Actions': ['SPY', 'QQQ'] + ASSET_POOLS['Actions_Intl'],
            'Bonds': ASSET_POOLS['Bonds'][:6] + ['EMB'],
            'Commodities': ['GLD', 'USO', 'DBC']
        }
    
    # Par défaut on renvoie le standard
    return get_universe("Standard (12)")

TICKER_TO_CATEGORY = {}

for category, tickers in ASSET_POOLS.items():
    # On nettoie le nom de la catégorie pour l'affichage
    # Ex: 'Actions_US' -> 'Actions', 'Actions_Intl' -> 'Actions'
    clean_cat = category.split('_')[0] 
    
    for ticker in tickers:
        TICKER_TO_CATEGORY[ticker] = clean_cat

def get_asset_class(ticker):
    """
    Retourne la catégorie d'un ticker (Actions, Bonds, Commodities).
    Renvoie 'Autre' si le ticker n'est pas dans l'univers connu.
    """
    # On gère le cas des tickers composés ou spéciaux si besoin
    # Ici on fait une recherche directe
    return TICKER_TO_CATEGORY.get(ticker, "Autre")