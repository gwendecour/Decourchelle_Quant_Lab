# ==============================================================================
# UNIVERSE CONFIGURATION & ASSET METADATA
# ==============================================================================

ASSET_DESCRIPTIONS = {
    # US EQUITIES
    'SPY': 'S&P 500 (US Large Cap)',
    'QQQ': 'Nasdaq 100 (Tech)',
    'IWM': 'Russell 2000 (Small Cap)',
    'XLE': 'Energy Sector',
    'XLF': 'Financials Sector',
    'XLK': 'Technology Sector',
    'XLV': 'Healthcare Sector',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLU': 'Utilities Sector',
    'XLI': 'Industrials Sector',
    'XLB': 'Materials Sector',

    # INTERNATIONAL EQUITIES
    'EFA': 'Dev. Markets (ex-US)',
    'EEM': 'Emerging Markets',
    'VGK': 'Europe Stocks',
    'EWJ': 'Japan Stocks',
    'MCHI': 'China Stocks',
    'INDA': 'India Stocks',

    # FIXED INCOME & BONDS
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

    # COMMODITIES
    'GLD': 'Gold',
    'SLV': 'Silver',
    'USO': 'Oil (WTI Crude)',
    'DBA': 'Agriculture',
    'DBC': 'Commodities Index',
    'UNG': 'Natural Gas',
    'COPX': 'Copper Miners',
    'PALL': 'Palladium'
}

ASSET_POOLS = {
    'Actions_US': ['SPY', 'QQQ', 'IWM', 'XLE', 'XLF', 'XLK', 'XLV', 'XLY', 'XLP', 'XLU', 'XLI', 'XLB'],
    'Actions_Intl': ['EFA', 'EEM', 'VGK', 'EWJ', 'MCHI', 'INDA'],
    'Bonds': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'BNDX', 'EMB', 'TIP', 'AGG', 'MUB'],
    'Commodities': ['GLD', 'SLV', 'USO', 'DBA', 'DBC', 'UNG', 'COPX', 'PALL']
}

def get_asset_name(ticker):
    """
    Returns the human-readable description for a given ticker symbol.
    """
    name = ASSET_DESCRIPTIONS.get(ticker)
    if name:
        return f"{ticker} | {name}"
    return ticker

def get_universe(preset_name="Standard (12)"):
    """
    Returns a dictionary of selected tickers grouped by asset class based on the UI preset.
    """
    if preset_name == "Standard (12)":
        return {
            'Actions': ['SPY', 'QQQ', 'XLE', 'XLK'],
            'Bonds': ['TLT', 'IEF', 'LQD', 'HYG'],
            'Commodities': ['GLD', 'SLV', 'USO', 'DBA']
        }
    
    elif preset_name == "Large (24)":
        return {
            'Actions': ASSET_POOLS['Actions_US'][:8], 
            'Bonds': ASSET_POOLS['Bonds'][:8],       
            'Commodities': ASSET_POOLS['Commodities'] 
        }

    elif preset_name == "No Commodities":
        return {
            'Actions': ['SPY', 'QQQ', 'IWM', 'XLE', 'XLF', 'XLK'],
            'Bonds': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'TIP']
        }
        
    elif preset_name == "Global Macro (Max)":
        return {
            'Actions': ['SPY', 'QQQ'] + ASSET_POOLS['Actions_Intl'],
            'Bonds': ASSET_POOLS['Bonds'][:6] + ['EMB'],
            'Commodities': ['GLD', 'USO', 'DBC']
        }
    
    return get_universe("Standard (12)")

TICKER_TO_CATEGORY = {}
for category, tickers in ASSET_POOLS.items():
    clean_cat = category.split('_')[0] 
    for ticker in tickers:
        TICKER_TO_CATEGORY[ticker] = clean_cat

def get_asset_class(ticker):
    """
    Returns the broad asset class categorisation for a ticker (e.g., 'Actions', 'Bonds').
    """
    return TICKER_TO_CATEGORY.get(ticker, "Other")