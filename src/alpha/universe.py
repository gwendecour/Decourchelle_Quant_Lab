ASSET_POOLS = {
    'Actions_US': ['SPY', 'QQQ', 'IWM', 'XLE', 'XLF', 'XLK', 'XLV', 'XLY', 'XLP', 'XLU', 'XLI', 'XLB'],
    'Actions_Intl': ['EFA', 'EEM', 'VGK', 'EWJ', 'MCHI', 'INDA'],
    'Bonds': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'BNDX', 'EMB', 'TIP', 'AGG', 'MUB'],
    'Commodities': ['GLD', 'SLV', 'USO', 'DBA', 'DBC', 'UNG', 'COPX', 'PALL']
}

def get_universe(preset_name="Standard (12)"):
    """
    Renvoie le dictionnaire {Catégorie: [Tickers]} selon le preset choisi.
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
            'Actions': ASSET_POOLS['Actions_US'][:6],
            'Bonds': ASSET_POOLS['Bonds'][:6]
        }
    
    # ... Tu peux ajouter autant de presets que tu veux
    return get_universe("Standard (12)")