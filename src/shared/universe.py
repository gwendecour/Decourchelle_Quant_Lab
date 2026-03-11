# ==============================================================================
# UNIVERSE CONFIGURATION & ASSET METADATA
# ==============================================================================

ASSET_DESCRIPTIONS = {
    # SINGLE STOCKS
    'NVDA': 'NVIDIA Corp', 'MC.PA': 'LVMH (Euronext)', 'WMT': 'Walmart Inc.',
    'JPM': 'JPMorgan Chase', 'JNJ': 'Johnson & Johnson', 'XOM': 'ExxonMobil', 'BA': 'Boeing',

    # US EQUITIES - Broad
    'SPY': 'S&P 500 ETF', 'QQQ': 'Nasdaq 100 ETF', 'IWM': 'Russell 2000 ETF',
    'DIA': 'Dow Jones Ind. ETF', 'VTI': 'Total Stock Market', 'VOO': 'S&P 500 (Vanguard)',
    'IVV': 'S&P 500 (iShares)', 'RSP': 'S&P 500 Equal Weight',
    
    # US EQUITIES - Sectors
    'XLE': 'Energy Sector', 'XLF': 'Financials Sector', 'XLK': 'Technology Sector',
    'XLV': 'Healthcare Sector', 'XLY': 'Consumer Discretionary', 'XLP': 'Consumer Staples',
    'XLU': 'Utilities Sector', 'XLI': 'Industrials Sector', 'XLB': 'Materials Sector',
    'XLRE': 'Real Estate Sector', 'XLC': 'Communication Services',
    
    # US EQUITIES - Factors/Style
    'MTUM': 'Momentum Factor ETF', 'VLUE': 'Value Factor ETF', 'QUAL': 'Quality Factor ETF',
    'USMV': 'Min Volatility ETF', 'VIG': 'Dividend Appreciation', 'SCHD': 'US Dividend Equity',
    'VYM': 'High Dividend Yield', 'SDY': 'S&P Dividend ETF', 'IWD': 'Russell 1000 Value',
    'IWF': 'Russell 1000 Growth', 'IJR': 'Core S&P Small-Cap', 'IWC': 'Micro-Cap ETF',

    # INTERNATIONAL EQUITIES
    'EFA': 'EAFE Developed Markets', 'VEA': 'Developed Markets ex-US', 'IEFA': 'Core EAFE',
    'EEM': 'Emerging Markets', 'VWO': 'Emerging Markets', 'IEMG': 'Core Emerging Markets',
    'VGK': 'Europe Stocks', 'EZU': 'Eurozone Stocks', 'EWJ': 'Japan Stocks',
    'MCHI': 'China Stocks', 'INDA': 'India Stocks', 'EWZ': 'Brazil Stocks',
    'EWT': 'Taiwan Stocks', 'EWY': 'South Korea Stocks', 'EWH': 'Hong Kong Stocks',
    'EWC': 'Canada Stocks', 'EWA': 'Australia Stocks', 'EWL': 'Switzerland Stocks',
    'EWG': 'Germany Stocks', 'EWQ': 'France Stocks', 'EWP': 'Spain Stocks',
    'EWD': 'Sweden Stocks', 'EWW': 'Mexico Stocks', 'ECH': 'Chile Stocks',
    'EPHE': 'Philippines Stocks', 'EIDO': 'Indonesia Stocks',

    # BONDS
    'AGG': 'US Aggregate Bond', 'BND': 'Total Return Bond', 'TLT': 'US Treasury 20y+',
    'IEF': 'US Treasury 7-10y', 'SHY': 'US Treasury 1-3y', 'GOVT': 'US Treasury Bonds',
    'VGIT': 'Intermediate Treasury', 'VCIT': 'Intermediate Corporate', 'VCSH': 'Short-Term Corporate',
    'LQD': 'Inv. Grade Corp Bonds', 'HYG': 'High Yield Corp Bonds', 'JNK': 'High Yield Bonds',
    'USIG': 'Broad Inv. Grade', 'IGSB': 'Short-Term Inv. Grade', 'SJNK': 'Short-Term High Yield',
    'BNDX': 'Intl Bonds (Hedged)', 'EMB': 'Emerging Markets Bonds (USD)', 'VWOB': 'Emerging Markets Bonds',
    'PCY': 'Emerging Markets Sovereign', 'EBND': 'Local Currency EM Bonds', 'TIP': 'TIPS (Inflation Protected)',
    'VTIP': 'Short-Term TIPS', 'MUB': 'National Muni Bond', 'MBB': 'Mortgage-Backed Bond',
    'CWB': 'Convertible Bonds', 'FALN': 'Fallen Angels High Yield',

    # COMMODITIES
    'DBC': 'Commodity Index Tracking', 'PDBC': 'Optimum Yield Commodity', 'GSG': 'GSCI Commodity-Indexed',
    'COMT': 'Commodities Strategy', 'FTGC': 'Global Commodities Strategy', 'GLD': 'Gold',
    'IAU': 'Gold (Mini)', 'SLV': 'Silver', 'SIVR': 'Physical Silver', 'PPLT': 'Physical Platinum',
    'PALL': 'Physical Palladium', 'SGOL': 'Physical Swiss Gold', 'USO': 'WTI Crude Oil',
    'BNO': 'Brent Oil', 'UGA': 'Gasoline', 'UNG': 'Natural Gas', 'DBA': 'Agriculture',
    'WEAT': 'Wheat', 'CORN': 'Corn', 'SOYB': 'Soybeans', 'CPER': 'Copper',
    'JJC': 'Copper Index', 'LIT': 'Lithium & Battery Tech', 'URA': 'Uranium', 'COPX': 'Copper Miners',

    # REAL ESTATE
    'VNQ': 'Real Estate (US)', 'SCHH': 'US REIT ETF', 'IYR': 'US Real Estate',
    'USRT': 'Core US REIT', 'REM': 'Mortgage Real Estate', 'MORT': 'VanEck Mortgage REIT',
    'VNQI': 'Global ex-US Real Estate', 'RWX': 'Intl Real Estate', 'REET': 'Global Real Estate',

    # CRYPTO
    'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum', 'BNB-USD': 'Binance Coin',
    'SOL-USD': 'Solana', 'XRP-USD': 'XRP', 'ADA-USD': 'Cardano',
    'AVAX-USD': 'Avalanche', 'DOGE-USD': 'Dogecoin', 'DOT-USD': 'Polkadot',
    'LINK-USD': 'Chainlink', 'MATIC-USD': 'Polygon', 'LTC-USD': 'Litecoin',

    # MUTUAL FUNDS
    'FXAIX': 'Fidelity 500 Index', 'VFIAX': 'Vanguard 500 Index', 'VTSAX': 'Vanguard Total Stock Market',
    'VSMAX': 'Vanguard Small-Cap Index', 'VGSLX': 'Vanguard Real Estate Index', 'VTIAX': 'Vanguard Total Intl Stock',
    'PTTRX': 'PIMCO Total Return', 'PONAX': 'PIMCO Income Fund', 'PRWCX': 'T. Rowe Price Capital App',
    'VWENX': 'Vanguard Wellington', 'FDGRX': 'Fidelity Growth Company', 'FCNTX': 'Fidelity Contrafund',

    # ALTERNATIVES & VOLATILITY
    'VIXY': 'VIX Short-Term Futures', 'VXX': 'iPath Series B S&P 500 VIX', 'UVXY': 'ProShares Ultra VIX (1.5x)',
    'SVXY': 'ProShares Short VIX (-0.5x)', 'QAI': 'IQ Hedge Multi-Strategy', 'MNA': 'IQ MacKay Shields Merger Arb',
    'BTAL': 'AGFiQ US Market Neutral Anti-Beta', 'RLY': 'SPDR SSGA Multi-Asset Real Return',
    'HTUS': 'Hull Tactical US', 'WTRE': 'WisdomTree Managed Futures Strategy',

    # FX PAIRS
    'EURUSD=X': 'EUR/USD', 'JPY=X': 'USD/JPY (Inverse)', 'GBPUSD=X': 'GBP/USD',
    'AUDUSD=X': 'AUD/USD', 'CAD=X': 'USD/CAD (Inverse)', 'CHF=X': 'USD/CHF (Inverse)',
    'NZDUSD=X': 'NZD/USD', 'CNY=X': 'USD/CNY (Inverse)', 'MXN=X': 'USD/MXN (Inverse)',
    'BRL=X': 'USD/BRL (Inverse)', 'ZAR=X': 'USD/ZAR (Inverse)', 'INR=X': 'USD/INR (Inverse)'
}

ASSET_POOLS = {
    'Equities US': [
        'NVDA', 'WMT', 'JPM', 'JNJ', 'XOM', 'BA',
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'IVV', 'RSP',                     
        'XLE', 'XLF', 'XLK', 'XLV', 'XLY', 'XLP', 'XLU', 'XLI', 'XLB', 'XLRE', 'XLC', 
        'MTUM', 'VLUE', 'QUAL', 'USMV', 'VIG', 'SCHD', 'VYM', 'SDY',                
        'IWD', 'IWF', 'IJR', 'IWC'                                           
    ],
    'Equities Intl': [
        'MC.PA',
        'EFA', 'VEA', 'IEFA', 'EEM', 'VWO', 'IEMG',                                 
        'VGK', 'EZU', 'EWJ', 'MCHI', 'INDA', 'EWZ', 'EWT', 'EWY', 'EWH', 'EWC',     
        'EWA', 'EWL', 'EWG', 'EWQ', 'EWP', 'EWD', 'EWW', 'ECH', 'EPHE', 'EIDO'      
    ],
    'Bonds': [
        'TLT', 'IEF', 'SHY', 'AGG', 'BND', 'GOVT', 'VGIT', 'VCIT', 'VCSH',          
        'LQD', 'HYG', 'JNK', 'USIG', 'IGSB', 'SJNK',                                
        'BNDX', 'EMB', 'VWOB', 'PCY', 'EBND',                                       
        'TIP', 'VTIP', 'MUB', 'MBB', 'CWB', 'FALN'                                  
    ],
    'Commodities': [
        'GLD', 'SLV', 'DBC', 'PDBC', 'GSG', 'COMT', 'FTGC',                                       
        'IAU', 'SIVR', 'PPLT', 'PALL', 'SGOL',                        
        'USO', 'BNO', 'UGA', 'UNG',                                                 
        'DBA', 'WEAT', 'CORN', 'SOYB',                                              
        'CPER', 'JJC', 'LIT', 'URA', 'COPX'                                         
    ],
    'Real Estate': [
        'VNQ', 'SCHH', 'XLRE', 'IYR', 'USRT', 'REM', 'MORT',                        
        'VNQI', 'RWX', 'REET'                                                       
    ],
    'Crypto': [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 
        'AVAX-USD', 'DOGE-USD', 'DOT-USD', 'LINK-USD', 'MATIC-USD', 'LTC-USD'
    ],
    'Mutual Funds': [
        'FXAIX', 'VFIAX', 'VTSAX', 'VSMAX', 'VGSLX', 'VTIAX', 
        'PTTRX', 'PONAX', 'PRWCX', 'VWENX', 'FDGRX', 'FCNTX'
    ],
    'Alternatives & Volatility': [
        'VIXY', 'VXX', 'UVXY', 'SVXY',                                              
        'QAI', 'MNA', 'BTAL', 'RLY', 'HTUS', 'WTRE'                                 
    ],
    'FX': [
        'EURUSD=X', 'JPY=X', 'GBPUSD=X', 'AUDUSD=X', 'CAD=X', 'CHF=X',
        'NZDUSD=X', 'CNY=X', 'MXN=X', 'BRL=X', 'ZAR=X', 'INR=X'
    ]
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
    if preset_name == "Small (6)":
        return {
            'Actions': ['SPY', 'QQQ'],
            'Bonds': ['TLT', 'IEF'],
            'Commodities': ['GLD', 'SLV']
        }
    
    elif preset_name == "Standard (12)":
        return {
            'Actions': ['SPY', 'QQQ', 'XLE', 'XLK'],
            'Bonds': ['TLT', 'IEF', 'LQD', 'HYG'],
            'Commodities': ['GLD', 'SLV', 'USO', 'DBA']
        }
    
    elif preset_name == "Large (24)":
        return {
            'Actions': ASSET_POOLS['Equities US'][:8], 
            'Bonds': ASSET_POOLS['Bonds'][:8],       
            'Commodities': ASSET_POOLS['Commodities'][:8] 
        }
        
    elif preset_name == "Global Macro (Max - 48)":
        return {
            'Actions': ASSET_POOLS['Equities US'][:16],
            'Bonds': ASSET_POOLS['Bonds'][:16],
            'Commodities': ASSET_POOLS['Commodities'][:16]
        }
    
    return get_universe("Standard (12)")

TICKER_TO_CATEGORY = {}
for category, tickers in ASSET_POOLS.items():
    clean_cat = category 
    for ticker in tickers:
        TICKER_TO_CATEGORY[ticker] = clean_cat

def get_asset_class(ticker):
    """
    Returns the broad asset class categorisation for a ticker (e.g., 'Actions', 'Bonds').
    """
    return TICKER_TO_CATEGORY.get(ticker, "Other")