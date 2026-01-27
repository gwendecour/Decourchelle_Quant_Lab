import yfinance as yf
import numpy as np
import pandas as pd

class MarketData:
    
    @staticmethod
    def get_spot(ticker):
        try:
            stock = yf.Ticker(ticker)
            # Tente de récupérer le prix temps réel (Fast Info)
            try:
                price = stock.fast_info.last_price
            except:
                price = None
                
            # Si échec, repli sur la clôture de la veille
            if price is None:
                hist = stock.history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                else:
                    return None 
            return price
        except Exception as e:
            print(f"Error fetching Spot for {ticker}: {e}")
            return None

    @staticmethod
    def get_volatility(ticker, window="1y"):
        """
        Calcule la volatilité historique annualisée (Close-to-Close).
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=window)
            
            if hist.empty:
                return 0.20 # Valeur par défaut si échec
            
            # Calcul des rendements log : ln(Pt / Pt-1)
            hist['Log_Ret'] = np.log(hist['Close'] / hist['Close'].shift(1))
            
            # Écart-type annualisé
            annualized_vol = hist['Log_Ret'].std() * np.sqrt(252)
            
            if np.isnan(annualized_vol):
                return 0.20
                
            return annualized_vol
        except Exception as e:
            print(f"Error volatility: {e}")
            return 0.20

    @staticmethod
    def get_dividend_yield(ticker):
        """
        Récupère le rendement du dividende et corrige l'échelle (ex: 2.46 -> 0.0246).
        """
        try:
            stock = yf.Ticker(ticker)
            div_yield = stock.info.get('dividendYield', 0.0)
            
            if div_yield is None:
                return 0.0
            
            # SÉCURITÉ CRITIQUE :
            # Si Yahoo renvoie "2.46" pour 2.46%, on divise par 100.
            # On assume qu'aucun dividende ne dépasse 50% (0.5).
            if div_yield > 0.5:
                div_yield = div_yield / 100.0
                
            return div_yield 
        except Exception as e:
            return 0.0
        
    @staticmethod
    def get_risk_free_rate(ticker="^TNX"):
        """
        Récupère le taux sans risque via le CBOE 10-Year Treasury Note Yield (^TNX).
        Yahoo donne le taux en %, ex: 4.20. Nous le convertissons en 0.042.
        """
        try:
            # ^TNX est l'indice de référence du rendement
            bond = yf.Ticker(ticker)
            hist = bond.history(period="1d")
            
            if not hist.empty:
                # La valeur affichée est en pourcentage (ex: 4.25 pour 4.25%)
                yield_value = hist['Close'].iloc[-1]
                return yield_value / 100.0 
            
            return 0.03 # Fallback à 3%
        except Exception as e:
            print(f"Error fetching risk free rate: {e}")
            return 0.03
    
    @staticmethod
    def get_historical_data(ticker, start_date, end_date):
        try:
            # On suppose que yfinance est importé en haut ou ici
            # import yfinance as yf 
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty: 
                return None
            
            # CORRECTION ICI : On renvoie tout le DataFrame
            # Cela évite l'erreur "Series object has no attribute 'columns'" dans l'interface
            return df 
            
        except Exception as e:
            return None

    @staticmethod
    def get_clean_multiticker_data(tickers, start_date, end_date):
        """
        Télécharge, nettoie et calcule un score de qualité des données.
        Version robuste aux changements de colonnes de yfinance.
        """
        try:
            # Téléchargement
            raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            
            if raw_data.empty:
                print("Erreur: Le DataFrame téléchargé est vide.")
                return None, None

            # Sélection intelligente de la colonne de prix
            # yf.download renvoie un MultiIndex si plusieurs tickers.
            if 'Adj Close' in raw_data.columns:
                raw_df = raw_data['Adj Close']
            elif 'Close' in raw_data.columns:
                raw_df = raw_data['Close']
                print("Warning: 'Adj Close' non trouvé, utilisation de 'Close'.")
            else:
                print(f"Colonnes disponibles : {raw_data.columns}")
                return None, None

            # Diagnostic avant nettoyage
            total_points = raw_df.size
            missing_values_per_ticker = raw_df.isna().sum()
            total_missing = missing_values_per_ticker.sum()
            
            ffill_ratio = (total_missing / total_points) * 100 if total_points > 0 else 0
            
            # Nettoyage
            clean_df = raw_df.ffill().dropna(how='all')
            
            metadata = {
                'global_ffill_rate': ffill_ratio,
                'is_reliable': ffill_ratio < 5.0
            }
            
            return clean_df, metadata

        except Exception as e:
            print(f"Erreur critique dans get_clean_multiticker_data: {e}")
            return None, None