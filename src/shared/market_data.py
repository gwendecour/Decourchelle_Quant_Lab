import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

class MarketData:
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_spot(ticker):
        try:
            stock = yf.Ticker(ticker)
            # Attempt to retrieve real-time price (Fast Info)
            try:
                price = stock.fast_info.last_price
            except:
                price = None
                
            # Fallback to the previous close
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
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_volatility(ticker, window="1y"):
        """
        Calculates annualized historical volatility (Close-to-Close).
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=window)
            
            if hist.empty:
                return 0.20 # Default value on failure
            
            # Log returns: ln(Pt / Pt-1)
            hist['Log_Ret'] = np.log(hist['Close'] / hist['Close'].shift(1))
            
            # Annualized standard deviation
            annualized_vol = hist['Log_Ret'].std() * np.sqrt(252)
            
            if np.isnan(annualized_vol):
                return 0.20
                
            return annualized_vol
        except Exception as e:
            print(f"Error volatility: {e}")
            return 0.20

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_dividend_yield(ticker):
        """
        Retrieves the dividend yield and standardizes the scale (e.g., 2.46 -> 0.0246).
        """
        try:
            stock = yf.Ticker(ticker)
            div_yield = stock.info.get('dividendYield', 0.0)
            
            if div_yield is None:
                return 0.0
            
            # Scale adjustment if Yahoo returns percentages instead of decimals
            if div_yield > 0.5:
                div_yield = div_yield / 100.0
                
            return div_yield 
        except Exception as e:
            return 0.0
        
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_risk_free_rate(ticker="^TNX"):
        """
        Retrieves the risk-free rate via the CBOE 10-Year Treasury Note Yield (^TNX).
        """
        try:
            bond = yf.Ticker(ticker)
            hist = bond.history(period="1d")
            
            if not hist.empty:
                yield_value = hist['Close'].iloc[-1]
                return yield_value / 100.0 
            
            return 0.03 # Fallback to 3%
        except Exception as e:
            print(f"Error fetching risk free rate: {e}")
            return 0.03
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_historical_data(ticker, start_date, end_date):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty: 
                return None
            
            return df 
            
        except Exception as e:
            return None

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_clean_multiticker_data(tickers, start_date, end_date):
        """
        Downloads, cleans, and calculates a data quality score for multiple tickers.
        """
        try:
            # Download
            raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            
            if raw_data.empty:
                print("Error: Downloaded DataFrame is empty.")
                return None, None

            if 'Adj Close' in raw_data.columns:
                raw_df = raw_data['Adj Close']
            elif 'Close' in raw_data.columns:
                raw_df = raw_data['Close']
            else:
                return None, None

            # Diagnostics
            total_points = raw_df.size
            missing_values_per_ticker = raw_df.isna().sum()
            total_missing = missing_values_per_ticker.sum()
            
            ffill_ratio = (total_missing / total_points) * 100 if total_points > 0 else 0
            
            # Cleaning
            clean_df = raw_df.ffill().dropna(how='all')
            
            metadata = {
                'global_ffill_rate': ffill_ratio,
                'is_reliable': ffill_ratio < 5.0
            }
            
            return clean_df, metadata

        except Exception as e:
            print(f"Critical error in get_clean_multiticker_data: {e}")
            return None, None