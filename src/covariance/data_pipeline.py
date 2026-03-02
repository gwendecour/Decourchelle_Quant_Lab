import pandas as pd
import numpy as np
from src.shared.market_data import MarketData

class MarketDataProvider:
    """
    Service responsible for fetching clean 'Ground Truth' market data.
    Uses the shared MarketData component to retrieve consecutive daily prices 
    for highly liquid assets, establishing a baseline before corruption.
    """
    def __init__(self, tickers: list[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Retrieves historical adjusted closes for the initialized tickers and dates.
        Returns a cleaned, forward-filled DataFrame representing the 'Ground Truth'.
        """
        clean_df, metadata = MarketData.get_clean_multiticker_data(
            self.tickers, self.start_date, self.end_date
        )
        if clean_df is None or clean_df.empty:
            raise ValueError(f"Failed to fetch clean market data for tickers: {self.tickers}")
        return clean_df

class DataCorruptor:
    """
    Engine responsible for injecting missing values (NaNs) into a clean DataFrame.
    Simulates a degrading liquidity environment, such as untraded corporate bonds
    or trading halts, through tunable probabilistic approaches.
    """
    def __init__(self, missing_rate: float, method: str = 'MCAR', target_tickers: list[str] = None):
        """
        Args:
            missing_rate (float): Fraction of data to drop (e.g., 0.2 means 20% data deleted).
            method (str): Corruption architecture. 
                          'MCAR' (Missing Completely At Random) removes single ticks.
                          'MAR' (Missing At Random) simulates consecutive missing days (blocks).
            target_tickers (list[str]): List of column names to corrupt. If None, corrupts all.
        """
        if not (0 <= missing_rate < 1):
            raise ValueError("missing_rate must be between 0 and 1 (exclusive of 1).")
        
        self.missing_rate = missing_rate
        self.method = method.upper()
        self.target_tickers = target_tickers
        
    def corrupt(self, df: pd.DataFrame, random_state: int = None) -> pd.DataFrame:
        """
        Injects NaN values into the DataFrame based on the configured method.
        Returns a new corrupted DataFrame, keeping the original intact.
        """
        corrupted_df = df.copy()
        
        if self.missing_rate == 0:
            return corrupted_df
            
        if random_state is not None:
            np.random.seed(random_state)
            
        # Determine which columns to corrupt
        cols_to_corrupt = self.target_tickers if self.target_tickers is not None else df.columns
        cols_to_corrupt = [c for c in cols_to_corrupt if c in df.columns]
            
        if self.method == 'MCAR':
            # Missing Completely At Random: drop points independently
            for col in cols_to_corrupt:
                mask = np.random.rand(len(corrupted_df)) < self.missing_rate
                corrupted_df.loc[mask, col] = np.nan
            
        elif self.method == 'MAR':
            # Missing At Random (Block/Sequential missing): 
            # drops in short consecutive blocks (e.g., 1 to 3 days for non-synchronous trading)
            for col in cols_to_corrupt:
                n_points = len(corrupted_df)
                n_missing_target = int(n_points * self.missing_rate)
                
                if n_missing_target > 0:
                    n_missing_actual = 0
                    max_attempts = n_missing_target * 3
                    attempts = 0
                    
                    while n_missing_actual < n_missing_target and attempts < max_attempts:
                        hole_length = np.random.randint(1, 4)  # Small holes: 1 to 3 days max
                        start_idx = np.random.randint(0, max(1, n_points - hole_length))
                        
                        slice_to_nan = slice(start_idx, start_idx + hole_length)
                        col_idx = corrupted_df.columns.get_loc(col)
                        
                        current_nans = corrupted_df.iloc[slice_to_nan, col_idx].isna().sum()
                        corrupted_df.iloc[slice_to_nan, col_idx] = np.nan
                        
                        added_nans = hole_length - current_nans
                        n_missing_actual += added_nans
                        attempts += 1
                        
        else:
            raise ValueError(f"Unknown corruption method: {self.method}. Use 'MCAR' or 'MAR'.")
            
        return corrupted_df
