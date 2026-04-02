"""
Script to download historical market data using yfinance.
"""

import os
import yfinance as yf
import pandas as pd
from tqdm import tqdm
from typing import List

def download_data(tickers: List[str], start_date: str, end_date: str, output_dir: str = 'data') -> None:
    """
    Downloads historical data for the given tickers and saves them as CSVs.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")

    for ticker in tqdm(tickers, desc="Downloading Tickers"):
        try:
            # Download data using yfinance
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if data.empty:
                print(f"\nWarning: No data found for {ticker}.")
                continue

            # Handle potential MultiIndex columns returned by newer yfinance versions
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]

            available_cols = data.columns.tolist()
            
            # yfinance sometimes doesn't return 'Adj Close', fallback to 'Close' if missing
            close_col_name = 'Adj Close' if 'Adj Close' in available_cols else 'Close'
            
            cols_to_keep = ['Open', 'High', 'Low', close_col_name, 'Volume']
            
            missing_cols = [c for c in cols_to_keep if c not in available_cols]
            if missing_cols:
                print(f"\nError: Missing columns {missing_cols} for {ticker}.")
                continue
                
            df = data[cols_to_keep].copy()
            
            # Rename columns to target format
            df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                close_col_name: 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            # Rename index to 'timestamp'
            df.index.name = 'timestamp'
            
            # Ensure columns are in the correct order
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Save to CSV
            file_path = os.path.join(output_dir, f"{ticker}.csv")
            df.to_csv(file_path)
            
        except Exception as e:
            print(f"\nError downloading {ticker}: {e}")

if __name__ == '__main__':
    # Example usage for testing and validation
    sample_tickers = ['AAPL', 'TSLA', 'BTC-USD', 'INVALID_TICKER']
    download_data(
        tickers=sample_tickers,
        start_date='2020-01-01',
        end_date='2023-01-01',
        output_dir='data'
    )
