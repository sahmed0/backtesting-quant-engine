import os
import pandas as pd
import pytest
from downloader import download_data

def test_market_data_file_formatting(tmp_path):
    """
    Test that downloaded market data files have the correct formatting
    and headers: timestamp, open, high, low, close, volume.
    """
    # Define a small date range and a single ticker to minimise download time
    tickers = ['AAPL']
    start_date = '2026-01-01'
    end_date = '2026-01-05'
    
    # Run the download function, saving to the pytest temporary directory
    download_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        output_dir=str(tmp_path)
    )
    
    # Construct the expected file path
    file_path = tmp_path / 'AAPL.csv'
    
    # Assert that the file was created
    assert file_path.exists(), "The CSV file was not created."
    
    # Read the CSV file using pandas
    df = pd.read_csv(file_path)
    
    # Define the expected columns
    expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Check if the actual columns match the expected columns
    actual_columns = df.columns.tolist()
    assert actual_columns == expected_columns, f"Expected columns {expected_columns}, but got {actual_columns}"
    
    # Also check that the data is not empty
    assert not df.empty, "The downloaded data file is empty."
