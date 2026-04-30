import pandas as pd

def load_and_clean_data(file_path):
    """
    Load transaction data and perform basic cleaning.
    """
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    # Ensure numerical types
    df['transaction_amount'] = pd.to_numeric(df['transaction_amount'])
    df['success'] = pd.to_numeric(df['success'])
    return df

def get_latest_date(df):
    """
    Get the most recent transaction date for recency calculation.
    """
    return df['date'].max()
