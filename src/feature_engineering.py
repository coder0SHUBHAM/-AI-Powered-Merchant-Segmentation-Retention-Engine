import pandas as pd
import numpy as np

def create_features(df):
    """
    Perform RFM and extra feature engineering.
    """
    latest_date = df['date'].max()
    
    # Group by merchant_id
    merchant_groups = df.groupby('merchant_id')
    
    # 1. RFM Features
    rfm = merchant_groups.agg({
        'date': lambda x: (latest_date - x.max()).days, # Recency
        'transaction_id': 'count',                      # Frequency
        'transaction_amount': 'sum'                     # Monetary
    }).rename(columns={
        'date': 'recency',
        'transaction_id': 'frequency',
        'transaction_amount': 'monetary'
    })
    
    # 2. Extra Features
    # Payment failure rate
    failure_rate = merchant_groups.apply(lambda x: 1 - (x['success'].sum() / len(x)))
    rfm['failure_rate'] = failure_rate
    
    # Average transaction value
    rfm['avg_ticket_size'] = rfm['monetary'] / rfm['frequency']
    
    # UPI vs Card ratio (simplified: ratio of UPI to all transactions)
    upi_ratio = merchant_groups.apply(lambda x: (x['payment_method'] == 'UPI').sum() / len(x))
    rfm['upi_ratio'] = upi_ratio
    
    # Peak usage time (simplified: most frequent hour)
    peak_hour = merchant_groups.apply(lambda x: x['date'].dt.hour.mode()[0] if not x['date'].dt.hour.mode().empty else 0)
    rfm['peak_hour'] = peak_hour

    # Churn Label (for training)
    # churn = 1 if recency > 30 days
    rfm['churn'] = (rfm['recency'] > 30).astype(int)
    
    return rfm.reset_index()
