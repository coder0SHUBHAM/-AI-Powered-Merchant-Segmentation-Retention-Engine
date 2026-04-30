import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(num_merchants=200, num_transactions=5000):
    np.random.seed(42)
    
    merchant_ids = [f"M_{1000 + i}" for i in range(num_merchants)]
    payment_methods = ['UPI', 'Card', 'NetBanking', 'Wallet']
    
    data = []
    end_date = datetime(2026, 4, 30)
    start_date = end_date - timedelta(days=180)
    
    for i in range(num_transactions):
        m_id = np.random.choice(merchant_ids)
        # Give some merchants more transactions to create natural clusters
        if int(m_id.split('_')[1]) % 10 == 0:
            m_id = np.random.choice(merchant_ids[:20]) # Top merchants
            
        t_id = f"T_{100000 + i}"
        # Random date within the last 6 months
        days_offset = np.random.randint(0, 180)
        t_date = start_date + timedelta(days=days_offset)
        
        # Amount: mostly small, some large
        amount = np.random.exponential(scale=500) + 10
        
        # Success rate: generally high but varies
        success = 1 if np.random.random() < 0.92 else 0
        
        # Payment method
        p_method = np.random.choice(payment_methods, p=[0.5, 0.3, 0.1, 0.1])
        
        data.append([m_id, t_id, t_date.strftime('%Y-%m-%d %H:%M:%S'), amount, success, p_method])
        
    df = pd.DataFrame(data, columns=['merchant_id', 'transaction_id', 'date', 'transaction_amount', 'success', 'payment_method'])
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/transactions.csv', index=False)
    print(f"Generated {len(df)} transactions for {num_merchants} merchants.")

if __name__ == "__main__":
    generate_synthetic_data()
