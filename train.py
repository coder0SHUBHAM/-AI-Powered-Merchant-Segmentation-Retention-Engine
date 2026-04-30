import os
import joblib
import pandas as pd
from src.preprocessing import load_and_clean_data
from src.feature_engineering import create_features
from src.segmentation import train_segmentation
from src.prediction import train_churn_model

def main():
    print("🚀 Starting Training Pipeline...")
    
    # 1. Load Data
    data_path = 'data/transactions.csv'
    if not os.path.exists(data_path):
        print(f"❌ Data file not found at {data_path}")
        return
    
    df = load_and_clean_data(data_path)
    print(f"✅ Loaded {len(df)} transactions.")
    
    # 2. Feature Engineering
    print("⚙️ Engineering features...")
    features_df = create_features(df)
    print(f"✅ Created features for {len(features_df)} merchants.")
    
    # 3. Segmentation (KMeans)
    print("🤖 Training Segmentation model...")
    kmeans, scaler, segmented_df = train_segmentation(features_df)
    
    # 4. Prediction (Random Forest)
    print("🤖 Training Churn Prediction model...")
    rf_model = train_churn_model(segmented_df)
    
    # 5. Save Models and Processed Data
    print("💾 Saving models and data...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(kmeans, 'models/kmeans.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(rf_model, 'models/rf_model.pkl')
    
    segmented_df.to_csv('data/processed_merchants.csv', index=False)
    print("✅ Pipeline complete! Models saved in models/ and processed data in data/.")

if __name__ == "__main__":
    main()
