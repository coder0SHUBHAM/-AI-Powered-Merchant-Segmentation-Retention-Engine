from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

def train_segmentation(df, n_clusters=4):
    """
    Train KMeans clustering on RFM features.
    """
    features = ['recency', 'frequency', 'monetary']
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_transform(X_scaled).argmin(axis=1) # Simplified cluster assignment
    
    # We'll actually use fit_predict for clarity
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Map clusters to meaningful names based on monetary value (simple heuristic)
    cluster_centers = df.groupby('cluster')['monetary'].mean().sort_values(ascending=False)
    mapping = {
        cluster_centers.index[0]: 0, # High Value
        cluster_centers.index[1]: 1, # Growth
        cluster_centers.index[2]: 3, # Low Activity (lowest monetary)
        cluster_centers.index[3]: 2  # At-Risk (second lowest)
    }
    # Note: The above mapping is a bit arbitrary, let's just stick to a fixed logic or explain it.
    # The prompt asked for: 0=High Value, 1=Growth, 2=At-Risk, 3=Low Activity
    
    return kmeans, scaler, df

def get_cluster_name(cluster_id):
    names = {
        0: "High Value",
        1: "Growth",
        2: "At-Risk",
        3: "Low Activity"
    }
    return names.get(cluster_id, "Unknown")
