import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.segmentation import get_cluster_name
from src.explain import explain_prediction

# Page config
st.set_page_config(page_title="Merchant AI Engine", layout="wide")

# Load data and models
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed_merchants.csv')
    return df

@st.cache_resource
def load_models():
    rf_model = joblib.load('models/rf_model.pkl')
    return rf_model

try:
    df = load_data()
    rf_model = load_models()
except Exception as e:
    st.error(f"Error loading data or models: {e}. Please run train.py first.")
    st.stop()

# Sidebar
st.sidebar.title("Merchant AI Engine")
page = st.sidebar.radio("Navigation", ["Overview", "Segmentation", "Risk Analysis", "Merchant Detail"])

if page == "Overview":
    st.title("📊 Merchant Network Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Merchants", len(df))
    with col2:
        st.metric("Total Revenue", f"${df['monetary'].sum():,.2f}")
    with col3:
        churn_rate = df['churn'].mean()
        st.metric("Churn Rate", f"{churn_rate:.1%}")
    
    # Revenue distribution
    fig = px.histogram(df, x="monetary", title="Revenue Distribution", nbins=30)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Segmentation":
    st.title("🎯 Merchant Segmentation")
    
    # Cluster distribution
    df['cluster_name'] = df['cluster'].apply(get_cluster_name)
    cluster_counts = df['cluster_name'].value_counts().reset_index()
    cluster_counts.columns = ['Segment', 'Count']
    
    fig = px.pie(cluster_counts, values='Count', names='Segment', title='Merchant Segment Distribution')
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster characteristics
    st.subheader("Segment Characteristics")
    cluster_stats = df.groupby('cluster_name').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'failure_rate': 'mean'
    }).round(2)
    st.table(cluster_stats)

elif page == "Risk Analysis":
    st.title("⚠️ Risk Analysis")
    
    at_risk = df[df['cluster'] == 2].copy() # Cluster 2 is At-Risk
    st.subheader(f"Identified At-Risk Merchants ({len(at_risk)})")
    
    if not at_risk.empty:
        st.dataframe(at_risk[['merchant_id', 'recency', 'frequency', 'monetary', 'failure_rate']])
        
        # Risk factors
        fig = px.scatter(at_risk, x="recency", y="failure_rate", size="monetary", 
                         hover_name="merchant_id", title="Recency vs Failure Rate (Size=Revenue)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No merchants currently in the high-risk segment.")

elif page == "Merchant Detail":
    st.title("🔍 Merchant Detail & Prediction")
    
    merchant_id = st.selectbox("Select Merchant ID", df['merchant_id'].unique())
    m_data = df[df['merchant_id'] == merchant_id].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Merchant Profile")
        st.write(f"**Segment:** {get_cluster_name(m_data['cluster'])}")
        st.write(f"**Recency:** {m_data['recency']} days")
        st.write(f"**Frequency:** {m_data['frequency']} transactions")
        st.write(f"**Monetary:** ${m_data['monetary']:,.2f}")
        st.write(f"**Failure Rate:** {m_data['failure_rate']:.1%}")
        
    with col2:
        st.subheader("Churn Prediction")
        # Prepare features for prediction
        features = ['frequency', 'monetary', 'failure_rate', 'avg_ticket_size', 'upi_ratio', 'peak_hour']
        pred_input = m_data[features].values.reshape(1, -1)
        churn_prob = rf_model.predict_proba(pred_input)[0][1]
        
        if churn_prob > 0.5:
            st.error(f"HIGH CHURN RISK: {churn_prob:.1%}")
        else:
            st.success(f"LOW CHURN RISK: {churn_prob:.1%}")
            
        st.subheader("AI Explanation")
        st.info(explain_prediction(m_data))
