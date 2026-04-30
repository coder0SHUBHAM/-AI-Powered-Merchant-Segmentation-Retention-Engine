# AI-Powered Merchant Segmentation & Retention Engine for Payment Platforms

## Project Overview

This project develops an end-to-end AI system designed for payment platforms (similar to Razorpay) to analyze merchant transaction data, segment merchants, predict churn, and provide actionable business insights. The system leverages RFM (Recency, Frequency, Monetary) analysis, KMeans clustering for segmentation, and RandomForest for churn prediction, all presented through an interactive Streamlit dashboard.

## Architecture Explanation

The project follows a modular architecture, separating concerns into distinct components:

-   **`data/`**: Stores the raw and processed transaction data.
    -   `transactions.csv`: Raw synthetic transaction data.
    -   `processed_merchants.csv`: Processed merchant features and labels.
-   **`models/`**: Stores trained machine learning models.
    -   `kmeans.pkl`: Trained KMeans clustering model.
    -   `rf_model.pkl`: Trained RandomForest churn prediction model.
    -   `scaler.pkl`: Trained StandardScaler for feature scaling.
-   **`src/`**: Contains the core Python modules for data processing and model logic.
    -   `preprocessing.py`: Handles data loading and initial cleaning.
    -   `feature_engineering.py`: Implements RFM and additional feature creation.
    -   `segmentation.py`: Contains KMeans model training and cluster assignment logic.
    -   `prediction.py`: Manages RandomForest churn model training and prediction.
    -   `explain.py`: Provides human-readable explanations for merchant predictions.
-   **`app/`**: Houses the Streamlit web application.
    -   `streamlit_app.py`: The main Streamlit application with multiple pages for dashboard visualization.
-   **`train.py`**: The main script to execute the entire training pipeline, from data loading to model saving.
-   **`requirements.txt`**: Lists all necessary Python dependencies.
-   **`README.md`**: This documentation file.

## Project Goal

The primary goal is to provide payment platforms with a robust tool to:
1.  Process and analyze transaction data.
2.  Segment merchants into meaningful groups (e.g., High Value, Growth, At-Risk, Low Activity).
3.  Predict which merchants are likely to churn.
4.  Offer business insights and explanations for predictions.
5.  Visualize key metrics and insights through a user-friendly Streamlit dashboard.

## Setup Instructions

To set up and run this project locally, follow these steps:

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd merchant-ai-engine
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### 1. Generate Synthetic Data (if not already present)

If you don't have `data/transactions.csv`, you can generate synthetic data using the provided script:

```bash
python3 generate_data.py
```

This will create `data/transactions.csv` with sample transaction records.

### 2. Train the Models

Run the training pipeline to generate features, train the segmentation and prediction models, and save them to the `models/` directory. This will also create `data/processed_merchants.csv`.

```bash
python3 train.py
```

### 3. Launch the Streamlit Dashboard

Navigate to the `app/` directory and run the Streamlit application:

```bash
cd app
streamlit run streamlit_app.py
```

This will open the dashboard in your web browser, typically at `http://localhost:8501`.

## Screenshots (Placeholders)

### Overview Page

![Overview Page Screenshot Placeholder](images/overview_page.png)

### Segmentation Page

![Segmentation Page Screenshot Placeholder](images/segmentation_page.png)

### Risk Analysis Page

![Risk Analysis Page Screenshot Placeholder](images/risk_analysis_page.png)

### Merchant Detail Page

![Merchant Detail Page Screenshot Placeholder](images/merchant_detail_page.png)
