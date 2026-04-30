from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_churn_model(df):
    """
    Train RandomForest to predict churn.
    """
    # Features for churn prediction
    features = ['frequency', 'monetary', 'failure_rate', 'avg_ticket_size', 'upi_ratio', 'peak_hour']
    X = df[features]
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    print("Churn Model Performance:")
    print(classification_report(y_test, rf.predict(X_test)))
    
    return rf
