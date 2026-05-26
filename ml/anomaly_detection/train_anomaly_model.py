import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest

# Load dataset
df = pd.read_csv(
    "ml/datasets/curated/featured_walmart.csv"
)

# Features for anomaly detection
X = df[
    [
        "Weekly_Sales",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment"
    ]
]

# Create model
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

# Train model
model.fit(X)

# Save model
joblib.dump(
    model,
    "ml/saved_models/anomaly_model.pkl"
)

print("Anomaly detection model trained successfully!")