import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "datasets" / "curated" / "featured_walmart.csv"
MODEL_PATH = BASE_DIR.parent / "saved_models" / "demand_forecast.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Features
X = df[
    [
        "Store",
        "Holiday_Flag",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Year",
        "Month",
        "Day",
        "Week"
    ]
]

# Target variable
y = df["Weekly_Sales"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)

print(f"Mean Absolute Error: {mae}")

# Save trained model
joblib.dump(
    model,
    MODEL_PATH
)

print("Forecasting model trained successfully!")