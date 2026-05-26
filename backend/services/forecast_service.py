import joblib
import pandas as pd

# Load trained model
model = joblib.load(
    "ml/saved_models/demand_forecast.pkl"
)

def predict_sales(data):

    input_data = pd.DataFrame(
        [data]
    )

    prediction = model.predict(input_data)

    return float(prediction[0])