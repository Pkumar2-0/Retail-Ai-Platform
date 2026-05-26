import joblib
import pandas as pd

# Load anomaly model
model = joblib.load(
    "ml/saved_models/anomaly_model.pkl"
)

def detect_anomaly(data):

    input_data = pd.DataFrame(
        [data]
    )

    prediction = model.predict(input_data)

    # -1 = anomaly
    # 1 = normal

    if prediction[0] == -1:
        return "Anomaly Detected"

    return "Normal Sales Pattern"