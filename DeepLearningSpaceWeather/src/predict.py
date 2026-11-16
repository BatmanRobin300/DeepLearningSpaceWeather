import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "space_weather_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "models", "space_weather_lstm.keras")


def predict_latest():
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    feature_scaler = joblib.load(os.path.join(OUTPUT_DIR, "feature_scaler.pkl"))
    target_scaler = joblib.load(os.path.join(OUTPUT_DIR, "target_scaler.pkl"))
    df = pd.read_csv(DATA_PATH)
    features = ["solar_wind_speed_km_s", "proton_density_n_cm3", "bz_nt", "temperature_k", "dynamic_pressure_npa"]
    latest_window = df[features].tail(24)
    scaled_window = feature_scaler.transform(latest_window)
    x_latest = scaled_window.reshape(1, 24, len(features))
    model = load_model(MODEL_PATH)
    prediction_scaled = model.predict(x_latest, verbose=0)
    prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]
    risk = "LOW"
    if prediction >= 5:
        risk = "HIGH"
    elif prediction >= 4:
        risk = "MODERATE"
    print("Loading trained model...")
    print("Predicting latest geomagnetic activity...")
    print(f"Predicted Kp Index: {prediction:.2f}")
    print(f"Storm Risk Level: {risk}")
    print("Prediction complete.")


if __name__ == "__main__":
    predict_latest()
