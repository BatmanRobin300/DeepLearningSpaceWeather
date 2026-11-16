import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "models", "space_weather_lstm.keras")


def evaluate_model():
    x_test = np.load(os.path.join(OUTPUT_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))
    target_scaler = joblib.load(os.path.join(OUTPUT_DIR, "target_scaler.pkl"))
    model = load_model(MODEL_PATH)
    predictions_scaled = model.predict(x_test).reshape(-1)
    actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    mae = float(mean_absolute_error(actual, predictions))
    r2 = float(r2_score(actual, predictions))
    np.save(os.path.join(OUTPUT_DIR, "actual_kp.npy"), actual)
    np.save(os.path.join(OUTPUT_DIR, "predicted_kp.npy"), predictions)
    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as file:
        file.write(f"RMSE: {rmse:.4f}\n")
        file.write(f"MAE: {mae:.4f}\n")
        file.write(f"R2: {r2:.4f}\n")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")


if __name__ == "__main__":
    evaluate_model()
