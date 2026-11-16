import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "space_weather_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FEATURES = ["solar_wind_speed_km_s", "proton_density_n_cm3", "bz_nt", "temperature_k", "dynamic_pressure_npa"]
TARGET = "kp_index"
WINDOW_SIZE = 24
TEST_SIZE = 0.2


def create_sequences(features, targets, window_size):
    x_values = []
    y_values = []
    for index in range(window_size, len(features)):
        x_values.append(features[index - window_size:index])
        y_values.append(targets[index])
    return np.array(x_values), np.array(y_values)


def preprocess_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print("Cleaning missing values...")
    df = df.dropna().reset_index(drop=True)
    split_index = int(len(df) * (1 - TEST_SIZE))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index - WINDOW_SIZE:]
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    print("Scaling data...")
    train_features = feature_scaler.fit_transform(train_df[FEATURES])
    test_features = feature_scaler.transform(test_df[FEATURES])
    train_targets = target_scaler.fit_transform(train_df[[TARGET]]).reshape(-1)
    test_targets = target_scaler.transform(test_df[[TARGET]]).reshape(-1)
    print("Creating sequences...")
    x_train, y_train = create_sequences(train_features, train_targets, WINDOW_SIZE)
    x_test, y_test = create_sequences(test_features, test_targets, WINDOW_SIZE)
    np.save(os.path.join(OUTPUT_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "x_test.npy"), x_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
    joblib.dump(feature_scaler, os.path.join(OUTPUT_DIR, "feature_scaler.pkl"))
    joblib.dump(target_scaler, os.path.join(OUTPUT_DIR, "target_scaler.pkl"))
    metadata = {"features": FEATURES, "target": TARGET, "window_size": WINDOW_SIZE, "rows": len(df), "train_sequences": int(len(x_train)), "test_sequences": int(len(x_test))}
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as file:
        json.dump(metadata, file, indent=4)
    print("Done.")
    print(f"Training sequences: {x_train.shape}")
    print(f"Testing sequences: {x_test.shape}")


if __name__ == "__main__":
    preprocess_data()
