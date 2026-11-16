import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "space_weather_dataset.csv")


def create_dataset(rows=3000, seed=42):
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2023-01-01", periods=rows, freq="h")
    t = np.arange(rows)
    speed = 420 + 65 * np.sin(t / 115) + 40 * np.sin(t / 37) + rng.normal(0, 22, rows)
    density = 6 + 2.2 * np.sin(t / 59) + rng.normal(0, 1.1, rows)
    bz = rng.normal(0, 3.2, rows) + 2.5 * np.sin(t / 83)
    temperature = 90000 + 18000 * np.sin(t / 71) + rng.normal(0, 6500, rows)
    pressure = density * (speed / 400) ** 2
    storm_centers = [520, 1120, 1840, 2470]
    kp = 2.2 + 0.006 * (speed - 400) + 0.15 * density - 0.45 * np.minimum(bz, 0) + rng.normal(0, 0.28, rows)
    for center in storm_centers:
        pulse = np.exp(-((t - center) ** 2) / (2 * 32 ** 2))
        speed += 145 * pulse
        density += 6.5 * pulse
        bz -= 10.5 * pulse
        temperature += 45000 * pulse
        pressure = density * (speed / 400) ** 2
        kp += 3.2 * pulse
    kp = np.clip(kp, 0, 9)
    sym_h = -15 - kp * 23 + rng.normal(0, 10, rows)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "solar_wind_speed_km_s": np.round(speed, 2),
        "proton_density_n_cm3": np.round(np.clip(density, 0.1, None), 2),
        "bz_nt": np.round(bz, 2),
        "temperature_k": np.round(np.clip(temperature, 1000, None), 2),
        "dynamic_pressure_npa": np.round(np.clip(pressure, 0.1, None), 2),
        "kp_index": np.round(kp, 2),
        "sym_h_nt": np.round(sym_h, 2)
    })
    df.to_csv(DATA_PATH, index=False)
    print(f"Created dataset: {DATA_PATH}")
    print(df.head())


if __name__ == "__main__":
    create_dataset()
