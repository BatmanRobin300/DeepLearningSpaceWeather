import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_lstm_model

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "space_weather_lstm.keras")


def train_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    x_train = np.load(os.path.join(OUTPUT_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(OUTPUT_DIR, "y_train.npy"))
    x_test = np.load(os.path.join(OUTPUT_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))
    model = build_lstm_model((x_train.shape[1], x_train.shape[2]))
    model.summary()
    early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True)
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30, batch_size=32, callbacks=[early_stop, checkpoint], verbose=1)
    model.save(MODEL_PATH)
    np.save(os.path.join(OUTPUT_DIR, "training_history.npy"), history.history)
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
