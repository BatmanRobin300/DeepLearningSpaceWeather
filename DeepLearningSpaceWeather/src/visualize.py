import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def plot_training_history():
    history = np.load(os.path.join(OUTPUT_DIR, "training_history.npy"), allow_pickle=True).item()
    plt.figure(figsize=(10, 5))
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()


def plot_predictions():
    actual = np.load(os.path.join(OUTPUT_DIR, "actual_kp.npy"))
    predicted = np.load(os.path.join(OUTPUT_DIR, "predicted_kp.npy"))
    points = min(500, len(actual))
    plt.figure(figsize=(12, 5))
    plt.plot(actual[:points], label="Actual Kp")
    plt.plot(predicted[:points], label="Predicted Kp")
    plt.xlabel("Test Time Step")
    plt.ylabel("Kp Index")
    plt.title("Actual vs Predicted Geomagnetic Activity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "prediction_vs_actual.png"))
    plt.close()
    plt.figure(figsize=(6, 6))
    plt.scatter(actual, predicted, alpha=0.6)
    plt.xlabel("Actual Kp")
    plt.ylabel("Predicted Kp")
    plt.title("Prediction Scatter Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "prediction_scatter.png"))
    plt.close()


def create_visualizations():
    plot_training_history()
    plot_predictions()
    print("Saved loss_curve.png")
    print("Saved prediction_vs_actual.png")
    print("Saved prediction_scatter.png")


if __name__ == "__main__":
    create_visualizations()
