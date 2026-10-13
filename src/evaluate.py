import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from data_loader import load_raw_data
from preprocess import DEFAULT_FEATURES, create_test_sequences
from model import LSTMRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def compute_nasa_score(y_true, y_pred):
    """
    NASA C-MAPSS Asymmetric Scoring Function:
    Penalizes late predictions (y_pred > y_true) exponentially more heavily
    than early predictions (y_pred < y_true).
    """
    d = y_pred - y_true
    score = 0.0
    for diff in d:
        if diff < 0:
            score += np.exp(-diff / 13.0) - 1.0
        else:
            score += np.exp(diff / 10.0) - 1.0
    return float(score)


def evaluate_model(seq_length=30):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    model_path = os.path.join(MODEL_DIR, "best_lstm_model.pt")

    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Model or scaler not found. Please run train.py first.")

    print(f"[Evaluate] Loading scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)

    _, raw_test, rul_df = load_raw_data()
    y_true = rul_df["true_rul"].values.astype(np.float32)

    # Scale test data
    test_scaled = raw_test.copy()
    test_scaled[DEFAULT_FEATURES] = scaler.transform(raw_test[DEFAULT_FEATURES])

    # Extract sequences
    X_test, valid_units = create_test_sequences(test_scaled, DEFAULT_FEATURES, seq_length=seq_length)
    print(f"[Evaluate] Generated test sequences for {len(valid_units)} engines: shape {X_test.shape}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(input_dim=len(DEFAULT_FEATURES), hidden_dim_1=64, hidden_dim_2=32, fc_dim=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        inputs = torch.from_numpy(X_test).to(device)
        y_pred = model(inputs).cpu().numpy()

    # Clip negative predictions to 0 (RUL cannot be negative)
    y_pred = np.maximum(y_pred, 0.0)

    # Compute Metrics
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    nasa_score = compute_nasa_score(y_true, y_pred)

    print("\n==================================================")
    print("            FINAL BENCHMARK EVALUATION            ")
    print("==================================================")
    print(f" Total Engines Evaluated: {len(y_true)}")
    print(f" Mean Absolute Error (MAE): {mae:.2f} cycles")
    print(f" Root Mean Squared Error (RMSE): {rmse:.2f} cycles")
    print(f" NASA Asymmetric Score: {nasa_score:.2f}")
    print("==================================================")

    # Save summary metrics to JSON
    metrics = {
        "num_engines": int(len(y_true)),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "NASA_Score": round(nasa_score, 2)
    }
    with open(os.path.join(RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions dataframe
    results_df = pd.DataFrame({
        "unit_id": valid_units,
        "true_rul": y_true,
        "pred_rul": y_pred,
        "error": y_pred - y_true,
        "abs_error": np.abs(y_pred - y_true)
    })
    results_df.to_csv(os.path.join(RESULTS_DIR, "engine_predictions.csv"), index=False)

    # Plot 1: True vs Predicted RUL sorted by True RUL
    sorted_df = results_df.sort_values(by="true_rul").reset_index(drop=True)
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_df["true_rul"], label="True RUL", color="black", linewidth=2.5)
    plt.plot(sorted_df["pred_rul"], label="Predicted RUL (LSTM)", color="#e74c3c", marker="o", markersize=3.5, linestyle="None", alpha=0.85)
    plt.fill_between(
        range(len(sorted_df)),
        sorted_df["true_rul"] - mae,
        sorted_df["true_rul"] + mae,
        color="gray",
        alpha=0.15,
        label=f"MAE Band (±{mae:.1f} cycles)"
    )
    plt.xlabel("Engine Index (Sorted by Ground Truth RUL)")
    plt.ylabel("Remaining Useful Life (Cycles)")
    plt.title(f"C-MAPSS FD001: True vs Predicted RUL (100 Test Engines) | RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    p1 = os.path.join(RESULTS_DIR, "actual_vs_predicted_rul.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 2: Prediction Error Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(results_df["error"], kde=True, bins=20, color="#2980b9")
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    plt.axvline(np.mean(results_df["error"]), color="orange", linestyle=":", linewidth=2, label=f"Mean Error ({np.mean(results_df['error']):.2f})")
    plt.xlabel("Prediction Error: Pred RUL - True RUL (Cycles)")
    plt.ylabel("Engine Count")
    plt.title("RUL Prediction Error Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    p2 = os.path.join(RESULTS_DIR, "rul_error_distribution.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[Plots Saved] Generated {p1} and {p2}")
    return metrics, results_df


if __name__ == "__main__":
    evaluate_model()
