import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data_loader import load_raw_data
from preprocess import (
    DEFAULT_FEATURES,
    add_remaining_useful_life,
    fit_and_scale_features,
    create_train_sequences,
    create_test_sequences
)
from model import LSTMRegressor, OverfittedLSTMRegressor

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(dataloader.dataset)


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * len(y_batch)
    return total_loss / len(dataloader.dataset)


def run_training(
    epochs=50,
    batch_size=64,
    lr=1e-3,
    seq_length=30,
    max_rul=125,
    patience=10,
    val_split_engines=20
):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    print("==================================================")
    print(" INDUSTRY 4.0 PREDICTIVE MAINTENANCE: LSTM RUL   ")
    print("==================================================")
    print(f"Loading C-MAPSS dataset...")
    raw_train, raw_test, ground_truth_rul = load_raw_data()
    
    # Label clipping
    train_df = add_remaining_useful_life(raw_train, max_rul=max_rul, clip_rul=True)
    
    # Split train units: e.g. units 1..80 for train, 81..100 for validation
    total_units = train_df["unit_id"].nunique()
    split_cutoff = total_units - val_split_engines
    
    tr_split = train_df[train_df["unit_id"] <= split_cutoff].copy()
    val_split = train_df[train_df["unit_id"] > split_cutoff].copy()
    
    # Scale features
    tr_scaled, val_scaled, scaler = fit_and_scale_features(
        tr_split, val_split, feature_cols=DEFAULT_FEATURES, scaler_path=scaler_path
    )
    # Also scale raw_test
    _, test_scaled, _ = fit_and_scale_features(
        tr_split, raw_test, feature_cols=DEFAULT_FEATURES
    )

    # Sequence creation
    X_train, y_train = create_train_sequences(tr_scaled, DEFAULT_FEATURES, seq_length)
    X_val, y_val = create_train_sequences(val_scaled, DEFAULT_FEATURES, seq_length)
    print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False
    )

    model = LSTMRegressor(
        input_dim=len(DEFAULT_FEATURES),
        hidden_dim_1=64,
        hidden_dim_2=32,
        fc_dim=16,
        dropout_prob=0.2
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4
    )

    best_val_loss = float("inf")
    best_model_path = os.path.join(MODEL_DIR, "best_lstm_model.pt")
    history = {"train_loss": [], "val_loss": []}
    epochs_no_improve = 0

    print("\n--- Starting Training Loop ---")
    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train RMSE: {np.sqrt(tr_loss):.2f} cycles | Val RMSE: {np.sqrt(val_loss):.2f} cycles")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Early Stopping] Triggered at epoch {epoch}. Restoring best checkpoint.")
                break

    # Save training curves
    plt.figure(figsize=(9, 5))
    plt.plot(np.sqrt(history["train_loss"]), label="Train RMSE", color="#1f77b4", linewidth=2)
    plt.plot(np.sqrt(history["val_loss"]), label="Val RMSE", color="#ff7f0e", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (Cycles)")
    plt.title("LSTM Training & Validation Loss Curves (Regularized with Early Stopping)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_curve_path = os.path.join(RESULTS_DIR, "training_loss_curve.png")
    plt.savefig(loss_curve_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot Saved] Training curve: {loss_curve_path}")

    # Save training metadata
    meta = {
        "seq_length": seq_length,
        "max_rul": max_rul,
        "features": DEFAULT_FEATURES,
        "best_val_rmse": float(np.sqrt(best_val_loss)),
        "epochs_trained": len(history["train_loss"])
    }
    with open(os.path.join(MODEL_DIR, "train_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return model, history


if __name__ == "__main__":
    run_training(epochs=40, batch_size=64, lr=1e-3, seq_length=30, max_rul=125)
