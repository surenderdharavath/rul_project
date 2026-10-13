import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data_loader import load_raw_data
from preprocess import (
    DEFAULT_FEATURES,
    add_remaining_useful_life,
    fit_and_scale_features,
    create_train_sequences
)
from model import LSTMRegressor, OverfittedLSTMRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def run_comparison(epochs=25, seq_length=30):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(" EXPERIMENT: DIAGNOSING AND FIXING OVERFITTING ")
    print("=" * 60)

    raw_train, _, _ = load_raw_data()

    # 1. Unclipped Data (Raw linear RUL) for Initial Overfitted Model
    df_unclipped = add_remaining_useful_life(raw_train, clip_rul=False)
    
    # 2. Clipped Data (Piecewise linear RUL max 125) for Proposed Model
    df_clipped = add_remaining_useful_life(raw_train, max_rul=125, clip_rul=True)

    # Train / Val splits (Engines 1-80 vs 81-100)
    tr_raw = df_unclipped[df_unclipped["unit_id"] <= 80].copy()
    val_raw = df_unclipped[df_unclipped["unit_id"] > 80].copy()

    tr_clip = df_clipped[df_clipped["unit_id"] <= 80].copy()
    val_clip = df_clipped[df_clipped["unit_id"] > 80].copy()

    # Scalers
    tr_raw_sc, val_raw_sc, _ = fit_and_scale_features(tr_raw, val_raw, DEFAULT_FEATURES)
    tr_clip_sc, val_clip_sc, _ = fit_and_scale_features(tr_clip, val_clip, DEFAULT_FEATURES)

    # Sequences
    X_tr_raw, y_tr_raw = create_train_sequences(tr_raw_sc, DEFAULT_FEATURES, seq_length, target_col="RUL")
    X_val_raw, y_val_raw = create_train_sequences(val_raw_sc, DEFAULT_FEATURES, seq_length, target_col="RUL")

    X_tr_clip, y_tr_clip = create_train_sequences(tr_clip_sc, DEFAULT_FEATURES, seq_length, target_col="RUL_clipped")
    X_val_clip, y_val_clip = create_train_sequences(val_clip_sc, DEFAULT_FEATURES, seq_length, target_col="RUL_clipped")

    loader_raw_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr_raw), torch.from_numpy(y_tr_raw)), batch_size=64, shuffle=True)
    loader_raw_val = DataLoader(TensorDataset(torch.from_numpy(X_val_raw), torch.from_numpy(y_val_raw)), batch_size=64, shuffle=False)

    loader_clip_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr_clip), torch.from_numpy(y_tr_clip)), batch_size=64, shuffle=True)
    loader_clip_val = DataLoader(TensorDataset(torch.from_numpy(X_val_clip), torch.from_numpy(y_val_clip)), batch_size=64, shuffle=False)

    # Model A: Initial Overfitted (3 deep layers, no dropout, unclipped RUL)
    print("\n[Phase 1] Training Initial Model (Unclipped RUL, Deep Network, No Regularization)...")
    model_a = OverfittedLSTMRegressor(input_dim=len(DEFAULT_FEATURES), hidden_dim_1=128, hidden_dim_2=64, hidden_dim_3=32).to(device)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    hist_a_tr, hist_a_val = [], []
    for ep in range(1, epochs + 1):
        model_a.train()
        l_tr = 0.0
        for x, y in loader_raw_tr:
            x, y = x.to(device), y.to(device)
            opt_a.zero_grad()
            p = model_a(x)
            loss = crit(p, y)
            loss.backward()
            opt_a.step()
            l_tr += loss.item() * len(y)
        l_tr /= len(loader_raw_tr.dataset)

        model_a.eval()
        l_val = 0.0
        with torch.no_grad():
            for x, y in loader_raw_val:
                x, y = x.to(device), y.to(device)
                p = model_a(x)
                l_val += crit(p, y).item() * len(y)
        l_val /= len(loader_raw_val.dataset)

        hist_a_tr.append(np.sqrt(l_tr))
        hist_a_val.append(np.sqrt(l_val))
        if ep % 5 == 0 or ep == epochs:
            print(f"  Model A Epoch [{ep:02d}/{epochs:02d}] - Train RMSE: {np.sqrt(l_tr):.2f} | Val RMSE: {np.sqrt(l_val):.2f}")

    # Model B: Proposed Regularized (2 compact layers, dropout 0.2, clipped RUL 125)
    print("\n[Phase 2] Training Proposed Model (Piecewise Clipped RUL, Compact LSTM + Dropout)...")
    model_b = LSTMRegressor(input_dim=len(DEFAULT_FEATURES), hidden_dim_1=64, hidden_dim_2=32, fc_dim=16, dropout_prob=0.2).to(device)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3, weight_decay=1e-5)

    hist_b_tr, hist_b_val = [], []
    for ep in range(1, epochs + 1):
        model_b.train()
        l_tr = 0.0
        for x, y in loader_clip_tr:
            x, y = x.to(device), y.to(device)
            opt_b.zero_grad()
            p = model_b(x)
            loss = crit(p, y)
            loss.backward()
            opt_b.step()
            l_tr += loss.item() * len(y)
        l_tr /= len(loader_clip_tr.dataset)

        model_b.eval()
        l_val = 0.0
        with torch.no_grad():
            for x, y in loader_clip_val:
                x, y = x.to(device), y.to(device)
                p = model_b(x)
                l_val += crit(p, y).item() * len(y)
        l_val /= len(loader_clip_val.dataset)

        hist_b_tr.append(np.sqrt(l_tr))
        hist_b_val.append(np.sqrt(l_val))
        if ep % 5 == 0 or ep == epochs:
            print(f"  Model B Epoch [{ep:02d}/{epochs:02d}] - Train RMSE: {np.sqrt(l_tr):.2f} | Val RMSE: {np.sqrt(l_val):.2f}")

    # Side-by-side plot demonstrating before and after
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(hist_a_tr, label="Train RMSE", color="#e74c3c", linewidth=2)
    ax1.plot(hist_a_val, label="Val RMSE", color="#8e44ad", linewidth=2, linestyle="--")
    ax1.set_title("BEFORE FIX: Initial Overfitted Model\n(Unclipped RUL, Deep Network, Diverging Val Loss)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("RMSE (Cycles)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(hist_b_tr, label="Train RMSE", color="#27ae60", linewidth=2)
    ax2.plot(hist_b_val, label="Val RMSE", color="#2980b9", linewidth=2, linestyle="--")
    ax2.set_title("AFTER FIX: Proposed Regularized Model\n(Piecewise Clipped RUL + Dropout + Compact Architecture)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RMSE (Cycles)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig_path = os.path.join(RESULTS_DIR, "overfitting_comparison.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[Comparison Complete] Saved before-and-after plot to: {fig_path}")


if __name__ == "__main__":
    run_comparison(epochs=20)
