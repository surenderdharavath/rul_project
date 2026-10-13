import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 14 Informative sensors identified by variance analysis in C-MAPSS FD001
# Flat / constant sensors dropped: s_1, s_5, s_6, s_10, s_16, s_18, s_19, setting_3
DEFAULT_FEATURES = [
    "s_2", "s_3", "s_4", "s_7", "s_8", "s_9",
    "s_11", "s_12", "s_13", "s_14", "s_15", "s_17",
    "s_20", "s_21"
]


def add_remaining_useful_life(df, max_rul=125, clip_rul=True):
    """
    Computes linear RUL for training data where engines run to failure.
    Applies piecewise linear clipping if clip_rul is True.
    """
    df = df.copy()
    # Compute max cycle for each unit
    max_cycles = df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]
    df = df.merge(max_cycles, on="unit_id", how="left")
    
    # Linear RUL: max_cycle - current_cycle
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    
    # Piecewise linear clipping (standard practice in prognostic literature)
    if clip_rul and max_rul is not None:
        df["RUL_clipped"] = df["RUL"].clip(upper=max_rul)
    else:
        df["RUL_clipped"] = df["RUL"]
        
    return df


def fit_and_scale_features(train_df, test_df, feature_cols=DEFAULT_FEATURES, scaler_path=None):
    """
    Fits MinMaxScaler on train_df feature columns and transforms both train and test.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"[Preprocess] Scaler saved to {scaler_path}")

    return train_scaled, test_scaled, scaler


def create_train_sequences(df, feature_cols=DEFAULT_FEATURES, seq_length=30, target_col="RUL_clipped"):
    """
    Generates sliding window sequences for training.
    Returns:
        X: np.ndarray of shape (N_samples, seq_length, num_features)
        y: np.ndarray of shape (N_samples,)
    """
    X_list, y_list = [], []
    for unit_id, group in df.groupby("unit_id"):
        data = group[feature_cols].values
        targets = group[target_col].values
        num_records = len(data)
        if num_records >= seq_length:
            for i in range(seq_length, num_records + 1):
                X_list.append(data[i - seq_length:i, :])
                y_list.append(targets[i - 1])
                
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def create_test_sequences(test_df, feature_cols=DEFAULT_FEATURES, seq_length=30):
    """
    Extracts the final window of length seq_length for each test engine.
    Returns:
        X_test: np.ndarray of shape (num_test_engines, seq_length, num_features)
        valid_units: list of unit_ids corresponding to each row
    """
    X_list, valid_units = [], []
    for unit_id, group in test_df.groupby("unit_id"):
        data = group[feature_cols].values
        if len(data) >= seq_length:
            X_list.append(data[-seq_length:, :])
            valid_units.append(unit_id)
        else:
            # Pad with the earliest reading if shorter than seq_length
            pad_len = seq_length - len(data)
            pad_block = np.repeat(data[0:1, :], pad_len, axis=0)
            padded = np.vstack([pad_block, data])
            X_list.append(padded)
            valid_units.append(unit_id)

    return np.array(X_list, dtype=np.float32), valid_units


if __name__ == "__main__":
    from data_loader import load_raw_data
    tr, te, rul = load_raw_data()
    tr = add_remaining_useful_life(tr, max_rul=125)
    tr_sc, te_sc, sc = fit_and_scale_features(tr, te)
    X_tr, y_tr = create_train_sequences(tr_sc, seq_length=30)
    X_te, units = create_test_sequences(te_sc, seq_length=30)
    print(f"X_train shape: {X_tr.shape}, y_train shape: {y_tr.shape}")
    print(f"X_test shape: {X_te.shape}, Units evaluated: {len(units)}")
