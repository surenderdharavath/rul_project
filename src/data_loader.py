import os
import requests
import pandas as pd

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

MIRROR_URLS = {
    "train_FD001.txt": [
        "https://raw.githubusercontent.com/Azure/lstms_for_predictive_maintenance/master/Dataset/train_FD001.txt",
        "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/train_FD001.txt"
    ],
    "test_FD001.txt": [
        "https://raw.githubusercontent.com/Azure/lstms_for_predictive_maintenance/master/Dataset/test_FD001.txt",
        "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/test_FD001.txt"
    ],
    "RUL_FD001.txt": [
        "https://raw.githubusercontent.com/Azure/lstms_for_predictive_maintenance/master/Dataset/RUL_FD001.txt",
        "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/RUL_FD001.txt"
    ]
}

INDEX_COLS = ["unit_id", "cycle"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS


def download_cmapss_data(data_dir=RAW_DATA_DIR):
    """Downloads NASA C-MAPSS FD001 dataset if not present."""
    os.makedirs(data_dir, exist_ok=True)
    for filename, urls in MIRROR_URLS.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"[DataLoader] Downloading {filename}...")
            downloaded = False
            for url in urls:
                try:
                    resp = requests.get(url, timeout=20)
                    if resp.status_code == 200 and len(resp.content) > 50:
                        with open(file_path, "wb") as f:
                            f.write(resp.content)
                        print(f"[DataLoader] Successfully saved {filename} ({len(resp.content)} bytes)")
                        downloaded = True
                        break
                except Exception as e:
                    print(f"[DataLoader] Failed from {url}: {e}")
            if not downloaded:
                raise RuntimeError(f"Could not download {filename} from available mirrors.")
        else:
            print(f"[DataLoader] Found {filename} locally.")


def load_raw_data(data_dir=RAW_DATA_DIR):
    """
    Loads raw train, test, and ground-truth RUL DataFrames.
    Returns:
        train_df: pd.DataFrame
        test_df: pd.DataFrame
        rul_df: pd.DataFrame
    """
    download_cmapss_data(data_dir)
    
    train_file = os.path.join(data_dir, "train_FD001.txt")
    test_file = os.path.join(data_dir, "test_FD001.txt")
    rul_file = os.path.join(data_dir, "RUL_FD001.txt")

    train_df = pd.read_csv(train_file, sep=r"\s+", header=None)
    # If file has trailing empty column due to trailing spaces, trim to 26 columns
    train_df = train_df.iloc[:, :26]
    train_df.columns = ALL_COLS

    test_df = pd.read_csv(test_file, sep=r"\s+", header=None)
    test_df = test_df.iloc[:, :26]
    test_df.columns = ALL_COLS

    rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["true_rul"])

    return train_df, test_df, rul_df


if __name__ == "__main__":
    tr, te, rul = load_raw_data()
    print("Train shape:", tr.shape)
    print("Test shape:", te.shape)
    print("RUL shape:", rul.shape)
