Predictive Maintenance & Remaining Useful Life (RUL) Estimation

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-ff4b4b.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/NASA-C--MAPSS%20FD001-blue.svg)](https://ti.arc.nasa.gov/)

 
> Predicting Remaining Useful Life of Industrial Equipment using LSTM-based Deep Learning on the NASA C-MAPSS Turbofan Degradation Dataset.

---

## 📌 Project Overview & Industry 4.0 Context
In modern smart manufacturing and aviation operations, unexpected equipment downtime causes catastrophic production losses and safety risks. Traditional preventive maintenance replaces expensive components prematurely based on static schedules. 

This project implements a **Cyber-Physical Prognostic System** that learns degradation trajectories directly from multivariate sensor streams and predicts remaining operating cycles before failure.

### Key Highlights
- **NASA C-MAPSS FD001 Benchmark:** 100 run-to-failure engines in training, 100 test engines with unknown cutoff cycles.
- **Sensor Pruning:** Variance-based removal of 7 uninformative/flat sensors.
- **Piecewise Linear RUL Target Function:** Upper bound clipping at $RUL_{max} = 125$ to prevent learning false degradation during healthy cycles.
- **Overfitting Resolution:** Demonstrates before-and-after results comparing an overparameterized 3-layer LSTM vs. a regularized 2-layer LSTM with dropout and early stopping.
- **Benchmark Performance:** Achieves **MAE ~10.6 cycles** and **RMSE ~14.05 cycles** across 100 test engines.
- **Interactive Industrial UI:** Live Streamlit dashboard for fleet health monitoring, countdown timers, and multi-sensor telemetry inspection.

---

## 🏗️ Project Architecture

```
predictive_maintenance_rul/
├── data/
│   └── raw/                       # C-MAPSS FD001 text files (auto-downloaded)
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
├── models/                        # Saved model weights & scaler
│   ├── best_lstm_model.pt
│   ├── scaler.pkl
│   └── train_metadata.json
├── results/                       # High-resolution evaluation charts & metrics
│   ├── actual_vs_predicted_rul.png
│   ├── rul_error_distribution.png
│   ├── training_loss_curve.png
│   ├── overfitting_comparison.png
│   ├── engine_predictions.csv
│   └── test_metrics.json
├── src/
│   ├── data_loader.py             # Auto-downloads and parses raw sensor streams
│   ├── preprocess.py              # Pruning, MinMaxScaler, sliding window sequences
│   ├── model.py                   # Regularized LSTM architecture
│   ├── train.py                   # Training loop, early stopping, LR scheduler
│   ├── evaluate.py                # MAE, RMSE, and NASA asymmetric score calculation
│   └── overfitting_experiment.py  # Before vs After overfitting proof experiment
├── app.py                         # Streamlit fleet monitoring web application
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Activate Environment
```bash
# Windows
.\.venv\Scripts\activate
```

### 2. Run Data Ingestion & Training Pipeline
Automatically downloads the C-MAPSS dataset, applies piecewise clipping, scales features, and trains the regularized LSTM:
```bash
python src/train.py
```

### 3. Evaluate on 100 Unseen Test Turbofans
Computes MAE, RMSE, and NASA Asymmetric Penalty Score, generating evaluation plots:
```bash
python src/evaluate.py
```

### 4. Run Overfitting Diagnosis Experiment (Before & After)
Reproduces the side-by-side training curves proving how label clipping, architecture downsizing, and early stopping resolved overfitting:
```bash
python src/overfitting_experiment.py
```

### 5. Launch the Streamlit Industrial Dashboard
```bash
streamlit run app.py
```

---

## 📊 Evaluation Metrics
- **Mean Absolute Error (MAE)** 
- **Root Mean Squared Error (RMSE)** 
- **NASA Asymmetric Scoring Function**

