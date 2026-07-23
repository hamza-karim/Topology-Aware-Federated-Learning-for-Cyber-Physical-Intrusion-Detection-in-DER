"""
Step 2 (metrics): Precision / Recall / F1 / AUC / FPR for the physical-fault
scenarios, computed the SAME way as the paper's existing Table V — a
per-zone detection threshold calibrated as a percentile of the gamma-adjusted
score on CLEAN TRAINING data (matches FL/Server/test_intact.py's per-zone
metrics section exactly), then applied to each category's test windows.

Each fault/attack category is scored against "normal" windows as the negative
class (same evaluation logic as the paper's per-zone/system evaluation).

Run in the same container as 05_run_inference_real_intact.py (needs the real
INTACT models + scalers again for the training-side threshold calibration).

  cd /app/src && python3 06_compute_detection_metrics.py

Writes:
  /app/src/results/fault_tolerance_detection_metrics.csv
"""
import os
import re
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam

DATA_DIR = "/app/src/data"
MODELS_DIR = "/app/src/models"
RESULTS_DIR = "/app/src/results"

TRAIN_PATH = os.path.join(DATA_DIR, "centralized_train_combined.csv")
DATASETS = {
    "normal_attack": os.path.join(DATA_DIR, "centralized_test_combined.csv"),
    "line_trip":     os.path.join(DATA_DIR, "centralized_test_line_trip.csv"),
    "der_dropout":   os.path.join(DATA_DIR, "centralized_test_der_dropout.csv"),
}
ZONE_ATTACK_CSV = {
    "zone2": os.path.join(DATA_DIR, "zone2_test_stealthy.csv"),
    "zone3": os.path.join(DATA_DIR, "zone3_test_stealthy.csv"),
    "zone4": os.path.join(DATA_DIR, "zone4_test_stealthy.csv"),
}
ADM_PATH = os.path.join(DATA_DIR, "zone_admittance.csv")
RUN_CONFIG_PATH = os.path.join(MODELS_DIR, "intact_run_config.json")

WINDOW_SIZE = 30
NUM_FEATURES = 36
BATCH_SIZE = 64
# Same per-zone calibration percentiles as FL/Server/test_intact.py
ZONE_OPTIMAL_PCTS = {"zone1": 99.4, "zone2": 99.1, "zone3": 99.4, "zone4": 99.9}

ZONE_NAMES = ["zone1", "zone2", "zone3", "zone4"]
ZONE_BUSES = {
    "zone1": range(1, 9), "zone2": range(9, 18),
    "zone3": range(18, 25), "zone4": range(25, 33),
}


def build_model():
    inp = Input(shape=(WINDOW_SIZE, NUM_FEATURES))
    x = LSTM(32, activation="tanh", return_sequences=True)(inp)
    x = LSTM(64, activation="tanh", return_sequences=False)(x)
    x = RepeatVector(WINDOW_SIZE)(x)
    x = LSTM(64, activation="tanh", return_sequences=True)(x)
    x = LSTM(32, activation="tanh", return_sequences=True)(x)
    x = TimeDistributed(Dense(NUM_FEATURES))(x)
    model = Model(inp, x)
    model.compile(optimizer=Adam(learning_rate=0.005730), loss="mse")
    return model


def load_weights(model, path):
    data = np.load(path)
    keys = sorted(data.files, key=lambda k: int(k.split("_")[-1]))
    model.set_weights([data[k] for k in keys])
    return model


def get_zone_columns(columns, zone_id):
    buses = ZONE_BUSES[zone_id]
    pattern = re.compile(r"_bus(" + "|".join(str(b) for b in buses) + r")$")
    return [c for c in columns if pattern.search(c)]


def pad_to_n(data, n=NUM_FEATURES):
    if data.shape[1] < n:
        pad = np.zeros((data.shape[0], n - data.shape[1]), dtype=data.dtype)
        return np.concatenate([data, pad], axis=1)
    return data


def create_windows(data):
    return np.array([data[i:i + WINDOW_SIZE] for i in range(len(data) - WINDOW_SIZE + 1)])


def zone_errors(model, df, zone_id, scaler=None):
    cols = get_zone_columns(list(df.columns), zone_id)
    data = df[cols].values.astype(np.float32)
    if scaler is None:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    data = pad_to_n(data)
    windows = create_windows(data)
    pred = model.predict(windows, batch_size=BATCH_SIZE, verbose=0)
    return np.mean(np.mean(np.square(windows - pred), axis=2), axis=1), scaler


def load_admittance_weights(path, gamma_default=0.3):
    gamma = gamma_default
    if os.path.exists(RUN_CONFIG_PATH):
        with open(RUN_CONFIG_PATH) as f:
            cfg = json.load(f)
        gamma = cfg.get("gamma", gamma_default)
    df = pd.read_csv(path)
    zone_bus_sets = {z: set(ZONE_BUSES[z]) for z in ZONE_NAMES}
    raw = {z: {z2: 0.0 for z2 in ZONE_NAMES} for z in ZONE_NAMES}
    for _, row in df.iterrows():
        fb, tb, adm = int(row["from_bus"]), int(row["to_bus"]), float(row["admittance"])
        fz = next((z for z, b in zone_bus_sets.items() if fb in b), None)
        tz = next((z for z, b in zone_bus_sets.items() if tb in b), None)
        if fz and tz and fz != tz:
            raw[fz][tz] += adm
            raw[tz][fz] += adm
    W = {}
    for z in ZONE_NAMES:
        total = sum(raw[z][z2] for z2 in ZONE_NAMES if z2 != z)
        W[z] = {z2: raw[z][z2] / total for z2 in ZONE_NAMES if z2 != z}
    return W, gamma


def window_any(rowvals, n_windows):
    return np.array([1 if np.any(rowvals[i:i + WINDOW_SIZE] == 1) else 0 for i in range(n_windows)])


def window_mode_faulttype(vals, n_windows):
    out = []
    for i in range(n_windows):
        seg = vals[i:i + WINDOW_SIZE]
        nz = [v for v in seg if v != "none"]
        out.append(nz[0] if nz else "none")
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading INTACT models...")
    models = {}
    for z in ZONE_NAMES:
        m = build_model()
        m = load_weights(m, os.path.join(MODELS_DIR, f"intact_{z}_final_weights.npz"))
        models[z] = m

    W, gamma = load_admittance_weights(ADM_PATH)
    print(f"gamma = {gamma}")

    train_df = pd.read_csv(TRAIN_PATH)
    drop = [c for c in train_df.columns if "timestamp" in c.lower() or c.lower() == "time"]
    train_df.drop(columns=drop, inplace=True, errors="ignore")
    train_df.drop(columns=["attack_label"], inplace=True, errors="ignore")

    print("Computing per-zone training errors and calibrated thresholds...")
    train_errs, scalers = {}, {}
    for z in ZONE_NAMES:
        train_errs[z], scalers[z] = zone_errors(models[z], train_df, z, scaler=None)

    adj_threshold = {}
    for z in ["zone2", "zone3", "zone4"]:
        e = train_errs[z]
        e_hat = np.zeros_like(e)
        for other, w in W[z].items():
            e_hat += w * train_errs[other]
        train_score = e + gamma * (e - e_hat)
        adj_threshold[z] = float(np.percentile(train_score, ZONE_OPTIMAL_PCTS[z]))
        print(f"  {z}: adj_threshold={adj_threshold[z]:.6f}  "
              f"(train score mean={train_score.mean():.6f}, {ZONE_OPTIMAL_PCTS[z]}th pct)")

    metric_rows = []

    for ds_name, path in DATASETS.items():
        print(f"\n=== Dataset: {ds_name} ===")
        df = pd.read_csv(path)
        feat_df = df.drop(columns=[c for c in ["attack_label", "fault_label", "fault_type"]
                                    if c in df.columns])
        n_windows = len(df) - WINDOW_SIZE + 1

        attack_row = df["attack_label"].values if "attack_label" in df.columns else np.zeros(len(df), dtype=int)
        fault_row = df["fault_label"].values if "fault_label" in df.columns else np.zeros(len(df), dtype=int)
        fault_type_row = df["fault_type"].values if "fault_type" in df.columns else np.array(["none"] * len(df))
        attack_win = window_any(attack_row, n_windows)
        fault_win = window_any(fault_row, n_windows)
        fault_type_win = window_mode_faulttype(fault_type_row, n_windows)

        # Zone-specific attack windows (was THIS zone actually the target),
        # only meaningful for the normal_attack dataset.
        zone_attack_win = {}
        if ds_name == "normal_attack":
            for z, path_z in ZONE_ATTACK_CSV.items():
                zdf = pd.read_csv(path_z)
                assert (zdf["timestamp"].values == df["timestamp"].values).all(), \
                    f"timestamp misalignment: {path_z}"
                zone_attack_win[z] = window_any(zdf["attack_label"].values, n_windows)

        errs = {}
        for z in ZONE_NAMES:
            errs[z], _ = zone_errors(models[z], feat_df, z, scaler=scalers[z])

        for z in ["zone2", "zone3", "zone4"]:
            e = errs[z]
            e_hat = np.zeros_like(e)
            for other, w in W[z].items():
                e_hat += w * errs[other]
            score = e + gamma * (e - e_hat)

            category = np.array(["normal"] * n_windows, dtype=object)
            for i in range(n_windows):
                if fault_win[i]:
                    category[i] = fault_type_win[i]
                elif attack_win[i] and ds_name == "normal_attack":
                    category[i] = "attack_own_zone" if zone_attack_win[z][i] else "attack_other_zone"

            normal_mask = category == "normal"
            for cat in sorted(set(category) - {"normal"}):
                cat_mask = category == cat
                eval_mask = normal_mask | cat_mask
                y_true = cat_mask[eval_mask].astype(int)
                y_score = score[eval_mask]
                preds = (y_score > adj_threshold[z]).astype(int)

                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true, preds, labels=[0, 1], zero_division=0)
                fpr = float(np.sum((preds == 1) & (y_true == 0)) / max(np.sum(y_true == 0), 1))
                try:
                    auc = float(roc_auc_score(y_true, y_score))
                except ValueError:
                    auc = float("nan")

                metric_rows.append({
                    "zone": z, "category": cat, "n_positive": int(y_true.sum()),
                    "n_negative": int((1 - y_true).sum()), "threshold": adj_threshold[z],
                    "precision": prec[1], "recall": rec[1], "f1": f1[1],
                    "auc": auc, "fpr": fpr,
                })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(os.path.join(RESULTS_DIR, "fault_tolerance_detection_metrics.csv"), index=False)
    print("\n=== Detection metrics (Table V style) ===")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
