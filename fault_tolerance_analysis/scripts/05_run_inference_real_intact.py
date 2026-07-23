"""
Step 4/5 (validated version): Cross-zone consistency inference on normal /
cyberattack / line-trip / DER-dropout data, using the ACTUAL trained
INTACT-personalized weights (intact_zone{1-4}_final_weights.npz) and the
same admittance-weighted consistency math from FL/Server/test_intact.py.

Meant to run INSIDE the same Docker container that trained these weights,
so there's no Keras/TensorFlow version mismatch to fight.

Expected container paths (adjust the CONFIG block below if yours differ):
  /app/src/models/intact_zone{1-4}_final_weights.npz   (already there)
  /app/src/models/intact_run_config.json               (already there)
  /app/src/data/centralized_train_combined.csv          (already there, for scaler fitting)
  /app/src/data/centralized_test_combined.csv           (already there, normal + existing cyberattacks)
  /app/src/data/zone_admittance.csv                      (already there)
  /app/src/data/zone2_test_stealthy.csv                  (COPY IN, for per-zone attack labels)
  /app/src/data/zone3_test_stealthy.csv                  (COPY IN)
  /app/src/data/zone4_test_stealthy.csv                  (COPY IN)
  /app/src/data/centralized_test_line_trip.csv           (COPY IN, generated locally)
  /app/src/data/centralized_test_der_dropout.csv         (COPY IN, generated locally)

Files to copy into the container before running (from the local
fault_tolerance_analysis/data/ and IDS DATASET/FL_DATA/ folders):
  centralized_test_line_trip.csv
  centralized_test_der_dropout.csv
  zone2_test_stealthy.csv, zone3_test_stealthy.csv, zone4_test_stealthy.csv
    (only needed if not already present in the container's /app/src/data/)

Run inside the container:
  docker cp 05_run_inference_real_intact.py <container>:/app/src/
  docker cp centralized_test_line_trip.csv <container>:/app/src/data/
  docker cp centralized_test_der_dropout.csv <container>:/app/src/data/
  docker exec -it <container> bash
  cd /app/src && python 05_run_inference_real_intact.py

Writes (inside container, copy back out afterward):
  /app/src/results/fault_tolerance_window_scores.csv
  /app/src/results/fault_tolerance_summary_by_category.csv
"""
import os
import re
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# ── CONFIG — adjust paths here if your container layout differs ─────────────
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
NUM_FEATURES = 36  # INTACT models are uniformly built with 36 input features for all zones
BATCH_SIZE = 64

ZONE_NAMES = ["zone1", "zone2", "zone3", "zone4"]
ZONE_BUSES = {
    "zone1": range(1, 9), "zone2": range(9, 18),
    "zone3": range(18, 25), "zone4": range(25, 33),
}


# ── Model / data helpers (mirrors FL/Server/test_intact.py exactly) ─────────

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


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading INTACT models and weights...")
    models = {}
    for z in ZONE_NAMES:
        n = z.replace("zone", "")
        m = build_model()
        m = load_weights(m, os.path.join(MODELS_DIR, f"intact_{z}_final_weights.npz"))
        models[z] = m

    W, gamma = load_admittance_weights(ADM_PATH)
    print(f"gamma = {gamma}")
    for z in ZONE_NAMES:
        print(f"  W[{z}] = {W[z]}")

    print("\nFitting per-zone scalers on centralized_train_combined.csv (matches original methodology)...")
    train_df = pd.read_csv(TRAIN_PATH)
    drop = [c for c in train_df.columns if "timestamp" in c.lower() or c.lower() == "time"]
    train_df.drop(columns=drop, inplace=True, errors="ignore")
    train_df.drop(columns=["attack_label"], inplace=True, errors="ignore")

    scalers = {}
    for z in ZONE_NAMES:
        _, scalers[z] = zone_errors(models[z], train_df, z, scaler=None)

    all_rows = []

    for ds_name, path in DATASETS.items():
        print(f"\n=== Dataset: {ds_name} ===")
        df = pd.read_csv(path)
        drop = [c for c in df.columns if "timestamp" in c.lower() or c.lower() == "time"]
        drop = [c for c in drop if c != "timestamp"]  # keep timestamp for alignment checks below
        feat_df = df.drop(columns=[c for c in ["attack_label", "fault_label", "fault_type"]
                                    if c in df.columns])

        n_windows = len(df) - WINDOW_SIZE + 1
        attack_row = df["attack_label"].values if "attack_label" in df.columns else np.zeros(len(df), dtype=int)
        fault_row = df["fault_label"].values if "fault_label" in df.columns else np.zeros(len(df), dtype=int)
        fault_type_row = df["fault_type"].values if "fault_type" in df.columns else np.array(["none"] * len(df))

        attack_win = window_any(attack_row, n_windows)
        fault_win = window_any(fault_row, n_windows)
        fault_type_win = window_mode_faulttype(fault_type_row, n_windows)

        zone_attack_win = {}
        if ds_name == "normal_attack":
            for z, path_z in ZONE_ATTACK_CSV.items():
                zdf = pd.read_csv(path_z)
                assert (zdf["timestamp"].values == df["timestamp"].values).all(), \
                    f"timestamp misalignment: {path_z}"
                zone_attack_win[z] = window_any(zdf["attack_label"].values, n_windows)

        errs = {}
        for z in ZONE_NAMES:
            print(f"  computing errors for {z}...")
            errs[z], _ = zone_errors(models[z], feat_df, z, scaler=scalers[z])

        for z in ZONE_NAMES:  # include zone1 too, to match Eq. 11's MAX across all four zones
            e = errs[z]
            e_hat = np.zeros_like(e)
            for other, w in W[z].items():
                e_hat += w * errs[other]
            mismatch = e - e_hat
            score = e + gamma * mismatch

            for i in range(n_windows):
                if attack_win[i] and ds_name == "normal_attack":
                    # zone1 has no DER and is never itself attacked -- always a bystander
                    is_own = zone_attack_win[z][i] if z in zone_attack_win else False
                    category = "attack_own_zone" if is_own else "attack_other_zone"
                elif fault_win[i]:
                    category = fault_type_win[i]
                else:
                    category = "normal"
                all_rows.append({
                    "dataset": ds_name, "zone": z, "window_idx": i,
                    "category": category, "e": e[i], "e_hat": e_hat[i],
                    "mismatch": mismatch[i], "score": score[i],
                })

    results = pd.DataFrame(all_rows)
    results.to_csv(os.path.join(RESULTS_DIR, "fault_tolerance_window_scores.csv"), index=False)
    print(f"\nSaved {len(results)} window-level rows.")

    results_dedup = results[~((results["category"] == "normal") & (results["dataset"] != "normal_attack"))]
    summary = results_dedup.groupby(["zone", "category"]).agg(
        n=("e", "size"),
        e_mean=("e", "mean"), e_median=("e", "median"),
        mismatch_mean=("mismatch", "mean"), mismatch_median=("mismatch", "median"),
        score_mean=("score", "mean"), score_median=("score", "median"),
    ).reset_index()
    summary.to_csv(os.path.join(RESULTS_DIR, "fault_tolerance_summary_by_category.csv"), index=False)
    print("\n=== Summary by zone and category (REAL INTACT weights) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
