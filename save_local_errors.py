"""
save_local_errors.py — run this ONCE in the same environment where the local
models were trained (correct sklearn / Keras versions).

It saves per-window reconstruction errors and labels for the centralized model
and each zone-local model as .npy files so compare_models.py can load them
without re-running Keras inference.

Usage (from project root):
    python save_local_errors.py
"""

import os
import re
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(SCRIPT_DIR, "ML model", "models")
OUT_DIR     = os.path.join(SCRIPT_DIR, "ML model", "results", "local")
TEST_CSV    = os.path.join(SCRIPT_DIR, "FL", "Server", "centralized_test_combined.csv")

WINDOW_SIZE = 30
ZONE_BUSES  = {
    'zone1': range(1, 9),
    'zone2': range(9, 18),
    'zone3': range(18, 25),
    'zone4': range(25, 33),
}

os.makedirs(OUT_DIR, exist_ok=True)


def get_zone_columns(columns, zone_id):
    buses   = ZONE_BUSES[zone_id]
    pattern = re.compile(r'_bus(' + '|'.join(str(b) for b in buses) + r')$')
    return [col for col in columns if pattern.search(col)]


def create_windows(data):
    return np.array([data[i:i + WINDOW_SIZE] for i in range(len(data) - WINDOW_SIZE + 1)])


def window_labels_from_rows(labels):
    return np.array([
        1 if np.any(labels[i:i + WINDOW_SIZE] == 1) else 0
        for i in range(len(labels) - WINDOW_SIZE + 1)
    ])


def mse_errors(model, windows):
    pred = model.predict(windows, batch_size=32, verbose=0)
    return np.mean(np.mean(np.square(windows - pred), axis=2), axis=1)


def main():
    print("Loading test data...")
    df         = pd.read_csv(TEST_CSV)
    raw_labels = df['attack_label'].values
    drop_cols  = ['attack_label'] + [c for c in df.columns
                                     if 'timestamp' in c.lower() or c.lower() == 'time']
    feature_df = df.drop(columns=drop_cols)
    labels     = window_labels_from_rows(raw_labels)
    print(f"  {len(labels)} windows | Normal: {(labels==0).sum()} | Attack: {(labels==1).sum()}")

    # ── Centralized ──────────────────────────────────────────────────────────
    print("\nCentralized model...")
    model  = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'centralized_lstm.keras'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'centralized_scaler.pkl'))
    data   = scaler.transform(feature_df.values.astype(np.float32))
    errs   = mse_errors(model, create_windows(data))
    np.save(os.path.join(OUT_DIR, 'centralized_errors.npy'), errs)
    np.save(os.path.join(OUT_DIR, 'centralized_labels.npy'), labels)
    print(f"  Saved  errors shape={errs.shape}  mean={errs.mean():.6f}")

    # ── Zone-local models ─────────────────────────────────────────────────────
    for zone_id in ['zone1', 'zone2', 'zone3', 'zone4']:
        print(f"\n{zone_id} local model...")
        model     = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, f'{zone_id}_local_lstm.keras'))
        scaler    = joblib.load(os.path.join(MODELS_DIR, f'{zone_id}_scaler.pkl'))
        zone_cols = get_zone_columns(list(feature_df.columns), zone_id)
        data      = scaler.transform(feature_df[zone_cols].values.astype(np.float32))
        errs      = mse_errors(model, create_windows(data))
        np.save(os.path.join(OUT_DIR, f'{zone_id}_local_errors.npy'), errs)
        np.save(os.path.join(OUT_DIR, f'{zone_id}_local_labels.npy'), labels)
        print(f"  Saved  errors shape={errs.shape}  mean={errs.mean():.6f}")

    print(f"\nAll saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
