# Ditto (Li et al., ICML 2021) — post-training per-zone personalisation.
#
# Ditto uses the FedAvg global model as an anchor θ and fine-tunes a separate
# personalized model v per zone by minimising:
#
#   L(v) = MSE(v(x), x) + (lambda/2) * ||v - θ||²
#
# The proximal term keeps v close to the global model while allowing zone-level
# adaptation.  Unlike INTACT, there is NO topology-aware mixing — each zone
# is tuned purely on its own data.
#
# Pre-requisite: FedAvg training must be complete (fedavg_final_weights.npz).
#
# Run (inside server container):
#   python3 ditto_personalize.py                    # default lambda=1.0
#   DITTO_LAMBDA=0.5 python3 ditto_personalize.py
#   DITTO_LAMBDA=0.1 DITTO_EPOCHS=60 python3 ditto_personalize.py
#
# To sweep lambdas:
#   for L in 0.1 0.5 1.0; do DITTO_LAMBDA=$L python3 ditto_personalize.py; done

import os
import re
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

random.seed(42); np.random.seed(42); tf.random.set_seed(42)

WINDOW_SIZE  = 30
NUM_FEATURES = 36
BATCH_SIZE   = 32

ZONE_NAMES = ['zone1', 'zone2', 'zone3', 'zone4']
ZONE_BUSES = {
    'zone1': range(1, 9),
    'zone2': range(9, 18),
    'zone3': range(18, 25),
    'zone4': range(25, 33),
}

TRAIN_PATH = '/app/src/data/centralized_train_combined.csv'
MODELS_DIR = '/app/src/models'


def get_zone_columns(columns, zone_id):
    buses   = ZONE_BUSES[zone_id]
    pattern = re.compile(r'_bus(' + '|'.join(str(b) for b in buses) + r')$')
    return [col for col in columns if pattern.search(col)]


def pad_to_n(data, n=NUM_FEATURES):
    if data.shape[1] < n:
        pad = np.zeros((data.shape[0], n - data.shape[1]), dtype=data.dtype)
        return np.concatenate([data, pad], axis=1)
    return data


def create_windows(data):
    return np.array([data[i:i + WINDOW_SIZE] for i in range(len(data) - WINDOW_SIZE + 1)])


def build_model():
    inp = Input(shape=(WINDOW_SIZE, NUM_FEATURES))
    x   = LSTM(32, activation='tanh', return_sequences=True)(inp)
    x   = LSTM(64, activation='tanh', return_sequences=False)(x)
    x   = RepeatVector(WINDOW_SIZE)(x)
    x   = LSTM(64, activation='tanh', return_sequences=True)(x)
    x   = LSTM(32, activation='tanh', return_sequences=True)(x)
    x   = TimeDistributed(Dense(NUM_FEATURES))(x)
    model = Model(inp, x)
    model.compile(optimizer=Adam(learning_rate=0.005730), loss='mse')
    return model


def load_weights(model, path):
    data = np.load(path)
    keys = sorted(data.files, key=lambda k: int(k.split('_')[-1]))
    model.set_weights([data[k] for k in keys])
    return model


def load_zone_data(train_df, zone_id):
    cols   = get_zone_columns(list(train_df.columns), zone_id)
    data   = train_df[cols].values.astype(np.float32)
    scaler = MinMaxScaler()
    data   = scaler.fit_transform(data)
    data   = pad_to_n(data)
    return create_windows(data), scaler


def ditto_fine_tune(global_weights, x_train, ditto_lambda, epochs, batch_size):
    """Fine-tune personalized model v with Ditto objective.

    Initialises v = θ (global model), then minimises:
        MSE(v(x), x) + (lambda/2) * ||v - θ||²

    Returns personalised weights, final Ditto loss, and pure MSE on train data.
    """
    model = build_model()
    model.set_weights([w.copy() for w in global_weights])

    # Freeze a copy of θ as constants for the proximal term
    theta = [tf.constant(w, dtype=tf.float32) for w in global_weights]

    def ditto_loss(y_true, y_pred):
        mse  = tf.reduce_mean(tf.square(y_true - y_pred))
        prox = tf.add_n([
            tf.reduce_sum(tf.square(w - g))
            for w, g in zip(model.trainable_variables, theta)
        ])
        return mse + (ditto_lambda / 2.0) * prox

    model.compile(optimizer=Adam(learning_rate=0.005730), loss=ditto_loss)
    history = model.fit(
        x_train, x_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    ditto_obj_loss = float(history.history['loss'][-1])

    # Recompile with plain MSE to measure pure reconstruction quality
    model.compile(optimizer=model.optimizer, loss='mse')
    mse_loss = float(model.evaluate(x_train, x_train, batch_size=batch_size, verbose=0))

    return model.get_weights(), ditto_obj_loss, mse_loss


def main():
    ditto_lambda       = float(os.environ.get('DITTO_LAMBDA', '1.0'))
    personalize_epochs = int(os.environ.get('DITTO_EPOCHS', '60'))  # 20 rounds × 3 local epochs
    prefix             = f'ditto_lambda{ditto_lambda}'

    print("=" * 60, flush=True)
    print("  Ditto Personalisation", flush=True)
    print(f"  lambda  = {ditto_lambda}", flush=True)
    print(f"  epochs  = {personalize_epochs}", flush=True)
    print(f"  prefix  = {prefix}", flush=True)
    print("=" * 60, flush=True)

    # ── Load global FedAvg weights ────────────────────────────────────────────
    global_weights_path = os.path.join(MODELS_DIR, 'fedavg_final_weights.npz')
    if not os.path.exists(global_weights_path):
        print(f"ERROR: {global_weights_path} not found. Run FedAvg training first.",
              flush=True)
        return

    tmp_model = build_model()
    tmp_model = load_weights(tmp_model, global_weights_path)
    global_weights = tmp_model.get_weights()
    print(f"Loaded global FedAvg weights from {global_weights_path}", flush=True)

    # ── Load training data ────────────────────────────────────────────────────
    print("Loading training data...", flush=True)
    train_df  = pd.read_csv(TRAIN_PATH)
    drop_cols = [c for c in train_df.columns
                 if 'timestamp' in c.lower() or c.lower() == 'time' or c == 'attack_label']
    train_df  = train_df.drop(columns=drop_cols, errors='ignore')
    print(f"Training rows: {len(train_df)}  features: {len(train_df.columns)}", flush=True)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Per-zone Ditto fine-tuning ─────────────────────────────────────────────
    results = {}
    for zid in ZONE_NAMES:
        print(f"\nPersonalising {zid}  (lambda={ditto_lambda}, epochs={personalize_epochs})...",
              flush=True)
        x_train, _ = load_zone_data(train_df, zid)
        print(f"  Training windows: {len(x_train)}", flush=True)

        pers_weights, ditto_obj, mse_obj = ditto_fine_tune(
            global_weights, x_train, ditto_lambda, personalize_epochs, BATCH_SIZE
        )

        weights_path = os.path.join(MODELS_DIR, f'{prefix}_{zid}_final_weights.npz')
        np.savez(weights_path, *pers_weights)
        results[zid] = {'ditto_loss': ditto_obj, 'mse_loss': mse_obj}
        print(f"  Saved: {weights_path}", flush=True)
        print(f"  Ditto objective loss: {ditto_obj:.6f}  |  MSE only: {mse_obj:.6f}",
              flush=True)

    # ── Save run config ───────────────────────────────────────────────────────
    run_cfg = {
        'strategy':           'ditto',
        'ditto_lambda':        ditto_lambda,
        'personalize_epochs':  personalize_epochs,
        'global_model':        'fedavg_final_weights.npz',
        'fl_rounds':           20,
        'local_epochs':        3,
        'zone_results':        results,
    }
    cfg_path = os.path.join(MODELS_DIR, f'{prefix}_run_config.json')
    with open(cfg_path, 'w') as f:
        json.dump(run_cfg, f, indent=2)
    print(f"\nRun config saved to {cfg_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("Ditto personalisation complete.", flush=True)
    print(f"Next: DITTO_LAMBDA={ditto_lambda} python3 test_ditto.py", flush=True)
    print("=" * 60, flush=True)


if __name__ == '__main__':
    main()
