# Run on server device AFTER training is complete:
# docker exec -it <server_container> bash
# cd /app/src && python test_server.py

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    precision_recall_fscore_support,
)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam

WINDOW_SIZE = 30
NUM_FEATURES = 36
FL_ROUNDS = 10
LOCAL_EPOCHS = 5
NUM_CLIENTS = 4
BATCH_SIZE = 32
OPTIMAL_PERCENTILE = 99.4

ZONE_BUSES = {
    'zone1': range(1, 9),    # buses 1-8,   32 features
    'zone2': range(9, 18),   # buses 9-17,  36 features
    'zone3': range(18, 25),  # buses 18-24, 28 features
    'zone4': range(25, 33),  # buses 25-32, 32 features
}

TRAIN_PATH   = '/app/src/data/centralized_train_combined.csv'
DATA_PATH    = '/app/src/data/centralized_test_combined.csv'
WEIGHTS_PATH = '/app/src/models/fedavg_final_weights.npz'
RESULTS_DIR  = '/app/src/results'


def get_zone_columns(columns, zone_id):
    buses = ZONE_BUSES[zone_id]
    pattern = re.compile(r'_bus(' + '|'.join(str(b) for b in buses) + r')$')
    return [col for col in columns if pattern.search(col)]


def build_model():
    inp = Input(shape=(WINDOW_SIZE, NUM_FEATURES))
    x = LSTM(32, activation='tanh', return_sequences=True)(inp)
    x = LSTM(64, activation='tanh', return_sequences=False)(x)
    x = RepeatVector(WINDOW_SIZE)(x)
    x = LSTM(64, activation='tanh', return_sequences=True)(x)
    x = LSTM(32, activation='tanh', return_sequences=True)(x)
    x = TimeDistributed(Dense(NUM_FEATURES))(x)
    model = Model(inp, x)
    model.compile(optimizer=Adam(learning_rate=0.0011), loss='mse')
    return model


def load_weights(model, path):
    data = np.load(path)
    keys = sorted(data.files, key=lambda k: int(k.split('_')[-1]))
    model.set_weights([data[k] for k in keys])
    return model


def create_windows(data):
    return np.array([data[i:i + WINDOW_SIZE] for i in range(len(data) - WINDOW_SIZE + 1)])


def pad_to_n(data, n=NUM_FEATURES):
    if data.shape[1] < n:
        pad = np.zeros((data.shape[0], n - data.shape[1]), dtype=data.dtype)
        return np.concatenate([data, pad], axis=1)
    return data


def window_labels_from_rows(labels):
    return np.array(
        [1 if np.any(labels[i:i + WINDOW_SIZE] == 1) else 0
         for i in range(len(labels) - WINDOW_SIZE + 1)]
    )


def zone_reconstruction_errors(model, scaled_all, feature_cols, zone_id):
    zone_cols = get_zone_columns(feature_cols, zone_id)
    col_idx = [feature_cols.index(c) for c in zone_cols]
    zone_data = pad_to_n(scaled_all[:, col_idx])
    windows = create_windows(zone_data)
    pred = model.predict(windows, batch_size=BATCH_SIZE, verbose=0)
    return np.mean(np.mean(np.square(windows - pred), axis=2), axis=1)


def format_summary(threshold, prec, rec, f1, support, auc):
    sep = '=' * 60
    w = 14  # name column width: len("Replay Attack") + 1
    col = 9
    header = f"{'':>{w}} {'precision':>{col}} {'recall':>{col}} {'f1-score':>{col}} {'support':>{col}}"
    normal_row = (
        f"{'Normal':>{w}} {prec[0]:>{col}.2f} {rec[0]:>{col}.2f}"
        f" {f1[0]:>{col}.2f} {int(support[0]):>{col}d}"
    )
    attack_row = (
        f"{'Replay Attack':>{w}} {prec[1]:>{col}.2f} {rec[1]:>{col}.2f}"
        f" {f1[1]:>{col}.2f} {int(support[1]):>{col}d}"
    )
    return (
        f"{sep}\n"
        f"FINAL RESULTS — FEDAVG — FL LSTM Autoencoder\n"
        f"FL rounds     : {FL_ROUNDS}\n"
        f"Local epochs  : {LOCAL_EPOCHS}\n"
        f"Num clients   : {NUM_CLIENTS}\n"
        f"{sep}\n"
        f"Optimal threshold: {threshold:.6f} ({OPTIMAL_PERCENTILE}th percentile)\n"
        f"Detection metrics:\n"
        f"{header}\n"
        f"{normal_row}\n"
        f"{attack_row}\n"
        f"AUC-ROC  : {auc:.4f}\n"
        f"{sep}\n"
    )


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading test data...", flush=True)
    df = pd.read_csv(DATA_PATH)
    raw_labels = df['attack_label'].values
    drop_cols = ['attack_label'] + [
        c for c in df.columns if 'timestamp' in c.lower() or c.lower() == 'time'
    ]
    feature_df = df.drop(columns=drop_cols)
    feature_cols = list(feature_df.columns)
    print(f"Test rows: {len(feature_df)}, Features: {len(feature_cols)}", flush=True)

    scaler = MinMaxScaler()
    scaled_all = scaler.fit_transform(feature_df.values.astype(np.float32))

    window_labels = window_labels_from_rows(raw_labels)
    n_normal = int((window_labels == 0).sum())
    n_attack = int((window_labels == 1).sum())
    print(f"Windows: {len(window_labels)} | Normal: {n_normal} | Attack: {n_attack}", flush=True)

    print("Building model and loading weights...", flush=True)
    model = build_model()
    model = load_weights(model, WEIGHTS_PATH)

    # ── Threshold from training data (all normal) ──────────────────
    print("Computing threshold from training data...", flush=True)
    train_df = pd.read_csv(TRAIN_PATH)
    train_drop = [c for c in train_df.columns if 'timestamp' in c.lower() or c.lower() == 'time']
    train_df = train_df.drop(columns=train_drop)
    train_scaler = MinMaxScaler()
    scaled_train = train_scaler.fit_transform(train_df.values.astype(np.float32))
    train_cols = list(train_df.columns)

    train_zone_errors = []
    for zone_id in ['zone1', 'zone2', 'zone3', 'zone4']:
        errs = zone_reconstruction_errors(model, scaled_train, train_cols, zone_id)
        train_zone_errors.append(errs)
    train_errors = np.mean(np.stack(train_zone_errors, axis=1), axis=1)
    threshold = float(np.percentile(train_errors, OPTIMAL_PERCENTILE))
    print(f"Threshold ({OPTIMAL_PERCENTILE}th pct of training errors): {threshold:.6f}", flush=True)

    # ── Inference on test data ─────────────────────────────────────
    print("Running inference on test data...", flush=True)
    zone_errors = []
    for zone_id in ['zone1', 'zone2', 'zone3', 'zone4']:
        errs = zone_reconstruction_errors(model, scaled_all, feature_cols, zone_id)
        zone_errors.append(errs)
        print(f"  {zone_id}: mean error {errs.mean():.6f}", flush=True)

    # Average reconstruction error across all zones per window
    errors = np.mean(np.stack(zone_errors, axis=1), axis=1)

    preds = (errors > threshold).astype(int)

    prec, rec, f1, support = precision_recall_fscore_support(
        window_labels, preds, labels=[0, 1]
    )
    fpr = float(
        np.sum((preds == 1) & (window_labels == 0)) / np.sum(window_labels == 0)
    )
    auc = float(roc_auc_score(window_labels, errors))

    summary = format_summary(threshold, prec, rec, f1, support, auc)
    print(summary, flush=True)
    print(f"FPR: {fpr:.4f}", flush=True)

    with open(os.path.join(RESULTS_DIR, 'fedavg_final_summary.txt'), 'w') as fh:
        fh.write(summary)

    model.save(os.path.join(RESULTS_DIR, 'fedavg_global_model.keras'))
    np.save(os.path.join(RESULTS_DIR, 'fedavg_threshold.npy'), threshold)

    # Save raw scores for compare_models.py
    np.save(os.path.join(RESULTS_DIR, 'fedavg_errors.npy'), errors)
    np.save(os.path.join(RESULTS_DIR, 'fedavg_labels.npy'), window_labels)
    np.save(os.path.join(RESULTS_DIR, 'fedavg_zone_errors.npy'),
            np.stack(zone_errors, axis=1))

    # Figure: error distribution + error over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ax1.hist(
        errors[window_labels == 0], bins=100, alpha=0.6,
        color='steelblue', label='Normal', density=True,
    )
    ax1.hist(
        errors[window_labels == 1], bins=100, alpha=0.6,
        color='tomato', label='Replay Attack', density=True,
    )
    ax1.axvline(
        threshold, color='black', linestyle='--',
        label=f'Threshold ({OPTIMAL_PERCENTILE}th pct)',
    )
    ax1.set_xlabel('Reconstruction Error (MSE)')
    ax1.set_ylabel('Density')
    ax1.set_title('Reconstruction Error Distribution')
    ax1.legend()

    idx = np.arange(len(errors))
    ax2.plot(
        idx[window_labels == 0], errors[window_labels == 0],
        '.', markersize=1, color='steelblue', label='Normal',
    )
    ax2.plot(
        idx[window_labels == 1], errors[window_labels == 1],
        '.', markersize=1, color='tomato', label='Replay Attack',
    )
    ax2.axhline(threshold, color='black', linestyle='--', label='Threshold')
    ax2.set_xlabel('Window Index')
    ax2.set_ylabel('Reconstruction Error (MSE)')
    ax2.set_title('Reconstruction Error Over Time')
    ax2.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'fedavg_fig_detection.png'), dpi=150)
    plt.close(fig)

    # Figure: ROC curve
    fpr_v, tpr_v, _ = roc_curve(window_labels, errors)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_v, tpr_v, color='#d7191c', linewidth=2, label=f'FL FedAvg (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — FL Global Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'fedavg_fig_roc.png'), dpi=150)
    plt.close(fig)

    # Figure: Precision-Recall curve
    prec_v, rec_v, _ = precision_recall_curve(window_labels, errors)
    ap = average_precision_score(window_labels, errors)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec_v, prec_v, color='#d7191c', linewidth=2, label=f'FL FedAvg (AP={ap:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve — FL Global Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'fedavg_fig_pr.png'), dpi=150)
    plt.close(fig)

    # Figure: per-zone error boxplot (normal vs attack)
    zone_errors_arr = np.stack(zone_errors, axis=1)
    zone_names = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4']
    normal_data = [zone_errors_arr[window_labels == 0, i] for i in range(4)]
    attack_data = [zone_errors_arr[window_labels == 1, i] for i in range(4)]
    positions_n = np.arange(1, 9, 2, dtype=float)
    positions_a = np.arange(2, 10, 2, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    bp1 = ax.boxplot(normal_data, positions=positions_n, widths=0.7,
                     patch_artist=True, boxprops=dict(facecolor='steelblue', alpha=0.6))
    bp2 = ax.boxplot(attack_data, positions=positions_a, widths=0.7,
                     patch_artist=True, boxprops=dict(facecolor='tomato', alpha=0.6))
    ax.set_xticks((positions_n + positions_a) / 2)
    ax.set_xticklabels(zone_names)
    ax.set_ylabel('Reconstruction Error (MSE)')
    ax.set_title('Per-Zone Reconstruction Error — FL Global Model')
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Normal', 'Attack'])
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'fedavg_fig_zone_errors.png'), dpi=150)
    plt.close(fig)

    print(f"All results saved to {RESULTS_DIR}", flush=True)


if __name__ == '__main__':
    main()
