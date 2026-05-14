# Run on server device AFTER intact training is complete:
# docker exec -it <server_container> bash
# cd /app/src && python test_intact.py

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_fscore_support,
    precision_recall_curve, average_precision_score,
)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam

WINDOW_SIZE      = 30
NUM_FEATURES     = 36
BATCH_SIZE       = 32
OPTIMAL_PCT      = 99.4

ZONE_NAMES = ['zone1', 'zone2', 'zone3', 'zone4']
ZONE_BUSES = {
    'zone1': range(1, 9),
    'zone2': range(9, 18),
    'zone3': range(18, 25),
    'zone4': range(25, 33),
}

TRAIN_PATH  = '/app/src/data/centralized_train_combined.csv'
TEST_PATH   = '/app/src/data/centralized_test_combined.csv'
MODELS_DIR  = '/app/src/models'
RESULTS_DIR = '/app/src/results'


# ── Model helpers ─────────────────────────────────────────────────────────────

def build_model():
    inp = Input(shape=(WINDOW_SIZE, NUM_FEATURES))
    x = LSTM(32, activation='tanh', return_sequences=True)(inp)
    x = LSTM(64, activation='tanh', return_sequences=False)(x)
    x = RepeatVector(WINDOW_SIZE)(x)
    x = LSTM(64, activation='tanh', return_sequences=True)(x)
    x = LSTM(32, activation='tanh', return_sequences=True)(x)
    x = TimeDistributed(Dense(NUM_FEATURES))(x)
    model = Model(inp, x)
    model.compile(optimizer=Adam(learning_rate=0.005730), loss='mse')
    return model


def load_weights(model, path):
    data = np.load(path)
    keys = sorted(data.files, key=lambda k: int(k.split('_')[-1]))
    model.set_weights([data[k] for k in keys])
    return model


# ── Data helpers ──────────────────────────────────────────────────────────────

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


def window_labels_from_rows(labels):
    return np.array([
        1 if np.any(labels[i:i + WINDOW_SIZE] == 1) else 0
        for i in range(len(labels) - WINDOW_SIZE + 1)
    ])


def zone_errors(model, df, zone_id, scaler=None):
    """Compute per-window reconstruction errors for one zone."""
    cols      = get_zone_columns(list(df.columns), zone_id)
    data      = df[cols].values.astype(np.float32)
    if scaler is None:
        scaler = MinMaxScaler()
        data   = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    data    = pad_to_n(data)
    windows = create_windows(data)
    pred    = model.predict(windows, batch_size=BATCH_SIZE, verbose=0)
    return np.mean(np.mean(np.square(windows - pred), axis=2), axis=1), scaler


# ── Admittance weights (for consistency check) ────────────────────────────────

def load_admittance_weights(models_dir=MODELS_DIR):
    """Load gamma and W from the intact run_config.json saved by the server."""
    cfg_path = os.path.join(models_dir, 'intact_run_config.json')
    gamma = 0.3
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg  = json.load(f)
        gamma = cfg.get('gamma', 0.3)

    # Rebuild W from zone_admittance.csv (same logic as Server.py)
    adm_path = os.path.join(os.path.dirname(__file__), 'data', 'zone_admittance.csv')
    zone_bus_sets = {
        'zone1': set(range(1, 9)),
        'zone2': set(range(9, 18)),
        'zone3': set(range(18, 25)),
        'zone4': set(range(25, 33)),
    }
    df  = pd.read_csv(adm_path)
    raw = {z: {z2: 0.0 for z2 in ZONE_NAMES} for z in ZONE_NAMES}
    for _, row in df.iterrows():
        fb  = int(row['from_bus'])
        tb  = int(row['to_bus'])
        adm = float(row['admittance'])
        fz  = next((z for z, b in zone_bus_sets.items() if fb in b), None)
        tz  = next((z for z, b in zone_bus_sets.items() if tb in b), None)
        if fz and tz and fz != tz:
            raw[fz][tz] += adm
            raw[tz][fz] += adm
    W = {}
    for z in ZONE_NAMES:
        total = sum(raw[z][z2] for z2 in ZONE_NAMES if z2 != z)
        W[z]  = {z2: raw[z][z2] / total for z2 in ZONE_NAMES if z2 != z}
    return W, gamma


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load run config
    fl_rounds    = 10
    local_epochs = 5
    alpha        = 0.5
    gamma        = 0.3
    cfg_path = os.path.join(MODELS_DIR, 'intact_run_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            run_cfg      = json.load(f)
        fl_rounds    = run_cfg.get('fl_rounds',    fl_rounds)
        local_epochs = run_cfg.get('local_epochs', local_epochs)
        alpha        = run_cfg.get('alpha',        alpha)
        gamma        = run_cfg.get('gamma',        gamma)
    print(f"INTACT  alpha={alpha}  gamma={gamma}  rounds={fl_rounds}  epochs={local_epochs}",
          flush=True)

    W, _ = load_admittance_weights()

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data...", flush=True)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    for df in (train_df, test_df):
        drop = [c for c in df.columns if 'timestamp' in c.lower() or c.lower() == 'time']
        df.drop(columns=drop, inplace=True)

    raw_labels   = test_df['attack_label'].values
    window_labels = window_labels_from_rows(raw_labels)
    test_feat    = test_df.drop(columns=['attack_label'])
    train_feat   = train_df.drop(columns=['attack_label'], errors='ignore')

    n_normal = int((window_labels == 0).sum())
    n_attack = int((window_labels == 1).sum())
    print(f"Test windows: {len(window_labels)}  Normal: {n_normal}  Attack: {n_attack}",
          flush=True)

    # ── Per-zone inference ─────────────────────────────────────────────────────
    zone_error_arrays = {}   # zone_id -> ndarray of errors (len = n_windows)
    zone_thresholds   = {}
    zone_models       = {}

    for zid in ZONE_NAMES:
        weights_path = os.path.join(MODELS_DIR, f'intact_{zid}_final_weights.npz')
        if not os.path.exists(weights_path):
            print(f"  [SKIP] {zid} — weights not found at {weights_path}", flush=True)
            continue

        print(f"  Loading personalised model for {zid}...", flush=True)
        model = build_model()
        model = load_weights(model, weights_path)
        zone_models[zid] = model

        # Threshold from training data (fit scaler on train, reuse on test)
        train_errs, scaler = zone_errors(model, train_feat, zid)
        threshold = float(np.percentile(train_errs, OPTIMAL_PCT))
        zone_thresholds[zid] = threshold
        print(f"    Threshold ({OPTIMAL_PCT}th pct): {threshold:.6f}", flush=True)

        # Test errors (reuse scaler fitted on train)
        test_errs, _ = zone_errors(model, test_feat, zid, scaler=scaler)
        zone_error_arrays[zid] = test_errs
        print(f"    Test errors — mean: {test_errs.mean():.6f}  max: {test_errs.max():.6f}",
              flush=True)

    if len(zone_error_arrays) < 4:
        print("ERROR: not all 4 zone models found. Run intact training first.", flush=True)
        return

    # ── Cross-zone consistency check ──────────────────────────────────────────
    # final_score[i,t] = local_error[i,t] + gamma*(local_error[i,t] - neighbour_avg[i,t])
    # Amplifies isolated spikes (replay attack) and dampens global disturbances.
    zone_final_scores = {}
    for zid in ZONE_NAMES:
        local    = zone_error_arrays[zid]
        neighbour_avg = np.zeros_like(local)
        for other, w in W[zid].items():
            neighbour_avg += w * zone_error_arrays[other]
        # spatial mismatch term
        mismatch = local - neighbour_avg
        zone_final_scores[zid] = local + gamma * mismatch

    # Save arrays
    np.save(os.path.join(RESULTS_DIR, 'intact_window_labels.npy'), window_labels)
    for zid in ZONE_NAMES:
        np.save(os.path.join(RESULTS_DIR, f'intact_{zid}_errors.npy'),
                zone_error_arrays[zid])
        np.save(os.path.join(RESULTS_DIR, f'intact_{zid}_final_scores.npy'),
                zone_final_scores[zid])
        np.save(os.path.join(RESULTS_DIR, f'intact_{zid}_threshold.npy'),
                zone_thresholds[zid])

    # ── Per-zone metrics ───────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("INTACT — Per-Zone Detection Results", flush=True)
    print("=" * 60, flush=True)

    zone_metrics = {}
    for zid in ZONE_NAMES:
        scores    = zone_final_scores[zid]
        threshold = zone_thresholds[zid]

        # Adjust threshold to account for the gamma amplification on training data
        local_train_errs, _ = zone_errors(zone_models[zid], train_feat, zid)
        neighbour_train_avg  = np.zeros_like(local_train_errs)
        for other, w in W[zid].items():
            other_train, _ = zone_errors(zone_models[other], train_feat, zid)
            # Use local proxy: other zone's errors on their own columns
            other_cols  = get_zone_columns(list(train_feat.columns), other)
            other_data  = train_feat[other_cols].values.astype(np.float32)
            s           = MinMaxScaler()
            other_data  = pad_to_n(s.fit_transform(other_data))
            other_wins  = create_windows(other_data)
            other_pred  = zone_models[other].predict(other_wins, batch_size=BATCH_SIZE, verbose=0)
            other_errs  = np.mean(np.mean(np.square(other_wins - other_pred), axis=2), axis=1)
            neighbour_train_avg += w * other_errs

        train_local, _ = zone_errors(zone_models[zid], train_feat, zid)
        train_scores   = train_local + gamma * (train_local - neighbour_train_avg)
        adj_threshold  = float(np.percentile(train_scores, OPTIMAL_PCT))

        preds = (scores > adj_threshold).astype(int)
        prec, rec, f1, support = precision_recall_fscore_support(
            window_labels, preds, labels=[0, 1], zero_division=0
        )
        fpr = float(np.sum((preds == 1) & (window_labels == 0)) / np.sum(window_labels == 0))
        auc = float(roc_auc_score(window_labels, scores))

        zone_metrics[zid] = {'prec': prec[1], 'rec': rec[1], 'f1': f1[1],
                              'auc': auc, 'fpr': fpr, 'threshold': adj_threshold}
        print(f"  {zid}  Prec={prec[1]:.3f}  Rec={rec[1]:.3f}  F1={f1[1]:.3f}  "
              f"AUC={auc:.4f}  FPR={fpr:.4f}", flush=True)

    # ── System average ─────────────────────────────────────────────────────────
    avg_prec = np.mean([m['prec'] for m in zone_metrics.values()])
    avg_rec  = np.mean([m['rec']  for m in zone_metrics.values()])
    avg_f1   = np.mean([m['f1']   for m in zone_metrics.values()])
    avg_auc  = np.mean([m['auc']  for m in zone_metrics.values()])
    avg_fpr  = np.mean([m['fpr']  for m in zone_metrics.values()])

    print()
    print(f"  SYSTEM AVG  Prec={avg_prec:.3f}  Rec={avg_rec:.3f}  F1={avg_f1:.3f}  "
          f"AUC={avg_auc:.4f}  FPR={avg_fpr:.4f}", flush=True)
    print("=" * 60, flush=True)

    # Save summary
    summary_lines = [
        "=" * 60,
        "FINAL RESULTS — INTACT (Topology-Aware FL)",
        f"FL rounds    : {fl_rounds}",
        f"Local epochs : {local_epochs}",
        f"Alpha        : {alpha}",
        f"Gamma        : {gamma}",
        "=" * 60,
    ]
    for zid, m in zone_metrics.items():
        summary_lines.append(
            f"  {zid}  Prec={m['prec']:.3f}  Rec={m['rec']:.3f}  "
            f"F1={m['f1']:.3f}  AUC={m['auc']:.4f}  FPR={m['fpr']:.4f}"
        )
    summary_lines += [
        "-" * 60,
        f"  SYSTEM AVG  Prec={avg_prec:.3f}  Rec={avg_rec:.3f}  "
        f"F1={avg_f1:.3f}  AUC={avg_auc:.4f}  FPR={avg_fpr:.4f}",
        "=" * 60,
    ]
    with open(os.path.join(RESULTS_DIR, 'intact_final_summary.txt'), 'w') as fh:
        fh.write('\n'.join(summary_lines))

    # ── ROC figure — all zones + system ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ['#d7191c', '#ff7f00', '#4daf4a', '#984ea3']
    for (zid, m), color in zip(zone_metrics.items(), colors):
        fpr_v, tpr_v, _ = roc_curve(window_labels, zone_final_scores[zid])
        ax.plot(fpr_v, tpr_v, color=color, lw=1.4, ls='--',
                label=f'{zid}  (AUC={m["auc"]:.4f})')

    # System-level score: average of all 4 final scores
    system_scores = np.mean([zone_final_scores[z] for z in ZONE_NAMES], axis=0)
    fpr_v, tpr_v, _ = roc_curve(window_labels, system_scores)
    sys_auc = roc_auc_score(window_labels, system_scores)
    ax.plot(fpr_v, tpr_v, color='#2c7bb6', lw=2.5, ls='-',
            label=f'INTACT System  (AUC={sys_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Random')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — INTACT Per-Zone + System', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'intact_fig_roc.png'), dpi=150)
    plt.close(fig)

    print(f"\nAll results saved to {RESULTS_DIR}", flush=True)


if __name__ == '__main__':
    main()
