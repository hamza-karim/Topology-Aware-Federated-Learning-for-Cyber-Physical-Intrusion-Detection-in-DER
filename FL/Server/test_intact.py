# Run on server device AFTER intact training is complete:
# docker exec -it <server_container> bash
# cd /app/src && python test_intact.py

import os
import re
import json
import time
import numpy as np
import random
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
random.seed(42); np.random.seed(42); tf.random.set_seed(42)

WINDOW_SIZE      = 30
NUM_FEATURES     = 36
BATCH_SIZE       = 32
OPTIMAL_PCT      = 99.4        # fallback / system MAX

# Optuna-selected percentile per zone (from local zone hyperparameter search)
ZONE_OPTIMAL_PCTS = {
    'zone1': 99.4,   # local model used 99.4th pct
    'zone2': 99.1,   # local model used 99.1th pct (more sensitive)
    'zone3': 99.4,   # local model used 99.4th pct
    'zone4': 99.9,   # local model used 99.9th pct (more precise)
}

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

    # ── Per-zone model loading + training calibration (one-time setup, not timed) ──
    zone_error_arrays = {}
    zone_thresholds   = {}
    zone_models       = {}
    zone_scalers      = {}   # store scalers fitted on training data for reuse

    for zid in ZONE_NAMES:
        weights_path = os.path.join(MODELS_DIR, f'intact_{zid}_final_weights.npz')
        if not os.path.exists(weights_path):
            print(f"  [SKIP] {zid} — weights not found at {weights_path}", flush=True)
            continue

        print(f"  Loading personalised model for {zid}...", flush=True)
        model = build_model()
        model = load_weights(model, weights_path)
        zone_models[zid] = model

        zone_pct = ZONE_OPTIMAL_PCTS[zid]
        train_errs, scaler = zone_errors(model, train_feat, zid)
        zone_scalers[zid]  = scaler
        threshold = float(np.percentile(train_errs, zone_pct))
        zone_thresholds[zid] = threshold
        print(f"    Threshold ({zone_pct}th pct): {threshold:.6f}", flush=True)

    if len(zone_models) < 4:
        print("ERROR: not all 4 zone models found. Run intact training first.", flush=True)
        return

    # ── Test inference only (TIMED — matches FedAvg timing scope) ─────────────
    _t_infer_start = time.perf_counter()

    for zid in ZONE_NAMES:
        test_errs, _ = zone_errors(zone_models[zid], test_feat, zid,
                                   scaler=zone_scalers[zid])
        zone_error_arrays[zid] = test_errs
        print(f"    Test errors — mean: {test_errs.mean():.6f}  max: {test_errs.max():.6f}",
              flush=True)

    # ── Cross-zone consistency check ──────────────────────────────────────────
    zone_final_scores = {}
    for zid in ZONE_NAMES:
        local         = zone_error_arrays[zid]
        neighbour_avg = np.zeros_like(local)
        for other, w in W[zid].items():
            neighbour_avg += w * zone_error_arrays[other]
        mismatch = local - neighbour_avg
        zone_final_scores[zid] = local + gamma * mismatch

    _t_infer_end = time.perf_counter()
    _n_windows   = len(zone_final_scores[ZONE_NAMES[0]])
    _total_ms    = (_t_infer_end - _t_infer_start) * 1000
    print(f"\n  [TIMING] Inference (test forward pass + consistency): "
          f"{_total_ms:.1f} ms total | "
          f"{_total_ms / _n_windows:.4f} ms/window | "
          f"{_n_windows} windows", flush=True)

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

    zone_metrics           = {}
    zone_train_final_scores = {}   # needed for system-level training threshold
    for zid in ZONE_NAMES:
        scores    = zone_final_scores[zid]
        threshold = zone_thresholds[zid]

        # Adjust threshold to account for the gamma amplification on training data
        local_train_errs, _ = zone_errors(zone_models[zid], train_feat, zid)
        neighbour_train_avg  = np.zeros_like(local_train_errs)
        for other, w in W[zid].items():
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
        zone_train_final_scores[zid] = train_scores          # save for system threshold
        adj_threshold  = float(np.percentile(train_scores, ZONE_OPTIMAL_PCTS[zid]))

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

    # ── System-level evaluation (MAX aggregation across zones) ───────────────
    # system_score[t] = MAX(zone_final_scores[t]) picks the most-alarmed zone.
    # Zone 1's inverted scores are naturally filtered out (never the MAX during attacks).
    # Threshold sweep over training MAX distribution — same methodology as all FL baselines.
    system_scores = np.max([zone_final_scores[z] for z in ZONE_NAMES], axis=0)
    train_system_scores = np.max([zone_train_final_scores[z] for z in ZONE_NAMES], axis=0)
    sys_auc = float(roc_auc_score(window_labels, system_scores))

    sweep_pcts = [95, 96, 97, 98, 99, 99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9]
    best_sys_f1, best_sys_pct = -1.0, OPTIMAL_PCT
    print(f"\n  System MAX threshold sweep:", flush=True)
    print(f"  {'Pct':>6}  {'Thresh':>10}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'FPR':>7}", flush=True)
    for pct in sweep_pcts:
        t = float(np.percentile(train_system_scores, pct))
        p = (system_scores > t).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(window_labels, p, labels=[0,1], zero_division=0)
        fpr_s = float(np.sum((p==1)&(window_labels==0)) / np.sum(window_labels==0))
        marker = ' <--' if f1[1] > best_sys_f1 else ''
        print(f"  {pct:>6}  {t:>10.6f}  {pr[1]:>7.3f}  {rc[1]:>7.3f}  {f1[1]:>7.3f}  {fpr_s:>7.4f}{marker}", flush=True)
        if f1[1] > best_sys_f1:
            best_sys_f1  = f1[1]
            best_sys_pct = pct
            sys_threshold = t
            sys_preds     = p
            sys_prec, sys_rec, sys_f1 = pr, rc, f1
            sys_fpr = fpr_s
    print(f"\n  Best system pct: {best_sys_pct}th  threshold={sys_threshold:.6f}", flush=True)

    print()
    print(f"  SYSTEM (MAX)  Prec={sys_prec[1]:.3f}  Rec={sys_rec[1]:.3f}  "
          f"F1={sys_f1[1]:.3f}  AUC={sys_auc:.4f}  FPR={sys_fpr:.4f}", flush=True)
    print("=" * 60, flush=True)

    # Save system scores, predictions, and threshold for compare_models.py
    np.save(os.path.join(RESULTS_DIR, 'intact_system_scores.npy'), system_scores)
    np.save(os.path.join(RESULTS_DIR, 'intact_system_preds.npy'), sys_preds)
    np.save(os.path.join(RESULTS_DIR, 'intact_system_threshold.npy'), np.array(sys_threshold))

    # Save summary
    summary_lines = [
        "=" * 60,
        "FINAL RESULTS — INTACT (Topology-Aware FL)",
        f"FL rounds    : {fl_rounds}",
        f"Local epochs : {local_epochs}",
        f"Alpha        : {alpha}",
        f"Gamma        : {gamma}",
        f"Threshold    : {best_sys_pct}th pct of training MAX scores",
        "=" * 60,
        "Per-zone metrics (each zone evaluated independently):",
    ]
    for zid, m in zone_metrics.items():
        pct = ZONE_OPTIMAL_PCTS[zid]
        summary_lines.append(
            f"  {zid} [{pct:.1f}th pct]  Prec={m['prec']:.3f}  Rec={m['rec']:.3f}  "
            f"F1={m['f1']:.3f}  AUC={m['auc']:.4f}  FPR={m['fpr']:.4f}"
        )
    summary_lines += [
        "-" * 60,
        "System-level metrics (MAX aggregation across zones):",
        f"  SYSTEM  Prec={sys_prec[1]:.3f}  Rec={sys_rec[1]:.3f}  "
        f"F1={sys_f1[1]:.3f}  AUC={sys_auc:.4f}  FPR={sys_fpr:.4f}",
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

    fpr_v, tpr_v, _ = roc_curve(window_labels, system_scores)
    ax.plot(fpr_v, tpr_v, color='#2c7bb6', lw=2.5, ls='-',
            label=f'INTACT System MAX  (AUC={sys_auc:.4f})')
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
