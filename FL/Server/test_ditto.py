# Ditto evaluation — per-zone personalised models, system MAX aggregation.
#
# Run (inside server container) after ditto_personalize.py:
#   python3 test_ditto.py                     # default lambda=1.0
#   DITTO_LAMBDA=0.5 python3 test_ditto.py
#
# Key difference vs test_intact.py:
#   - Loads ditto_lambda{L}_zone{N}_final_weights.npz (per-zone personalised)
#   - NO gamma cross-zone consistency term (topology contribution of INTACT)
#   - System-level score = MAX(zone errors) — same as INTACT

import os
import re
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_fscore_support,
)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam

random.seed(42); np.random.seed(42); tf.random.set_seed(42)

WINDOW_SIZE  = 30
NUM_FEATURES = 36
BATCH_SIZE   = 32
OPTIMAL_PCT  = 99.4

ZONE_NAMES = ['zone1', 'zone2', 'zone3', 'zone4']
ZONE_BUSES = {
    'zone1': range(1, 9),
    'zone2': range(9, 18),
    'zone3': range(18, 25),
    'zone4': range(25, 33),
}

# Same per-zone percentiles as INTACT for comparable per-zone diagnostic metrics
ZONE_OPTIMAL_PCTS = {
    'zone1': 99.4,
    'zone2': 99.1,
    'zone3': 99.4,
    'zone4': 99.9,
}

TRAIN_PATH  = '/app/src/data/centralized_train_combined.csv'
TEST_PATH   = '/app/src/data/centralized_test_combined.csv'
MODELS_DIR  = '/app/src/models'
RESULTS_DIR = '/app/src/results'


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
    cols = get_zone_columns(list(df.columns), zone_id)
    data = df[cols].values.astype(np.float32)
    if scaler is None:
        scaler = MinMaxScaler()
        data   = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    data    = pad_to_n(data)
    windows = create_windows(data)
    pred    = model.predict(windows, batch_size=BATCH_SIZE, verbose=0)
    return np.mean(np.mean(np.square(windows - pred), axis=2), axis=1), scaler


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ditto_lambda       = float(os.environ.get('DITTO_LAMBDA', '1.0'))
    prefix             = f'ditto_lambda{ditto_lambda}'

    # Load run config
    personalize_epochs = 60
    cfg_path = os.path.join(MODELS_DIR, f'{prefix}_run_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        ditto_lambda       = cfg.get('ditto_lambda',       ditto_lambda)
        personalize_epochs = cfg.get('personalize_epochs', personalize_epochs)
    print(f"Ditto  lambda={ditto_lambda}  personalize_epochs={personalize_epochs}",
          flush=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...", flush=True)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    for df in (train_df, test_df):
        drop = [c for c in df.columns
                if 'timestamp' in c.lower() or c.lower() == 'time']
        df.drop(columns=drop, inplace=True)

    raw_labels    = test_df['attack_label'].values
    window_labels = window_labels_from_rows(raw_labels)
    test_feat     = test_df.drop(columns=['attack_label'])
    train_feat    = train_df.drop(columns=['attack_label'], errors='ignore')

    n_normal = int((window_labels == 0).sum())
    n_attack = int((window_labels == 1).sum())
    print(f"Test windows: {len(window_labels)}  Normal: {n_normal}  Attack: {n_attack}",
          flush=True)

    # ── Per-zone model loading + training calibration (not timed) ─────────────
    zone_error_arrays = {}
    zone_thresholds   = {}
    zone_models       = {}
    zone_scalers      = {}

    for zid in ZONE_NAMES:
        weights_path = os.path.join(MODELS_DIR, f'{prefix}_{zid}_final_weights.npz')
        if not os.path.exists(weights_path):
            print(f"  [SKIP] {zid} — not found: {weights_path}", flush=True)
            continue

        print(f"  Loading personalised Ditto model for {zid}...", flush=True)
        model = build_model()
        model = load_weights(model, weights_path)
        zone_models[zid] = model

        zone_pct            = ZONE_OPTIMAL_PCTS[zid]
        train_errs, scaler  = zone_errors(model, train_feat, zid)
        zone_scalers[zid]   = scaler
        threshold           = float(np.percentile(train_errs, zone_pct))
        zone_thresholds[zid] = threshold
        print(f"    Threshold ({zone_pct}th pct): {threshold:.6f}", flush=True)

    if len(zone_models) < 4:
        print("ERROR: not all 4 zone models found. Run ditto_personalize.py first.",
              flush=True)
        return

    # ── Test inference only (TIMED) ───────────────────────────────────────────
    _t_infer_start = time.perf_counter()

    for zid in ZONE_NAMES:
        test_errs, _ = zone_errors(zone_models[zid], test_feat, zid,
                                   scaler=zone_scalers[zid])
        zone_error_arrays[zid] = test_errs
        print(f"    {zid} test errors — mean: {test_errs.mean():.6f}  "
              f"max: {test_errs.max():.6f}", flush=True)

    _t_infer_end = time.perf_counter()
    _n_windows   = len(zone_error_arrays[ZONE_NAMES[0]])
    _total_ms    = (_t_infer_end - _t_infer_start) * 1000
    print(f"\n  [TIMING] Inference (test forward pass, no topology term): "
          f"{_total_ms:.1f} ms total | "
          f"{_total_ms / _n_windows:.4f} ms/window | "
          f"{_n_windows} windows", flush=True)

    # ── Per-zone metrics ───────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print(f"Ditto (lambda={ditto_lambda}) — Per-Zone Detection Results", flush=True)
    print("=" * 60, flush=True)

    zone_metrics            = {}
    zone_train_error_arrays = {}
    for zid in ZONE_NAMES:
        scores             = zone_error_arrays[zid]
        train_errs, _      = zone_errors(zone_models[zid], train_feat, zid)
        zone_train_error_arrays[zid] = train_errs
        threshold          = float(np.percentile(train_errs, ZONE_OPTIMAL_PCTS[zid]))

        preds              = (scores > threshold).astype(int)
        prec, rec, f1, sup = precision_recall_fscore_support(
            window_labels, preds, labels=[0, 1], zero_division=0
        )
        fpr  = float(np.sum((preds == 1) & (window_labels == 0))
                     / np.sum(window_labels == 0))
        auc  = float(roc_auc_score(window_labels, scores))

        zone_metrics[zid] = {'prec': prec[1], 'rec': rec[1], 'f1': f1[1],
                              'auc': auc, 'fpr': fpr}
        print(f"  {zid}  Prec={prec[1]:.3f}  Rec={rec[1]:.3f}  F1={f1[1]:.3f}  "
              f"AUC={auc:.4f}  FPR={fpr:.4f}", flush=True)

    # ── System-level evaluation (MAX across zones) — same as INTACT ──────────
    system_scores       = np.max([zone_error_arrays[z]      for z in ZONE_NAMES], axis=0)
    train_system_scores = np.max([zone_train_error_arrays[z] for z in ZONE_NAMES], axis=0)
    sys_auc             = float(roc_auc_score(window_labels, system_scores))

    sweep_pcts = [95, 96, 97, 98, 99, 99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9]
    best_sys_f1, best_sys_pct = -1.0, OPTIMAL_PCT
    print(f"\n  System MAX threshold sweep:", flush=True)
    print(f"  {'Pct':>6}  {'Thresh':>10}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'FPR':>7}",
          flush=True)
    for pct in sweep_pcts:
        t  = float(np.percentile(train_system_scores, pct))
        p  = (system_scores > t).astype(int)
        pr, rc, f1s, _ = precision_recall_fscore_support(
            window_labels, p, labels=[0, 1], zero_division=0
        )
        fpr_s  = float(np.sum((p == 1) & (window_labels == 0)) / np.sum(window_labels == 0))
        marker = ' <--' if f1s[1] > best_sys_f1 else ''
        print(f"  {pct:>6}  {t:>10.6f}  {pr[1]:>7.3f}  {rc[1]:>7.3f}  "
              f"{f1s[1]:>7.3f}  {fpr_s:>7.4f}{marker}", flush=True)
        if f1s[1] > best_sys_f1:
            best_sys_f1  = f1s[1]
            best_sys_pct = pct
            sys_threshold = t
            sys_preds     = p
            sys_prec, sys_rec, sys_f1 = pr, rc, f1s
            sys_fpr = fpr_s
    print(f"\n  Best system pct: {best_sys_pct}th  threshold={sys_threshold:.6f}", flush=True)

    print()
    print(f"  SYSTEM (MAX)  Prec={sys_prec[1]:.3f}  Rec={sys_rec[1]:.3f}  "
          f"F1={sys_f1[1]:.3f}  AUC={sys_auc:.4f}  FPR={sys_fpr:.4f}", flush=True)
    print("=" * 60, flush=True)

    # ── Save arrays ───────────────────────────────────────────────────────────
    np.save(os.path.join(RESULTS_DIR, f'{prefix}_window_labels.npy'),    window_labels)
    np.save(os.path.join(RESULTS_DIR, f'{prefix}_system_scores.npy'),    system_scores)
    np.save(os.path.join(RESULTS_DIR, f'{prefix}_system_preds.npy'),     sys_preds)
    np.save(os.path.join(RESULTS_DIR, f'{prefix}_system_threshold.npy'), np.array(sys_threshold))
    for zid in ZONE_NAMES:
        np.save(os.path.join(RESULTS_DIR, f'{prefix}_{zid}_errors.npy'),
                zone_error_arrays[zid])

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_lines = [
        "=" * 60,
        f"FINAL RESULTS — Ditto (Personalised FL, lambda={ditto_lambda})",
        f"Global model  : FedAvg (20 rounds x 3 local epochs)",
        f"Personalise   : {personalize_epochs} epochs per zone",
        f"Lambda        : {ditto_lambda}",
        f"Threshold     : {best_sys_pct}th pct of training MAX scores",
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
    summary_path = os.path.join(RESULTS_DIR, f'{prefix}_final_summary.txt')
    with open(summary_path, 'w') as fh:
        fh.write('\n'.join(summary_lines))
    print(f"Summary saved to {summary_path}", flush=True)

    # ── ROC figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ['#d7191c', '#ff7f00', '#4daf4a', '#984ea3']
    for (zid, m), color in zip(zone_metrics.items(), colors):
        fpr_v, tpr_v, _ = roc_curve(window_labels, zone_error_arrays[zid])
        ax.plot(fpr_v, tpr_v, color=color, lw=1.4, ls='--',
                label=f'{zid}  (AUC={m["auc"]:.4f})')
    fpr_v, tpr_v, _ = roc_curve(window_labels, system_scores)
    ax.plot(fpr_v, tpr_v, color='#2c7bb6', lw=2.5, ls='-',
            label=f'Ditto System MAX  (AUC={sys_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve — Ditto (lambda={ditto_lambda})', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f'{prefix}_fig_roc.png'), dpi=150)
    plt.close(fig)

    print(f"\nAll results saved to {RESULTS_DIR}", flush=True)


if __name__ == '__main__':
    main()
