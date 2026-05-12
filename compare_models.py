"""
compare_models.py  — Generate paper comparison figures for all model variants.

Run on your laptop AFTER:
  1. FL training is complete on the Jetson (50 rounds)
  2. test_server.py has run on AGX 04
  3. You have copied the FL results to your laptop:

     mkdir -p "ML model/results/fl"
     scp jetson@10.226.44.86:"~/fl/results/fedavg_*.npy" "ML model/results/fl/"
     scp jetson@10.226.44.86:"~/fl/models/fedavg_training_log.csv" "ML model/results/fl/"

Usage:
  python compare_models.py
"""

import os
import re
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    precision_recall_fscore_support,
)
import tensorflow as tf

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
ML_MODELS_DIR  = os.path.join(SCRIPT_DIR, "ML model", "models")
FL_RESULTS_DIR = os.path.join(SCRIPT_DIR, "ML model", "results", "fl")
TEST_CSV       = os.path.join(SCRIPT_DIR, "FL", "Server", "centralized_test_combined.csv")
OUT_DIR        = os.path.join(SCRIPT_DIR, "ML model", "results", "comparison")
FL_TRAIN_LOG   = os.path.join(FL_RESULTS_DIR, "fedavg_training_log.csv")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FL_RESULTS_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
WINDOW_SIZE  = 30
OPTIMAL_PCT  = 99.4
PERCENTILES  = [95, 96, 97, 98, 99, 99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9]

ZONE_BUSES = {
    'zone1': range(1, 9),
    'zone2': range(9, 18),
    'zone3': range(18, 25),
    'zone4': range(25, 33),
}

COLORS = {
    'Centralized':  '#2c7bb6',
    'Zone 1 Local': '#74c476',
    'Zone 2 Local': '#41ab5d',
    'Zone 3 Local': '#238b45',
    'Zone 4 Local': '#005a32',
    'FL FedAvg':    '#d7191c',
}

MODEL_ORDER = ['Centralized', 'Zone 1 Local', 'Zone 2 Local',
               'Zone 3 Local', 'Zone 4 Local', 'FL FedAvg']


# ── Helpers ────────────────────────────────────────────────────────────────
def get_zone_columns(columns, zone_id):
    buses = ZONE_BUSES[zone_id]
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


def compute_metrics(errors, labels):
    thr = float(np.percentile(errors[labels == 0], OPTIMAL_PCT))
    preds = (errors > thr).astype(int)
    prec, rec, f1, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0
    )
    auc = roc_auc_score(labels, errors)
    fpr = float(np.sum((preds == 1) & (labels == 0)) / np.sum(labels == 0))
    return {
        'auc': auc, 'f1': f1[1], 'recall': rec[1],
        'precision': prec[1], 'fpr': fpr, 'threshold': thr,
    }


def sweep_metrics(errors, labels):
    normal_errors = errors[labels == 0]
    rows = []
    for pct in PERCENTILES:
        thr = float(np.percentile(normal_errors, pct))
        preds = (errors > thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, labels=[0, 1], zero_division=0
        )
        rows.append({'pct': pct, 'f1': f1[1], 'recall': rec[1], 'precision': prec[1]})
    return rows


# ── Data loading ───────────────────────────────────────────────────────────
def load_test_data():
    df = pd.read_csv(TEST_CSV)
    raw_labels = df['attack_label'].values
    drop_cols = ['attack_label'] + [
        c for c in df.columns if 'timestamp' in c.lower() or c.lower() == 'time'
    ]
    feature_df = df.drop(columns=drop_cols)
    window_labels = window_labels_from_rows(raw_labels)
    return feature_df, window_labels


# ── Inference helpers ──────────────────────────────────────────────────────
def get_centralized_errors(feature_df):
    model  = tf.keras.models.load_model(os.path.join(ML_MODELS_DIR, 'centralized_lstm.keras'))
    scaler = joblib.load(os.path.join(ML_MODELS_DIR, 'centralized_scaler.pkl'))
    data    = scaler.transform(feature_df.values.astype(np.float32))
    windows = create_windows(data)
    return mse_errors(model, windows)


def get_local_zone_errors(feature_df, zone_id):
    model  = tf.keras.models.load_model(
        os.path.join(ML_MODELS_DIR, f'{zone_id}_local_lstm.keras')
    )
    scaler    = joblib.load(os.path.join(ML_MODELS_DIR, f'{zone_id}_scaler.pkl'))
    zone_cols = get_zone_columns(list(feature_df.columns), zone_id)
    data      = scaler.transform(feature_df[zone_cols].values.astype(np.float32))
    windows   = create_windows(data)
    return mse_errors(model, windows)


def get_fl_errors():
    errors_path = os.path.join(FL_RESULTS_DIR, 'fedavg_errors.npy')
    labels_path = os.path.join(FL_RESULTS_DIR, 'fedavg_labels.npy')
    if not os.path.exists(errors_path):
        print(f"  [SKIP] FL errors not found. Copy from Jetson:")
        print(f'    mkdir -p "ML model/results/fl"')
        print(f'    scp jetson@10.226.44.86:"~/fl/results/fedavg_*.npy" "ML model/results/fl/"')
        print(f'    scp jetson@10.226.44.86:"~/fl/models/fedavg_training_log.csv" "ML model/results/fl/"')
        return None, None
    return np.load(errors_path), np.load(labels_path)


# ── Figure 1: ROC curves ───────────────────────────────────────────────────
def fig_roc(all_errors, all_labels, window_labels):
    fig, ax = plt.subplots(figsize=(8, 7))
    for name in MODEL_ORDER:
        errors = all_errors.get(name)
        if errors is None:
            continue
        labels = all_labels.get(name, window_labels)
        fpr_v, tpr_v, _ = roc_curve(labels, errors)
        auc = roc_auc_score(labels, errors)
        lw = 2.5 if name in ('Centralized', 'FL FedAvg') else 1.2
        ls = '-'  if name in ('Centralized', 'FL FedAvg') else '--'
        ax.plot(fpr_v, tpr_v, color=COLORS[name], linewidth=lw,
                linestyle=ls, label=f'{name}  (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Model Variants', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_roc_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  ✓ {os.path.basename(path)}')


# ── Figure 2: Precision-Recall curves ─────────────────────────────────────
def fig_pr(all_errors, all_labels, window_labels):
    fig, ax = plt.subplots(figsize=(8, 7))
    for name in MODEL_ORDER:
        errors = all_errors.get(name)
        if errors is None:
            continue
        labels = all_labels.get(name, window_labels)
        prec_v, rec_v, _ = precision_recall_curve(labels, errors)
        ap = average_precision_score(labels, errors)
        lw = 2.5 if name in ('Centralized', 'FL FedAvg') else 1.2
        ls = '-'  if name in ('Centralized', 'FL FedAvg') else '--'
        ax.plot(rec_v, prec_v, color=COLORS[name], linewidth=lw,
                linestyle=ls, label=f'{name}  (AP={ap:.4f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves — All Model Variants', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_pr_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  ✓ {os.path.basename(path)}')


# ── Figure 3: Bar chart comparison ────────────────────────────────────────
def fig_bar(all_errors, all_labels, window_labels):
    metrics_list = []
    names_used = []
    for name in MODEL_ORDER:
        errors = all_errors.get(name)
        if errors is None:
            continue
        labels = all_labels.get(name, window_labels)
        m = compute_metrics(errors, labels)
        metrics_list.append(m)
        names_used.append(name)

    metric_keys   = ['auc',    'f1',        'recall',     'precision']
    metric_labels = ['AUC-ROC', 'F1 (Attack)', 'Recall (Attack)', 'Precision (Attack)']

    x     = np.arange(len(names_used))
    width = 0.2
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        vals = [m[key] for m in metrics_list]
        bars = ax.bar(x + i * width, vals, width, label=label)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names_used, rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Detection Performance — All Model Variants', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_bar_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  ✓ {os.path.basename(path)}')


# ── Figure 4: Threshold sweep ──────────────────────────────────────────────
def fig_sweep(all_errors, all_labels, window_labels):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for name in MODEL_ORDER:
        errors = all_errors.get(name)
        if errors is None:
            continue
        labels = all_labels.get(name, window_labels)
        rows   = sweep_metrics(errors, labels)
        pcts   = [r['pct'] for r in rows]
        lw = 2.5 if name in ('Centralized', 'FL FedAvg') else 1.2
        ls = '-'  if name in ('Centralized', 'FL FedAvg') else '--'
        axes[0].plot(pcts, [r['f1']     for r in rows], color=COLORS[name],
                     linewidth=lw, linestyle=ls, label=name)
        axes[1].plot(pcts, [r['recall'] for r in rows], color=COLORS[name],
                     linewidth=lw, linestyle=ls, label=name)

    for ax, ylabel, title in zip(
        axes,
        ['F1 Score (Attack)', 'Recall (Attack)'],
        ['F1 vs Threshold Percentile', 'Recall vs Threshold Percentile'],
    ):
        ax.axvline(OPTIMAL_PCT, color='gray', linestyle=':', linewidth=1,
                   label=f'Optimal ({OPTIMAL_PCT}th pct)')
        ax.set_xlabel('Threshold Percentile', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_threshold_sweep.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  ✓ {os.path.basename(path)}')


# ── Figure 5: Error distribution comparison ────────────────────────────────
def fig_distributions(all_errors, all_labels, window_labels):
    key_models = [n for n in ['Centralized', 'FL FedAvg'] if all_errors.get(n) is not None]
    if not key_models:
        print('  [SKIP] fig_error_distributions — no models available')
        return

    fig, axes = plt.subplots(1, len(key_models), figsize=(7 * len(key_models), 5),
                             sharey=False)
    if len(key_models) == 1:
        axes = [axes]

    for ax, name in zip(axes, key_models):
        errors = all_errors[name]
        labels = all_labels.get(name, window_labels)
        thr    = float(np.percentile(errors[labels == 0], OPTIMAL_PCT))
        ax.hist(errors[labels == 0], bins=80, alpha=0.65, density=True,
                color='steelblue', label='Normal')
        ax.hist(errors[labels == 1], bins=80, alpha=0.65, density=True,
                color='tomato', label='Replay Attack')
        ax.axvline(thr, color='black', linestyle='--', linewidth=1.2,
                   label=f'Threshold ({OPTIMAL_PCT}th pct)')
        ax.set_xlabel('Reconstruction Error (MSE)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(name, fontsize=12)
        ax.legend(fontsize=9)

    fig.suptitle('Reconstruction Error Distributions', fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_error_distributions.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  ✓ {os.path.basename(path)}')


# ── Figure 6: FL training convergence ─────────────────────────────────────
def fig_convergence():
    if not os.path.exists(FL_TRAIN_LOG):
        print(f'  [SKIP] fig_fl_convergence — training log not found at {FL_TRAIN_LOG}')
        return
    df = pd.read_csv(FL_TRAIN_LOG)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df['round'], df['loss'], color='#d7191c', linewidth=2,
            marker='o', markersize=3)
    ax.set_xlabel('FL Round', fontsize=12)
    ax.set_ylabel('Aggregated Loss (MSE)', fontsize=12)
    ax.set_title('Federated Learning Training Convergence (50 Rounds)', fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_fl_convergence.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  ✓ {os.path.basename(path)}')


# ── Figure 7: Per-zone FL error analysis ──────────────────────────────────
def fig_zone_analysis():
    zone_path  = os.path.join(FL_RESULTS_DIR, 'fedavg_zone_errors.npy')
    labels_path = os.path.join(FL_RESULTS_DIR, 'fedavg_labels.npy')
    if not os.path.exists(zone_path):
        print(f'  [SKIP] fig_fl_zone_analysis — zone errors not found at {zone_path}')
        return

    zone_errors = np.load(zone_path)   # (n_windows, 4)
    labels      = np.load(labels_path)
    zone_names  = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: boxplot normal vs attack per zone
    normal_data = [zone_errors[labels == 0, i] for i in range(4)]
    attack_data = [zone_errors[labels == 1, i] for i in range(4)]
    pos_n = np.arange(1, 9, 2, dtype=float)
    pos_a = np.arange(2, 10, 2, dtype=float)
    bp1 = axes[0].boxplot(normal_data, positions=pos_n, widths=0.7,
                           patch_artist=True,
                           boxprops=dict(facecolor='steelblue', alpha=0.65))
    bp2 = axes[0].boxplot(attack_data, positions=pos_a, widths=0.7,
                           patch_artist=True,
                           boxprops=dict(facecolor='tomato', alpha=0.65))
    axes[0].set_xticks((pos_n + pos_a) / 2)
    axes[0].set_xticklabels(zone_names)
    axes[0].set_ylabel('Reconstruction Error (MSE)', fontsize=11)
    axes[0].set_title('Per-Zone Error Distribution (FL Global Model)', fontsize=12)
    axes[0].legend([bp1['boxes'][0], bp2['boxes'][0]], ['Normal', 'Replay Attack'],
                   fontsize=9)

    # Right: mean error per zone
    means_n = [zone_errors[labels == 0, i].mean() for i in range(4)]
    means_a = [zone_errors[labels == 1, i].mean() for i in range(4)]
    x = np.arange(4)
    w = 0.35
    axes[1].bar(x - w / 2, means_n, w, label='Normal',        color='steelblue', alpha=0.8)
    axes[1].bar(x + w / 2, means_a, w, label='Replay Attack', color='tomato',    alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(zone_names)
    axes[1].set_ylabel('Mean Reconstruction Error (MSE)', fontsize=11)
    axes[1].set_title('Mean Per-Zone Error — Normal vs Attack', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_fl_zone_analysis.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  ✓ {os.path.basename(path)}')


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print('\n' + '=' * 60)
    print('  Model Comparison Figure Generator')
    print('=' * 60)

    print('\nLoading test data...')
    feature_df, window_labels = load_test_data()
    n_normal = int((window_labels == 0).sum())
    n_attack = int((window_labels == 1).sum())
    print(f'  {len(window_labels)} windows | Normal: {n_normal} | Attack: {n_attack}')

    all_errors = {}
    all_labels = {}

    print('\nRunning inference — Centralized model...')
    try:
        all_errors['Centralized'] = get_centralized_errors(feature_df)
        m = compute_metrics(all_errors['Centralized'], window_labels)
        print(f'  AUC={m["auc"]:.4f}  F1={m["f1"]:.4f}  Recall={m["recall"]:.4f}')
    except Exception as e:
        print(f'  [SKIP] {e}')
        all_errors['Centralized'] = None

    zone_map = {
        'zone1': 'Zone 1 Local', 'zone2': 'Zone 2 Local',
        'zone3': 'Zone 3 Local', 'zone4': 'Zone 4 Local',
    }
    for zone_id, zone_name in zone_map.items():
        print(f'\nRunning inference — {zone_name}...')
        try:
            all_errors[zone_name] = get_local_zone_errors(feature_df, zone_id)
            m = compute_metrics(all_errors[zone_name], window_labels)
            print(f'  AUC={m["auc"]:.4f}  F1={m["f1"]:.4f}  Recall={m["recall"]:.4f}')
        except Exception as e:
            print(f'  [SKIP] {e}')
            all_errors[zone_name] = None

    print('\nLoading FL errors...')
    fl_errors, fl_labels = get_fl_errors()
    all_errors['FL FedAvg'] = fl_errors
    if fl_labels is not None:
        all_labels['FL FedAvg'] = fl_labels
        m = compute_metrics(fl_errors, fl_labels)
        print(f'  AUC={m["auc"]:.4f}  F1={m["f1"]:.4f}  Recall={m["recall"]:.4f}')

    print(f'\nGenerating figures → {OUT_DIR}')
    fig_roc(all_errors, all_labels, window_labels)
    fig_pr(all_errors, all_labels, window_labels)
    fig_bar(all_errors, all_labels, window_labels)
    fig_sweep(all_errors, all_labels, window_labels)
    fig_distributions(all_errors, all_labels, window_labels)
    fig_convergence()
    fig_zone_analysis()

    print('\nDone.\n')


if __name__ == '__main__':
    main()
