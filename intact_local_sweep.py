"""
intact_local_sweep.py
=====================
Offline analysis using saved INTACT .npy files — no AGX or retraining needed.

Four analyses in one script:
  1. Gamma sweep        : recompute final scores for gamma in [0.0 .. 1.5]
  2. DER-only MAX       : system score from DER zones (2/3/4) only vs all-zones MAX
  3. Threshold strategy : fixed-pct vs Youden-J vs F1-max
  4. Percentile sweep   : pct in [97.0 .. 99.9] at best gamma

Thresholds here are calibrated on test-normal windows (labels==0).
This is intentional for the sweep — once you pick the best gamma/pct,
validate it with a training-data threshold run on AGX.

Run:  python intact_local_sweep.py
Outputs saved to:  ML model/results/comparison/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_fscore_support,
    f1_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
FL_DIR         = os.path.join(SCRIPT_DIR, "ML model", "results", "fl")
OUT_DIR        = os.path.join(SCRIPT_DIR, "ML model", "results", "comparison")
ADM_PATH       = os.path.join(SCRIPT_DIR, "FL", "Server", "zone_admittance.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
ZONE_NAMES = ['zone1', 'zone2', 'zone3', 'zone4']
DER_ZONES  = ['zone2', 'zone3', 'zone4']   # zones with DER generation
ZONE_BUSES = {
    'zone1': set(range(1, 9)),
    'zone2': set(range(9, 18)),
    'zone3': set(range(18, 25)),
    'zone4': set(range(25, 33)),
}

GAMMAS      = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
PERCENTILES = [97.0, 97.5, 98.0, 98.5, 99.0, 99.2, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9]
CURRENT_GAMMA = 0.3
CURRENT_PCT   = 99.4


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_admittance():
    df = pd.read_csv(ADM_PATH)
    raw = {z: {z2: 0.0 for z2 in ZONE_NAMES} for z in ZONE_NAMES}
    for _, row in df.iterrows():
        fb  = int(row['from_bus'])
        tb  = int(row['to_bus'])
        adm = float(row['admittance'])
        fz  = next((z for z, b in ZONE_BUSES.items() if fb in b), None)
        tz  = next((z for z, b in ZONE_BUSES.items() if tb in b), None)
        if fz and tz and fz != tz:
            raw[fz][tz] += adm
            raw[tz][fz] += adm
    W = {}
    for z in ZONE_NAMES:
        total = sum(raw[z][z2] for z2 in ZONE_NAMES if z2 != z)
        W[z]  = {z2: raw[z][z2] / total for z2 in ZONE_NAMES if z2 != z}
    return W


def load_zone_errors():
    errs = {}
    for zid in ZONE_NAMES:
        errs[zid] = np.load(os.path.join(FL_DIR, f'intact_{zid}_errors.npy'))
    labels = np.load(os.path.join(FL_DIR, 'intact_window_labels.npy'))
    return errs, labels


def final_scores(zone_errors, W, gamma):
    scores = {}
    for zid in ZONE_NAMES:
        local = zone_errors[zid]
        nb_avg = sum(w * zone_errors[other] for other, w in W[zid].items())
        scores[zid] = local + gamma * (local - nb_avg)
    return scores


def system_max(zone_scores, zones=None):
    zones = zones or ZONE_NAMES
    return np.max([zone_scores[z] for z in zones], axis=0)


def pct_threshold(scores, labels, pct):
    return float(np.percentile(scores[labels == 0], pct))


def f1_optimal_threshold(scores, labels):
    """Sweep candidate thresholds, return the one maximising attack F1."""
    candidates = np.unique(np.percentile(scores, np.linspace(0, 100, 1000)))
    best_f1, best_thr = -1.0, candidates[0]
    for thr in candidates:
        preds = (scores > thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, labels=[0, 1], zero_division=0)
        if f1[1] > best_f1:
            best_f1, best_thr = f1[1], thr
    return best_thr, best_f1


def youden_threshold(scores, labels):
    """Youden's J statistic: maximises TPR - FPR on ROC curve."""
    fpr_v, tpr_v, thresholds = roc_curve(labels, scores)
    j = tpr_v - fpr_v
    best_idx = np.argmax(j)
    return float(thresholds[best_idx])


def metrics(scores, labels, threshold):
    preds = (scores > threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0)
    fpr = float(np.sum((preds == 1) & (labels == 0)) / np.sum(labels == 0))
    auc = float(roc_auc_score(labels, scores))
    return {'auc': auc, 'prec': prec[1], 'rec': rec[1],
            'f1': f1[1], 'fpr': fpr, 'thr': threshold}


def print_row(label, m):
    print(f"  {label:<30}  AUC={m['auc']:.4f}  F1={m['f1']:.4f}  "
          f"Rec={m['rec']:.4f}  Prec={m['prec']:.4f}  FPR={m['fpr']:.4f}")


# ── Analysis 1 — Gamma sweep ──────────────────────────────────────────────────

def sweep_gamma(zone_errors, W, labels):
    print("\n" + "=" * 70)
    print("GAMMA SWEEP  (threshold = 99.4th pct of test-normal MAX scores)")
    print("=" * 70)
    print(f"  {'gamma':<8}  {'AUC':>6}  {'F1':>6}  {'Recall':>7}  "
          f"{'Prec':>6}  {'FPR':>6}  {'Threshold':>10}")
    print("  " + "-" * 62)

    rows = []
    for g in GAMMAS:
        fscores = final_scores(zone_errors, W, g)
        sys_s   = system_max(fscores)
        thr     = pct_threshold(sys_s, labels, CURRENT_PCT)
        m       = metrics(sys_s, labels, thr)
        marker  = "  <-- current" if g == CURRENT_GAMMA else ""
        print(f"  {g:<8.2f}  {m['auc']:>6.4f}  {m['f1']:>6.4f}  "
              f"{m['rec']:>7.4f}  {m['prec']:>6.4f}  {m['fpr']:>6.4f}  "
              f"{thr:>10.6f}{marker}")
        rows.append({'gamma': g, **m})

    print("\nGAMMA SWEEP  (threshold = F1-optimal on test data)")
    print(f"  {'gamma':<8}  {'AUC':>6}  {'F1':>6}  {'Recall':>7}  "
          f"{'Prec':>6}  {'FPR':>6}")
    print("  " + "-" * 55)
    rows_opt = []
    for g in GAMMAS:
        fscores = final_scores(zone_errors, W, g)
        sys_s   = system_max(fscores)
        thr, _  = f1_optimal_threshold(sys_s, labels)
        m       = metrics(sys_s, labels, thr)
        marker  = "  <-- current" if g == CURRENT_GAMMA else ""
        print(f"  {g:<8.2f}  {m['auc']:>6.4f}  {m['f1']:>6.4f}  "
              f"{m['rec']:>7.4f}  {m['prec']:>6.4f}  {m['fpr']:>6.4f}{marker}")
        rows_opt.append({'gamma': g, **m})

    return rows, rows_opt


# ── Analysis 2 — DER-only MAX vs all-zones MAX ────────────────────────────────

def sweep_der_vs_all(zone_errors, W, labels):
    print("\n" + "=" * 70)
    print("DER-ONLY MAX  vs  ALL-ZONES MAX  (gamma=0.3, F1-optimal threshold)")
    print("=" * 70)

    for g in [0.1, 0.2, 0.3, 0.5, 0.7]:
        fscores = final_scores(zone_errors, W, g)

        sys_all = system_max(fscores, ZONE_NAMES)
        thr_all, _ = f1_optimal_threshold(sys_all, labels)
        m_all   = metrics(sys_all, labels, thr_all)

        sys_der = system_max(fscores, DER_ZONES)
        thr_der, _ = f1_optimal_threshold(sys_der, labels)
        m_der   = metrics(sys_der, labels, thr_der)

        print(f"\n  gamma={g:.1f}")
        print_row("All zones MAX", m_all)
        print_row("DER-only MAX (z2/z3/z4)", m_der)


# ── Analysis 3 — Threshold strategy comparison ────────────────────────────────

def sweep_threshold_strategy(zone_errors, W, labels):
    print("\n" + "=" * 70)
    print("THRESHOLD STRATEGY COMPARISON  (gamma=0.3)")
    print("=" * 70)

    fscores = final_scores(zone_errors, W, CURRENT_GAMMA)
    sys_s   = system_max(fscores)
    auc     = float(roc_auc_score(labels, sys_s))
    print(f"  AUC (threshold-independent): {auc:.4f}\n")

    # Fixed percentile (current)
    thr = pct_threshold(sys_s, labels, CURRENT_PCT)
    m   = metrics(sys_s, labels, thr)
    print_row(f"Fixed {CURRENT_PCT}th pct (current)", m)

    # F1-optimal
    thr, _ = f1_optimal_threshold(sys_s, labels)
    m = metrics(sys_s, labels, thr)
    print_row("F1-optimal threshold", m)

    # Youden's J
    thr = youden_threshold(sys_s, labels)
    m   = metrics(sys_s, labels, thr)
    print_row("Youden's J threshold", m)

    # Several fixed FPR targets
    for target_fpr in [0.01, 0.02, 0.05]:
        fpr_v, tpr_v, thresholds = roc_curve(labels, sys_s)
        idx = np.argmin(np.abs(fpr_v - target_fpr))
        thr = float(thresholds[idx])
        m   = metrics(sys_s, labels, thr)
        print_row(f"Fixed FPR<={target_fpr:.0%}", m)


# ── Analysis 4 — Percentile sweep at best gamma ───────────────────────────────

def sweep_percentiles(zone_errors, W, labels, gamma=CURRENT_GAMMA):
    print("\n" + "=" * 70)
    print(f"PERCENTILE SWEEP  (gamma={gamma}, threshold from test-normal MAX)")
    print("=" * 70)
    print(f"  {'pct':<7}  {'AUC':>6}  {'F1':>6}  {'Recall':>7}  "
          f"{'Prec':>6}  {'FPR':>6}")
    print("  " + "-" * 50)

    fscores = final_scores(zone_errors, W, gamma)
    sys_s   = system_max(fscores)
    rows = []
    for pct in PERCENTILES:
        thr = pct_threshold(sys_s, labels, pct)
        m   = metrics(sys_s, labels, thr)
        marker = "  <-- current" if pct == CURRENT_PCT else ""
        print(f"  {pct:<7.1f}  {m['auc']:>6.4f}  {m['f1']:>6.4f}  "
              f"{m['rec']:>7.4f}  {m['prec']:>6.4f}  {m['fpr']:>6.4f}{marker}")
        rows.append({'pct': pct, **m})
    return rows


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_gamma_sweep(rows_pct, rows_opt):
    gammas = [r['gamma'] for r in rows_pct]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # AUC (same for both threshold strategies, use pct rows)
    axes[0].plot(gammas, [r['auc'] for r in rows_pct],
                 'o-', color='#2c7bb6', linewidth=2, markersize=6)
    axes[0].axvline(CURRENT_GAMMA, color='gray', ls=':', lw=1.2, label=f'γ={CURRENT_GAMMA}')
    axes[0].set_xlabel('Gamma (γ)', fontsize=12)
    axes[0].set_ylabel('AUC-ROC', fontsize=12)
    axes[0].set_title('AUC vs Gamma\n(threshold-independent)', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # F1 — both threshold strategies
    axes[1].plot(gammas, [r['f1'] for r in rows_pct],
                 's-', color='#d7191c', linewidth=2, markersize=6, label=f'99.4th pct threshold')
    axes[1].plot(gammas, [r['f1'] for r in rows_opt],
                 '^--', color='#1a9641', linewidth=2, markersize=6, label='F1-optimal threshold')
    axes[1].axvline(CURRENT_GAMMA, color='gray', ls=':', lw=1.2, label=f'γ={CURRENT_GAMMA}')
    axes[1].set_xlabel('Gamma (γ)', fontsize=12)
    axes[1].set_ylabel('F1 Score (Attack)', fontsize=12)
    axes[1].set_title('F1 vs Gamma', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Recall vs FPR trade-off for each gamma (F1-optimal threshold)
    axes[2].scatter([r['fpr'] for r in rows_opt],
                    [r['rec'] for r in rows_opt],
                    c=gammas, cmap='RdYlGn_r', s=80, zorder=3)
    for r in rows_opt:
        axes[2].annotate(f"γ={r['gamma']}", (r['fpr'], r['rec']),
                         textcoords='offset points', xytext=(5, 3), fontsize=8)
    axes[2].set_xlabel('False Positive Rate', fontsize=12)
    axes[2].set_ylabel('Recall (Attack)', fontsize=12)
    axes[2].set_title('Recall vs FPR Trade-off\n(F1-optimal threshold)', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('INTACT Gamma Sweep Analysis', fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'intact_gamma_sweep.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  [OK] {os.path.basename(path)}')


def plot_percentile_sweep(rows):
    pcts = [r['pct'] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pcts, [r['f1']  for r in rows], 'o-', color='#d7191c',  lw=2, ms=6, label='F1 (Attack)')
    ax.plot(pcts, [r['rec'] for r in rows], 's-', color='#2c7bb6',  lw=2, ms=6, label='Recall')
    ax.plot(pcts, [r['fpr'] for r in rows], '^-', color='#ff7f00',  lw=2, ms=6, label='FPR')
    ax.plot(pcts, [r['auc'] for r in rows], 'D-', color='#984ea3',  lw=2, ms=6, label='AUC')
    ax.axvline(CURRENT_PCT, color='gray', ls=':', lw=1.5, label=f'Current ({CURRENT_PCT}th pct)')
    ax.set_xlabel('Threshold Percentile', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'INTACT Metrics vs Threshold Percentile  (γ={CURRENT_GAMMA})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'intact_percentile_sweep.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  [OK] {os.path.basename(path)}')


def plot_roc_gamma(zone_errors, W, labels):
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.get_cmap('plasma', len(GAMMAS))
    for i, g in enumerate(GAMMAS):
        fscores = final_scores(zone_errors, W, g)
        sys_s   = system_max(fscores)
        fpr_v, tpr_v, _ = roc_curve(labels, sys_s)
        auc = roc_auc_score(labels, sys_s)
        lw  = 2.5 if g == CURRENT_GAMMA else 1.4
        ls  = '-'  if g == CURRENT_GAMMA else '--'
        ax.plot(fpr_v, tpr_v, color=cmap(i), lw=lw, ls=ls,
                label=f'γ={g:.1f}  (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — INTACT System MAX at Different Gamma', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'intact_roc_gamma.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  [OK] {os.path.basename(path)}')


# ── Analysis 5 — Local Optuna thresholds applied to INTACT raw errors ─────────

# Thresholds from Optuna-tuned standalone zone models (zone*_final_summary.txt)
# Optimal percentiles found by Optuna for each local zone model
# (from zone*_final_summary.txt — these are the operating points, not the MSE values)
OPTUNA_PERCENTILES = {
    'zone1': 99.4,   # from zone1_final_summary.txt
    'zone2': 99.1,   # from zone2_final_summary.txt  ← lower = more sensitive
    'zone3': 99.4,   # from zone3_final_summary.txt
    'zone4': 99.9,   # from zone4_final_summary.txt  ← higher = more precise
}

# Fallback: MSE values from local Optuna models (used only for comparison reference)
LOCAL_OPTUNA_THRESHOLDS = {
    'zone1': 0.006712,
    'zone2': 0.004597,
    'zone3': 0.005947,
    'zone4': 0.003344,
}


def sweep_local_optuna_thresholds(zone_errors, W, labels):
    print("\n" + "=" * 70)
    print("OPTUNA-PERCENTILE THRESHOLDS ON INTACT TRAINING DISTRIBUTION")
    print("Each zone uses the percentile Optuna selected for its local model,")
    print("applied to INTACT's own zone error distribution.")
    print("(Local approx uses test-normal windows; AGX run uses true training data.)")
    print("=" * 70)

    # Load INTACT per-zone saved thresholds (computed at 99.4th pct on training data)
    intact_train_thresholds = {}
    for zid in ZONE_NAMES:
        thr_path = os.path.join(FL_DIR, f'intact_{zid}_threshold.npy')
        intact_train_thresholds[zid] = float(np.load(thr_path)) if os.path.exists(thr_path) else None

    # Compute Optuna-percentile thresholds on INTACT's test-NORMAL distribution (local proxy)
    # On AGX these would be computed on training data instead
    optuna_pct_thresholds_testproxy = {}
    for zid in ZONE_NAMES:
        pct = OPTUNA_PERCENTILES[zid]
        normal_errs = zone_errors[zid][labels == 0]
        optuna_pct_thresholds_testproxy[zid] = float(np.percentile(normal_errs, pct))

    # Show the threshold comparison table
    print(f"\n  {'zone':<8}  {'Optuna pct':>10}  {'INTACT 99.4 (train)':>20}  "
          f"{'Optuna pct (test-proxy)':>23}  {'diff':>8}")
    print("  " + "-" * 78)
    for zid in ZONE_NAMES:
        pct       = OPTUNA_PERCENTILES[zid]
        thr_99_4  = intact_train_thresholds[zid] or 0
        thr_opt   = optuna_pct_thresholds_testproxy[zid]
        diff      = thr_opt - thr_99_4
        arrow     = "higher (stricter)" if diff > 0 else "lower  (looser)"
        print(f"  {zid:<8}  {pct:>10.1f}  {thr_99_4:>20.6f}  "
              f"{thr_opt:>23.6f}  {arrow}")

    # OR-logic system detection using Optuna percentiles on test-normal proxy
    print(f"\n  Results using Optuna percentile thresholds (test-normal proxy):")
    print(f"  {'gamma':<8}  {'AUC':>6}  {'F1':>6}  {'Recall':>7}  {'Prec':>6}  {'FPR':>6}")
    print("  " + "-" * 52)

    for g in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        fscores = final_scores(zone_errors, W, g)
        sys_s   = system_max(fscores)
        zone_alarms = {z: (zone_errors[z] > optuna_pct_thresholds_testproxy[z])
                       for z in ZONE_NAMES}
        sys_preds = np.any(list(zone_alarms.values()), axis=0).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, sys_preds, labels=[0, 1], zero_division=0)
        fpr = float(np.sum((sys_preds == 1) & (labels == 0)) / np.sum(labels == 0))
        auc = float(roc_auc_score(labels, sys_s))
        marker = "  <-- gamma=0.3" if g == CURRENT_GAMMA else ""
        print(f"  {g:<8.2f}  {auc:>6.4f}  {f1[1]:>6.4f}  {rec[1]:>7.4f}  "
              f"{prec[1]:>6.4f}  {fpr:>6.4f}{marker}")

    # Per-zone breakdown at gamma=0.3
    print(f"\n  Per-zone breakdown at gamma=0.3:")
    print(f"  {'zone':<8}  {'pct':>5}  {'threshold':>10}  {'FPR':>6}  {'Recall':>7}")
    print("  " + "-" * 48)
    fscores_03 = final_scores(zone_errors, W, 0.3)
    zone_alarms_03 = {z: (zone_errors[z] > optuna_pct_thresholds_testproxy[z])
                      for z in ZONE_NAMES}
    for zid in ZONE_NAMES:
        fpr_z = zone_alarms_03[zid][labels == 0].sum() / (labels == 0).sum()
        rec_z = zone_alarms_03[zid][labels == 1].sum() / (labels == 1).sum()
        print(f"  {zid:<8}  {OPTUNA_PERCENTILES[zid]:>5.1f}  "
              f"{optuna_pct_thresholds_testproxy[zid]:>10.6f}  "
              f"{fpr_z:>6.4f}  {rec_z:>7.4f}")

    # Final head-to-head comparison at gamma=0.3
    print(f"\n  HEAD-TO-HEAD COMPARISON at gamma=0.3:")
    fscores = final_scores(zone_errors, W, 0.3)
    sys_s   = system_max(fscores)
    auc     = float(roc_auc_score(labels, sys_s))

    # (a) current: training-calibrated 99.4 system MAX
    train_thr_path = os.path.join(FL_DIR, 'intact_system_threshold.npy')
    if os.path.exists(train_thr_path):
        m = metrics(sys_s, labels, float(np.load(train_thr_path)))
        print_row("Current (train 99.4 sys MAX)", m)

    # (b) Optuna-pct per-zone OR logic (test-normal proxy)
    sys_preds = np.any(list(zone_alarms_03.values()), axis=0).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, sys_preds, labels=[0, 1], zero_division=0)
    fpr = float(np.sum((sys_preds == 1) & (labels == 0)) / np.sum(labels == 0))
    print(f"  {'Optuna-pct OR logic (test proxy)':<30}  AUC={auc:.4f}  "
          f"F1={f1[1]:.4f}  Rec={rec[1]:.4f}  Prec={prec[1]:.4f}  FPR={fpr:.4f}")

    # (c) F1-optimal ceiling
    thr_opt, _ = f1_optimal_threshold(sys_s, labels)
    m = metrics(sys_s, labels, thr_opt)
    print_row("F1-optimal ceiling", m)

    print(f"\n  NOTE: Run test_intact.py on AGX with OPTUNA_PERCENTILES to get")
    print(f"  training-data calibrated results (test-normal proxy is approximate).")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading zone errors and admittance matrix...")
    zone_errors, labels = load_zone_errors()
    W = load_admittance()

    n_normal = int((labels == 0).sum())
    n_attack = int((labels == 1).sum())
    print(f"  {len(labels)} windows  |  Normal: {n_normal}  |  Attack: {n_attack}")

    # Run all analyses
    rows_pct, rows_opt = sweep_gamma(zone_errors, W, labels)
    sweep_der_vs_all(zone_errors, W, labels)
    sweep_threshold_strategy(zone_errors, W, labels)
    rows_pct_sweep = sweep_percentiles(zone_errors, W, labels, gamma=CURRENT_GAMMA)
    sweep_local_optuna_thresholds(zone_errors, W, labels)

    # Find best gamma by F1-optimal
    best = max(rows_opt, key=lambda r: r['f1'])
    print(f"\n{'='*70}")
    print(f"BEST GAMMA (F1-optimal threshold): gamma={best['gamma']:.2f}")
    print(f"  AUC={best['auc']:.4f}  F1={best['f1']:.4f}  "
          f"Recall={best['rec']:.4f}  FPR={best['fpr']:.4f}")
    print(f"\nRun test_intact.py on AGX with this gamma to get training-calibrated results.")
    print(f"{'='*70}\n")

    # Also run percentile sweep at best gamma if different from current
    if best['gamma'] != CURRENT_GAMMA:
        print(f"\nPercentile sweep at best gamma={best['gamma']:.2f}:")
        rows_best_gamma = sweep_percentiles(zone_errors, W, labels, gamma=best['gamma'])

    # Figures
    print("\nGenerating figures...")
    plot_gamma_sweep(rows_pct, rows_opt)
    plot_percentile_sweep(rows_pct_sweep)
    plot_roc_gamma(zone_errors, W, labels)
    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
