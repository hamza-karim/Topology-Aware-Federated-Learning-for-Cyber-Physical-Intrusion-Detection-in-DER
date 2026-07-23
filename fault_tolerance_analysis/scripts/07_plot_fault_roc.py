"""
Figure for the physical-fault subsection (VI.C): ROC curves for the
system-level (MAX-fusion, Eq. 11) detection of the line/feeder fault and
the DER dropout, matching the plotting style of the paper's existing Fig. 7.

Reads: fault_tolerance_analysis/results/agx_real_intact/fault_tolerance_window_scores.csv
Writes: fault_tolerance_analysis/results/fig_physical_fault_roc.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

df = pd.read_csv("../results/agx_real_intact/fault_tolerance_window_scores.csv")

# Exclude windows overlapping the ORIGINAL cyberattacks still present in these
# copied files -- otherwise correct detections of the old attack get counted
# as false positives against the new fault labels.
orig = pd.read_csv("../../IDS DATASET/FL_DATA/centralized_test_combined.csv")
attack_rows = orig.index[orig["attack_label"] == 1].tolist()
WINDOW = 30
touched_windows = set()
for r in attack_rows:
    lo = max(0, r - WINDOW + 1)
    for w in range(lo, r + 1):
        touched_windows.add(w)

fig, ax = plt.subplots(figsize=(7, 6))
colors = {"line_trip": "#d7191c", "der_dropout": "#2c7bb6"}
labels = {"line_trip": "Line/Feeder Fault", "der_dropout": "DER Dropout"}

for ds in ["line_trip", "der_dropout"]:
    sub = df[df["dataset"] == ds]
    piv_score = sub.pivot(index="window_idx", columns="zone", values="score")
    piv_cat = sub.pivot(index="window_idx", columns="zone", values="category")
    # MAX across all four zones, matching Eq. 11 exactly (zone1 included)
    system_score = piv_score[["zone1", "zone2", "zone3", "zone4"]].max(axis=1)
    y_true = (piv_cat["zone2"] != "normal").astype(int)

    keep = ~system_score.index.isin(touched_windows)
    system_score = system_score[keep]
    y_true = y_true[keep]

    fpr, tpr, _ = roc_curve(y_true, system_score)
    auc = roc_auc_score(y_true, system_score)
    ax.plot(fpr, tpr, color=colors[ds], lw=2.2, label=f"{labels[ds]} (AUC={auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — System-Level Physical Fault Detection", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("../results/fig_physical_fault_roc.png", dpi=150)
print("Saved ../results/fig_physical_fault_roc.png")
