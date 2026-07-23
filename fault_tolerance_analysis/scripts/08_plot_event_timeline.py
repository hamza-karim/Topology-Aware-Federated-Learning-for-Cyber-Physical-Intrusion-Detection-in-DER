"""
Event-timeline figure for VI.C: system-level score over time during a
representative DER-dropout event (Zone 4 / Bus 32), showing the score
crossing the detection threshold exactly when the fault window begins
and falling back afterward. Same visual idea as the paper's existing Fig. 5
(timeline view), applied to the system-level MAX-fusion score instead of a
single zone's raw error.

Reads: fault_tolerance_analysis/results/agx_real_intact/fault_tolerance_window_scores.csv
       fault_tolerance_analysis/data/centralized_test_der_dropout.csv (for timestamps)
Writes: fault_tolerance_analysis/results/fig_event_timeline_der_dropout.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYS_THRESHOLD = 0.005887286022305488
EVENT_ROW_START, EVENT_ROW_END = 3888, 3935  # der_dropout_bus32, confirmed above
PAD = 40  # windows of context before/after the event

scores_df = pd.read_csv("../results/agx_real_intact/fault_tolerance_window_scores.csv")
raw_df = pd.read_csv("../data/centralized_test_der_dropout.csv")
raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])

sub = scores_df[scores_df["dataset"] == "der_dropout"]
piv_score = sub.pivot(index="window_idx", columns="zone", values="score")
piv_cat = sub.pivot(index="window_idx", columns="zone", values="category")
system_score = piv_score[["zone1", "zone2", "zone3", "zone4"]].max(axis=1)
is_fault_window = (piv_cat["zone2"] == "der_dropout_bus32")  # any-row-touched convention

lo = max(0, EVENT_ROW_START - PAD)
hi = min(len(system_score) - 1, EVENT_ROW_END + PAD)
window_idxs = list(range(lo, hi + 1))
timestamps = raw_df.loc[window_idxs, "timestamp"].values
y = system_score.loc[window_idxs].values
fault_mask = is_fault_window.loc[window_idxs].values

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(timestamps, y, color="#2c7bb6", lw=1.4, label="System score (MAX fusion)")
ax.axhline(SYS_THRESHOLD, color="black", ls="--", lw=1.2, label=f"Detection threshold ({SYS_THRESHOLD:.4f})")

# Shade the windows actually counted as "fault" under the any-row-touched
# convention (Eq. 2) -- this starts up to 29 timesteps before the raw fault
# rows begin, since a window overlapping even the first affected row counts.
fault_ts = timestamps[fault_mask]
if len(fault_ts) > 0:
    ax.axvspan(fault_ts.min(), fault_ts.max(), color="red", alpha=0.15,
               label="Windows overlapping the DER dropout event (Bus 32)")
ax.axvline(raw_df.loc[EVENT_ROW_START, "timestamp"], color="darkred", ls=":", lw=1.3,
           label="Actual fault onset (row-level)")

ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("System-Level Anomaly Score", fontsize=12)
ax.set_title("System Score During a Representative DER Dropout Event (Zone 4, Bus 32)", fontsize=12)
ax.legend(loc="upper right", fontsize=7, framealpha=0.9, handlelength=1.5,
          labelspacing=0.3, borderpad=0.5)
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("../results/fig_event_timeline_der_dropout.png", dpi=150)
print("Saved ../results/fig_event_timeline_der_dropout.png")
