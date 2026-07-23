"""
Step 1: Validate that pandapower.networks.case33bw() reproduces the recorded
simulation_with_der.csv when fed the recorded per-bus P/Q as injections.

This is a read-only check against the existing dataset (IDS DATASET/simulation_with_der.csv).
It does not modify that file or any other existing project file.
"""
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

CSV_PATH = r"../../IDS DATASET/simulation_with_der.csv"

df = pd.read_csv(CSV_PATH)
print("rows:", len(df))

df["timestamp"] = pd.to_datetime(df["timestamp"])
noon_mask = (df["timestamp"].dt.hour == 12) & (df["timestamp"].dt.minute == 0)
candidates = df[noon_mask]
print(f"noon candidates: {len(candidates)}")

results = []
for i in range(min(5, len(candidates))):
    row = candidates.iloc[i]

    net = pn.case33bw()
    for idx in net.load.index:
        bus = int(net.load.at[idx, "bus"])
        net.load.at[idx, "p_mw"] = row[f"P_bus{bus}"]
        net.load.at[idx, "q_mvar"] = row[f"Q_bus{bus}"]

    pp.runpp(net)

    errs_v, errs_theta = [], []
    for bus in range(33):
        errs_v.append(abs(row[f"V_bus{bus}"] - net.res_bus.at[bus, "vm_pu"]))
        errs_theta.append(abs(row[f"THETA_bus{bus}"] - net.res_bus.at[bus, "va_degree"]))

    mv, xv = np.mean(errs_v), np.max(errs_v)
    mt, xt = np.mean(errs_theta), np.max(errs_theta)
    results.append((row["timestamp"], mv, xv, mt, xt))
    print(f"{row['timestamp']}  V err mean={mv:.6f} max={xv:.6f}  "
          f"THETA err mean={mt:.6f} max={xt:.6f}")

print("\nNoise tolerance reference (paper): sigma = 0.002 pu")
all_mv = [r[1] for r in results]
print(f"\nOverall mean V error across {len(results)} timestamps: {np.mean(all_mv):.6f}")
