"""
Step 3: DER-dropout fault scenario (main test).

Confirmed by correlation search: centralized_test_combined.csv rows map 1:1 by
row index (not by literal date string) to the first 4032 rows of the original
simulation_with_der.csv / simulation_no_der.csv (corr=0.994 at offset 0; the
test file has been re-dated and had noise added, but is otherwise the same
underlying simulation). This lets us pull the TRUE "no generation" value for
a given bus at a given moment directly from simulation_no_der.csv by matching
row index, rather than approximating it.

For each DER bus, at a chosen daytime event: replace ONLY that bus's P/Q with
its real no-DER (load-only) value at the same row index, keep every other bus
(including the other two still-active DER buses) at its real recorded with-DER
value, and rerun power flow so the rest of the network responds physically
correctly to that one inverter going dark.

Reads: IDS DATASET/FL_DATA/centralized_test_combined.csv (read-only)
       IDS DATASET/simulation_no_der.csv (read-only)
Writes: fault_tolerance_analysis/data/centralized_test_der_dropout.csv (new file)
"""
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

TEST_SRC = r"../../IDS DATASET/FL_DATA/centralized_test_combined.csv"
NODER_SRC = r"../../IDS DATASET/simulation_no_der.csv"
OUT = r"../data/centralized_test_der_dropout.csv"

# der_bus, event start ROW INDEX in the test file (= same row index in simulation_no_der.csv)
# Chosen on days with NO existing replay attacks and NOT reused by the line-trip script
# (which used Jan 7/8/9). Using Jan 6, Jan 12, Jan 19 at solar-peak hour instead.
EVENTS = [
    {"der_bus": 17, "start": "2026-01-06 12:00:00"},
    {"der_bus": 24, "start": "2026-01-12 12:00:00"},
    {"der_bus": 32, "start": "2026-01-19 12:00:00"},
]
WINDOW_LEN = 48  # 4 hours, matches existing attack / line-trip duration

test_df = pd.read_csv(TEST_SRC)
test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
noder_df = pd.read_csv(NODER_SRC)  # same row-index frame as the original full simulation

df_out = test_df.copy()
df_out["fault_label"] = 0
df_out["fault_type"] = "none"

all_buses = list(range(1, 33))

for ev in EVENTS:
    db = ev["der_bus"]
    start_idx = test_df.index[test_df["timestamp"] == pd.Timestamp(ev["start"])][0]
    window_idxs = range(start_idx, start_idx + WINDOW_LEN)

    print(f"\n=== Event: DER dropout at bus {db}, start {ev['start']} (row {start_idx}) ===")

    gen_contribs = []
    max_neighbor_dv = 0.0
    max_neighbor_dtheta = 0.0

    for i in window_idxs:
        row = test_df.loc[i]
        noder_row = noder_df.loc[i]  # same row index = same underlying simulation moment

        gen_contrib = noder_row[f"P_bus{db}"] - row[f"P_bus{db}"]
        gen_contribs.append(gen_contrib)

        net = pn.case33bw()
        for idx in net.load.index:
            bus = int(net.load.at[idx, "bus"])
            if bus == db:
                # DER offline: use the true load-only (no-DER) injection for this bus
                net.load.at[idx, "p_mw"] = noder_row[f"P_bus{bus}"]
                net.load.at[idx, "q_mvar"] = noder_row[f"Q_bus{bus}"]
            else:
                # everyone else, including the other two still-active DERs, stays real
                net.load.at[idx, "p_mw"] = row[f"P_bus{bus}"]
                net.load.at[idx, "q_mvar"] = row[f"Q_bus{bus}"]

        pp.runpp(net)

        for bus in all_buses:
            v_new = net.res_bus.at[bus, "vm_pu"]
            th_new = net.res_bus.at[bus, "va_degree"]
            if bus == db:
                df_out.at[i, f"V_bus{bus}"] = v_new
                df_out.at[i, f"THETA_bus{bus}"] = th_new
                df_out.at[i, f"P_bus{bus}"] = noder_row[f"P_bus{bus}"]
                df_out.at[i, f"Q_bus{bus}"] = noder_row[f"Q_bus{bus}"]
            else:
                dv = abs(v_new - row[f"V_bus{bus}"])
                dtheta = abs(th_new - row[f"THETA_bus{bus}"])
                max_neighbor_dv = max(max_neighbor_dv, dv)
                max_neighbor_dtheta = max(max_neighbor_dtheta, dtheta)
                df_out.at[i, f"V_bus{bus}"] = v_new
                df_out.at[i, f"THETA_bus{bus}"] = th_new
                # P/Q unchanged for buses whose injection we didn't touch

        df_out.at[i, "fault_label"] = 1
        df_out.at[i, "fault_type"] = f"der_dropout_bus{db}"

    print(f"Generation lost: mean={np.mean(gen_contribs):.4f} MW, "
          f"max={np.max(gen_contribs):.4f} MW")
    print(f"Max |dV| across all other buses: {max_neighbor_dv:.6f} pu")
    print(f"Max |dTHETA| across all other buses: {max_neighbor_dtheta:.6f} deg")

df_out.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}")
print(df_out["fault_type"].value_counts())
