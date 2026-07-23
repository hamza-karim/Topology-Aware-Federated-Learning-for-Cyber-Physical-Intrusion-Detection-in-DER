"""
Step 2: Line-trip fault scenario (sanity check).

For each DER zone, trip the single line feeding its DER bus (a leaf node on the
radial feeder, confirmed via topology inspection: bus 17 <- line(16,17), bus 24 <-
line(23,24), bus 32 <- line(31,32); each has no other in-service connection except
a normally-open tie line, so tripping the feeder line strands exactly that one bus).

For 48 consecutive rows (4 hours, matching the paper's existing attack window length),
recompute the network state with that line out of service, using the REAL recorded
P/Q for every other bus as power-flow input. The stranded bus reports V=0, THETA=0,
P=0, Q=0 (a de-energized junction). Neighbor buses are also recomputed and checked
for any measurable shift.

Reads: IDS DATASET/FL_DATA/centralized_test_combined.csv (read-only)
Writes: fault_tolerance_analysis/data/centralized_test_line_trip.csv (new file)
"""
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

SRC = r"../../IDS DATASET/FL_DATA/centralized_test_combined.csv"
OUT = r"../data/centralized_test_line_trip.csv"

# (der_bus, feeding line from_bus->to_bus, event start timestamp)
# Start timestamps chosen on days with NO existing replay attacks (attacks occupy
# Jan 10-11, 13-18); using Jan 7/8/9 at solar-peak hours instead.
EVENTS = [
    {"der_bus": 17, "from_bus": 16, "to_bus": 17, "start": "2026-01-07 11:00:00"},
    {"der_bus": 24, "from_bus": 23, "to_bus": 24, "start": "2026-01-08 11:00:00"},
    {"der_bus": 32, "from_bus": 31, "to_bus": 32, "start": "2026-01-09 11:00:00"},
]
WINDOW_LEN = 48  # 4 hours at 5-min resolution, matches existing attack duration

df = pd.read_csv(SRC)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df_out = df.copy()
df_out["fault_label"] = 0
df_out["fault_type"] = "none"

all_buses = list(range(1, 33))

for ev in EVENTS:
    db = ev["der_bus"]
    start_idx = df.index[df["timestamp"] == pd.Timestamp(ev["start"])][0]
    window_idxs = range(start_idx, start_idx + WINDOW_LEN)

    print(f"\n=== Event: line trip feeding bus {db} "
          f"({ev['from_bus']}->{ev['to_bus']}), start {ev['start']} ===")

    max_neighbor_dv = 0.0
    max_neighbor_dtheta = 0.0

    for i in window_idxs:
        row = df.loc[i]

        net = pn.case33bw()
        for idx in net.load.index:
            bus = int(net.load.at[idx, "bus"])
            net.load.at[idx, "p_mw"] = row[f"P_bus{bus}"]
            net.load.at[idx, "q_mvar"] = row[f"Q_bus{bus}"]

        line_idx = net.line[
            (net.line["from_bus"] == ev["from_bus"]) & (net.line["to_bus"] == ev["to_bus"])
        ].index[0]
        net.line.at[line_idx, "in_service"] = False

        pp.runpp(net)

        for bus in all_buses:
            v_new = net.res_bus.at[bus, "vm_pu"]
            th_new = net.res_bus.at[bus, "va_degree"]
            if bus == db:
                df_out.at[i, f"V_bus{bus}"] = 0.0
                df_out.at[i, f"THETA_bus{bus}"] = 0.0
                df_out.at[i, f"P_bus{bus}"] = 0.0
                df_out.at[i, f"Q_bus{bus}"] = 0.0
            else:
                if np.isnan(v_new):
                    continue  # shouldn't happen for non-stranded buses
                dv = abs(v_new - row[f"V_bus{bus}"])
                dtheta = abs(th_new - row[f"THETA_bus{bus}"])
                if bus in (ev["from_bus"],):
                    max_neighbor_dv = max(max_neighbor_dv, dv)
                    max_neighbor_dtheta = max(max_neighbor_dtheta, dtheta)
                df_out.at[i, f"V_bus{bus}"] = v_new
                df_out.at[i, f"THETA_bus{bus}"] = th_new
                # P/Q at non-stranded, non-load-changed buses are unaffected inputs;
                # leave as recorded (they are the specified injections, unchanged).

        df_out.at[i, "fault_label"] = 1
        df_out.at[i, "fault_type"] = f"line_trip_bus{db}"

    print(f"Max |dV| at immediate upstream neighbor bus {ev['from_bus']}: {max_neighbor_dv:.6f} pu")
    print(f"Max |dTHETA| at immediate upstream neighbor bus {ev['from_bus']}: {max_neighbor_dtheta:.6f} deg")

df_out.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}")
print(df_out["fault_type"].value_counts())
