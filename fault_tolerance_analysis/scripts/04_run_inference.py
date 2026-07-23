"""
Step 4: Cross-zone consistency inference on normal / cyberattack / line-trip /
DER-dropout data, using the EXISTING local-baseline LSTM autoencoders
(ML model/models/zone{1-4}_local_lstm.keras + their fitted scalers) and the
EXISTING zone_admittance.csv — no retraining, no Jetson access needed for
this pass. This replicates the Eq. 7-10 cross-zone consistency math from
FL/Server/test_intact.py, applied here to the local baseline models instead
of the INTACT-personalized weights, purely to test whether the MECHANISM
(not the specific trained model) responds differently to physical faults
than to cyberattacks.

Reads only (no existing files modified):
  ML model/models/zone{1-4}_local_lstm.keras, zone{1-4}_scaler.pkl
  IDS DATASET/zone_admittance.csv
  IDS DATASET/FL_DATA/centralized_test_combined.csv   (normal + existing cyberattacks)
  fault_tolerance_analysis/data/centralized_test_line_trip.csv
  fault_tolerance_analysis/data/centralized_test_der_dropout.csv

Writes:
  fault_tolerance_analysis/results/window_scores.csv
  fault_tolerance_analysis/results/summary_by_category.csv
"""
import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

WINDOW_SIZE = 30
NUM_FEATURES = 36
BATCH_SIZE = 64

MODELS_DIR = r"../../ML model/models"
ADM_PATH = r"../../IDS DATASET/zone_admittance.csv"

DATASETS = {
    "normal_attack": r"../../IDS DATASET/FL_DATA/centralized_test_combined.csv",
    "line_trip":     r"../data/centralized_test_line_trip.csv",
    "der_dropout":   r"../data/centralized_test_der_dropout.csv",
}

# Per-zone attack labels (zone-specific: was THIS zone the one actually attacked),
# row-aligned by timestamp with centralized_test_combined.csv (verified: exact match).
# Used only for the normal_attack dataset, to split "this zone attacked" from
# "another zone attacked, this one is a bystander" instead of the pooled
# system-wide attack_label.
ZONE_ATTACK_CSV = {
    "zone2": r"../../IDS DATASET/FL_DATA/zone2_test_stealthy.csv",
    "zone3": r"../../IDS DATASET/FL_DATA/zone3_test_stealthy.csv",
    "zone4": r"../../IDS DATASET/FL_DATA/zone4_test_stealthy.csv",
}

ZONE_NAMES = ["zone1", "zone2", "zone3", "zone4"]
ZONE_BUSES = {
    "zone1": range(1, 9),
    "zone2": range(9, 18),
    "zone3": range(18, 25),
    "zone4": range(25, 33),
}
GAMMA = 0.3  # matches paper's Eq. 10 consistency penalty coefficient


def get_zone_columns(columns, zone_id):
    buses = ZONE_BUSES[zone_id]
    pattern = re.compile(r"_bus(" + "|".join(str(b) for b in buses) + r")$")
    return [c for c in columns if pattern.search(c)]


def pad_to_n(data, n):
    if data.shape[1] < n:
        pad = np.zeros((data.shape[0], n - data.shape[1]), dtype=data.dtype)
        return np.concatenate([data, pad], axis=1)
    return data


def create_windows(data):
    return np.array([data[i:i + WINDOW_SIZE] for i in range(len(data) - WINDOW_SIZE + 1)])


def zone_errors(model, scaler, df, zone_id):
    # Local baseline models were each trained on their own zone's natural
    # bus-count * 4 features (32/36/28/32 for zones 1-4), not uniformly
    # padded to 36 like the INTACT-personalized models. Pad/match to
    # whatever THIS model actually expects.
    n_expected = model.input_shape[-1]
    cols = get_zone_columns(list(df.columns), zone_id)
    data = df[cols].values.astype(np.float32)
    data = scaler.transform(data)
    data = pad_to_n(data, n_expected)
    windows = create_windows(data)
    pred = model.predict(windows, batch_size=BATCH_SIZE, verbose=0)
    return np.mean(np.mean(np.square(windows - pred), axis=2), axis=1)


def load_admittance_weights(path):
    df = pd.read_csv(path)
    zone_bus_sets = {z: set(ZONE_BUSES[z]) for z in ZONE_NAMES}
    raw = {z: {z2: 0.0 for z2 in ZONE_NAMES} for z in ZONE_NAMES}
    for _, row in df.iterrows():
        fb, tb, adm = int(row["from_bus"]), int(row["to_bus"]), float(row["admittance"])
        fz = next((z for z, b in zone_bus_sets.items() if fb in b), None)
        tz = next((z for z, b in zone_bus_sets.items() if tb in b), None)
        if fz and tz and fz != tz:
            raw[fz][tz] += adm
            raw[tz][fz] += adm
    W = {}
    for z in ZONE_NAMES:
        total = sum(raw[z][z2] for z2 in ZONE_NAMES if z2 != z)
        W[z] = {z2: raw[z][z2] / total for z2 in ZONE_NAMES if z2 != z}
    return W


print("Loading models, scalers, admittance weights...")
models, scalers = {}, {}
for z in ZONE_NAMES:
    n = z.replace("zone", "")
    models[z] = tf.keras.models.load_model(os.path.join(MODELS_DIR, f"zone{n}_local_lstm.keras"))
    scalers[z] = joblib.load(os.path.join(MODELS_DIR, f"zone{n}_scaler.pkl"))
W = load_admittance_weights(ADM_PATH)
for z in ZONE_NAMES:
    print(f"  W[{z}] = {W[z]}")

all_rows = []

for ds_name, path in DATASETS.items():
    print(f"\n=== Dataset: {ds_name} ===")
    df = pd.read_csv(path)

    has_attack = "attack_label" in df.columns
    has_fault = "fault_label" in df.columns

    attack_row = df["attack_label"].values if has_attack else np.zeros(len(df), dtype=int)
    fault_row = df["fault_label"].values if has_fault else np.zeros(len(df), dtype=int)
    fault_type_row = df["fault_type"].values if "fault_type" in df.columns else np.array(["none"] * len(df))

    n_windows = len(df) - WINDOW_SIZE + 1

    def window_any(rowvals):
        return np.array([1 if np.any(rowvals[i:i + WINDOW_SIZE] == 1) else 0 for i in range(n_windows)])

    def window_mode_faulttype(vals):
        out = []
        for i in range(n_windows):
            seg = vals[i:i + WINDOW_SIZE]
            nz = [v for v in seg if v != "none"]
            out.append(nz[0] if nz else "none")
        return out

    attack_win = window_any(attack_row)
    fault_win = window_any(fault_row)
    fault_type_win = window_mode_faulttype(fault_type_row)

    # Zone-specific attack windows (only populated for the normal_attack dataset)
    zone_attack_win = {}
    if ds_name == "normal_attack":
        for z, path_z in ZONE_ATTACK_CSV.items():
            zdf = pd.read_csv(path_z)
            assert (zdf["timestamp"].values == df["timestamp"].values).all(), \
                f"timestamp misalignment between {path_z} and {path}"
            zone_attack_win[z] = window_any(zdf["attack_label"].values)

    errs = {}
    for z in ZONE_NAMES:
        print(f"  computing errors for {z}...")
        errs[z] = zone_errors(models[z], scalers[z], df, z)

    for z in ["zone2", "zone3", "zone4"]:  # DER zones only; zone1 used just as neighbor input
        e = errs[z]
        e_hat = np.zeros_like(e)
        for other, w in W[z].items():
            e_hat += w * errs[other]
        mismatch = e - e_hat
        score = e + GAMMA * mismatch

        for i in range(n_windows):
            if attack_win[i] and ds_name == "normal_attack":
                category = "attack_own_zone" if zone_attack_win[z][i] else "attack_other_zone"
            elif fault_win[i]:
                category = fault_type_win[i]  # e.g. line_trip_bus17, der_dropout_bus24
            else:
                category = "normal"
            all_rows.append({
                "dataset": ds_name, "zone": z, "window_idx": i,
                "category": category, "e": e[i], "e_hat": e_hat[i],
                "mismatch": mismatch[i], "score": score[i],
            })

results = pd.DataFrame(all_rows)
os.makedirs("../results", exist_ok=True)
results.to_csv("../results/window_scores.csv", index=False)
print(f"\nSaved {len(results)} window-level rows to ../results/window_scores.csv")

# Only count each fault/attack category once (it appears identically in every
# dataset's "normal" bucket, so restrict normal to the normal_attack dataset).
results_dedup = results[~((results["category"] == "normal") & (results["dataset"] != "normal_attack"))]

summary = results_dedup.groupby(["zone", "category"]).agg(
    n=("e", "size"),
    e_mean=("e", "mean"), e_median=("e", "median"),
    mismatch_mean=("mismatch", "mean"), mismatch_median=("mismatch", "median"),
    score_mean=("score", "mean"), score_median=("score", "median"),
).reset_index()
summary.to_csv("../results/summary_by_category.csv", index=False)
print("\n=== Summary by zone and category ===")
print(summary.to_string(index=False))
