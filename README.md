# Topology-Aware Federated Learning for Cyber-Physical Intrusion Detection in DER

A privacy-preserving intrusion detection system (IDS) for smart grid Distributed Energy Resource (DER) networks. Each zone of an IEEE 33-bus power grid is monitored by a dedicated edge device (NVIDIA Jetson). A federated LSTM autoencoder is trained across all zones using the Flower framework — no raw data leaves any device.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Attack Model](#3-attack-model)
4. [Dataset](#4-dataset)
5. [Zone Topology](#5-zone-topology)
6. [Model Architecture](#6-model-architecture)
7. [Directory Structure](#7-directory-structure)
8. [Baseline Results](#8-baseline-results)
9. [FL Implementation Details](#9-fl-implementation-details)
10. [Step 1 — Build Docker Images](#step-1--build-docker-images)
11. [Step 2 — Push to Docker Hub](#step-2--push-to-docker-hub)
12. [Step 3 — Deploy on Edge Devices](#step-3--deploy-on-edge-devices)
13. [Step 4 — Configure start_clients.sh](#step-4--configure-start_clientssh-one-time-setup)
14. [Step 5 — Run FL Training](#step-5--run-fl-training)
15. [Step 6 — Evaluate the Global Model](#step-6--evaluate-the-global-model)
16. [Dependencies](#16-dependencies)

---

## 1. Project Overview

Modern power grids increasingly integrate DER components (solar, battery storage, EV chargers). These cyber-physical systems are vulnerable to data injection attacks that manipulate sensor readings to disrupt grid operation. Centralized detection approaches require sharing raw grid measurements, which violates operator privacy and creates a single point of failure.

This project addresses that by:

- Partitioning the IEEE 33-bus grid into **4 topology-aware zones**
- Assigning each zone to a **Jetson edge device** that trains locally
- Using **Federated Averaging (FedAvg)** to aggregate a shared global model without exposing raw data
- Detecting **stealthy context-aware replay attacks** using an LSTM autoencoder trained on normal operation only

Three model variants are compared:

| Variant | Description |
|---|---|
| **Centralized** | Single model trained on all 128 features from all zones — privacy-violating upper bound |
| **Local** | One model per zone, trained only on that zone's data — no collaboration |
| **Federated (FedAvg)** | One global model trained collaboratively across all zones — privacy-preserving |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   IEEE 33-Bus Grid                  │
│  Zone 1 (bus 1-8) │ Zone 2 (bus 9-17)              │
│  Zone 3 (bus 18-24)│ Zone 4 (bus 25-32)            │
└────────┬──────────┴────────┬────────────────────────┘
         │  local sensor data│
  ┌──────▼──────┐     ┌──────▼──────┐
  │  Jetson #1  │     │  Jetson #2  │   ...×4
  │  FL Client  │     │  FL Client  │
  │  (zone1)    │     │  (zone2)    │
  └──────┬──────┘     └──────┬──────┘
         │   model weights only (no raw data)
         └──────────┬─────────┘
               ┌────▼─────┐
               │  Jetson  │
               │ FL Server│
               │ (FedAvg) │
               └──────────┘
```

Training flow per round:
1. Server broadcasts current global model weights to all clients
2. Each client trains locally for N epochs on its zone data
3. Clients send updated weights back to the server
4. Server aggregates with FedAvg (weighted average by sample count)
5. Repeat for 50 rounds

---

## 3. Attack Model

**Attack type:** Stealthy context-aware replay attack

The attacker records legitimate sensor measurements during a reference window and replays them at a later time to mask abnormal grid behaviour. Two design techniques make the attack hard to detect:

- **Statistical window selection** — the replayed window is chosen to closely match the statistical profile of the current operating point ([Shen & Qin 2024])
- **Cross-fade boundary blending** — the transition between real and replayed data is smoothed over 12 steps (1 hour) to avoid abrupt discontinuities ([Erba et al. 2020])

The attack injects all four measurement types simultaneously: **P, Q, |V|, θ** (active power, reactive power, voltage magnitude, voltage angle).

---

## 4. Dataset

The dataset is generated from a power flow simulation of the IEEE 33-bus distribution network with DER integration.

| Property | Value |
|---|---|
| Sampling interval | 5 minutes |
| Train rows | 8,064 (≈28 days of normal operation) |
| Test rows | 4,032 (≈14 days, mixed normal + attack) |
| Features | 128 (4 measurements × 32 buses) |
| Normal windows (test) | 3,720 |
| Attack windows (test) | 312 |
| Noise std | 0.002 (measurement noise) |

**Feature naming convention:**
```
V_bus{n}     voltage magnitude at bus n
P_bus{n}     active power at bus n
Q_bus{n}     reactive power at bus n
THETA_bus{n} voltage angle at bus n
```

---

## 5. Zone Topology

The 32 buses are partitioned into 4 zones based on grid topology. Each zone is assigned to one Jetson edge device.

| Zone | Buses | Features | Jetson | ZONE_ID |
|---|---|---|---|---|
| Zone 1 | 1 – 8 | 32 | Client 1 | `zone1` |
| Zone 2 | 9 – 17 | 36 | Client 2 | `zone2` |
| Zone 3 | 18 – 24 | 28 | Client 3 | `zone3` |
| Zone 4 | 25 – 32 | 32 | Client 4 | `zone4` |

All zones are **zero-padded to 36 features** before feeding to the model so all clients share an identical model architecture and FedAvg can average weights directly.

---

## 6. Model Architecture

LSTM encoder-decoder autoencoder trained on **normal data only**. At inference, high reconstruction error flags anomalies.

```
Input:  (30 timesteps × 36 features)
        │
        ├─ LSTM(32, tanh, return_sequences=True)
        ├─ LSTM(64, tanh, return_sequences=False)   ← bottleneck
        ├─ RepeatVector(30)
        ├─ LSTM(64, tanh, return_sequences=True)
        ├─ LSTM(32, tanh, return_sequences=True)
        └─ TimeDistributed(Dense(36))
Output: (30 timesteps × 36 features)   reconstruction

Loss:        MSE
Optimizer:   Adam(lr=0.0011)
Threshold:   99.4th percentile of reconstruction error on normal windows
```

---

## 7. Directory Structure

```
project/
├── IDS DATASET/
│   ├── IDS_TRAINING&TEST_DATA/
│   │   ├── zone1_train.csv ... zone4_train.csv   per-zone training sets
│   │   └── zone1_test.csv  ... zone4_test.csv    per-zone test sets
│   ├── simulation_with_der.csv                   raw simulation output (DER on)
│   ├── simulation_no_der.csv                     raw simulation output (DER off)
│   └── eda_simulation.ipynb                      exploratory data analysis
│
├── ML model/
│   ├── centralized_lstm.ipynb     centralized baseline (all zones, single model)
│   ├── lstm_intact_v4.ipynb       per-zone local models
│   ├── models/                    saved .keras models, scalers, thresholds
│   └── results/                   detection metrics, threshold sweeps, figures
│
└── FL/
    ├── Client/
    │   ├── Client.py                      FL client (NumPyClient, LSTM autoencoder)
    │   ├── requirements.txt
    │   ├── Dockerfile
    │   └── centralized_train_combined.csv train data (all zones, 8064 rows, 128 features)
    └── Server/
        ├── Server.py                      FL server (FedAvg, saves per-round weights)
        ├── test_server.py                 post-training evaluation script
        ├── requirements.txt
        ├── Dockerfile
        └── centralized_test_combined.csv  test data (all zones, 4032 rows, attack_label)
```

---

## 8. Baseline Results

### Centralized model (privacy-violating upper bound)

Trained on all 128 features from all zones combined.

```
Optimal threshold : 0.004599  (99.4th percentile)

               precision    recall  f1-score   support
       Normal       0.97      0.94      0.95      3371
Replay Attack       0.72      0.83      0.77       631

AUC-ROC : 0.9280
```

### Per-zone local models (no collaboration)

Each zone trains independently on its own data only.

| Zone | AUC-ROC | Precision (attack) | Recall (attack) | F1 (attack) |
|---|---|---|---|---|
| Zone 1 | 0.8477 | 0.44 | 0.52 | 0.47 |
| Zone 2 | 0.9029 | 0.44 | 0.69 | 0.54 |
| Zone 3 | 0.9227 | 0.74 | 0.75 | 0.74 |
| Zone 4 | 0.9847 | 0.68 | 0.94 | 0.79 |

Local models underperform on attack recall because each zone sees only a fraction of the grid — the FL global model is expected to close this gap by sharing learned representations across zones.

---

## 9. FL Implementation Details

| Parameter | Value |
|---|---|
| Framework | Flower (flwr) 1.5.0 |
| Strategy | FedAvg |
| FL rounds | 50 |
| Local epochs per round | 5 |
| Batch size | 32 |
| Window size | 30 timesteps |
| Min clients per round | 4 |
| Aggregation | Weighted average by number of training windows |

**Weight saving:** the server saves `fedavg_round_{n}_weights.npz` after every round and keeps `fedavg_final_weights.npz` updated as a convenience copy of the latest round. This lets you load any intermediate checkpoint.

**Evaluation:** `test_server.py` loads the final weights, rebuilds the model, evaluates on each zone's test features separately, and averages the reconstruction errors across zones to produce a single anomaly score per window.

---

## Step 1 — Build Docker Images

Build on a Jetson device (ARM64 native) or cross-compile with `--platform linux/arm64` from x86.

> **Note:** The base image `nvcr.io/nvidia/l4t-ml:r36.2.0-py3` is ARM64-only. Building on an x86 machine requires QEMU and may be slow. Building directly on a Jetson is recommended.

```bash
# Clone the repo on the build machine
git clone <repo-url>
cd Topology-Aware-Federated-Learning-for-Cyber-Physical-Intrusion-Detection-in-DER/FL

# Build server image
docker build --platform linux/arm64 \
  -t hamzakarim07/flwr_server_intact:latest \
  -f Server/Dockerfile Server

# Build client image
docker build --platform linux/arm64 \
  -t hamzakarim07/flwr_client_intact:latest \
  -f Client/Dockerfile Client
```

The Dockerfile automatically moves any `.csv` files into `/app/src/data/` during build so the Python scripts find them at the expected path.

---

## Step 2 — Push to Docker Hub

```bash
docker login
docker push hamzakarim07/flwr_server_intact:latest
docker push hamzakarim07/flwr_client_intact:latest
```

---

## Step 3 — Deploy on Edge Devices

### Devices

| Device | IP | Role |
|---|---|---|
| AGX 04 (c2sragx04) | `10.226.44.86` | FL Server |
| Nano 07 (c2srnano07) | `10.226.47.0` | Client — zone1 |
| Nano 08 (c2srnano08) | `10.226.47.108` | Client — zone2 |
| Nano 10 (hamzakarim) | `10.226.46.8` | Client — zone3 |
| Nano 13 (hamzakarim) | `10.226.47.64` | Client — zone4 |

### AGX 04 — pull and start server container

```bash
ssh jetson@10.226.44.86

docker pull hamzakarim07/flwr_server_intact:latest

docker run -d \
  --name flwr-server \
  --runtime=nvidia \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -p 8080:8080 \
  -v ~/fl/models:/app/src/models \
  -v ~/fl/results:/app/src/results \
  hamzakarim07/flwr_server_intact:latest
```

### Each Nano — pull client image once

```bash
# Run on each Nano (or SSH in)
docker pull hamzakarim07/flwr_client_intact:latest
```

> Client containers are started automatically by `start_clients.sh` — no manual `docker run` needed on the Nanos.

---

## Step 4 — `start_clients.sh` (already configured)

`start_clients.sh` in the project root is pre-configured with all device IPs. No changes needed — just run it.

If your SSH username differs from `jetson`, edit the one line at the top:

```bash
SSH_USER="jetson"    # change if needed
```

---

## Step 5 — Run FL Training

### 5a. Start the server (SSH into AGX 04)

```bash
ssh jetson@10.226.44.86
docker exec -it flwr-server bash
cd /app/src && python3 Server.py
```

Fill in the prompts (press Enter to accept defaults):

```
==================================================
  FL Server Configuration
==================================================
Port [8080]:
FL rounds [50]:
Min clients per round [4]:
Local epochs (sent to clients) [5]:
Model dir [/app/src/models]:
==================================================
```

The server is now waiting for all 4 Nanos to connect.

### 5b. Start all 4 clients from your laptop (one command)

```bash
bash start_clients.sh
```

The script SSHes into all 4 Nanos in parallel and starts each client container with the correct zone and server address — no prompts:

```
========================================
  Starting FL clients
  Server : AGX 04 @ 10.226.44.86:8080
========================================

  ✓ Nano 07 (10.226.47.0)   → zone1
  ✓ Nano 08 (10.226.47.108) → zone2
  ✓ Nano 10 (10.226.46.8)   → zone3
  ✓ Nano 13 (10.226.47.64)  → zone4

========================================
  All clients started — watch logs:
========================================
  ssh jetson@10.226.47.0 'docker logs -f flwr-client-zone1'
  ...
```

Training starts automatically on the server as soon as all 4 clients connect.

### 5c. Watch training progress

**Server terminal:**
```
Round   1 | Clients: 4 | Aggregated Loss: 0.012345
Round   2 | Clients: 4 | Aggregated Loss: 0.009876
...
Round  50 | Clients: 4 | Aggregated Loss: 0.003210
```

**Any client log:**
```bash
ssh jetson@10.226.47.0 'docker logs -f flwr-client-zone1'
```
```
Round 1 | Zone zone1 | Loss: 0.011234
Round 2 | Zone zone1 | Loss: 0.008901
...
```

Per-round weights are saved to `~/fl/models/` on the server host:
```
fedavg_round_1_weights.npz
...
fedavg_round_50_weights.npz
fedavg_final_weights.npz
```

---

## Step 6 — Evaluate the Global Model

Run on the **server Jetson** after training completes:

```bash
ssh jetson@10.226.44.86
docker exec -it flwr-server bash
cd /app/src && python3 test_server.py
```

This script:
1. Loads `fedavg_final_weights.npz`
2. Rebuilds the LSTM autoencoder
3. Evaluates on each zone's test features (same preprocessing as training)
4. Averages reconstruction errors across all 4 zones per window
5. Sets the anomaly threshold at the **99.4th percentile** of normal-window errors
6. Runs a threshold sweep over 14 percentile values

**Output files** (saved to `/app/src/results/`, mounted to `~/fl/results/` on the host):

| File | Contents |
|---|---|
| `fedavg_final_summary.txt` | Precision, recall, F1, AUC-ROC at optimal threshold |
| `fedavg_threshold_sweep.txt` | Metrics across 14 percentile thresholds |
| `fedavg_global_model.keras` | Full saved Keras model |
| `fedavg_threshold.npy` | Threshold value for deployment |
| `fedavg_fig_detection.png` | Error distribution + error-over-time plot |

---

## 15. Dependencies

**Python packages** (installed inside the Docker image):

```
flwr==1.5.0
tf-keras
numpy<2
pandas
scikit-learn
matplotlib
joblib
```

**Base image:** `nvcr.io/nvidia/l4t-ml:r36.2.0-py3`
- JetPack 6 (L4T r36.2)
- Python 3.10
- TensorFlow 2.16 (NVIDIA Jetson build, GPU-accelerated)
- CUDA 12.2

**Hardware:** NVIDIA Jetson (any JetPack 6 compatible board) × 5
- 1 server device
- 4 client devices (one per zone)
