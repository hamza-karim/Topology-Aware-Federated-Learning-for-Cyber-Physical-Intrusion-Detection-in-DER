#!/bin/bash
# start_clients.sh — run from your laptop to start all 4 FL clients
#
# Usage:  bash start_clients.sh
#
# Prerequisites:
#   - SSH passwordless auth already configured (no password needed)
#   - Docker image already pulled on each Nano:
#       docker pull hamzakarim07/flwr_client_intact:latest
#   - Server (AGX) container already running and Server.py started

# ── Device configuration ──────────────────────────────────────
SSH_USER="jetson"                 # change if your SSH username differs
SERVER_IP="10.226.44.86"         # AGX 04 — FL server
IMAGE="hamzakarim07/flwr_client_intact:latest"

#         IP               Zone     Hostname
CLIENT_IPS=("10.226.47.0"  "10.226.47.108"  "10.226.46.8"   "10.226.47.64")
ZONES=(    "zone1"          "zone2"           "zone3"          "zone4"      )
NAMES=(    "Nano 07"        "Nano 08"         "Nano 10"        "Nano 13"    )
# ─────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

start_client() {
    local ip=$1
    local zone=$2
    local display=$3
    local name="flwr-client-${zone}"

    ssh -o StrictHostKeyChecking=no \
        -o ConnectTimeout=10 \
        "${SSH_USER}@${ip}" bash <<EOF
docker rm -f ${name} 2>/dev/null && echo "  Removed old container"

docker run -d \
  --name ${name} \
  --runtime=nvidia \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e ZONE_ID=${zone} \
  -e SERVER_ADDRESS=${SERVER_IP}:8080 \
  -v ~/fl/models:/app/src/models \
  ${IMAGE}
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ ${display} (${ip}) → ${zone}${NC}"
    else
        echo -e "${RED}  ✗ ${display} (${ip}) → failed${NC}"
    fi
}

echo ""
echo -e "${CYAN}========================================"
echo "  Starting FL clients"
echo "  Server : AGX 04 @ ${SERVER_IP}:8080"
echo -e "========================================${NC}"
echo ""

for i in "${!CLIENT_IPS[@]}"; do
    start_client "${CLIENT_IPS[$i]}" "${ZONES[$i]}" "${NAMES[$i]}" &
done

wait

echo ""
echo -e "${CYAN}========================================"
echo "  All clients started — watch logs:"
echo -e "========================================${NC}"
for i in "${!CLIENT_IPS[@]}"; do
    echo "  ssh ${SSH_USER}@${CLIENT_IPS[$i]} 'docker logs -f flwr-client-${ZONES[$i]}'"
done
echo ""
