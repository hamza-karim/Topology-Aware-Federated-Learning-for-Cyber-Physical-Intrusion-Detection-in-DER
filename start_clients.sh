#!/bin/bash
# start_clients.sh — run from your laptop to start or stop all 4 FL clients
#
# Usage:
#   bash start_clients.sh          # start all clients
#   bash start_clients.sh stop     # stop all client containers

SERVER_IP="10.226.44.86"
IMAGE="hamzakarim07/flwr_client_intact:latest"

#           User           Host/IP          Zone    Display
USERS=(  "c2srnano07"        "c2srnano08"        "hamzakarim"   "hamzakarim"  )
HOSTS=(  "10.226.47.0"    "10.226.47.108"    "10.226.46.8"  "10.226.47.64")
ZONES=(  "zone1"         "zone2"         "zone3"         "zone4"       )
NAMES=(  "Nano 07"       "Nano 08"       "Nano 10"       "Nano 13"     )

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"

stop_client() {
    local user=$1
    local host=$2
    local zone=$3
    local display=$4
    local name="flwr-client-${zone}"

    $SSH ${user}@${host} "docker rm -f ${name} 2>/dev/null && echo removed"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ ${display} (${host}) → stopped${NC}"
    else
        echo -e "${RED}  ✗ ${display} (${host}) → failed (SSH key not set up?)${NC}"
    fi
}

start_client() {
    local user=$1
    local host=$2
    local zone=$3
    local display=$4
    local name="flwr-client-${zone}"

    $SSH ${user}@${host} "
        docker rm -f ${name} 2>/dev/null
        docker run -d \
          --name ${name} \
          --runtime=nvidia \
          --gpus all \
          -e NVIDIA_VISIBLE_DEVICES=all \
          -e ZONE_ID=${zone} \
          -e SERVER_ADDRESS=${SERVER_IP}:8080 \
          -v ~/fl/models:/app/src/models \
          ${IMAGE}
    "

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ ${display} (${host}) → ${zone}${NC}"
    else
        echo -e "${RED}  ✗ ${display} (${host}) → failed (SSH key not set up?)${NC}"
    fi
}

if [ "$1" == "stop" ]; then
    echo ""
    echo -e "${CYAN}========================================"
    echo "  Stopping FL clients"
    echo -e "========================================${NC}"
    echo ""

    for i in "${!HOSTS[@]}"; do
        stop_client "${USERS[$i]}" "${HOSTS[$i]}" "${ZONES[$i]}" "${NAMES[$i]}" &
    done

    wait
    echo ""

else
    echo ""
    echo -e "${CYAN}========================================"
    echo "  Starting FL clients"
    echo "  Server : AGX 04 @ ${SERVER_IP}:8080"
    echo -e "========================================${NC}"
    echo ""

    for i in "${!HOSTS[@]}"; do
        start_client "${USERS[$i]}" "${HOSTS[$i]}" "${ZONES[$i]}" "${NAMES[$i]}" &
    done

    wait

    echo ""
    echo -e "${CYAN}========================================"
    echo "  All clients started — watch logs:"
    echo -e "========================================${NC}"
    for i in "${!HOSTS[@]}"; do
        echo "  ssh ${USERS[$i]}@${HOSTS[$i]} 'docker logs -f flwr-client-${ZONES[$i]}'"
    done
    echo ""
fi
