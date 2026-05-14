#!/bin/bash
# fetch_fl_results.sh — copy FL results from AGX 04 to laptop
#
# Usage:
#   ./fetch_fl_results.sh          # fetch latest results
#   ./fetch_fl_results.sh clean    # delete all previously fetched results

SERVER_USER="c2sragx04"
SERVER_IP="10.226.44.86"
REMOTE_RESULTS="~/fl/results"
REMOTE_MODELS="~/fl/models"
LOCAL_DIR="ML model/results/fl"

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

if [ "$1" == "clean" ]; then
    echo ""
    echo -e "${CYAN}========================================"
    echo "  Cleaning fetched FL results"
    echo "  ${LOCAL_DIR}"
    echo -e "========================================${NC}"
    echo ""
    if [ -d "$LOCAL_DIR" ]; then
        rm -rf "$LOCAL_DIR"
        echo -e "${GREEN}  ✓ ${LOCAL_DIR} removed${NC}"
    else
        echo "  Nothing to clean — ${LOCAL_DIR} does not exist"
    fi
    echo ""
    exit 0
fi

mkdir -p "$LOCAL_DIR"

echo ""
echo -e "${CYAN}========================================"
echo "  Fetching FL results from AGX 04"
echo "  ${SERVER_USER}@${SERVER_IP}"
echo -e "========================================${NC}"
echo ""

# Copy all result files for all strategies
echo "  Copying results/..."
scp -o BatchMode=yes -o ConnectTimeout=10 \
    "${SERVER_USER}@${SERVER_IP}:${REMOTE_RESULTS}/fedavg_*" \
    "${SERVER_USER}@${SERVER_IP}:${REMOTE_RESULTS}/fedprox_*" \
    "${SERVER_USER}@${SERVER_IP}:${REMOTE_RESULTS}/fedadam_*" \
    "${SERVER_USER}@${SERVER_IP}:${REMOTE_RESULTS}/intact_*" \
    "$LOCAL_DIR/" 2>/dev/null
echo -e "${GREEN}  [OK] results copied (missing files skipped)${NC}"

# Copy training logs and run configs for all strategies
echo "  Copying training logs and run configs..."
scp -o BatchMode=yes -o ConnectTimeout=10 \
    "${SERVER_USER}@${SERVER_IP}:${REMOTE_MODELS}/fedavg_training_log.csv" \
    "${SERVER_USER}@${SERVER_IP}:${REMOTE_MODELS}/fedavg_run_config.json" \
    "$LOCAL_DIR/" 2>/dev/null
scp -o BatchMode=yes -o ConnectTimeout=10 \
    "${SERVER_USER}@${SERVER_IP}:${REMOTE_MODELS}/fedprox_*_training_log.csv" \
    "${SERVER_USER}@${SERVER_IP}:${REMOTE_MODELS}/fedprox_*_run_config.json" \
    "$LOCAL_DIR/" 2>/dev/null
scp -o BatchMode=yes -o ConnectTimeout=10 \
    "${SERVER_USER}@${SERVER_IP}:${REMOTE_MODELS}/fedadam_*_training_log.csv" \
    "${SERVER_USER}@${SERVER_IP}:${REMOTE_MODELS}/fedadam_*_run_config.json" \
    "$LOCAL_DIR/" 2>/dev/null
scp -o BatchMode=yes -o ConnectTimeout=10 \
    "${SERVER_USER}@${SERVER_IP}:${REMOTE_MODELS}/intact_training_log.csv" \
    "${SERVER_USER}@${SERVER_IP}:${REMOTE_MODELS}/intact_run_config.json" \
    "$LOCAL_DIR/" 2>/dev/null
echo -e "${GREEN}  [OK] logs + configs copied (missing files skipped)${NC}"

echo ""
echo -e "${CYAN}Files in ${LOCAL_DIR}:${NC}"
ls -lh "$LOCAL_DIR" 2>/dev/null | tail -n +2
echo ""
