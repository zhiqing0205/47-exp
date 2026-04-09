#!/bin/bash
# Serial execution of all 8 methods
# Usage: ./run_serial.sh <epoch> <tag>
set -e

EPOCH=${1:-2}
TAG=${2:-"e${EPOCH}"}
SEED=2

eval "$(conda shell.bash hook)"
conda activate bilevel

METHODS=("stocbio" "ttsa" "saba" "ma-soba" "bo-rep" "sustain" "vrbo" "accbo")

mkdir -p logs

echo "=== Serial run: epoch=$EPOCH seed=$SEED tag=$TAG | $(date) ==="

for method in "${METHODS[@]}"; do
    logfile="logs/${method}_${TAG}.log"
    echo ""
    echo ">>> Starting $method at $(date) -> $logfile"
    PYTHONUNBUFFERED=1 python main.py --methods "$method" --epoch $EPOCH --seed $SEED 2>&1 | tee "$logfile"
    status=$?
    if [ $status -ne 0 ]; then
        echo "!!! $method FAILED with exit code $status"
    else
        echo "<<< $method DONE at $(date)"
    fi
done

echo ""
echo "=========================================="
echo "=== ALL $TAG DONE at $(date) ==="
echo "=========================================="
echo ""
echo "=== Summary ==="
for method in "${METHODS[@]}"; do
    logfile="logs/${method}_${TAG}.log"
    last_test=$(grep "Test Acc" "$logfile" 2>/dev/null | tail -1)
    echo "  $method: $last_test"
done
