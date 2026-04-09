#!/bin/bash
# Run all 8 methods
# Usage: ./run_all.sh <epoch> <phase_name>
# Phase 1: epoch=2 (validation run)
# Phase 2: epoch=20 (full run)

EPOCH=${1:-2}
PHASE=${2:-"test"}
SEED=2

eval "$(conda shell.bash hook)"
conda activate bilevel

METHODS=("stocbio" "ttsa" "saba" "ma-soba" "bo-rep" "sustain" "vrbo" "accbo")

echo "=== Phase: $PHASE | Epoch: $EPOCH | Seed: $SEED | $(date) ==="

# Run 4 methods at a time
for batch in 0 1; do
    start=$((batch * 4))
    pids=()
    for i in $(seq 0 3); do
        idx=$((start + i))
        if [ $idx -ge ${#METHODS[@]} ]; then break; fi
        method=${METHODS[$idx]}
        logfile="logs/${method}_${PHASE}.log"
        echo "Starting $method -> $logfile"
        PYTHONUNBUFFERED=1 python main.py --methods "$method" --epoch $EPOCH --seed $SEED > "$logfile" 2>&1 &
        pids+=($!)
    done

    # Wait for this batch
    for pid in "${pids[@]}"; do
        wait $pid
        status=$?
        echo "PID $pid exited with status $status"
    done
    echo "=== Batch $((batch+1)) done at $(date) ==="
done

echo "=== All methods done for phase $PHASE at $(date) ==="
