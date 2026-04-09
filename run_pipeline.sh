#!/bin/bash
# Full pipeline: wait for batch1, run batch2 (epoch=2), then all 8 (epoch=20)
set -e

eval "$(conda shell.bash hook)"
conda activate bilevel

SEED=2

mkdir -p logs

echo "=== [1/3] Waiting for batch1 (stocbio,ttsa,saba,ma-soba epoch=2) to finish ==="
while ps aux | grep "python main.py" | grep -v grep | grep -q .; do
    sleep 30
    echo "  Still running $(ps aux | grep 'python main.py' | grep -v grep | wc -l) python processes at $(date +%H:%M:%S)"
done
echo "=== Batch1 done at $(date) ==="

# Show batch1 results
for f in logs/stocbio_run.log logs/ttsa_run.log logs/saba_run.log logs/ma_soba_run.log; do
    echo "  $f: $(tail -1 "$f" 2>/dev/null)"
done

echo ""
echo "=== [2/3] Running batch2: bo-rep, sustain, vrbo, accbo (epoch=2) ==="
BATCH2=("bo-rep" "sustain" "vrbo" "accbo")
pids=()
for method in "${BATCH2[@]}"; do
    logfile="logs/${method}_run.log"
    echo "  Starting $method -> $logfile"
    PYTHONUNBUFFERED=1 python main.py --methods "$method" --epoch 2 --seed $SEED > "$logfile" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait $pid
    echo "  PID $pid done (exit=$?)"
done
echo "=== Batch2 done at $(date) ==="
for method in "${BATCH2[@]}"; do
    echo "  $method: $(tail -1 "logs/${method}_run.log" 2>/dev/null)"
done

echo ""
echo "=== [3/3] Running ALL 8 methods with epoch=20 ==="
ALL_METHODS=("stocbio" "ttsa" "saba" "ma-soba" "bo-rep" "sustain" "vrbo" "accbo")

# Run in 2 batches of 4 to avoid GPU OOM
for batch in 0 1; do
    start=$((batch * 4))
    pids=()
    for i in $(seq 0 3); do
        idx=$((start + i))
        if [ $idx -ge ${#ALL_METHODS[@]} ]; then break; fi
        method=${ALL_METHODS[$idx]}
        logfile="logs/${method}_epoch20.log"
        echo "  Starting $method (epoch=20) -> $logfile"
        PYTHONUNBUFFERED=1 python main.py --methods "$method" --epoch 20 --seed $SEED > "$logfile" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait $pid
        echo "  PID $pid done (exit=$?)"
    done
    echo "  Batch $((batch+1)) of epoch=20 done at $(date)"
done

echo ""
echo "=========================================="
echo "=== ALL DONE at $(date) ==="
echo "=========================================="
echo ""
echo "=== Epoch=20 Results ==="
for method in "${ALL_METHODS[@]}"; do
    logfile="logs/${method}_epoch20.log"
    echo "--- $method ---"
    tail -5 "$logfile" 2>/dev/null
    echo ""
done
