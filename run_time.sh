#!/bin/bash
# Serial run all methods with main_time.py (includes per-epoch timing)
# Usage: bash run_time.sh [epoch] [seed]

cd "$(dirname "$0")"

eval "$(conda shell.bash hook)"
conda activate bilevel

EPOCH=${1:-20}
SEED=${2:-2}
METHODS=("stocbio" "ttsa" "saba" "ma-soba" "bo-rep" "sustain" "vrbo" "accbo")

mkdir -p logs

echo "=== main_time.py serial run | epoch=$EPOCH seed=$SEED | $(date) ==="

for m in "${METHODS[@]}"; do
    echo ""
    echo ">>> $m start $(date)"
    PYTHONUNBUFFERED=1 python main_time.py --methods "$m" --epoch $EPOCH --seed $SEED 2>&1 | tee "logs/${m}_time_e${EPOCH}.log"
    echo "<<< $m exit=$? $(date)"
done

echo ""
echo "=== ALL DONE $(date) ==="
python stats.py --output results_time.md 2>/dev/null
