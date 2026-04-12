#!/bin/bash
# One-click serial run for all 8 baseline methods
# Usage: bash run.sh [epoch] [seed]

cd "$(dirname "$0")"

eval "$(conda shell.bash hook)" 2>/dev/null
conda activate bilevel 2>/dev/null

EPOCH=${1:-20}
SEED=${2:-2}
METHODS=("stocbio" "ttsa" "saba" "ma-soba" "bo-rep" "sustain" "vrbo" "accbo")

mkdir -p logs

echo "=== Bilevel Optimization Experiment ==="
echo "Methods: ${METHODS[*]}"
echo "Epoch: $EPOCH | Seed: $SEED | Start: $(date)"
echo ""

for m in "${METHODS[@]}"; do
    echo ">>> [$m] start $(date)"
    PYTHONUNBUFFERED=1 python main.py --methods "$m" --epoch $EPOCH --seed $SEED 2>&1 | tee "logs/${m}_e${EPOCH}.log"
    status=$?
    if [ $status -ne 0 ]; then
        echo "!!! [$m] FAILED (exit=$status)"
    else
        echo "<<< [$m] done $(date)"
    fi
    echo ""
done

echo "=== All done $(date) ==="
python stats.py 2>/dev/null && echo "Report: results.md"
