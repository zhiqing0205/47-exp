#!/bin/bash
# Run selected methods serially
# Usage: bash methods/run.sh

cd "$(dirname "$0")/.."

eval "$(conda shell.bash hook)"
conda activate bilevel

EPOCH=20
SEED=2
methods=("nova2" "nova3" "meha" "s-pngbio" "accbo" "bo-rep")

mkdir -p logs

for m in "${methods[@]}"; do
    echo ">>> 正在运行方法: $m (epoch=$EPOCH, seed=$SEED) $(date)"
    PYTHONUNBUFFERED=1 python main.py --methods "$m" --epoch $EPOCH --seed $SEED 2>&1 | tee "logs/${m}_e${EPOCH}.log"
    echo "<<< $m 完成 exit=$? $(date)"
    echo ""
done

echo "=== 全部完成 $(date) ==="
python stats.py --output results.md
