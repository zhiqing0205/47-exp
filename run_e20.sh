#!/bin/bash
# Run all 8 methods, epoch=20, 2 at a time
eval "$(conda shell.bash hook)"
conda activate bilevel
mkdir -p logs

SEED=2
EPOCH=20
METHODS=("stocbio" "ttsa" "saba" "ma-soba" "bo-rep" "sustain" "vrbo" "accbo")

echo "=== epoch=$EPOCH seed=$SEED start $(date) ==="

for ((i=0; i<${#METHODS[@]}; i+=2)); do
    m1=${METHODS[$i]}
    m2=${METHODS[$((i+1))]}
    echo ""
    echo ">>> Batch: $m1 + $m2 start $(date)"

    PYTHONUNBUFFERED=1 python main.py --methods "$m1" --epoch $EPOCH --seed $SEED > "logs/${m1}_e20.log" 2>&1 &
    pid1=$!
    PYTHONUNBUFFERED=1 python main.py --methods "$m2" --epoch $EPOCH --seed $SEED > "logs/${m2}_e20.log" 2>&1 &
    pid2=$!

    wait $pid1
    s1=$?
    wait $pid2
    s2=$?

    echo "<<< $m1 exit=$s1, $m2 exit=$s2 at $(date)"
    echo "  $m1 last: $(tail -1 "logs/${m1}_e20.log")"
    echo "  $m2 last: $(tail -1 "logs/${m2}_e20.log")"
done

echo ""
echo "=== ALL DONE $(date) ==="
