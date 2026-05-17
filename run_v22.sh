#!/bin/bash
# One-click script: Phase 1 (multi-seed) + Phase 2 (gamma anneal search)

echo "============================================================"
echo "  Phase 1: Multi-seed validation with v20 best params"
echo "  (7 seeds × 20 epochs, looking for 76%+)"
echo "============================================================"

best_acc=0
for seed in 0 1 2 42 123 2024 2026; do
    echo ""
    echo "--- seed=$seed ---"
    python validate_v22.py --seed $seed
done

echo ""
echo "============================================================"
echo "  Phase 2: Gamma annealing search (760→380)"
echo "  (100 trials × 20 epochs)"
echo "============================================================"
echo ""
python tune_nova3.py --n_trials 100

echo ""
echo "============================================================"
echo "  ALL DONE"
echo "============================================================"
