#!/bin/bash
# Multi-seed reproducibility test with v20 best params
# Run 7 seeds × 20 epochs, report all results at the end

echo "============================================================"
echo "  NOVA3 Multi-Seed Reproducibility Test"
echo "  Params: v20 T92 best (75.96% with seed=2)"
echo "  Seeds: 0, 1, 2, 42, 123, 2024, 2026"
echo "============================================================"
echo ""

results_file="logs/multiseed_results.txt"
> "$results_file"

for seed in 0 1 2 42 123 2024 2026; do
    echo "--- Running seed=$seed ---"
    python validate_v22.py --seed $seed 2>&1 | tee -a "logs/multiseed_seed${seed}.log"
    echo ""
done

echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
echo ""
grep "Best:" logs/multiseed_seed*.log | sort -t: -k3 -rn
