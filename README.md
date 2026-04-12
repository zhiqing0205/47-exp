# Bilevel Optimization under Unbounded Smoothness

Benchmarking bilevel optimization algorithms on **data hyper-cleaning** (SNLI dataset + RNN).

> Based on: *"Bilevel Optimization under Unbounded Smoothness: A New Algorithm and Convergence Analysis"*

## Problem

Upper-level optimizes sample weights to minimize validation loss; lower-level trains an RNN classifier weighted by these sample weights. Label noise is injected to simulate corrupted training data.

## Methods

| Method | File | Type |
|--------|------|------|
| StocBio | `methods/stocbio.py` | Neumann series |
| TTSA | `methods/ttsa.py` | Two-timescale |
| SABA | `methods/saba.py` | SAGA-based |
| MA-SOBA | `methods/ma_soba.py` | Momentum-assisted |
| BO-REP | `methods/bo_rep.py` | Periodic updates |
| SUSTAIN | `methods/sustain.py` | STORM-based |
| VRBO | `methods/vrbo.py` | SPIDER variance reduction |
| AccBO | `methods/accbo.py` | Nesterov acceleration |
| MEHA | `methods/MEHA.py` | Penalty-based |
| NOVA2 | `methods/NOVA2.py` | Penalty + momentum |
| NOVA3 | `methods/NOVA3.py` | Penalty + normalized gradient |
| S-PNGBIO | `methods/s_pngbio.py` | Penalty + normalized gradient |

## Project Structure

```
.
├── main.py              # Entry point (training + per-epoch timing)
├── data_loader.py       # SNLI dataset loading
├── methods/
│   ├── RNN_net.py       # RNN model definitions
│   ├── run.sh           # Batch run script
│   └── *.py             # Method implementations
├── data/                # Pre-processed SNLI pkl files (Git LFS)
├── logs/                # Training outputs (.txt results, .pt checkpoints, .log console)
├── figures/             # Visualization output
├── stats.py             # Generate results report (markdown)
├── plot_results.py      # Generate visualization figures
├── tune_nova3.py        # Optuna hyperparameter tuning for NOVA3
└── requirements.txt
```

## Setup

```bash
conda create -n bilevel python=3.9 -y
conda activate bilevel
pip install -r requirements.txt
```

## Usage

### Run a single method

```bash
python main.py --methods <method> --epoch 20 --seed 2
```

Available: `stocbio`, `ttsa`, `saba`, `ma-soba`, `bo-rep`, `sustain`, `vrbo`, `accbo`, `meha`, `nova2`, `nova3`, `s-pngbio`

### Run all methods

```bash
bash methods/run.sh
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--methods` | `stocbio` | Method name |
| `--epoch` | `20` | Number of epochs |
| `--seed` | `2` | Random seed |
| `--noise_rate` | `0.1` | Label noise rate |
| `--inner_batch_size` | `512` | Training batch size |

### Generate results report

```bash
python stats.py                      # -> results.md
python stats.py --output report.md   # custom output path
```

### Generate figures

```bash
python plot_results.py    # -> figures/fig1~fig6
```

### Hyperparameter tuning (NOVA3)

```bash
python tune_nova3.py --n_trials 50 --epoch 5     # quick search
python tune_nova3.py --n_trials 200 --epoch 20    # thorough search
```

## Output Format

Each run produces 3 files in `logs/`:

| File | Content |
|------|---------|
| `{method}_outlr{lr}_inlr{lr}_seed{s}_{date}.txt` | Results dict (config, acc, loss, time, epoch_times) |
| `{method}_outlr{lr}_inlr{lr}_seed{s}_{date}.pt` | PyTorch checkpoint (train/test acc & loss arrays) |
| `{method}_e20.log` | Console output (training steps, per-epoch summary) |

## Console Output

Per epoch:
```
[epoch/epochs]:0/20
Step 0 | Task Loss: 6.17 | Acc: 0.00
Step 100 | Task Loss: 1.81 | Acc: 0.32
...
  Train Loss: 1.1486 | Train Acc: 0.3893
  Test  Loss: 1.0562 | Test  Acc: 0.4417
  Time: train=85.3s test=12.1s total=97.4s cumul=97.4s (0.03h)
```
