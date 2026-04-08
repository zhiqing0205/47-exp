# Bilevel Optimization under Unbounded Smoothness

This repository implements and benchmarks multiple bilevel optimization algorithms on **data hyper-cleaning** tasks using the SNLI dataset with RNN models.

> Based on the paper: *"Bilevel Optimization under Unbounded Smoothness: A New Algorithm and Convergence Analysis"*

## Problem Setting

Bilevel optimization for data hyper-cleaning: the upper-level optimizes sample weights (lambda) to minimize validation loss, while the lower-level trains an RNN classifier weighted by these sample weights. Label noise is injected at a configurable rate to simulate corrupted training data.

## Methods

| Method | File | Description |
|--------|------|-------------|
| StocBio | `methods/stocbio.py` | Stochastic bilevel optimizer with Neumann series |
| TTSA | `methods/ttsa.py` | Two-timescale stochastic approximation |
| SABA | `methods/saba.py` | SAGA-based bilevel approximation |
| MA-SOBA | `methods/ma_soba.py` | Momentum-assisted single-loop bilevel optimizer |
| BO-REP | `methods/bo_rep.py` | Bilevel optimizer with initialization refinement and periodic updates |
| SUSTAIN | `methods/sustain.py` | Variance-reduced STORM-based bilevel optimizer |
| VRBO | `methods/vrbo.py` | Variance-reduced bilevel optimization with SPIDER |
| AccBO | `methods/accbo.py` | Accelerated bilevel optimizer with Nesterov momentum |
| MEHA | `methods/MEHA.py` | Penalty-based bilevel method (single-loop) |
| NOVA2 | `methods/NOVA2.py` | Penalty-based bilevel method with momentum |
| S-PNGBIO | `methods/s_pngbio.py` | Penalty-based normalized gradient bilevel method |

## Project Structure

```
.
├── main.py              # Entry point, argument parsing, training loop
├── data_loader.py       # SNLI / Sent140 dataset loading and preprocessing
├── methods/
│   ├── RNN_net.py       # RNN / NLIRNN model definitions
│   ├── stocbio.py
│   ├── ttsa.py
│   ├── saba.py
│   ├── ma_soba.py
│   ├── bo_rep.py
│   ├── sustain.py
│   ├── vrbo.py
│   ├── accbo.py
│   ├── MEHA.py
│   ├── NOVA2.py
│   └── s_pngbio.py
├── data/                # Pre-processed SNLI pickle files
├── logs/                # Training logs and results
├── requirements.txt
└── README.md
```

## Setup

```bash
conda create -n bilevel python=3.9 -y
conda activate bilevel
pip install -r requirements.txt
```

## Usage

Run a specific method:

```bash
python main.py --methods <method_name> --epoch 20 --seed 2
```

Available method names: `stocbio`, `ttsa`, `saba`, `ma-soba`, `bo-rep`, `sustain`, `vrbo`, `accbo`, `meha`, `nova2`, `s-pngbio`

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--methods` | `stocbio` | Method to run |
| `--epoch` | `20` | Number of outer epochs |
| `--seed` | `2` | Random seed |
| `--noise_rate` | `0.1` | Label noise rate |
| `--inner_batch_size` | `16` | Training batch size |
| `--data` | `snli` | Dataset (snli) |

### Run All Methods

```bash
# Serial execution (recommended)
for method in stocbio ttsa saba ma-soba bo-rep sustain vrbo accbo meha nova2 s-pngbio; do
    python main.py --methods $method --epoch 20 --seed 2
done
```

## Data

The SNLI dataset is pre-processed and stored as pickle files in `data/`. The pre-processing includes GloVe word embeddings (300d) and label noise injection.

## Output

Training logs are saved to `logs/` with format:
`{method}_outlr{lr}_inlr{lr}_seed{seed}_{date}.txt`

Each log contains: experiment config, train/test accuracy, train/test loss, and total time.
